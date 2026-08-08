"""Work out where on the floor each of a model's numbered keypoints sits.

basketball-court-detection-2 returns 33 keypoints named "01" to "41" and nothing
says what those numbers mean. Without that table the keypoints cannot produce a
homography, however accurate they are -- and this model is the accurate one, at
19px on Summer League against 60px for the one whose table we do have.

So bootstrap it. The reloc2 model has a known table (court_model.py), which gives
a court-to-image homography for a frame. Push the other model's keypoints back
through its inverse and they land in court metres. Do that on many frames and
average.

The bootstrap is much less accurate than the answer needs to be, and that is
fine. Court landmarks are metres apart; the decode only has to say *which*
landmark each number is, after which the exact canonical coordinate replaces the
estimate. A metre of slop identifies a landmark perfectly well.

    python pipeline/decode_keypoints.py --video web/media/nba.mp4

What makes it trustworthy is the spread, not the mean: a keypoint index that is
genuinely one court position lands in the same place from every frame. One that
scatters is either misdetected or means something that is not a fixed point, and
gets dropped rather than guessed.
"""

import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

from config import secret
from court_model import COURT_LENGTH, COURT_WIDTH, RELOC2

ENDPOINT = "https://detect.roboflow.com"
PROJECT = "basketball-court-detection-2"

# Every landmark a numbered keypoint could plausibly be, in metres. Wider than
# court_model's eighteen because the other model has thirty-three, so it must
# name things reloc2 does not.
def candidate_landmarks():
    L, W = COURT_LENGTH, COURT_WIDTH
    named = {}
    for x, tag in ((0.0, "w"), (L, "e")):
        for y, side in ((0.0, "corner_s"), (0.91, "3pt_s"), (5.18, "key_s"),
                        (10.0, "key_n"), (14.1, "3pt_n"), (W, "corner_n")):
            named[f"{tag}_{side}"] = (x, y)
    for x, tag in ((5.79, "w"), (L - 5.79, "e")):
        named[f"{tag}_ft_s"] = (x, 5.18)
        named[f"{tag}_ft_n"] = (x, 10.0)
        named[f"{tag}_ft_mid"] = (x, 7.59)
    named["half_s"] = (L / 2, 0.0)
    named["half_n"] = (L / 2, W)
    named["centre"] = (L / 2, W / 2)
    named["circle_s"] = (L / 2, W / 2 - 1.8)
    named["circle_n"] = (L / 2, W / 2 + 1.8)
    return named


def hosted_keypoints(image, version, api_key, box_conf, kp_conf):
    buf = base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]).decode()
    r = requests.post(f"{ENDPOINT}/{PROJECT}/{version}",
                      params={"api_key": api_key, "confidence": box_conf}, data=buf,
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=90)
    r.raise_for_status()
    preds = sorted(r.json().get("predictions", []), key=lambda p: -p["confidence"])
    if not preds:
        return {}
    return {k.get("class_name") or k.get("class"): (k["x"], k["y"])
            for k in preds[0]["keypoints"] if k.get("confidence", 0) >= kp_conf}


def reference_homography(model, image, kp_conf, min_points=6, max_rms=25.0):
    """court -> image, from the model whose table we already know."""
    res = model.predict(image, verbose=False)[0]
    if res.keypoints is None or res.keypoints.conf is None or not len(res.keypoints.conf):
        return None
    i = int(res.boxes.conf.cpu().numpy().argmax()) if res.boxes is not None else 0
    xy = res.keypoints.xy[i].cpu().numpy()
    conf = res.keypoints.conf[i].cpu().numpy()

    src, dst = [], []
    for k, ((x, y), c) in enumerate(zip(xy, conf)):
        if c >= kp_conf and k < len(RELOC2):
            src.append(RELOC2[k])
            dst.append((float(x), float(y)))
    if len(src) < min_points:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 8.0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(np.float32(src).reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt((np.linalg.norm(proj - np.float32(dst), axis=1) ** 2).mean()))
    return H if rms <= max_rms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--reference", default="runs/pose/out/train/court/weights/best.pt")
    ap.add_argument("--version", type=int, default=22)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--box-conf", type=int, default=20)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--max-spread", type=float, default=1.5,
                    help="metres of scatter above which a keypoint is not one place")
    ap.add_argument("--out", default="out/keypoint_map.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    key = secret("ROBOFLOW_API_KEY")
    ref = YOLO(args.reference)
    landmarks = candidate_landmarks()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    seen = defaultdict(list)
    used = 0
    for idx in range(0, total, args.every):
        if used >= args.limit:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        H = reference_homography(ref, frame, args.kp_conf)
        if H is None:
            continue
        target = hosted_keypoints(frame, args.version, key, args.box_conf, args.kp_conf)
        if not target:
            continue
        Hinv = np.linalg.inv(H)
        pts = np.float32(list(target.values())).reshape(-1, 1, 2)
        court = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
        for name, (cx, cy) in zip(target, court):
            if np.isfinite(cx) and np.isfinite(cy):
                seen[name].append((float(cx), float(cy)))
        used += 1
        if used % 10 == 0:
            print(f"  {used} frames decoded...")
    cap.release()
    print(f"used {used} frames where the reference model solved cleanly\n")

    table, rejected = {}, []
    print(f"{'kp':>4} {'n':>4} {'median (m)':>18} {'spread':>8}  nearest landmark")
    for name, pts in sorted(seen.items(), key=lambda kv: int(kv[0])):
        p = np.array(pts)
        if len(p) < 4:
            rejected.append((name, len(p), None, "too few sightings"))
            continue
        med = np.median(p, axis=0)
        spread = float(np.median(np.linalg.norm(p - med, axis=1)))
        best, dist = min(((k, float(np.hypot(*(np.array(v) - med))))
                          for k, v in landmarks.items()), key=lambda t: t[1])
        flag = ""
        if spread > args.max_spread:
            flag = "  <- scattered, dropped"
            rejected.append((name, len(p), spread, "scattered"))
        elif dist > 2.0:
            flag = f"  <- {dist:.1f}m from anything, dropped"
            rejected.append((name, len(p), spread, "no landmark"))
        else:
            table[name] = {"landmark": best, "court": list(landmarks[best]),
                           "estimate": [round(float(med[0]), 2), round(float(med[1]), 2)],
                           "spread_m": round(spread, 2), "sightings": len(p)}
        print(f"{name:>4} {len(p):>4} {str((round(med[0],1), round(med[1],1))):>18} "
              f"{spread:7.2f}m  {best} ({dist:.1f}m){flag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "model": f"{PROJECT}/{args.version}",
        "decoded_via": args.reference, "frames_used": used,
        "keypoints": table,
    }, indent=2))
    print(f"\n{len(table)} of {len(seen)} keypoints decoded, {len(rejected)} dropped")
    print(f"wrote {args.out}")
    if table:
        worst = max(table.values(), key=lambda v: v["spread_m"])
        print(f"worst accepted scatter {worst['spread_m']}m, against landmarks metres apart --")
        print("the decode only has to name the landmark, and the exact court coordinate")
        print("then replaces the estimate.")


if __name__ == "__main__":
    main()
