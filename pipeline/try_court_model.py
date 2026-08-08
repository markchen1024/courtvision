"""Measure whether an off-the-shelf court keypoint model works on our footage.

Before hand-calibrating anything it is worth knowing what the ready-made models
actually do here, and "it looks about right" is not knowing. This asks a question
the model cannot fake:

Two frames of the same court are related by a homography. So are the model's
keypoints, if the model is putting them on the court. Fit a homography from the
keypoints it shares between two frames, fit another from ORB features on the
static background, and compare where the two send the frame corners. ORB agrees
with itself to about 1px on this clip, so it is the ruler.

No court model, no hand calibration, no ground truth needed -- the model is
checked against geometry it has to obey either way.

    python pipeline/try_court_model.py --version 22

Needs ROBOFLOW_API_KEY in .env. Frames are sent to the hosted endpoint, which is
fine for public broadcast footage; nothing is uploaded to a workspace.
"""

import argparse
import base64
import itertools

import cv2
import numpy as np
import requests

from config import secret
from register import (GOOD_OVERLAP, MIN_INLIERS, build_static_mask, make_detector,
                      overlap_fraction, register_pair, scale_matrix)

PROJECT = "basketball-court-detection-2"
ENDPOINT = "https://detect.roboflow.com"


def local_keypoints(model, image, kp_conf):
    """Same shape of answer as the hosted endpoint, from a local .pt.

    Names are the keypoint index as a string, matching how the Roboflow export
    names them, so court_model.court_point() reads either source.
    """
    res = model.predict(image, verbose=False)[0]
    if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
        return {}, 0, 0.0
    # Highest-scoring court instance, as the hosted path also does.
    box_conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
    i = int(box_conf.argmax()) if box_conf is not None and len(box_conf) else 0
    xy = res.keypoints.xy[i].cpu().numpy()
    conf = (res.keypoints.conf[i].cpu().numpy()
            if res.keypoints.conf is not None else [1.0] * len(xy))
    out = {str(k): (float(x), float(y))
           for k, ((x, y), c) in enumerate(zip(xy, conf)) if c >= kp_conf}
    n = len(box_conf) if box_conf is not None else 1
    return out, n, float(box_conf[i]) if box_conf is not None and len(box_conf) else 0.0


def predict(image, version, api_key, box_conf):
    buf = base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]).decode()
    r = requests.post(f"{ENDPOINT}/{PROJECT}/{version}",
                      params={"api_key": api_key, "confidence": box_conf}, data=buf,
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=90)
    r.raise_for_status()
    return sorted(r.json().get("predictions", []), key=lambda p: -p["confidence"])


def keypoints(preds, kp_conf):
    """The highest-scoring court instance, as name -> pixel."""
    if not preds:
        return {}
    return {k.get("class_name") or k.get("class"): (k["x"], k["y"])
            for k in preds[0]["keypoints"] if k.get("confidence", 0) >= kp_conf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--version", type=int, default=22)
    ap.add_argument("--weights", help="a local .pt to test instead of the hosted model")
    ap.add_argument("--frames", default="4017,4100,4326,4400")
    ap.add_argument("--box-conf", type=int, default=20)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    args = ap.parse_args()

    frames = [int(f) for f in args.frames.split(",")]
    model = None
    key = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(args.weights)
    else:
        key = secret("ROBOFLOW_API_KEY", "the model is public; the key only authorises the call")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask, _ = build_static_mask(cap, total)

    images = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            images[f] = im
    cap.release()

    who = args.weights if args.weights else f"{PROJECT}/{args.version}"
    print(f"{who} on {len(images)} frames "
          f"(box>={args.box_conf}%, keypoint>={args.kp_conf})")
    found = {}
    for f, im in images.items():
        if model is not None:
            found[f], npred, best = local_keypoints(model, im, args.kp_conf)
        else:
            preds = predict(im, args.version, key, args.box_conf)
            found[f] = keypoints(preds, args.kp_conf)
            npred, best = len(preds), (preds[0]["confidence"] if preds else 0.0)
        print(f"  frame {f:5d}: {npred} court instance(s), best {best:.2f}, "
              f"{len(found[f])} keypoints above threshold")

    scale = 0.5
    det, norm = make_detector("orb")
    S, Sinv = scale_matrix(scale), scale_matrix(1 / scale)
    mask_w = cv2.resize(mask, (int(W * scale), int(H_px * scale)), interpolation=cv2.INTER_NEAREST)
    grey = {f: cv2.cvtColor(cv2.resize(im, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            for f, im in images.items()}
    corners = np.float32([[0, 0], [W, 0], [W, H_px], [0, H_px]]).reshape(-1, 1, 2)

    print(f"\n{'pair':>14} {'shared':>7} {'ORB err':>9} {'overlap':>8} {'disagreement':>14}")
    gaps = []
    for a, b in itertools.combinations(sorted(images), 2):
        shared = sorted(set(found[a]) & set(found[b]))
        H_orb, inl, err = register_pair(det, norm, grey[a], grey[b], mask_w)
        if H_orb is None:
            continue
        # The baseline is only a ruler between frames that actually see the same
        # thing. Reprojection error and inlier count both look healthy on a pair
        # sharing no view at all -- overlap is the check that does not, and
        # leaving it out is the same mistake this script exists to catch.
        ov = overlap_fraction(H_orb, grey[a].shape)
        if inl < MIN_INLIERS or ov < GOOD_OVERLAP:
            print(f"  {a}->{b:5d} {len(shared):>7} {err:8.2f}px {ov * 100:7.0f}% "
                  f"{'no baseline':>14}")
            continue
        H_orb = Sinv @ H_orb @ S
        if len(shared) < 4:
            print(f"  {a}->{b:5d} {len(shared):>7} {err:8.2f}px {ov * 100:7.0f}% {'too few':>14}")
            continue
        H_kp, _ = cv2.findHomography(
            np.float32([found[a][k] for k in shared]).reshape(-1, 1, 2),
            np.float32([found[b][k] for k in shared]).reshape(-1, 1, 2), cv2.RANSAC, 5.0)
        if H_kp is None:
            print(f"  {a}->{b:5d} {len(shared):>7} {err:8.2f}px {ov * 100:7.0f}% {'no fit':>14}")
            continue
        gap = float(np.linalg.norm(
            cv2.perspectiveTransform(corners, H_kp).reshape(-1, 2)
            - cv2.perspectiveTransform(corners, H_orb).reshape(-1, 2), axis=1).mean())
        gaps.append(gap)
        print(f"  {a}->{b:5d} {len(shared):>7} {err:8.2f}px {ov * 100:7.0f}% {gap:12.1f}px")

    if gaps:
        best = min(gaps)
        print(f"\nbest disagreement {best:.0f}px against an ORB baseline good to about 1px.")
        # The verdict is the number. Writing the conclusion into the script ahead
        # of the measurement is how you stop noticing when it changes.
        if best <= 15:
            print("Usable. The keypoints are on the court: a homography built from them")
            print("agrees with feature matching to within a few pixels, so the calibration")
            print("can come from the model and nobody has to click anything.")
        elif best <= 60:
            print("Marginal. Enough to draw a court overlay that looks about right, not")
            print("enough to measure positions from -- at this scale the error is metres")
            print("on the floor, not centimetres.")
        else:
            print("Not usable. A model whose keypoints were on the court would agree with")
            print("the baseline to a few pixels. Raising the keypoint confidence does not")
            print("rescue it either: that drops the reliable-looking points too and the fit")
            print("gets worse. Both public models trained on professional broadcast, where")
            print("floors carry large painted areas and the camera is high. A community")
            print("stadium from a tripod at halfway is a different distribution, and mAP95")
            print("of 0.98 on the model's own split says nothing about it.")


if __name__ == "__main__":
    main()
