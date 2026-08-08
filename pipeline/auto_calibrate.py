"""Solve the court homography per frame, with nobody clicking anything.

Two things had to exist first: a keypoint model accurate enough on this footage
(basketball-court-detection-2 v22, measured at 19px on this clip against a 1px
baseline) and a table saying where each of its numbered keypoints sits on the
floor (decode_keypoints.py). With both, every frame stands alone --

    keypoints -> court positions -> cv2.findHomography

-- and none of the machinery for a panning camera is needed. No reference frame,
no feature-matching chain, no calibration to propagate. Broadcast cuts stop
mattering too: a frame either shows enough court or it does not, and the ones
that do not are marked rather than guessed at.

    python pipeline/auto_calibrate.py --video web/media/nba.mp4

Three guards, because a homography that fits its own points can still be wrong,
which is the failure this project keeps running into:

  coverage   the convex hull of the court points used. Four landmarks clustered
             at one end fit perfectly and extrapolate into nonsense.
  rms        how far the fit misses the points it was given.
  wobble     how far this frame's homography sits from the midpoint of its two
             neighbours', measured over the part of the court the keypoints
             actually cover. Two things had to be got right for this to mean
             anything. Comparing with the previous frame alone measures the camera
             panning, which is not an error and swamps everything, so it is the
             second difference. And measuring at the court corners measures
             extrapolation into a region the model never saw -- 70px median there
             against 6px where the points are. This is the only check of the three
             that tests something the fit was not fitted to, and it is the one
             that separates the 90% of frames good to 7cm from the 10% that are
             wrong by metres.
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import requests

from config import secret

ENDPOINT = "https://detect.roboflow.com"
PROJECT = "basketball-court-detection-2"

MIN_POINTS = 5          # four is the minimum, five means the fit can disagree
MAX_RMS = 12.0          # pixels, against points the model places to about 20cm
MIN_COVERAGE = 40.0     # square metres of court spanned by the points used


def fetch_keypoints(cache_path, video, version, every, limit, api_key, box_conf, kp_conf):
    """Keypoints per sampled frame, cached: each frame is a network round trip."""
    if Path(cache_path).exists():
        c = json.loads(Path(cache_path).read_text())
        if c.get("video") == video and c.get("every") == every and c.get("version") == version:
            print(f"keypoints: reusing {cache_path} ({len(c['frames'])} frames)")
            return {int(k): v for k, v in c["frames"].items()}

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = list(range(0, total, every))[:limit]
    print(f"querying {PROJECT}/{version} for {len(targets)} frames...")

    out = {}
    for n, idx in enumerate(targets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        buf = base64.b64encode(cv2.imencode(".jpg", frame,
                                            [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode()
        r = requests.post(f"{ENDPOINT}/{PROJECT}/{version}",
                          params={"api_key": api_key, "confidence": box_conf}, data=buf,
                          headers={"Content-Type": "application/x-www-form-urlencoded"},
                          timeout=90)
        r.raise_for_status()
        preds = sorted(r.json().get("predictions", []), key=lambda p: -p["confidence"])
        rows = []
        if preds:
            rows = [{"name": k.get("class_name") or k.get("class"),
                     "x": k["x"], "y": k["y"], "conf": k.get("confidence", 0)}
                    for k in preds[0]["keypoints"] if k.get("confidence", 0) >= kp_conf]
        out[idx] = rows
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(targets)} frames...")

    cap.release()
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).write_text(json.dumps({
        "video": video, "version": version, "every": every,
        "frames": {str(k): v for k, v in out.items()},
    }))
    print(f"cached to {cache_path}")
    return out


def solve_frame(rows, kmap):
    """(H court->image, rms px, coverage m^2, points used) or a reason it failed."""
    src, dst = [], []
    for r in rows:
        entry = kmap.get(r["name"])
        if entry:
            src.append(entry["court"])
            dst.append((r["x"], r["y"]))
    if len(src) < MIN_POINTS:
        return None, f"{len(src)} decoded keypoints in view"

    src_a, dst_a = np.float32(src), np.float32(dst)
    coverage = float(cv2.contourArea(cv2.convexHull(src_a)))
    if coverage < MIN_COVERAGE:
        return None, f"points span only {coverage:.0f}m2 of court"

    H, _ = cv2.findHomography(src_a, dst_a, cv2.RANSAC, 8.0)
    if H is None:
        return None, "no homography from these points"
    proj = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt((np.linalg.norm(proj - dst_a, axis=1) ** 2).mean()))
    if rms > MAX_RMS:
        return None, f"rms {rms:.1f}px, the points contradict each other"
    return (H, rms, coverage, len(src), src), None


COURT_BOX = np.float32([[0, 0], [28, 0], [28, 15], [0, 15]]).reshape(-1, 1, 2)


def probe_points(hull):
    """Where to measure. Court corners are the wrong place: a broadcast camera
    sees about a tenth of the floor at a time, so the corners are extrapolated
    far outside anything the model looked at, and measuring there reads 70px
    median where the same frames read 6px inside the region the keypoints cover.
    That is the difference between a calibration that wobbles by three quarters
    of a metre and one that wobbles by seven centimetres, and only the second one
    is about players."""
    if hull is None or len(hull) < 3:
        return COURT_BOX
    return np.float32(hull).reshape(-1, 1, 2)


def court_corners(H, pts=COURT_BOX):
    """Where this homography puts the probe points, in image pixels."""
    p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return p if np.isfinite(p).all() else None


def wobble(prev_H, H, next_H, pts=COURT_BOX):
    """Departure from smooth camera motion, in pixels.

    A steady pan moves the corners a long way between samples, and comparing
    consecutive frames measures that rather than any error -- on this clip it
    reads 178px median and drowns everything. Halfway between the neighbours is
    where a smoothly moving camera puts this frame, so the distance from there is
    what is left when the motion is taken out.
    """
    a, b, c = (court_corners(prev_H, pts), court_corners(H, pts),
               court_corners(next_H, pts))
    if a is None or b is None or c is None:
        return float("inf")
    return float(np.linalg.norm(b - (a + c) / 2, axis=1).mean())


def bridge_gaps(solved, every, max_span):
    """Fill short runs of unsolved frames by interpolating the court's corners.

    75 gaps in 891 sampled frames is what a broadcast looks like -- a replay, a
    bench shot, a moment where too little floor is in view. Left alone they make
    the top-down view blink, and they break every track across them: of 140
    tracks, 80 were born at a gap rather than because the tracker lost anyone.

    Interpolating the four court corners and re-solving is well behaved where
    averaging homography matrices is not, and it is defensible only because the
    camera has already been measured to move smoothly -- wobble a median of 4px.

    Only short gaps, and every filled frame is marked. A frame where the model
    saw no court has no court position, and inventing one across a ten-second
    replay would be exactly the quiet fabrication this pipeline exists to avoid.
    """
    idx = {r["frame"]: i for i, r in enumerate(solved)}
    anchors = [r for r in solved if r.get("solved")]
    filled = 0
    for a, b in zip(anchors, anchors[1:]):
        span = (b["frame"] - a["frame"]) // every - 1
        if span < 1 or span > max_span:
            continue
        ca = court_corners(np.array(a["H_court_to_image"], np.float64))
        cb = court_corners(np.array(b["H_court_to_image"], np.float64))
        if ca is None or cb is None:
            continue
        for k in range(1, span + 1):
            frame = a["frame"] + k * every
            i = idx.get(frame)
            if i is None or solved[i].get("solved"):
                continue
            t = k / (span + 1)
            corners = (1 - t) * ca + t * cb
            H, _ = cv2.findHomography(COURT_BOX.reshape(-1, 2), corners.astype(np.float32))
            if H is None:
                continue
            solved[i].update({"solved": True, "interpolated": True,
                              "H_court_to_image": H.tolist(),
                              "reason": f"bridged a {span}-sample gap"})
            filled += 1
    return filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--keypoint-map", default="out/keypoint_map.json")
    ap.add_argument("--version", type=int, default=22)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--box-conf", type=int, default=20)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--max-jump", type=float, default=0.0,
                    help="pixels of departure from smooth motion to flag (default 30)")
    ap.add_argument("--max-bridge", type=int, default=3,
                    help="samples of unsolved frames to interpolate across; 0 disables")
    ap.add_argument("--cache", default="out/court_keypoints.json")
    ap.add_argument("--out", default="out/auto_calibration.json")
    args = ap.parse_args()

    kmap = json.loads(Path(args.keypoint_map).read_text())["keypoints"]
    print(f"keypoint map: {len(kmap)} decoded landmarks from {args.keypoint_map}")

    key = secret("ROBOFLOW_API_KEY")
    frames = fetch_keypoints(args.cache, args.video, args.version, args.every,
                             args.limit, key, args.box_conf, args.kp_conf)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()

    solved, failures = [], {}
    for idx in sorted(frames):
        got, why = solve_frame(frames[idx], kmap)
        if got is None:
            failures[why.split(",")[0].split(" in view")[0]] = \
                failures.get(why.split(",")[0].split(" in view")[0], 0) + 1
            solved.append({"frame": idx, "solved": False, "reason": why})
            continue
        H, rms, coverage, n, used = got
        solved.append({"frame": idx, "solved": True, "H_court_to_image": H.tolist(),
                       "rms_px": round(rms, 2), "coverage_m2": round(coverage, 1),
                       "points": n, "court_points": used})

    if args.max_bridge:
        filled = bridge_gaps(solved, args.every, args.max_bridge)
        print(f"\nbridged {filled} frames across gaps of up to {args.max_bridge} "
              f"samples ({args.max_bridge * args.every / fps:.1f}s), marked as interpolated")

    # Temporal check last, since it needs both neighbours to exist first.
    jumps = []
    for i, r in enumerate(solved):
        if not r["solved"] or i == 0 or i + 1 >= len(solved):
            continue
        a, c = solved[i - 1], solved[i + 1]
        if not (a["solved"] and c["solved"]):
            continue
        pts = probe_points(r.get("court_points"))
        w = wobble(np.array(a["H_court_to_image"], np.float64),
                   np.array(r["H_court_to_image"], np.float64),
                   np.array(c["H_court_to_image"], np.float64), pts)
        r["wobble_px"] = round(w, 1)
        jumps.append(w)

    ok = [r for r in solved if r["solved"]]
    # Interpolated frames were never fitted to anything, so they have no rms or
    # coverage and must not be averaged in with the frames that were.
    fitted = [r for r in ok if not r.get("interpolated")]
    bridged = [r for r in ok if r.get("interpolated")]
    print(f"\n{len(ok)}/{len(solved)} frames solved ({100 * len(ok) / max(1, len(solved)):.0f}%)")
    for why, n in sorted(failures.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d} failed: {why}")

    if bridged:
        print(f"  of those, {len(fitted)} solved from keypoints and {len(bridged)} "
              f"interpolated across short gaps")
    if fitted:
        rms = np.array([r["rms_px"] for r in fitted])
        cov = np.array([r["coverage_m2"] for r in fitted])
        pts = np.array([r["points"] for r in fitted])
        print(f"\n  points per frame  median {np.median(pts):.0f}  min {pts.min()}  max {pts.max()}")
        print(f"  fit rms           median {np.median(rms):.1f}px  p90 {np.percentile(rms, 90):.1f}px")
        print(f"  coverage          median {np.median(cov):.0f}m2  min {cov.min():.0f}m2")

    if jumps:
        j = np.array(jumps)
        # A threshold read off the same data it polices adapts to whatever is
        # wrong with it. Fixed at 30px -- a third of a metre on this floor --
        # with the median reported so the choice can be argued with.
        limit = args.max_jump or 30.0
        bad = [r for r in ok if r.get("wobble_px", 0) > limit]
        for r in bad:
            r["suspect"] = True
        print(f"  wobble            median {np.median(j):.0f}px  p90 {np.percentile(j, 90):.0f}px")
        print(f"  {len(bad)} of {len(jumps)} frames depart from smooth motion by more than "
              f"{limit:.0f}px")
        print("    (departure from the midpoint of the neighbours, so a steady pan reads")
        print("     zero -- this is the only check here that tests something the fit was")
        print("     not fitted to)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "fps": fps, "every": args.every,
        "source": f"{PROJECT}/{args.version} + {args.keypoint_map}",
        "width": shape[1], "height": shape[0],
        "frames": solved,
    }, indent=2))
    print(f"\nwrote {args.out}")
    print("Positions from this are good to roughly 20cm on the floor, which is a shot")
    print("chart but not a distance-covered figure -- that error accumulates per frame.")


if __name__ == "__main__":
    main()
