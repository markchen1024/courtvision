"""Measure what the camera actually does, before building anything on top of it.

The plan only works if the camera rotates about a fixed point, because then every
frame maps to a chosen reference frame by a single homography. This measures that
directly: match background features against a reference frame and report how well
a homography explains the motion, and how far the view has swung.

Two things must be masked out first or the estimate is garbage:
  the burned-in scoreboard and channel logo, which are painted in screen space and
  therefore do NOT move when the camera pans, and the players, who move
  independently of the camera.

    python pipeline/check_camera_motion.py --video web/media/game.mp4
"""

import argparse

import cv2
import numpy as np

# Overlay regions as fractions of the frame: (x0, y0, x1, y1).
# Scoreboard strip along the bottom, broadcaster logo top right.
OVERLAYS = [
    (0.20, 0.86, 0.80, 1.00),
    (0.90, 0.00, 1.00, 0.15),
]


def background_mask(shape, boxes=None):
    h, w = shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    for x0, y0, x1, y1 in OVERLAYS:
        mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
    for x0, y0, x1, y1 in boxes or []:
        pad = 12
        mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 0
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{args.video}: {w}x{h}, {total} frames, {total/fps:.0f}s\n")

    orb = cv2.ORB_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # reference frame: the middle of the clip, so the pan reaches it from both sides
    ref_idx = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, ref = cap.read()
    if not ok:
        raise SystemExit("could not read the reference frame")
    ref_grey = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    kp_ref, des_ref = orb.detectAndCompute(ref_grey, background_mask(ref.shape))
    print(f"reference frame {ref_idx} ({ref_idx/fps:.0f}s): {len(kp_ref)} background features\n")

    centre = np.array([[[w / 2, h / 2]]], np.float32)
    print(f"{'time':>6} {'matches':>8} {'inliers':>8} {'inlier %':>9} {'centre shift px':>16}")
    print("-" * 54)

    shifts = []
    for i in range(args.samples):
        idx = int(i * (total - 1) / max(1, args.samples - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(grey, background_mask(frame.shape))
        if des is None or des_ref is None or len(kp) < 10:
            print(f"{idx/fps:6.0f} {'too few features':>30}")
            continue

        # Lowe ratio test, then RANSAC. Inlier share is the real signal: a high
        # share means one homography explains the whole background, which is what
        # a camera rotating about a fixed point produces.
        pairs = matcher.knnMatch(des, des_ref, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            print(f"{idx/fps:6.0f} {len(good):8d} {'insufficient matches':>28}")
            continue

        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            print(f"{idx/fps:6.0f} {len(good):8d} {'no homography':>28}")
            continue

        n_in = int(inliers.sum())
        moved = cv2.perspectiveTransform(centre, H)[0][0]
        shift = float(np.linalg.norm(moved - np.array([w / 2, h / 2])))
        shifts.append(shift)
        print(f"{idx/fps:6.0f} {len(good):8d} {n_in:8d} {100*n_in/len(good):8.0f}% {shift:16.0f}")

    if shifts:
        print(f"\npan range across the clip: {max(shifts):.0f} px of view shift "
              f"({100*max(shifts)/w:.0f}% of frame width)")
        print("A high inlier share at every sample means one homography per frame is enough,")
        print("so the court can be calibrated once on the reference frame and carried everywhere.")


if __name__ == "__main__":
    main()
