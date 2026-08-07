"""Register every frame back to one calibrated reference frame.

The camera pans and zooms but never leaves its tripod. A camera that only rotates
about its optical centre maps between any two of its frames by a pure homography
-- no depth, no parallax, no scene model. So the court only has to be calibrated
once: compose the reference frame's court homography with the frame-to-frame one
and every frame is calibrated.

    H_court_to_frame = H_ref_to_frame @ H_court_to_ref

That is the whole idea, and it is why this survives on footage where the per-frame
court keypoint model finds nothing.

    python pipeline/register.py --ref 120 --every 5 --limit 600
    python pipeline/register.py --calibration out/calibration/frame_000120.json --preview

Two things are measured because two things can go wrong:

  overlap  -- how much of the reference frame is still in view. This is the real
              limit of the approach: pan far enough and there is nothing left to
              match against. Reported per frame so the failure is visible rather
              than silent.
  drift    -- where a frame can be registered both directly and by chaining
              through its neighbours, the two answers should agree. The gap is
              accumulated error, in pixels.

Burned-in broadcast graphics are masked out automatically. They are welded to
image coordinates, so they vote for "the camera did not move" no matter what it
did -- left in, they drag every homography toward the identity.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LIME = (49, 240, 200)
RED = (107, 114, 244)
TEAL = (168, 201, 59)
DIM = (111, 162, 154)
INK = (24, 22, 20)

MIN_INLIERS = 25          # below this a homography is noise wearing a suit
GOOD_OVERLAP = 0.30       # fraction of the reference frame still visible


def build_static_mask(cap, total, samples=40, threshold=2.0, out=None):
    """Mask out pixels that never change -- i.e. the scoreboard and the logo.

    Real background moves across the sensor as the camera pans, so anything with
    near-zero temporal variance is painted onto the video rather than filmed.
    """
    idxs = np.linspace(0, max(0, total - 1), samples).astype(int)
    stack = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    if len(stack) < 4:
        return None, 0.0

    std = np.stack(stack).std(axis=0)
    mask = (std > threshold).astype(np.uint8) * 255
    # A clock digit changes every second, so it passes the variance test while
    # still being welded to the frame. Close the excluded region to swallow those
    # holes, otherwise the live parts of the scoreboard stay in play.
    holes = cv2.morphologyEx(255 - mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    mask = 255 - holes
    # Overlays have soft edges and drop shadows; erode the usable area a little so
    # the fringe does not leak back in.
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    excluded = 1.0 - mask.mean() / 255.0

    if out is not None:
        cv2.imwrite(str(out), mask)
    return mask, excluded


def make_detector(kind, n=4000):
    if kind == "sift":
        return cv2.SIFT_create(nfeatures=n), cv2.NORM_L2
    return cv2.ORB_create(nfeatures=n), cv2.NORM_HAMMING


def register_pair(det, norm, ref, cur, mask=None):
    """Homography mapping reference-frame pixels to current-frame pixels."""
    kp1, des1 = det.detectAndCompute(ref, mask)
    kp2, des2 = det.detectAndCompute(cur, mask)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, 0, None

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(des1, des2, k=2)
    # Lowe's ratio test: a match is only trustworthy if it beats the runner-up.
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good), None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=5000)
    if H is None:
        return None, 0, None

    n_in = int(inliers.sum())
    err = None
    if n_in:
        keep = inliers.ravel().astype(bool)
        proj = cv2.perspectiveTransform(src[keep], H).reshape(-1, 2)
        err = float(np.linalg.norm(proj - dst[keep].reshape(-1, 2), axis=1).mean())
    return H, n_in, err


def overlap_fraction(H, shape):
    """How much of the reference frame lands inside the current frame."""
    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return 0.0
    # A camera that only rotates cannot mirror its own image, so a flipped quad
    # means the homography is junk. intersectConvexConvex would still return a
    # plausible-looking number for it.
    if cv2.contourArea(warped.astype(np.float32), oriented=True) <= 0:
        return 0.0
    frame_rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    inter, _ = cv2.intersectConvexConvex(warped.astype(np.float32), frame_rect)
    return max(0.0, float(inter) / float(w * h))


def corner_disagreement(Ha, Hb, shape):
    """Mean pixel gap between two homographies -- our units for drift."""
    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, Ha).reshape(-1, 2)
    b = cv2.perspectiveTransform(corners, Hb).reshape(-1, 2)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(np.linalg.norm(a - b, axis=1).mean())


def scale_matrix(s):
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)


def court_overlay(frame, H_court_to_frame):
    """Draw the court model through the composed homography."""
    from calibrate import court_polylines

    for poly in court_polylines():
        pts = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H_court_to_frame).reshape(-1, 2)
        if not np.isfinite(pts).all():
            continue
        cv2.polylines(frame, [np.clip(pts, -1e4, 1e4).astype(np.int32)], False, TEAL, 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--ref", type=int, default=None,
                    help="reference frame to hub on (default: the calibrated frame, else 0)")
    ap.add_argument("--every", type=int, default=5, help="register one frame in N")
    ap.add_argument("--limit", type=int, default=600, help="max frames to register")
    ap.add_argument("--features", choices=["orb", "sift"], default="orb")
    ap.add_argument("--no-mask", action="store_true",
                    help="keep burned-in graphics in play (to show what they cost)")
    ap.add_argument("--work-width", type=int, default=960, help="registration resolution")
    ap.add_argument("--calibration", help="calibration json; its frame becomes the reference")
    ap.add_argument("--preview", action="store_true", help="write an annotated mp4")
    ap.add_argument("--out", default="out/registration")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # The frame worth clicking and the frame worth hubbing on are different
    # questions -- one wants court markings in view, the other wants overlap with
    # the rest of the clip -- so they are allowed to be different frames.
    H_court_to_cal, cal_frame = None, None
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text())
        H_court_to_cal = np.array(cal["H_court_to_image"], np.float64)
        cal_frame = cal["frame"]
        print(f"calibration: frame {cal_frame}, {len(cal['points'])} points, "
              f"rms {cal['reprojection_rms_px']:.1f}px, coverage {cal['coverage_m2']:.0f}m2")
    if args.ref is None:
        args.ref = cal_frame if cal_frame is not None else 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"{args.video}: {W}x{H_px} @ {fps:.0f}fps, {total} frames")

    if args.no_mask:
        mask, excluded = None, 0.0
        print("burned-in graphics left in play (--no-mask)")
    else:
        print("finding burned-in graphics...")
        mask, excluded = build_static_mask(cap, total, out=outdir / "static_mask.png")
        print(f"  masked out {excluded * 100:.1f}% of the frame (scoreboard, logos)")

    scale = min(1.0, args.work_width / W)
    work = (int(W * scale), int(H_px * scale))
    work_shape = (work[1], work[0])
    mask_w = cv2.resize(mask, work, interpolation=cv2.INTER_NEAREST) if mask is not None else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.ref)
    ok, ref_frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read reference frame {args.ref}")
    ref_gray = cv2.cvtColor(cv2.resize(ref_frame, work), cv2.COLOR_BGR2GRAY)

    det, norm = make_detector(args.features)
    S, Sinv = scale_matrix(scale), scale_matrix(1.0 / scale)

    H_court_to_ref = H_court_to_cal
    if H_court_to_cal is not None and cal_frame != args.ref:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cal_frame)
        ok, cal_img = cap.read()
        if not ok:
            raise SystemExit(f"cannot read calibrated frame {cal_frame}")
        cal_gray = cv2.cvtColor(cv2.resize(cal_img, work), cv2.COLOR_BGR2GRAY)
        H_cal_ref, n_in, err = register_pair(det, norm, cal_gray, ref_gray, mask_w)
        if H_cal_ref is None or n_in < MIN_INLIERS:
            raise SystemExit(
                f"cannot register calibrated frame {cal_frame} onto reference {args.ref} "
                f"({n_in} inliers) - pick a reference closer to the calibrated frame")
        H_court_to_ref = (Sinv @ H_cal_ref @ S) @ H_court_to_cal
        print(f"  moved calibration from frame {cal_frame} onto reference {args.ref}: "
              f"{n_in} inliers, {err:.2f}px")

    targets = list(range(0, total, args.every))[:args.limit]
    print(f"registering {len(targets)} frames against frame {args.ref} using {args.features}")

    writer = None
    if args.preview:
        writer = cv2.VideoWriter(str(outdir / "preview.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H_px))

    results = []
    prev_gray, prev_H = None, None
    for n, idx in enumerate(targets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, work), cv2.COLOR_BGR2GRAY)

        H_direct, in_direct, err_direct = register_pair(det, norm, ref_gray, gray, mask_w)
        # None means the match failed; 0.0 would mean the reference genuinely left
        # the frame. Conflating them makes the overlap summary a lie.
        ov_direct = overlap_fraction(H_direct, work_shape) if H_direct is not None else None

        # Chained: reference -> previous -> current. Survives a pan that leaves the
        # reference behind, at the cost of accumulating every step's error.
        H_chain, in_chain = None, 0
        if prev_gray is not None and prev_H is not None:
            H_step, in_chain, _ = register_pair(det, norm, prev_gray, gray, mask_w)
            if H_step is not None:
                H_chain = H_step @ prev_H

        direct_ok = (H_direct is not None and in_direct >= MIN_INLIERS
                     and ov_direct is not None and ov_direct >= GOOD_OVERLAP)
        if direct_ok:
            H_work, method, inl = H_direct, "direct", in_direct
        elif H_chain is not None and in_chain >= MIN_INLIERS:
            H_work, method, inl = H_chain, "chained", in_chain
        else:
            results.append({"frame": idx, "method": "failed", "inliers": 0,
                            "inliers_direct": int(in_direct), "inliers_chained": int(in_chain),
                            "overlap": ov_direct, "drift_px": None})
            prev_gray, prev_H = gray, None
            continue

        drift = (corner_disagreement(H_direct, H_chain, work_shape)
                 if (H_direct is not None and H_chain is not None and direct_ok) else None)

        results.append({
            "frame": idx, "method": method, "inliers": int(inl),
            # Both counts, because "inliers" alone means whichever path won and
            # "overlap" always describes the direct one -- reading them as the
            # same measurement leads you astray.
            "inliers_direct": int(in_direct), "inliers_chained": int(in_chain),
            "overlap": ov_direct,
            "reproj_px": err_direct,
            "drift_px": drift,
            "H_ref_to_frame": (Sinv @ H_work @ S).tolist(),
        })

        if writer is not None:
            vis = frame.copy()
            H_full = Sinv @ H_work @ S
            if H_court_to_ref is not None:
                court_overlay(vis, H_full @ H_court_to_ref)
            colour = LIME if method == "direct" else RED
            cv2.rectangle(vis, (0, 0), (W, 34), INK, -1)
            ov_text = f"{ov_direct * 100:.0f}%" if ov_direct is not None else "no match"
            cv2.putText(vis, f"frame {idx}  {method}  inliers {inl}  overlap {ov_text}"
                             + (f"  drift {drift:.1f}px" if drift is not None else ""),
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
            writer.write(vis)

        prev_gray, prev_H = gray, H_work
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(targets)} frames...")

    cap.release()
    if writer is not None:
        writer.release()

    (outdir / "registration.json").write_text(json.dumps({
        "video": args.video, "reference_frame": args.ref, "every": args.every,
        "features": args.features, "work_width": args.work_width,
        "masked_fraction": excluded,
        "frames": results,
    }, indent=2))

    # ---- the numbers that decide whether this approach holds ----
    methods = {m: sum(1 for r in results if r["method"] == m) for m in ("direct", "chained", "failed")}
    print(f"\nregistered {len(results)} frames: "
          f"{methods['direct']} direct, {methods['chained']} chained, {methods['failed']} failed")

    ov = np.array([r["overlap"] for r in results if r["overlap"] is not None])
    no_match = sum(1 for r in results if r["overlap"] is None)
    if len(ov):
        print(f"overlap with reference: median {np.median(ov) * 100:.0f}%  "
              f"min {ov.min() * 100:.0f}%  (below {GOOD_OVERLAP * 100:.0f}% falls back to chaining)")
    if no_match:
        print(f"  {no_match} frames could not be matched to the reference at all")

    # Inlier count is not a validity check. A gym is full of repeated structure --
    # rows of ceiling lights, identical seats, the same banner twice -- and ORB
    # will match hundreds of them confidently between frames that share no view at
    # all. RANSAC then agrees, because that repetition is itself close to a
    # translation. Only the overlap test catches these.
    bogus = [r for r in results if r.get("inliers_direct", 0) >= MIN_INLIERS
             and r["overlap"] is not None and r["overlap"] < 0.05]
    if bogus:
        worst = max(bogus, key=lambda r: r["inliers_direct"])
        print(f"rejected {len(bogus)} confident-but-wrong direct fits "
              f"(worst: frame {worst['frame']}, {worst['inliers_direct']} inliers, "
              f"{worst['overlap'] * 100:.1f}% overlap)")

    errs = [r["reproj_px"] for r in results if r.get("reproj_px") is not None]
    if errs:
        print(f"registration reprojection: median {np.median(errs):.2f}px  p90 {np.percentile(errs, 90):.2f}px")

    drifts = [r["drift_px"] for r in results if r.get("drift_px") is not None]
    if drifts:
        d = np.array(drifts)
        print(f"direct vs chained disagreement: median {np.median(d):.1f}px  p90 {np.percentile(d, 90):.1f}px")
        print("  (this is what chaining costs when the reference goes out of view)")

    print(f"\nwrote {outdir / 'registration.json'}"
          + (f" and {outdir / 'preview.mp4'}" if args.preview else ""))
    if methods["failed"]:
        print(f"note: {methods['failed']} frames failed outright - check the pan range in the preview")
    if H_court_to_ref is None:
        print("note: no --calibration given, so nothing was mapped to court coordinates yet")


if __name__ == "__main__":
    main()
