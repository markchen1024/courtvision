"""Choose which frames are worth calibrating by hand.

One reference frame does not cover a three-minute clip: measured on this footage,
about half the frames can be registered directly to any single reference, and the
chained fallback diverges (a sign flip at frame 150 that never recovers). The fix
is several hand-calibrated anchors, so no frame is more than one short hop from a
trusted one.

Picking those anchors by eye does not work either. The frame that looks best is
usually mid-pan -- sharp-looking action, but the only court markings in view are
the halfway line and the centre circle, whose landmarks are all at x=14m. Four
collinear points have no homography at all.

So: sample candidates, register every pair, and read the overlap off real
measurements. Then it is a set cover -- pick the fewest anchors that leave no
frame stranded. Sharpness breaks ties, because a blurred frame cannot be clicked
accurately even when it is geometrically ideal.

    python pipeline/pick_anchors.py --candidates 36 --want 5

Output is a contact sheet plus the frame numbers to feed to calibrate.py. The
machine picks frames that are sharp and that span the camera's range; you still
have to look at the sheet and confirm each one actually shows enough court -- that
part needs judgement, and this script has none.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from register import (GOOD_OVERLAP, MIN_INLIERS, build_static_mask, make_detector,
                      overlap_fraction, register_pair, LIME, RED, DIM, INK)


def sharpness(gray):
    """Laplacian variance -- low means motion blur, which ruins click accuracy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def greedy_cover(reach, n, want):
    """Fewest anchors covering the most frames. Ties go to the sharper frame."""
    chosen, covered = [], set()
    while len(chosen) < want and len(covered) < n:
        best, best_gain = None, 0
        for i in range(n):
            if i in chosen:
                continue
            gain = len(reach[i] - covered)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None or best_gain == 0:
            break
        chosen.append(best)
        covered |= reach[best]
    return chosen, covered


def contact_sheet(frames, images, chosen, path, cols=6, cell=320):
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * (cell * 9 // 16 + 26), cols * cell, 3), 20, np.uint8)
    ch = cell * 9 // 16
    for k, (idx, img) in enumerate(zip(frames, images)):
        r, c = divmod(k, cols)
        y, x = r * (ch + 26), c * cell
        sheet[y + 26:y + 26 + ch, x:x + cell] = cv2.resize(img, (cell, ch))
        picked = k in chosen
        cv2.putText(sheet, f"f{idx}  {idx / 30:.1f}s" + ("  ANCHOR" if picked else ""),
                    (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    LIME if picked else DIM, 1, cv2.LINE_AA)
        if picked:
            cv2.rectangle(sheet, (x + 1, y + 27), (x + cell - 2, y + 25 + ch), LIME, 2)
    cv2.imwrite(str(path), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--candidates", type=int, default=36, help="frames to consider")
    ap.add_argument("--want", type=int, default=5, help="max anchors to recommend")
    ap.add_argument("--work-width", type=int, default=960)
    ap.add_argument("--out", default="out/anchors")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"{args.video}: {total} frames")

    mask, excluded = build_static_mask(cap, total, out=outdir / "static_mask.png")
    print(f"masked out {excluded * 100:.1f}% of the frame (burned-in graphics)")

    scale = min(1.0, args.work_width / W)
    idxs = np.linspace(0, total - 1, args.candidates).astype(int)
    mask_w = None

    frames, grays, images = [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        small = cv2.resize(img, None, fx=scale, fy=scale)
        if mask_w is None and mask is not None:
            mask_w = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        frames.append(int(i))
        images.append(img)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()

    n = len(frames)
    sharp = np.array([sharpness(g) for g in grays])
    shape = grays[0].shape
    print(f"registering {n * (n - 1) // 2} pairs among {n} candidates...")

    det, norm = make_detector("orb")
    # reach[i] = frames that anchor i can register directly, itself included
    reach = [{i} for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            H, inl, _ = register_pair(det, norm, grays[i], grays[j], mask_w)
            if H is None or inl < MIN_INLIERS:
                continue
            ov = overlap_fraction(H, shape)
            if ov >= GOOD_OVERLAP:
                # Overlap is symmetric enough at this threshold to treat the pair
                # as mutually reachable.
                reach[i].add(j)
                reach[j].add(i)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n} rows...")

    order = sorted(range(n), key=lambda i: (-len(reach[i]), -sharp[i]))
    chosen, covered = greedy_cover(reach, n, args.want)
    # Prefer the sharpest frame among equally-reaching neighbours of each pick.
    refined = []
    for c in chosen:
        pool = [k for k in reach[c] if reach[k] >= reach[c]] or [c]
        refined.append(max(pool, key=lambda k: sharp[k]))
    refined = sorted(set(refined))
    covered_refined = set().union(*(reach[k] for k in refined)) if refined else set()

    print(f"\nsharpness across candidates: median {np.median(sharp):.0f}  "
          f"range {sharp.min():.0f}-{sharp.max():.0f}")
    print(f"reach per candidate: median {np.median([len(r) for r in reach]):.0f} of {n} frames")
    print(f"\nrecommended anchors ({len(refined)}), covering {len(covered_refined)}/{n} candidates:")
    for k in refined:
        print(f"  frame {frames[k]:5d}  ({frames[k] / 30:6.1f}s)  reaches {len(reach[k]):2d}/{n}  "
              f"sharpness {sharp[k]:6.0f}")

    stranded = [frames[k] for k in range(n) if k not in covered_refined]
    if stranded:
        print(f"\n{len(stranded)} candidate frames reachable from none of them: {stranded}")
        print("  raise --want, or accept that those stretches stay uncalibrated")

    sheet = outdir / "candidates.jpg"
    contact_sheet(frames, images, set(refined), sheet)
    (outdir / "anchors.json").write_text(json.dumps({
        "video": args.video,
        "candidates": frames,
        "sharpness": sharp.tolist(),
        "reach": [sorted(r) for r in reach],
        "anchors": [frames[k] for k in refined],
        "covered": len(covered_refined),
    }, indent=2))

    print(f"\nwrote {sheet} - look at it before clicking anything")
    # What this script ranks is reach, which makes a good registration hub. It says
    # nothing about whether a frame can be calibrated: the most central pose tends
    # to be a mid-court view whose only landmarks sit on the halfway line, and four
    # collinear points have no homography. Pick the frame to click off the contact
    # sheet, by looking for a whole key with its corners clear of players.
    print("\nuse the top anchor as the registration hub:")
    print(f"  python pipeline/register.py --ref {frames[refined[0]]} --calibration <your calibration.json>")
    print("\nbut choose the frame to CLICK from the contact sheet, not from this ranking -")
    print("it needs a full key in view, which a central pose usually does not have.")


if __name__ == "__main__":
    main()
