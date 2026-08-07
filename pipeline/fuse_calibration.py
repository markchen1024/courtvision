"""Solve one court homography from points clicked across several frames.

A side-on view of one end cannot spread four landmarks far apart. Click the key
corners and the convex hull is 28 square metres -- seven percent of the court --
and the fit extrapolates wildly everywhere else. The frames that do show the
halfway line show only the halfway line, whose landmarks are all at x=14m and so
are exactly collinear. Neither frame can be calibrated alone.

But every frame of a panning camera is related to every other by a homography,
which register.py already measures. So the points do not have to come from one
frame: transport them all into a single hub frame and solve them together.

    H_court_to_hub  solved from  { H_frame_to_hub @ pixel : every click, every frame }

Click whatever each frame shows clearly -- four key corners here, the centre
circle there, the far baseline somewhere else -- and the union spans the court.

    python pipeline/calibrate.py --frame 4017 --partial
    python pipeline/calibrate.py --frame 4326 --partial
    python pipeline/fuse_calibration.py --ref 4326

Every transported point carries the registration error of its own frame on top of
the click error, so the per-point residuals below are the honest ones to read: a
point that only fits after being moved 40 pixels was either clicked wrong or came
from a frame that does not register well.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate import (BY_NAME, COURT_AREA, LINE_BY_NAME, MAX_RMS, MIN_COVERAGE,
                       draw_court, save_calibration, solve)
from register import (MIN_INLIERS, build_static_mask, make_detector, overlap_fraction,
                      register_pair, scale_matrix, LIME, RED)


def load_frames(cap, indices, work_scale):
    out = {}
    for i in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        small = cv2.resize(img, None, fx=work_scale, fy=work_scale)
        out[i] = (img, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--calibrations", default="out/calibration")
    ap.add_argument("--ref", type=int, help="hub frame (default: the one with most clicks)")
    ap.add_argument("--work-width", type=int, default=960)
    ap.add_argument("--out", default="out/calibration/fused.json")
    args = ap.parse_args()

    files = sorted(Path(args.calibrations).glob("frame_*.json"))
    if not files:
        raise SystemExit(f"no calibrations in {args.calibrations}")

    # A line is carried by its two clicked pixels: a homography maps lines to
    # lines and points on a line to points on its image, so transporting the two
    # clicks and re-fitting a line through them is exact. No separate H^-T path.
    sets = {}
    for f in files:
        d = json.loads(f.read_text())
        pts = {p["name"]: tuple(p["pixel"]) for p in d["points"]}
        lns = {l["name"]: (tuple(l["pixel"][0]), tuple(l["pixel"][1]))
               for l in d.get("lines", [])}
        if pts or lns:
            sets[d["frame"]] = (pts, lns)
        print(f"{f.name}: frame {d['frame']}, {len(pts)}pt {len(lns)}line"
              + ("  (partial)" if d.get("partial") else ""))
    if not sets:
        raise SystemExit("every calibration file is empty")

    hub = args.ref if args.ref is not None else max(
        sets, key=lambda k: len(sets[k][0]) + len(sets[k][1]))
    if hub not in sets:
        sets[hub] = ({}, {})  # a hub with no clicks of its own is fine
    print(f"\nhub frame: {hub}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask, _ = build_static_mask(cap, total)

    scale = min(1.0, args.work_width / W)
    S, Sinv = scale_matrix(scale), scale_matrix(1.0 / scale)
    frames = load_frames(cap, sets.keys(), scale)
    cap.release()
    if hub not in frames:
        raise SystemExit(f"cannot read hub frame {hub}")

    mask_w = (cv2.resize(mask, frames[hub][1].shape[::-1], interpolation=cv2.INTER_NEAREST)
              if mask is not None else None)
    det, norm = make_detector("orb")
    hub_gray = frames[hub][1]
    shape = hub_gray.shape

    merged, merged_lines, origin = {}, {}, {}
    for idx, (pts, lns) in sets.items():
        if not pts and not lns:
            continue
        if idx == hub:
            for n, p in pts.items():
                merged[n], origin[n] = p, (idx, 0.0)
            for n, seg in lns.items():
                merged_lines[n], origin[n] = seg, (idx, 0.0)
            print(f"  frame {idx:5d}: {len(pts)}pt {len(lns)}line used directly (this is the hub)")
            continue
        if idx not in frames:
            print(f"  frame {idx:5d}: unreadable, skipped")
            continue

        H, inl, err = register_pair(det, norm, frames[idx][1], hub_gray, mask_w)
        if H is None or inl < MIN_INLIERS:
            print(f"  frame {idx:5d}: cannot register onto the hub ({inl} inliers) - points dropped")
            continue
        ov = overlap_fraction(H, shape)
        H_full = Sinv @ H @ S

        flat = list(pts.values()) + [p for seg in lns.values() for p in seg]
        if flat:
            moved = cv2.perspectiveTransform(
                np.array([flat], np.float32), H_full).reshape(-1, 2)
            for n, p in zip(pts, moved[:len(pts)]):
                merged[n], origin[n] = (float(p[0]), float(p[1])), (idx, err or 0.0)
            rest = moved[len(pts):]
            for k, n in enumerate(lns):
                a, b = rest[2 * k], rest[2 * k + 1]
                merged_lines[n] = ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                origin[n] = (idx, err or 0.0)
        print(f"  frame {idx:5d}: {len(pts)}pt {len(lns)}line moved onto the hub "
              f"({inl} inliers, {ov * 100:.0f}% overlap, {err:.2f}px registration error)")

    if len(merged) + len(merged_lines) < 4:
        raise SystemExit(f"only {len(merged)}pt {len(merged_lines)}line in total; need four")

    H, rms, spread, residuals = solve(merged, merged_lines)
    if H is None:
        raise SystemExit("the merged correspondences still do not pin a homography down - "
                         "parallel lines only, or points sitting on the lines they pair with")

    print(f"\n{len(merged)}pt {len(merged_lines)}line from "
          f"{len({o[0] for o in origin.values()})} frames")
    print(f"reprojection rms {rms:.1f}px   coverage {spread:.0f}m2 "
          f"({spread / COURT_AREA * 100:.0f}% of court)")
    for name, err in sorted(residuals.items(), key=lambda kv: -kv[1]):
        src, reg = origin[name]
        flag = "  <-- re-click this one" if err > MAX_RMS else ""
        print(f"  {name:14s} {err:7.1f}px   from frame {src:5d} (reg {reg:.2f}px){flag}")

    if rms > MAX_RMS:
        print(f"\nFAILED: above {MAX_RMS:.0f}px the points contradict each other.")
    elif spread < MIN_COVERAGE:
        print(f"\nTHIN: under {MIN_COVERAGE:.0f}m2 the fit still guesses outside your points.")
    else:
        print("\nusable.")

    out = Path(args.out)
    save_calibration(out, args.video, hub, (frames[hub][0].shape), merged, H, rms, spread,
                     residuals, merged_lines)
    overlay = draw_court(frames[hub][0].copy(), H)
    for name, (px, py) in merged.items():
        bad = residuals.get(name, 0) > MAX_RMS
        cv2.circle(overlay, (int(px), int(py)), 6, RED if bad else LIME, -1)
    for name, seg in merged_lines.items():
        bad = residuals.get(name, 0) > MAX_RMS
        cv2.line(overlay, tuple(int(v) for v in seg[0]), tuple(int(v) for v in seg[1]),
                 RED if bad else LIME, 2, cv2.LINE_AA)
    cv2.imwrite(str(out.with_suffix(".jpg")), overlay)
    print(f"wrote {out} and {out.with_suffix('.jpg')}")
    print(f"\nnext: python pipeline/register.py --calibration {out} --ref {hub} --every 5 --preview")


if __name__ == "__main__":
    main()
