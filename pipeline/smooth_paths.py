"""Clean the viewer's court paths with the notebook's clean_paths.

The last unported piece of blog.roboflow.com/identify-basketball-players:
sports.clean_paths removes teleport-like jumps (linear interpolation over
the removed run) and Savitzky-Golay smooths each trajectory. The notebook
calls it once over a dense (T, P, 2) array; our tracks are fragments with
gaps, so each track's contiguous runs are cleaned independently and only
frames the track actually has get written back -- no invented positions.

One honest parameter note: the notebook runs at ~30Hz where its
smooth_window=9 spans 0.3s; our grid is 5Hz, where the same window spans
1.8s. Both the faithful window (9) and the time-equivalent one (5) get
measured; --window picks which to apply.

    python pipeline/smooth_paths.py                # measure only, render preview
    python pipeline/smooth_paths.py --apply        # write web/data/nba.json

Metrics per variant: wobble (median |second difference|, the calibration
validation metric), points edited by jump removal, and how far smoothing
moves unedited measurements (median/p95) -- the cost being paid for calm.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# the notebook's exact call
NB = dict(jump_sigma=3.5, min_jump_dist=0.6, max_jump_run=18,
          pad_around_runs=2, smooth_window=9, smooth_poly=2)


def runs_of(idx):
    """Contiguous runs in a sorted integer index list."""
    out, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            out.append((start, a))
            start = b
    out.append((start, idx[-1]))
    return out


def clean(doc, window):
    from sports import clean_paths

    frames = doc["frames"]
    by_track = {}
    for fi, f in enumerate(frames):
        for p in f["positions"]:
            by_track.setdefault(p["id"], []).append((fi, p))

    wob_before, wob_after, moved, edited_n, total_n = [], [], [], 0, 0
    stats_fallback = [0]
    for tid, samples in by_track.items():
        idx = [fi for fi, _ in samples]
        pts = {fi: p for fi, p in samples}
        for a, b in runs_of(idx):
            n = b - a + 1
            if n < 5:
                continue
            xy = np.array([[pts[i]["x"], pts[i]["y"]] for i in range(a, b + 1)],
                          float).reshape(-1, 1, 2)
            try:
                cleaned, edit = clean_paths(xy, **{**NB, "smooth_window": window})
            except ValueError:
                # upstream limitation: a jump run touching the segment's edge
                # leaves NaN that linear interpolation cannot anchor, and
                # savgol then refuses. Fall back to smoothing without jump
                # removal for that segment rather than dropping it.
                cleaned, edit = clean_paths(
                    xy, **{**NB, "smooth_window": window, "max_jump_run": 0})
                stats_fallback[0] += 1
            cleaned, edit = cleaned[:, 0, :], edit[:, 0]
            orig = xy[:, 0, :]
            wob_before.append(np.abs(np.diff(orig, 2, axis=0)).mean())
            wob_after.append(np.abs(np.diff(cleaned, 2, axis=0)).mean())
            moved.extend(np.linalg.norm((cleaned - orig)[~edit], axis=1))
            edited_n += int(edit.sum())
            total_n += n
            for k, i in enumerate(range(a, b + 1)):
                pts[i]["x"] = round(float(cleaned[k, 0]), 2)
                pts[i]["y"] = round(float(cleaned[k, 1]), 2)
    moved = np.array(moved)
    stats = {
        "wobble_before_m": float(np.median(wob_before)),
        "wobble_after_m": float(np.median(wob_after)),
        "edited_points": edited_n,
        "total_points": total_n,
        "smoothing_shift_median_m": float(np.median(moved)),
        "smoothing_shift_p95_m": float(np.percentile(moved, 95)),
        "edge_jump_fallbacks": stats_fallback[0],
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="web/data/nba.json")
    ap.add_argument("--window", type=int, default=NB["smooth_window"],
                    help="savgol window; 9 = notebook-faithful, 5 = same "
                         "seconds of smoothing at our 5Hz")
    ap.add_argument("--apply", action="store_true",
                    help="write the smoothed positions back (backup kept)")
    args = ap.parse_args()

    raw = json.loads(Path(args.data).read_text())

    if not args.apply:
        for w in (9, 5):
            doc = json.loads(json.dumps(raw))
            s = clean(doc, w)
            print(f"window={w}: wobble {s['wobble_before_m'] * 100:.1f}cm -> "
                  f"{s['wobble_after_m'] * 100:.1f}cm, "
                  f"{s['edited_points']}/{s['total_points']} points jump-edited, "
                  f"smoothing shifts median {s['smoothing_shift_median_m'] * 100:.1f}cm "
                  f"p95 {s['smoothing_shift_p95_m'] * 100:.1f}cm")
        print("\nmeasure-only; rerun with --apply --window N to write")
        return

    backup = Path("out") / (Path(args.data).stem + "_before_smooth.json")
    if not backup.exists():
        shutil.copy(args.data, backup)
        print(f"backup: {backup}")
    s = clean(raw, args.window)
    if "clean_paths" not in raw.get("source", ""):
        raw["source"] = raw.get("source", "") + f" · paths cleaned (clean_paths, window {args.window})"
    Path(args.data).write_text(json.dumps(raw), encoding="utf-8")
    print(f"applied window={args.window}: wobble "
          f"{s['wobble_before_m'] * 100:.1f}cm -> {s['wobble_after_m'] * 100:.1f}cm, "
          f"{s['edited_points']} points jump-edited, wrote {args.data}")


if __name__ == "__main__":
    main()
