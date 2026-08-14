"""Build a ground truth for who each track is, so identity work can be measured.

Eight gates went into identify.py, each added because a rendered frame was
wrong. Every one was justified on its own and none was measured: "is this
better?" was settled by cropping a frame and arguing. That is why they
accumulated -- with no metric, adding a rule is the only move that makes a
counter-example go away, and nothing tells you what it broke elsewhere.

The truth is per track, not per frame, and that is not a shortcut. A first
version sampled frames and asked for the jersey number in every detected box;
on this footage two or three of ten were legible in any given still, the rest
side-on, turned away or occluded. A human labeller has the same problem. It is
exactly why the pipeline votes over time instead of trusting one read, and a
truth file built from single frames would have been mostly guesses.

A track, on the other hand, is readable: over its life some frames show the
number, and those settle who it is for all of them. So each track gets a strip
of crops spread across its lifetime, and one of:

    "32"          this track is that player, throughout
    "mixed"       it followed more than one man -- an id switch
    "not-player"  a referee, a coach, someone in the crowd
    "?"           unreadable; the scorer skips it rather than guessing

Both metrics survive the change. Precision counts drawn labels whose track
truth agrees; coverage counts, per frame, how many of the ten men on court
carry a correct label. The denominator stays ten whatever tracker produced the
boxes, so SAM2 and SAM3 remain comparable even though each needs its own file.

    python pipeline/label_truth.py --video out/segments/X.mp4 \
        --boxes out/X_tracks.json --out eval/X_truth.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def strip(frames_by_idx, cap, tid, samples, cell=(300, 520)):
    """One track's crops across its life, tiled left to right, times burned in."""
    import cv2

    cw, ch = cell
    sheet = np.full((ch, cw * len(samples), 3), 24, np.uint8)
    for col, (f, box) in enumerate(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        s = min(cw / crop.shape[1], (ch - 28) / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)),
                                 max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_LANCZOS4)
        ox = col * cw + (cw - crop.shape[1]) // 2
        sheet[28:28 + crop.shape[0], ox:ox + crop.shape[1]] = crop
        cv2.putText(sheet, f"{f}", (col * cw + 6, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 230, 240), 2)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True, help="tracks JSON to label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=6,
                    help="crops per track, spread over its lifetime")
    ap.add_argument("--min-frames", type=int, default=30,
                    help="skip tracks shorter than this -- a fragment of a few "
                         "frames is never drawn and cannot be read anyway")
    ap.add_argument("--identities",
                    help="label only the tracks this run actually draws. Both "
                         "metrics need no more: precision judges drawn labels, "
                         "and a track with no label contributes nothing to "
                         "coverage either way. On the SAM3 output that is a "
                         "dozen tracks instead of eighty-one.")
    args = ap.parse_args()

    import cv2
    from progress import Progress

    def path(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    sidecar = json.loads(path(args.boxes).read_text(encoding="utf-8"))
    frames = {int(k): v for k, v in sidecar["frames"].items()}
    fps = sidecar.get("fps", 59.94)

    seen = {}
    for f, rows in frames.items():
        for r in rows:
            seen.setdefault(r["tid"], []).append((f, r["box"]))
    for v in seen.values():
        v.sort()

    keep = {t: v for t, v in seen.items() if len(v) >= args.min_frames}
    if args.identities:
        idn = json.loads(path(args.identities).read_text())["identities"]
        drawn = {int(k) for k, v in idn.items()
                 if v.get("number") and not v.get("ignored")}
        keep = {t: v for t, v in keep.items() if t in drawn}
    out = path(args.out)
    sheets = out.parent / "tracks" / Path(args.boxes).stem
    sheets.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path(args.video)))
    truth = {"video": Path(args.video).as_posix(),
             "boxes": Path(args.boxes).as_posix(), "fps": fps,
             "legend": {"<number>": "this track is that player throughout",
                        "mixed": "followed more than one man",
                        "not-player": "referee, coach, crowd",
                        "?": "unreadable; skipped by the scorer"},
             "tracks": {}}
    prog = Progress("label-truth", total=len(keep), video=args.video)
    for tid, rows in sorted(keep.items()):
        idx = np.linspace(0, len(rows) - 1, min(args.samples, len(rows)))
        picks = [rows[int(round(i))] for i in idx]
        cv2.imwrite(str(sheets / f"t{tid:04d}.jpg"),
                    strip(frames, cap, tid, picks),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        truth["tracks"][str(tid)] = {
            "player": None,
            "frames": [rows[0][0], rows[-1][0]],
            "seconds": [round(rows[0][0] / fps, 2), round(rows[-1][0] / fps, 2)],
            "n_frames": len(rows)}
        prog.step(note=f"track {tid}")
    cap.release()
    prog.done(note=f"{len(keep)} tracks")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(truth, indent=1))
    print(f"wrote {out}: {len(keep)} tracks to label "
          f"({len(seen) - len(keep)} shorter than {args.min_frames} frames skipped)")
    print(f"  strips in {sheets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
