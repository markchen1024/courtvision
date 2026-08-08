"""Track players with SAM2's video predictor, the Roboflow pipeline's choice.

Both cheap trackers lose everyone at every broadcast cut -- about 70 of the
remaining tracks are born at a gap, structurally, because neither IoU nor a
court-space gate can connect a player across a replay. SAM2 is the one candidate
with a mechanism for that: it matches by appearance memory, not by motion, so a
player who vanishes for two seconds and reappears is the same object to it.
That is the specific claim this run measures. It is also why the 1-2 FPS cost
is acceptable: this pipeline is offline by design.

    python pipeline/track_sam2.py --video web/media/nba.mp4

One honest limitation, marked rather than hidden: the ultralytics wrapper takes
its prompts on the first frame only, so the players tracked are the ones visible
there. Anyone who enters later -- a substitute, a player off-screen at t=0 --
does not get an identity. Good enough to answer the cross-cut question; not yet
a complete tracker.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from progress import Progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--dense", default="out/detections_nba_dense.json",
                    help="dense detections; the first frame's players become the prompts")
    ap.add_argument("--model", default="sam2.1_b.pt")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="kept low on purpose: a lost object must stay in the output "
                         "so the prompt-order identity mapping cannot shift")
    ap.add_argument("--min-prompt-conf", type=float, default=0.4)
    ap.add_argument("--out", default="out/tracks_sam2.json")
    args = ap.parse_args()

    dense = json.loads(Path(args.dense).read_text())
    first = dense["frames"].get("0", [])
    boxes = [r["box"] for r in first if r["conf"] >= args.min_prompt_conf]
    if len(boxes) < 4:
        raise SystemExit(f"only {len(boxes)} confident players on frame 0 to prompt with")
    print(f"prompting SAM2 with {len(boxes)} players from frame 0")

    from ultralytics.models.sam import SAM2VideoPredictor

    predictor = SAM2VideoPredictor(overrides=dict(
        conf=args.conf, task="segment", mode="predict", imgsz=1024,
        model=args.model, save=False, verbose=False))

    tracks = {}
    n = 0
    prog = Progress("sam2", total=None)
    partial = Path(args.out).with_suffix(".partial.json")
    for r in predictor(source=args.video, bboxes=boxes):
        rows = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            # Identity is the prompt slot. The wrapper does not expose ids, but it
            # does carry the object index in cls; fall back to position if not.
            cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
            for i, b in enumerate(xyxy):
                tid = int(cls[i]) if cls is not None else i
                rows.append({"tid": tid, "box": [float(v) for v in b]})
        tracks[n] = rows
        n += 1
        prog.step(note=f"frame {n}")
        if n % 500 == 0:
            print(f"  {n} frames...", flush=True)
            # A multi-hour run with no checkpoint is a bet nothing crashes.
            partial.write_text(json.dumps({"frames_done": n,
                "frames": {str(k): v for k, v in tracks.items()}}))

    fps = dense.get("fps", 25.0)
    lifetimes = {}
    for idx, rows in tracks.items():
        for t in rows:
            a, b = lifetimes.get(t["tid"], (idx, idx))
            lifetimes[t["tid"]] = (min(a, idx), max(b, idx))
    spans = np.array([(b - a) / fps for a, b in lifetimes.values()]) if lifetimes else np.array([0.0])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "every": 1, "fps": fps,
        "tracker": f"ultralytics SAM2VideoPredictor ({args.model})",
        "prompted": len(boxes),
        "frames": {str(k): v for k, v in tracks.items()},
    }))
    print(f"\nwrote {args.out}: {n} frames, {len(lifetimes)} of {len(boxes)} prompted "
          f"objects ever seen")
    print(f"  lifetime median {np.median(spans):.1f}s  max {spans.max():.1f}s "
          f"(clip is {n / fps:.0f}s; a lifetime near that means identity survived the cuts)")


if __name__ == "__main__":
    main()
