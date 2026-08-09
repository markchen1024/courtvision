"""Track players with SAM3's concept prompts, the BasketEvent way.

SAM3 (Meta, 2025-11) upgrades exactly the two structural limits measured on
SAM2 in this repo: a text prompt ("basketball player") replaces detector
boxes, and the concept keeps admitting NEW instances mid-video instead of
freezing the frame-0 roster. arXiv 2607.21267 (BasketEvent, SJTU) runs its
whole player-tracking stage this way. This script measures whether those
claims hold on our footage; output schema matches track_sam2.py so
identify.py and render_final.py consume it unchanged.

    python pipeline/track_sam3.py --video out/espn_test18.mp4 \
        --out out/espn_tracks_sam3.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from progress import Progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", default="basketball player",
                    help="the concept to track; BasketEvent uses "
                         "'basketball player on the court'")
    ap.add_argument("--model", default="sam3.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    predictor = SAM3VideoSemanticPredictor(overrides=dict(
        conf=0.25, task="segment", mode="predict", imgsz=1024,
        model=args.model, save=False, verbose=False))

    import cv2
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    tracks = {}
    n = 0
    prog = Progress("sam3", total=total)
    partial = Path(args.out).with_suffix(".partial.json")
    # stream=True for the same reason as track_sam2.py: without it every
    # result buffers until the end and nothing below runs during inference.
    for r in predictor(source=args.video, text=[args.text], stream=True):
        rows = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            if getattr(r.boxes, "id", None) is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
            elif r.boxes.cls is not None:
                ids = r.boxes.cls.cpu().numpy().astype(int)
            else:
                ids = np.arange(len(xyxy))
            for tid, b in zip(ids, xyxy):
                rows.append({"tid": int(tid), "box": [float(v) for v in b]})
        tracks[n] = rows
        n += 1
        prog.step(note=f"frame {n}, {len(rows)} tracked")
        if n % 500 == 0:
            partial.write_text(json.dumps({"frames_done": n,
                "frames": {str(k): v for k, v in tracks.items()}}))

    lifetimes = {}
    for idx, rows in tracks.items():
        for t in rows:
            a, b = lifetimes.get(t["tid"], (idx, idx))
            lifetimes[t["tid"]] = (min(a, idx), max(b, idx))
    spans = np.array([(b - a) / fps for a, b in lifetimes.values()]) \
        if lifetimes else np.array([0.0])
    born_late = sum(1 for a, _ in lifetimes.values() if a > 5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "every": 1, "fps": fps,
        "tracker": f"ultralytics SAM3VideoSemanticPredictor ({args.model}, "
                   f"text={args.text!r})",
        "frames": {str(k): v for k, v in tracks.items()},
    }))
    prog.done(note=f"{len(lifetimes)} ids")
    print(f"\nwrote {args.out}: {n} frames, {len(lifetimes)} ids "
          f"({born_late} first seen after frame 5 -- the mid-video pickups "
          f"SAM2 structurally cannot make)")
    print(f"  lifetime median {np.median(spans):.1f}s  max {spans.max():.1f}s "
          f"(clip is {n / fps:.0f}s)")


if __name__ == "__main__":
    main()
