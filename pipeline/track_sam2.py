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


def dedupe(rows, iou_thresh):
    """Greedy NMS over the prompt boxes, most confident first.

    The detector returned two boxes on one player on frame 0 of the NYK @ DET
    clip -- 143x275 at (169,357) at 0.93 and 134x259 at (171,353) at 0.51.
    SAM2 takes prompts by slot, so a duplicate does not merge later: it spends
    the whole clip tracking one man as two objects, and costs a slot that a
    player who was never prompted could have used.
    """
    kept = []
    for r in sorted(rows, key=lambda r: -r["conf"]):
        x1, y1, x2, y2 = r["box"]
        clash = False
        for k in kept:
            a1, b1, a2, b2 = k["box"]
            ix = max(0.0, min(x2, a2) - max(x1, a1))
            iy = max(0.0, min(y2, b2) - max(y1, b1))
            inter = ix * iy
            union = (x2-x1)*(y2-y1) + (a2-a1)*(b2-b1) - inter
            if union and inter / union > iou_thresh:
                clash = True
                break
        if not clash:
            kept.append(r)
    if len(kept) < len(rows):
        print(f"prompts: {len(rows)} -> {len(kept)} after de-duplication")
    return kept


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
    ap.add_argument("--prompt-iou", type=float, default=0.5,
                    help="drop a prompt overlapping a more confident one by "
                         "more than this: two prompts on one player split him "
                         "between two slots for the whole clip")
    ap.add_argument("--out", default="out/tracks_sam2.json")
    args = ap.parse_args()

    dense = json.loads(Path(args.dense).read_text())
    first = dense["frames"].get("0", [])
    kept = dedupe([r for r in first if r["conf"] >= args.min_prompt_conf],
                  args.prompt_iou)
    boxes = [r["box"] for r in kept]
    if len(boxes) < 4:
        raise SystemExit(f"only {len(boxes)} confident players on frame 0 to prompt with")
    print(f"prompting SAM2 with {len(boxes)} players from frame 0")

    from ultralytics.models.sam import SAM2VideoPredictor

    predictor = SAM2VideoPredictor(overrides=dict(
        conf=args.conf, task="segment", mode="predict", imgsz=1024,
        model=args.model, save=False, verbose=False))

    import cv2
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    tracks = {}
    n = 0
    prog = Progress("sam2", total=total)
    partial = Path(args.out).with_suffix(".partial.json")
    # stream=True is load-bearing: without it ultralytics buffers every result
    # and returns a list, so this loop -- progress, checkpoints, everything --
    # runs only after the whole video is done. That was the invisible 2.5h run.
    for r in predictor(source=args.video, bboxes=boxes, stream=True):
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

    prog.done(note=f"{n} frames, {len(lifetimes)} objects")
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
