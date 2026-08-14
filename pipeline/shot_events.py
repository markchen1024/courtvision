"""Detect shot events the tutorial's way, to compare against the hand tags.

The last unported piece of blog.roboflow.com/identify-basketball-players:
the same RF-DETR detector also classifies shot poses (player-jump-shot,
player-layup-dunk) and the made-basket moment (ball-in-basket), and
sports.basketball.ShotEventTracker turns those per-frame flags into shot
started / shot made events with debounce windows. No training, no new
models -- just the detector run densely plus a small state machine.

The hand-tagged events in web/data/nba.json are the baseline this gets
compared against. The comparison is the point: our position all along has
been that event recognition is the hard two-year part, and this measures
exactly how far the ready-made version gets on this footage.

    python pipeline/shot_events.py --video web/media/nba.mp4
"""

import argparse
import json
from pathlib import Path

import cv2

import config
from progress import Progress

DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
DETECTION_CONFIDENCE = 0.4
DETECTION_IOU = 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--every", type=int, default=1,
                    help="the notebook runs every frame; >1 trades recall for speed")
    ap.add_argument("--out", default="out/shot_events_auto.json")
    args = ap.parse_args()

    config.load_env()
    import os
    os.environ.setdefault("ROBOFLOW_API_KEY", config.secret("ROBOFLOW_API_KEY"))
    import supervision as sv
    from inference import get_model
    from sports.basketball import ShotEventTracker

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = get_model(model_id=DETECTION_MODEL_ID)
    tracker = ShotEventTracker(
        reset_time_frames=int(fps * 1.7),
        minimum_frames_between_starts=int(fps * 0.5),
        cooldown_frames_after_made=int(fps * 0.5),
    )

    events = []
    prog = Progress("shot-events", total=total // args.every, video=args.video,
                    artifact=args.out, meta={"every": args.every})
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            result = model.infer(frame, confidence=DETECTION_CONFIDENCE,
                                 iou_threshold=DETECTION_IOU)[0]
            det = sv.Detections.from_inference(result)
            names = det.data["class_name"]
            got = tracker.update(
                frame_index=idx,
                has_jump_shot=bool((names == "player-jump-shot").any()),
                has_layup_dunk=bool((names == "player-layup-dunk").any()),
                has_ball_in_basket=bool((names == "ball-in-basket").any()),
            )
            for e in got or []:
                rec = {"t": round(idx / fps, 2), "frame": idx, "event": str(e)}
                events.append(rec)
                print(f"  {rec['t']:7.2f}s  {rec['event']}", flush=True)
            prog.step(note=f"frame {idx}, {len(events)} events")
        idx += 1
    cap.release()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "fps": fps, "every": args.every,
        "model": DETECTION_MODEL_ID,
        "tracker": "sports.basketball.ShotEventTracker (notebook debounce windows)",
        "events": events,
    }))
    prog.done(note=f"{len(events)} events")
    print(f"\nwrote {args.out}: {len(events)} events over {idx / fps:.0f}s")


if __name__ == "__main__":
    main()
