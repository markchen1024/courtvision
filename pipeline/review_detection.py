"""Watch stage one on every frame, not just the one it runs on.

The pipeline only ever detects players on frame 0 -- that call decides who can
be identified for the rest of the clip. Which makes it worth seeing what the
same detector does on all the other frames: where it loses people, where it
finds people who are not players, and how steady the count is. None of that is
visible from a single frame, and all of it bears on whether prompting once is
enough.

Green   becomes a prompt
Red     struck out as a duplicate of a more confident box
Amber   detected, but not a player class (referee, rim, ball, number)

    python pipeline/review_detection.py --video web/media/det_final.mp4 \
        --out out/review_detection.mp4

Prints how often the count reached a full lineup, which is the same question
check_lineup.py asks of frame 0, asked of every frame.
"""

import argparse
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

import config
from check_lineup import CONFIDENCE, IOU_THRESHOLD, PLAYER_CLASS_IDS, dedupe
from progress import Progress

DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
GREEN, RED, AMBER = (0, 210, 0), (0, 0, 240), (0, 170, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--need", type=int, default=10)
    ap.add_argument("--source-offset", type=float, default=0.0,
                    help="seconds this clip starts at inside the original "
                         "recording, so the overlay names a timestamp that can "
                         "be scrubbed to in the full game rather than a frame "
                         "number that can only be found here")
    ap.add_argument("--show-others", action="store_true",
                    help="also draw referee/rim/ball/number detections")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    config.load_env()
    config.inference_env()

    import cv2
    import supervision as sv
    from inference import get_model

    def path(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    model = get_model(model_id=DETECTION_MODEL_ID)
    cap = cv2.VideoCapture(str(path(args.video)))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # cv2's mp4v plays poorly in browsers; pipe raw frames to ffmpeg for H.264,
    # the same way render_tracks.py does.
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE)

    counts = Counter()
    idx = 0
    prog = Progress("review-detection", total=total)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.infer(frame, confidence=CONFIDENCE,
                          iou_threshold=IOU_THRESHOLD)[0]
        det = sv.Detections.from_inference(res)
        is_player = np.isin(det.class_id, PLAYER_CLASS_IDS)
        players = det[is_player]
        keep = set(dedupe(players.xyxy.tolist(), players.confidence.tolist())) \
            if len(players) else set()
        counts[len(keep)] += 1

        if args.show_others:
            others = det[~is_player]
            for box, name in zip(others.xyxy, others.data.get("class_name", [])):
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), AMBER, 2)
                cv2.putText(frame, str(name), (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1)
        for i, (box, conf) in enumerate(zip(players.xyxy, players.confidence)):
            x1, y1, x2, y2 = (int(v) for v in box)
            kept = i in keep
            colour = GREEN if kept else RED
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(frame, f"{conf:.2f}" + ("" if kept else " DUP"),
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colour, 2)
        short = len(keep) < args.need
        t = idx / fps
        src = args.source_offset + t
        stamp = (f"{t:6.2f}s   src {int(src)//60:02d}:{src % 60:05.2f}"
                 if args.source_offset else f"{t:6.2f}s")
        cv2.putText(frame, f"f{idx}  {stamp}   {len(keep)} players"
                           + (f"  (short {args.need - len(keep)})" if short else ""),
                    (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 240) if short else (255, 255, 255), 3)
        ff.stdin.write(frame.tobytes())
        idx += 1
        prog.step(note=f"frame {idx}, {len(keep)} players")
    cap.release()
    ff.stdin.close()
    ff.wait()
    prog.done(note=f"{idx} frames")

    full = sum(n for c, n in counts.items() if c >= args.need)
    print(f"\nwrote {out}: {idx} frames")
    print(f"  {full} frames ({full/max(1,idx):.0%}) carry {args.need}+ players")
    if args.source_offset:
        print(f"  overlay timestamps are offset by {args.source_offset:.2f}s "
              f"to match the original recording")
    print(f"\n  {'players':>8} {'frames':>7}")
    for c in sorted(counts):
        bar = "#" * max(1, round(60 * counts[c] / idx))
        print(f"  {c:>8} {counts[c]:>7}  {bar}")


if __name__ == "__main__":
    main()
