"""Burn tracking boxes into a video for eyeball review.

Metrics said the SAM2 identities were perfect; the rendered frames said
otherwise. This makes that check routine: feed any tracks JSON (SAM2,
ByteTrack -- same schema) and get a playable mp4 with per-id colored boxes,
so "did the right box stay on the right player" is a thing you watch, not
a number you trust.

    python pipeline/render_tracks.py --video out/nba_test10.mp4 \
        --tracks out/tracks_sam2_10s.json --out out/review_sam2_10s.mp4

cv2's mp4v encode plays poorly in browsers, so frames are piped straight
to ffmpeg for H.264.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

import config
from progress import Progress

# distinct hues, dark-footage friendly (BGR)
COLORS = [(61, 204, 145), (72, 175, 240), (115, 115, 255), (64, 201, 255),
          (255, 120, 180), (80, 255, 255), (200, 200, 80), (255, 140, 255),
          (140, 255, 140), (100, 160, 255), (255, 200, 100), (180, 255, 220)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--labels", help="JSON with a top-level labels map "
                    "(tid -> text), e.g. identify.py output; falls back to tid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.tracks).read_text())
    frames = {int(k): v for k, v in data["frames"].items()}
    every = int(data.get("every", 1))
    labels = {}
    if args.labels:
        labels = {int(k): v for k, v in
                  json.loads(Path(args.labels).read_text())["labels"].items()}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    config.ensure_ffmpeg()
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE)

    prog = Progress("render", total=total)
    idx = 0
    last = []   # sparse tracks (every>1) hold their boxes until the next sample
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in frames:
            last = frames[idx]
        elif every == 1:
            last = []
        for t in last:
            x1, y1, x2, y2 = (int(v) for v in t["box"])
            c = COLORS[t["tid"] % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
            label = labels.get(t["tid"], str(t["tid"]))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), c, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
        cv2.putText(frame, f"{idx}  t={idx / fps:.1f}s", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        ff.stdin.write(frame.tobytes())
        idx += 1
        prog.step(note=f"frame {idx}")
    cap.release()
    ff.stdin.close()
    ff.wait()
    prog.done()
    print(f"wrote {args.out}: {idx} frames with boxes from {args.tracks}")


if __name__ == "__main__":
    main()
