"""Find the hard cuts, and the longest single-shot stretches between them.

SAM2 takes its prompts once and matches by appearance memory, so it holds
identity within a shot and loses it across one -- measured here and in
docs/tracking-comparison.md. The tutorial's demo footage is a single
continuous shot; a broadcast is not. Picking a segment therefore means
picking a stretch with no cut in it, which needs the cuts located to the
frame rather than guessed from a sparse sample.

No model for this. A hard cut replaces the whole frame at once, so the
colour histogram correlation between consecutive frames collapses, while a
pan or a player crossing barely moves it. Reading every frame sequentially
also avoids seeking, which is what made the sparse passes slow.

The correlation alone is not enough, and the first version of this was
wrong because of it: an arena is full of photographers' flashes and LED
boards, and a blown-out frame collapses the histogram exactly like a cut
does. It reported a cut every 1.3s on footage a human can see is one
continuous shot. The rendered pairs settled it -- same camera, same players,
same game clock, one frame washed white.

What separates them is persistence. A cut is permanent: the frame after it
still does not match the frame before. A flash is over in a frame or two and
the picture returns to what it was. So a collapse only counts once the
picture has failed to come back.

The remaining failure is the opposite one, and it is not fixed here: a cut
between two shots that happen to share a colour layout goes unseen. On this
broadcast a wide shot and a courtside close-up are both arena blue, crowd,
and pale wood, and one such cut survived both the whole-frame histogram and
the 3x3 grid. Treat the output as candidates and look at the frames before
committing a segment to a run -- eyes settle it in seconds, and every wrong
call here costs a multi-hour SAM2 pass.

    python pipeline/detect_cuts.py --video web/media/det.mp4
    python pipeline/detect_cuts.py --video "d:/game.mkv" --start-sec 1740 \
        --min-shot 20

Prints the longest cut-free stretches, longest first, as frame ranges and
timestamps ready to hand to ffmpeg.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from progress import Progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-sec", type=float, default=0.0,
                    help="0 reads to the end")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="histogram correlation below this is a cut; a pan "
                         "stays well above it")
    ap.add_argument("--flash-window", type=int, default=5,
                    help="frames the picture has to stay changed before a "
                         "collapse counts as a cut rather than a flash")
    ap.add_argument("--min-shot", type=float, default=8.0,
                    help="seconds; shorter shots are not worth reporting")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(fps * args.start_sec)
    end = total if not args.max_sec else min(total, start + int(fps * args.max_sec))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # A 3x3 grid, not one histogram over the frame. Whole-frame colour cannot
    # separate a wide shot from a courtside close-up here -- both are arena
    # blue, crowd, and pale wood, and the correlation stayed above threshold
    # across a cut we could see with our eyes. Per-cell histograms carry where
    # the colours are, so a cut moves every cell while a pan leaves several
    # recognisable.
    GX, GY, CELLS = 3, 3, 16

    def hist(frame):
        small = cv2.resize(frame, (321, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        cells = []
        for gy in range(GY):
            for gx in range(GX):
                cell = hsv[gy * 60:(gy + 1) * 60, gx * 107:(gx + 1) * 107]
                h = cv2.calcHist([cell], [0, 1], None, [CELLS, CELLS],
                                 [0, 180, 0, 256])
                cells.append(cv2.normalize(h, h).flatten())
        return cells

    def similarity(a, b):
        return float(np.mean([cv2.compareHist(x, y, cv2.HISTCMP_CORREL)
                              for x, y in zip(a, b)]))

    cuts = []
    flashes = 0
    prev = None
    pending = None          # (frame index, histogram from before the collapse)
    idx = start
    prog = Progress("detect-cuts", total=end - start)
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        h = hist(frame)
        if pending is not None:
            at, before = pending
            if similarity(before, h) >= args.threshold:
                flashes += 1          # picture came back: a flash, not a cut
                pending = None
            elif idx - at >= args.flash_window:
                cuts.append(at)       # still gone: a real cut
                pending = None
        elif prev is not None and \
                similarity(prev, h) < args.threshold:
            pending = (idx, prev)
        prev = h
        idx += 1
        if (idx - start) % 500 == 0:
            prog.step(500, note=f"frame {idx}, {len(cuts)} cuts, {flashes} flashes")
    if pending is not None:
        cuts.append(pending[0])
    cap.release()
    prog.done(note=f"{len(cuts)} cuts, {flashes} flashes")
    print(f"\nignored {flashes} flashes (picture returned within "
          f"{args.flash_window} frames)")

    bounds = [start] + cuts + [idx]
    shots = [(b - a, a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    print(f"\n{len(cuts)} cuts over {(idx - start) / fps:.0f}s "
          f"= a cut every {(idx - start) / fps / max(1, len(cuts)):.1f}s\n")

    long = sorted((s for s in shots if s[0] / fps >= args.min_shot), reverse=True)
    print(f"longest cut-free stretches (>= {args.min_shot:.0f}s):")
    for n, a, b in long[:args.top]:
        print(f"  {n / fps:6.1f}s  frames {a}-{b}  "
              f"start {a / fps:8.2f}s = {a / fps / 60:6.2f}min   "
              f"ffmpeg -ss {a / fps:.2f} -t {n / fps:.2f}")
    if not long:
        print("  none -- every shot is shorter than --min-shot")


if __name__ == "__main__":
    main()
