"""The tutorial's finished look: club-tinted player masks and name chips.

blog.roboflow.com/identify-basketball-players ends with players silhouetted
in team colours and '#11 BRUNSON' chips at their feet, drawn by
supervision's MaskAnnotator and RichLabelAnnotator with the Staatliches
font -- and this uses exactly those, not hand-rolled drawing. The one
adaptation: their masks fall out of SAM2-as-tracker, our tracks carry
boxes, so sam2.1_b runs here in image mode, prompted by each frame's
boxes. Tracker choice and finished look stay independent.

    python pipeline/render_final.py --start 1375 --end 2125 \
        --out out/final_30s.mp4

Chips only appear on tracks identify.py confirmed a number for; everyone
else gets the club tint (or grey when the cluster never settled). No
guessing on screen.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from progress import Progress

FONT = "out/fonts/Staatliches-Regular.ttf"   # the notebook's typeface
CHIP_H = 34          # a chip's height in pixels, for pushing overlaps apart
CLUB_COLOURS = {"warriors": "#FFC72C", "grizzlies": "#5D76A9",
                "knicks": "#F58426", "celtics": "#007A33",
                # Pistons red, not their blue: the Knicks wear blue on this
                # footage and two blues on one floor defeat the point.
                "pistons": "#C8102E"}
UNKNOWN_HEX = "#9AA0A6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--boxes", default="out/track_boxes_nba.json")
    ap.add_argument("--identities", default="out/identities_nba.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nvenc", action="store_true",
                    help="encode on the GPU (h264_nvenc) instead of libx264. "
                         "Off by default: ffmpeg lists h264_nvenc on this box "
                         "but cannot open it (driver 591.86), and the encode "
                         "is not the bottleneck anyway -- SAM2 runs per frame.")
    args = ap.parse_args()

    import supervision as sv
    from ultralytics import SAM

    sidecar = json.loads(Path(args.boxes).read_text())
    frames = {int(k): v for k, v in sidecar["frames"].items()}
    idn = json.loads(Path(args.identities).read_text())["identities"]
    idn = {int(k): v for k, v in idn.items()}

    clubs_present = sorted({v.get("club") for v in idn.values() if v.get("club")})
    club_order = clubs_present + [None]
    club_hex = [CLUB_COLOURS.get(c, UNKNOWN_HEX) for c in clubs_present] + [UNKNOWN_HEX]
    palette = sv.ColorPalette.from_hex(club_hex)
    mask_annotator = sv.MaskAnnotator(
        color=palette, opacity=0.5, color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.RichLabelAnnotator(
        font_path=FONT, font_size=26, color=palette,
        text_color=sv.Color.WHITE, text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.INDEX)

    # "Tim Hardaway Jr." ends in "Jr.", not in a surname, and taking the last
    # word rendered him as "#8 Jr." on screen. Same trap waiting in "Ronald
    # Holland II" and "Lindy Waters III".
    SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

    def surname_of(full):
        parts = full.split()
        tail = []
        while len(parts) > 1 and parts[-1].lower().rstrip(".") in SUFFIXES:
            tail.insert(0, parts.pop())
        return " ".join(parts[-1:] + tail)

    def chip(tid):
        v = idn.get(tid)
        if not v or not v["number"]:
            return None
        surname = surname_of(v["name"]) if v["name"] else ""
        return f"#{v['number']} {surname}".strip()

    sam = SAM("sam2.1_b.pt")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = args.start, min(args.end, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # The GPU is already busy running SAM2 per frame; NVENC costs it almost
    # nothing and takes the encode off the CPU, which is the other thing
    # feeding this pipe.
    codec = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
             if args.nvenc else
             ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast"])
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", *codec,
         "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)

    prog = Progress("final-render", total=end - start + 1)
    rows = []
    for f in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if f in frames:
            rows = [r for r in frames[f]
                    if not (idn.get(r["tid"]) or {}).get("ignored")]
        if rows:
            boxes = np.array([r["box"] for r in rows], np.float32)
            res = sam(frame, bboxes=boxes.tolist(), verbose=False)[0]
            masks = (res.masks.data.cpu().numpy() > 0.5) \
                if res.masks is not None else None
            clubs = np.array(
                [club_order.index((idn.get(r["tid"]) or {}).get("club"))
                 for r in rows])
            det = sv.Detections(
                xyxy=boxes,
                mask=masks if masks is not None and len(masks) == len(rows) else None,
                class_id=clubs)
            frame = mask_annotator.annotate(
                scene=frame, detections=det, custom_color_lookup=clubs)
            labels = [chip(r["tid"]) for r in rows]
            keep = np.array([i for i, t in enumerate(labels) if t])
            if len(keep):
                shown = det[keep]
                # Chips sit under a player's feet, and players stand next to
                # each other, so two chips land on the same pixels and one is
                # drawn over the other -- "#32 Towns" was unreadable behind
                # "#12 Harris". Push overlapping chips down a row each: the
                # anchor stays under the right player, only the text moves.
                order = np.argsort(shown.xyxy[:, 3])          # top of court first
                taken = []
                offsets = np.zeros(len(shown))
                for i in order:
                    x1, _, x2, y2 = shown.xyxy[i]
                    row = 0
                    while any(abs((y2 + row * CHIP_H) - ty) < CHIP_H
                              and not (x2 < tx1 or x1 > tx2)
                              for tx1, tx2, ty in taken):
                        row += 1
                    offsets[i] = row * CHIP_H
                    taken.append((x1, x2, y2 + row * CHIP_H))
                nudged = shown.xyxy.copy()
                nudged[:, 1] += offsets
                nudged[:, 3] += offsets
                shown = sv.Detections(xyxy=nudged, class_id=shown.class_id)
                frame = label_annotator.annotate(
                    scene=frame, detections=shown,
                    labels=[labels[i] for i in keep],
                    custom_color_lookup=clubs[keep])
        ff.stdin.write(frame.tobytes())
        prog.step(note=f"frame {f}")
    cap.release()
    ff.stdin.close()
    ff.wait()
    prog.done()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
