"""Harvest jersey-number training data from our own footage.

The number-region detector was trained on 13 games of 2025 playoff kits
and has seen zero Summer League jerseys -- measured: 6.5 regions/frame on
its home domain, 2.2 on ours. This collects the missing chapter: frames
from the ESPN recording with the detector's own low-confidence guesses
attached as pseudo-annotations, so the labeling session is mostly
confirm-and-fix rather than draw-from-scratch.

    python pipeline/harvest_numbers.py \
        --video "d:/夸克/NBA_20260719_GSW @ MEM_1080p60_ESPN.mkv"

Check where tip-off actually is before harvesting: a broadcast recording
opens with the pre-game show, and sampling from frame zero spends the whole
budget on studio desks. --start-sec skips it, --every-sec spreads the budget
across the whole game so the time-based split's test set is a genuinely
unseen stretch of play.

Outputs a COCO-format dataset (images/ + annotations.json, one class:
number) under out/harvest_numbers/, ready for Roboflow upload or direct
rfdetr training. Frames with zero detections are kept at a low rate on
purpose -- those are where the labeler adds what the model cannot see,
which is the whole point of the exercise.
"""

import argparse
import json
from pathlib import Path

import cv2

import config
from progress import Progress

DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PSEUDO_CONF = 0.1          # low on purpose: candidates for a human to judge
KEEP_EMPTY_EVERY = 12      # every Nth empty frame is kept as a hard negative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every-sec", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--start-sec", type=float, default=0.0,
                    help="skip the pre-game show; a broadcast recording can "
                         "open with half an hour of studio segments")
    ap.add_argument("--out", default="out/harvest_numbers")
    args = ap.parse_args()

    config.load_env()
    import os
    os.environ.setdefault("ROBOFLOW_API_KEY", config.secret("ROBOFLOW_API_KEY"))
    import supervision as sv
    from inference import get_model

    model = get_model(model_id=DETECTION_MODEL_ID)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * args.every_sec)
    start = int(fps * args.start_sec)

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id, kept, empties = 1, 0, 0
    prog = Progress("harvest", total=min(args.max_frames, (total - start) // step))
    for f in range(start, total, step):
        if kept >= args.max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        det = sv.Detections.from_inference(
            model.infer(frame, confidence=PSEUDO_CONF, iou_threshold=0.9)[0])
        nums = det[det.data["class_name"] == "number"]
        if len(nums) == 0:
            empties += 1
            if empties % KEEP_EMPTY_EVERY:
                continue
        name = f"f{f:07d}.jpg"
        cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": name,
                       "width": frame.shape[1], "height": frame.shape[0]})
        for xyxy, conf in zip(nums.xyxy, nums.confidence):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                "score_hint": round(float(conf), 3),
            })
            ann_id += 1
        kept += 1
        prog.step(note=f"frame {f}, {kept} kept, {ann_id - 1} boxes")
    cap.release()

    (out / "annotations.json").write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "number"}],
    }), encoding="utf-8")
    prog.done(note=f"{kept} frames")
    print(f"harvested {kept} frames, {ann_id - 1} pseudo-boxes -> {out}")
    print("next: upload to Roboflow for the confirm-and-fix pass, or split "
          "by TIME (not randomly) and train directly")


if __name__ == "__main__":
    main()
