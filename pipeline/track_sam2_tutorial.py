"""SAM2 tracking exactly as the tutorial does it, for a like-for-like comparison.

track_sam2.py is not the notebook's tracker, and the differences were only
visible once the notebook was read line by line:

  detector    the notebook prompts SAM2 with basketball-player-detection-3
              -ycjdo/4 -- the same model it uses everywhere else. Ours took its
              prompts from the dense cache in track_bytetrack.py, which calls
              project.py's detector: koppolusameer/rfdetr-basketball-player-ball
              -referee-detection, a different model off HuggingFace. On frame 0
              of the 30:05 segment the notebook's detector returns 11 players
              and ours returns 10.
  classes     PLAYER_CLASS_IDS = [3, 4, 5, 6, 7] -- player, player-in-possession,
              player-jump-shot, player-layup-dunk, player-shot-block. Ours kept
              only class_name == "player". They agree on frame 0 here, but a
              frame with someone mid-shot would not.
  thresholds  confidence 0.4, iou 0.9.
  weights     sam2.1_hiera_large through segment-anything-2-real-time's camera
              predictor. Ours ran sam2.1_b through the ultralytics wrapper.

    python pipeline/track_sam2_tutorial.py --video web/media/det_final.mp4 \
        --out out/tracks_sam2tut_detfinal.json

Output matches track_sam2.py's schema, so identify.py and render_final.py take
either one and the two can be scored against each other.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import config
from progress import Progress

SAM2_REPO = Path(r"C:\Users\fqche\segment-anything-2-real-time")
DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
CONFIDENCE = 0.4          # notebook cell 25
IOU_THRESHOLD = 0.9       # notebook cell 25
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]   # notebook cell 31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint",
                    default=str(SAM2_REPO / "checkpoints" / "sam2.1_hiera_large.pt"))
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    config.load_env()
    config.inference_env()   # key, cache path, GPU provider -- before the import

    import cv2
    import torch
    import supervision as sv
    from inference import get_model

    # hydra resolves the config path relative to the sam2 package root
    os.chdir(SAM2_REPO)
    from sam2.build_sam import build_sam2_camera_predictor

    predictor = build_sam2_camera_predictor(args.sam2_config, args.checkpoint)
    detector = get_model(model_id=DETECTION_MODEL_ID)

    video = Path(args.video)
    if not video.is_absolute():
        video = Path(__file__).resolve().parent.parent / video
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("empty video")

    result = detector.infer(frame, confidence=CONFIDENCE,
                            iou_threshold=IOU_THRESHOLD)[0]
    detections = sv.Detections.from_inference(result)
    detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)
    if len(detections) == 0:
        raise SystemExit("no players on frame 0 to prompt with")
    print(f"prompting SAM2 with {len(detections)} players from frame 0 "
          f"(classes {sorted(set(int(c) for c in detections.class_id))})")

    # SAM2Tracker.prompt_first_frame, notebook cell 40
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.load_first_frame(frame)
        for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
            predictor.add_new_prompt(frame_idx=0, obj_id=int(obj_id),
                                     bbox=np.asarray([xyxy], dtype=np.float32))

    tracks = {}
    idx = 0
    prog = Progress("sam2-tutorial", total=total or None)
    while True:
        # SAM2Tracker.propagate, notebook cell 40
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = predictor.track(frame)
        ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        masks = np.array([
            sv.filter_segments_by_distance(m, relative_distance=0.03, mode="edge")
            for m in masks
        ])
        xyxy = sv.mask_to_xyxy(masks=masks)

        rows = []
        for tid, box in zip(ids, xyxy):
            if box[2] > box[0] and box[3] > box[1]:   # a lost object yields an empty mask
                rows.append({"tid": int(tid), "box": [float(v) for v in box]})
        tracks[idx] = rows

        idx += 1
        prog.step(note=f"frame {idx}, {len(rows)} objects")
        if idx % 200 == 0:
            print(f"  {idx} frames...", flush=True)
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()

    lifetimes = {}
    for i, rows in tracks.items():
        for r in rows:
            a, b = lifetimes.get(r["tid"], (i, i))
            lifetimes[r["tid"]] = (min(a, i), max(b, i))
    prog.done(note=f"{idx} frames, {len(lifetimes)} objects")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent.parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "video": args.video, "every": 1, "fps": fps,
        "tracker": f"segment-anything-2-real-time camera predictor "
                   f"({Path(args.checkpoint).name})",
        "detector": DETECTION_MODEL_ID,
        "prompted": len(detections),
        "frames": {str(k): v for k, v in tracks.items()},
    }))
    spans = [(b - a) / fps for a, b in lifetimes.values()]
    print(f"wrote {out}: {idx} frames, {len(lifetimes)} of {len(detections)} "
          f"prompted objects ever seen")
    if spans:
        print(f"  lifetime median {sorted(spans)[len(spans)//2]:.1f}s  "
              f"max {max(spans):.1f}s (clip is {idx/fps:.0f}s)")


if __name__ == "__main__":
    main()
