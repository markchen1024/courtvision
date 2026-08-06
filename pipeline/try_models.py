"""Eyeball test for two off-the-shelf models before committing to either.

Neither has published metrics, so the only honest way to judge them is to run
them on our own footage and look. Samples frames from the clip, draws player /
ball / referee boxes and court keypoints, and writes an annotated preview plus a
few full-size stills.

    python pipeline/try_models.py --video web/media/game.mp4 --every 30

What to look for in the output:
  detector  — are all ten players boxed? is the referee a separate class or
              being counted as a player? do boxes survive occlusion clusters?
  keypoints — do the court points sit on the actual court markings, and do they
              stay put between frames? jitter here means the top-down view will
              wobble, which is worse than it sounds.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DETECTOR = "koppolusameer/rfdetr-basketball-player-ball-referee-detection"
KEYPOINTS = "koppolusameer/yolo11n-basketball-court-keypoints"

# Blueprint palette, so the preview matches the viewer
CLASS_COLOUR = {
    "player": (49, 240, 200),      # BGR — signal lime
    "ball": (107, 114, 244),       # danger red
    "referee": (168, 201, 59),     # teal
}
KP_COLOUR = (49, 240, 200)
DIM = (111, 162, 154)


def load_detector():
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    import torch

    proc = AutoImageProcessor.from_pretrained(DETECTOR)
    model = AutoModelForObjectDetection.from_pretrained(DETECTOR)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return proc, model


def run_detector(proc, model, frame, threshold):
    import torch

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([[frame.shape[0], frame.shape[1]]])
    res = proc.post_process_object_detection(outputs, threshold=threshold, target_sizes=target)[0]

    out = []
    for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
        name = model.config.id2label.get(int(label), str(int(label)))
        out.append((name, float(score), [int(v) for v in box.tolist()]))
    return out


def load_keypoints():
    from ultralytics import YOLO

    return YOLO(KEYPOINTS)


def run_keypoints(model, frame, threshold):
    res = model.predict(frame, verbose=False)[0]
    if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
        return []
    xy = res.keypoints.xy[0].cpu().numpy()
    conf = (
        res.keypoints.conf[0].cpu().numpy()
        if res.keypoints.conf is not None
        else np.ones(len(xy))
    )
    return [(i, float(c), (float(x), float(y))) for i, (c, (x, y)) in enumerate(zip(conf, xy)) if c >= threshold]


def annotate(frame, dets, kps):
    out = frame.copy()
    for name, score, (x1, y1, x2, y2) in dets:
        colour = CLASS_COLOUR.get(name.lower(), DIM)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        tag = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (12, 12, 12), 1, cv2.LINE_AA)

    for idx, conf, (x, y) in kps:
        cv2.circle(out, (int(x), int(y)), 6, KP_COLOUR, -1)
        cv2.circle(out, (int(x), int(y)), 6, (12, 12, 12), 1)
        cv2.putText(out, str(idx), (int(x) + 8, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, KP_COLOUR, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--every", type=int, default=30, help="sample one frame in N")
    ap.add_argument("--limit", type=int, default=120, help="max frames to process")
    ap.add_argument("--det-threshold", type=float, default=0.4)
    ap.add_argument("--kp-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="out/model_check")
    args = ap.parse_args()

    outdir = Path(args.out)
    (outdir / "stills").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.video}: {w}x{h} @ {fps:.0f}fps, {total} frames")

    print(f"loading {DETECTOR}")
    proc, det_model = load_detector()
    print("  classes:", det_model.config.id2label)
    print(f"loading {KEYPOINTS}")
    kp_model = load_keypoints()

    writer = cv2.VideoWriter(
        str(outdir / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h)
    )

    class_counts, per_frame_players, kp_counts, kp_positions = Counter(), [], [], {}
    processed = 0
    idx = 0

    while processed < args.limit:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every:
            idx += 1
            continue

        dets = run_detector(proc, det_model, frame, args.det_threshold)
        kps = run_keypoints(kp_model, frame, args.kp_threshold)

        for name, _, _ in dets:
            class_counts[name] += 1
        per_frame_players.append(sum(1 for n, _, _ in dets if n.lower() == "player"))
        kp_counts.append(len(kps))
        for i, _, (x, y) in kps:
            kp_positions.setdefault(i, []).append((x, y))

        vis = annotate(frame, dets, kps)
        writer.write(vis)
        if processed < 6:
            cv2.imwrite(str(outdir / "stills" / f"frame_{idx:06d}.jpg"), vis)

        processed += 1
        idx += 1
        if processed % 20 == 0:
            print(f"  {processed} frames…")

    cap.release()
    writer.release()

    print(f"\nprocessed {processed} frames")
    print("detections by class:", dict(class_counts))
    if per_frame_players:
        arr = np.array(per_frame_players)
        print(f"players per frame: mean {arr.mean():.1f}  min {arr.min()}  max {arr.max()}  (10 is right)")
    if kp_counts:
        arr = np.array(kp_counts)
        print(f"court keypoints per frame: mean {arr.mean():.1f}  min {arr.min()}  max {arr.max()}")

    # Jitter is the number that decides whether per-frame homography is usable.
    # The camera pans, so points genuinely move; what matters is whether a point
    # moves smoothly or jumps. Report the median frame-to-frame step per point.
    steps = []
    for i, pts in kp_positions.items():
        if len(pts) > 2:
            p = np.array(pts)
            steps.extend(np.linalg.norm(np.diff(p, axis=0), axis=1).tolist())
    if steps:
        s = np.array(steps)
        print(f"keypoint step between sampled frames: median {np.median(s):.1f}px  p90 {np.percentile(s, 90):.1f}px")
        print("  (large p90 relative to median = jitter, and the top-down court will wobble)")

    print(f"\nwrote {outdir / 'preview.mp4'} and {min(processed, 6)} stills in {outdir / 'stills'}")


if __name__ == "__main__":
    main()
