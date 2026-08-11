"""Do the signals we already have predict whether an OCR read is right?

The published state of the art filters unreadable crops before voting, with a
trained legibility classifier (Koshkina et al., CVPRW 2024: ResNet34, 94.5%).
Before training anything, this asks whether the free signals already attached
to every read -- the number region's size, the detector's confidence in it,
and how sharp the crop is -- separate the reads that agree with their track's
majority from the ones that do not.

There is no ground truth for individual reads, so agreement with the track's
own majority is the proxy. It is not perfect: where the majority is itself
wrong (Hart's 3 read as 9), a correct read counts as a disagreement. That
biases the measurement against finding a signal, which is the safe direction
-- a signal that shows up anyway is real.

    python pipeline/measure_legibility.py --video web/media/det_final.mp4 \
        --boxes out/tracks_sam2tut_detfinal.json --out out/legibility.json

Prints how each feature splits agreeing from disagreeing reads, and what a
threshold on it would cost and save.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config
from progress import Progress

DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
OCR_PROMPT = "Read the number."
NUMBER_CLASS_ID = 2
NUMBER_CONF = 0.2          # identify.py's floor
IOS_THRESHOLD = 0.9


def sharpness(crop):
    """Variance of the Laplacian -- the standard cheap blur measure. A crisp
    edge produces large second derivatives; motion blur flattens them."""
    import cv2
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", default="out/legibility.json")
    ap.add_argument("--stride", type=int, default=5)
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

    sidecar = json.loads(path(args.boxes).read_text(encoding="utf-8"))
    frames = {int(k): v for k, v in sidecar["frames"].items()}

    detector = get_model(model_id=DETECTION_MODEL_ID)
    ocr = get_model(model_id=OCR_MODEL_ID)

    cap = cv2.VideoCapture(str(path(args.video)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    grid = sorted(frames)[::args.stride]

    reads = []
    idx = 0
    prog = Progress("legibility", total=len(grid))
    want = set(grid)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            rows = frames.get(idx, [])
            if rows:
                h, w = frame.shape[:2]
                res = detector.infer(frame, confidence=NUMBER_CONF,
                                     iou_threshold=0.9)[0]
                nums = sv.Detections.from_inference(res)
                nums = nums[nums.class_id == NUMBER_CLASS_ID]
                for nbox, nconf in zip(nums.xyxy, nums.confidence):
                    # attach to the player whose box contains most of it
                    best_tid, best_ios = None, 0.0
                    nx1, ny1, nx2, ny2 = nbox
                    narea = max(1.0, (nx2-nx1) * (ny2-ny1))
                    for r in rows:
                        px1, py1, px2, py2 = r["box"]
                        ix = max(0.0, min(nx2, px2) - max(nx1, px1))
                        iy = max(0.0, min(ny2, py2) - max(ny1, py1))
                        ios = (ix * iy) / narea
                        if ios > best_ios:
                            best_ios, best_tid = ios, r["tid"]
                    if best_tid is None or best_ios < IOS_THRESHOLD:
                        continue
                    pad = sv.clip_boxes(sv.pad_boxes(
                        xyxy=np.array([nbox]), px=10, py=10), (w, h))[0]
                    crop = sv.crop_image(frame, pad)
                    if crop.size == 0:
                        continue
                    value = ocr.predict(crop, OCR_PROMPT)[0]
                    reads.append({
                        "tid": int(best_tid), "frame": idx,
                        "value": str(value),
                        "w": float(nx2 - nx1), "h": float(ny2 - ny1),
                        "conf": float(nconf),
                        "sharpness": sharpness(crop),
                    })
            prog.step(note=f"frame {idx}, {len(reads)} reads")
        idx += 1
    cap.release()
    prog.done(note=f"{len(reads)} reads")

    # proxy truth: the track's own majority
    by_tid = defaultdict(Counter)
    for r in reads:
        by_tid[r["tid"]][r["value"]] += 1
    majority = {t: c.most_common(1)[0][0] for t, c in by_tid.items()}
    for r in reads:
        r["agrees"] = r["value"] == majority[r["tid"]]

    path(args.out).write_text(json.dumps({"reads": reads,
                                          "majority": majority}),
                              encoding="utf-8")

    agree = [r for r in reads if r["agrees"]]
    differ = [r for r in reads if not r["agrees"]]
    print(f"\n{len(reads)} reads: {len(agree)} agree with their track's "
          f"majority, {len(differ)} do not ({len(differ)/max(1,len(reads)):.0%})")

    print(f"\n{'feature':<12} {'agreeing':>22} {'disagreeing':>22}")
    print(f"{'':12} {'p25':>7}{'median':>8}{'p75':>7} {'p25':>7}{'median':>8}{'p75':>7}")
    for key in ("h", "w", "conf", "sharpness"):
        a = np.array([r[key] for r in agree]) if agree else np.zeros(1)
        d = np.array([r[key] for r in differ]) if differ else np.zeros(1)
        print(f"{key:<12} "
              f"{np.percentile(a,25):>7.1f}{np.median(a):>8.1f}{np.percentile(a,75):>7.1f} "
              f"{np.percentile(d,25):>7.1f}{np.median(d):>8.1f}{np.percentile(d,75):>7.1f}")

    print("\nwhat a threshold would cost and save:")
    print(f"{'rule':<24} {'reads kept':>11} {'bad dropped':>12} {'good lost':>10}")
    for key, lo, hi in (("h", 14, 40), ("conf", 0.3, 0.7),
                        ("sharpness", 50, 400)):
        for t in np.linspace(lo, hi, 4):
            kept = [r for r in reads if r[key] >= t]
            bad_dropped = len(differ) - sum(1 for r in kept if not r["agrees"])
            good_lost = len(agree) - sum(1 for r in kept if r["agrees"])
            print(f"{key + ' >= ' + f'{t:.2f}':<24} {len(kept):>11} "
                  f"{bad_dropped:>12} {good_lost:>10}")


if __name__ == "__main__":
    main()
