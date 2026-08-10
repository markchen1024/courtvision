"""Render what a number detector actually does on the labelled test split.

The score line is three numbers over 30 frames; one box either way moves it
by half a point. That is not enough to decide anything on, and metrics have
misled this project before -- the SAM2 run scored perfectly while the
identities were visibly on the wrong players. So every scoring run gets a
render, and the render is what gets looked at.

Green   true positive, matched a labelled box at IoU >= 0.5
Red     false positive, the model saw a number that is not there
Yellow  false negative, a labelled number the model missed

    python pipeline/review_numbers.py --dataset out/det_numbers_ds \
        --ckpt out/rfdetr_det_numbers/checkpoint_best_ema.pth
    python pipeline/review_numbers.py --dataset out/det_numbers_ds  # baseline

Writes out/review_numbers_<tag>/ plus a per-frame tally, so the frames that
disagree most can be opened first.
"""

import argparse
import json
from pathlib import Path

import cv2

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 235), (0, 200, 235)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union else 0.0


def base_model():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import config, os
    config.load_env()
    os.environ.setdefault("ROBOFLOW_API_KEY", config.secret("ROBOFLOW_API_KEY"))
    import supervision as sv
    from inference import get_model
    m = get_model(model_id="basketball-player-detection-3-ycjdo/4")

    def predict(frame):
        det = sv.Detections.from_inference(
            m.infer(frame, confidence=0.2, iou_threshold=0.9)[0])
        return det[det.data["class_name"] == "number"].xyxy.tolist()
    return predict


def finetuned(ckpt, threshold):
    from rfdetr import RFDETRBase
    from PIL import Image
    model = RFDETRBase(pretrain_weights=str(ckpt))

    def predict(frame):
        det = model.predict(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            threshold=threshold)
        return det.xyxy.tolist()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None,
                    help="omit to review the base detector instead")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = Path(args.dataset) / args.split
    tag = "finetune" if args.ckpt else "baseline"
    out = Path(args.out or f"out/review_numbers_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    predict = finetuned(args.ckpt, args.threshold) if args.ckpt else base_model()

    coco = json.loads((ds / "_annotations.coco.json").read_text(encoding="utf-8"))
    gt_by = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by.setdefault(a["image_id"], []).append([x, y, x + w, y + h])

    tally, tp_all = [], 0
    fp_all = fn_all = 0
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        frame = cv2.imread(str(ds / im["file_name"]))
        dets = predict(frame)
        gts = gt_by.get(im["id"], [])
        matched, tp, fp = set(), 0, 0
        for d in dets:
            best, best_j = 0.0, None
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                v = iou(d, g)
                if v > best:
                    best, best_j = v, j
            if best >= 0.5:
                matched.add(best_j)
                tp += 1
                cv2.rectangle(frame, (int(d[0]), int(d[1])),
                              (int(d[2]), int(d[3])), GREEN, 2)
            else:
                fp += 1
                cv2.rectangle(frame, (int(d[0]), int(d[1])),
                              (int(d[2]), int(d[3])), RED, 2)
        fn = 0
        for j, g in enumerate(gts):
            if j not in matched:
                fn += 1
                cv2.rectangle(frame, (int(g[0]), int(g[1])),
                              (int(g[2]), int(g[3])), YELLOW, 2)
        tp_all += tp
        fp_all += fp
        fn_all += fn
        cv2.putText(frame, f"{tag}  {im['file_name']}  "
                            f"tp {tp}  fp {fp}  fn {fn}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(str(out / im["file_name"]), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        tally.append((fp + fn, im["file_name"], tp, fp, fn))

    prec = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0
    rec = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0
    print(f"{tag} on {args.split}: precision {prec:.1%}  recall {rec:.1%}  "
          f"(tp {tp_all}, fp {fp_all}, fn {fn_all})")
    print(f"\nworst frames (open these first):")
    for bad, name, tp, fp, fn in sorted(tally, reverse=True)[:8]:
        if bad == 0:
            break
        print(f"  {name}  tp {tp}  fp {fp}  fn {fn}")
    print(f"\nrendered -> {out}")


if __name__ == "__main__":
    main()
