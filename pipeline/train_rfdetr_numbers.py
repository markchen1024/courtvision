"""Fine-tune RF-DETR on our own jersey-number data. Run this yourself.

This is the counterpart to harvest_numbers.py: that script collected the
frames and pseudo-boxes from our footage, a human confirm-and-fix pass
turns them into labels, and this script does the split / baseline / train
/ evaluate loop. Designed to run on the 4080 box:

    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install rfdetr supervision pycocotools

    python pipeline/train_rfdetr_numbers.py --prepare   # time-based split
    python pipeline/train_rfdetr_numbers.py --baseline  # score BEFORE training
    python pipeline/train_rfdetr_numbers.py --train     # the actual fine-tune
    python pipeline/train_rfdetr_numbers.py --eval      # score AFTER

Two decisions worth being able to defend out loud:

  time-based split   frames two seconds apart are near-duplicates; a random
                     split would leak them across train/test and inflate the
                     score. The last 20% of the GAME is the test set -- the
                     model is judged on minutes it has never seen.
  single class       only the 'number' region is trained. The ten-class
                     detector stays as-is; this model exists to fix the one
                     measured gap (2.2 regions/frame on SL kits vs 6.5 on
                     the playoff kits it was trained on).
"""

import argparse
import json
import shutil
from pathlib import Path

SRC = Path("out/harvest_numbers")
DS = Path("out/sl_numbers_ds")           # rfdetr expects train/valid/test dirs
CKPT_DIR = Path("out/rfdetr_numbers")
TRAIN, VALID, TEST = 0.7, 0.1, 0.2       # fractions of GAME TIME, in order


def coco_subset(coco, image_ids):
    ids = set(image_ids)
    return {
        "images": [i for i in coco["images"] if i["id"] in ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        "categories": coco["categories"],
    }


def prepare(src=SRC, ds=DS):
    coco = json.loads((src / "annotations.json").read_text(encoding="utf-8"))
    images = sorted(coco["images"], key=lambda i: i["file_name"])  # time order
    n = len(images)
    cuts = [int(n * TRAIN), int(n * (TRAIN + VALID))]
    splits = {"train": images[:cuts[0]],
              "valid": images[cuts[0]:cuts[1]],
              "test": images[cuts[1]:]}
    for name, imgs in splits.items():
        d = ds / name
        d.mkdir(parents=True, exist_ok=True)
        for im in imgs:
            shutil.copy(src / "images" / im["file_name"], d / im["file_name"])
        (d / "_annotations.coco.json").write_text(
            json.dumps(coco_subset(coco, [i["id"] for i in imgs])),
            encoding="utf-8")
        boxes = sum(1 for a in coco["annotations"]
                    if a["image_id"] in {i["id"] for i in imgs})
        print(f"{name}: {len(imgs)} frames, {boxes} boxes "
              f"(game-time order, no shuffling)")


def score(model_ref, ds=DS, split="test"):
    """Recall/precision of number regions against the labeled split."""
    coco = json.loads((ds / split / "_annotations.coco.json")
                      .read_text(encoding="utf-8"))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])

    import cv2
    tp = fp = fn = 0
    for im in coco["images"]:
        frame = cv2.imread(str(ds / split / im["file_name"]))
        dets = model_ref(frame)   # -> Nx4 xyxy
        gts = [[x, y, x + w, y + h] for x, y, w, h in by_img.get(im["id"], [])]
        matched = set()
        for d in dets:
            best_iou, best_j = 0.0, None
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                ix1, iy1 = max(d[0], g[0]), max(d[1], g[1])
                ix2, iy2 = min(d[2], g[2]), min(d[3], g[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = ((d[2]-d[0])*(d[3]-d[1]) + (g[2]-g[0])*(g[3]-g[1])
                         - inter)
                iou = inter / union if union else 0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5:
                matched.add(best_j)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(matched)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"{split}: precision {prec:.1%}  recall {rec:.1%}  "
          f"(tp {tp}, fp {fp}, fn {fn})")


def baseline(ds=DS):
    """The tutorial's ten-class detector, before any training of ours."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import config
    config.load_env()
    config.inference_env()   # key, cache path, GPU provider -- before the import
    import supervision as sv
    from inference import get_model
    m = get_model(model_id="basketball-player-detection-3-ycjdo/4")

    def ref(frame):
        det = sv.Detections.from_inference(
            m.infer(frame, confidence=0.2, iou_threshold=0.9)[0])
        return det[det.data["class_name"] == "number"].xyxy.tolist()
    score(ref, ds)


def train(ds=DS, ckpt_dir=CKPT_DIR):
    from rfdetr import RFDETRBase
    model = RFDETRBase()
    model.train(dataset_dir=str(ds), epochs=50, batch_size=8,
                grad_accum_steps=2, lr=1e-4, output_dir=str(ckpt_dir))


def evaluate(ds=DS, ckpt_dir=CKPT_DIR):
    from rfdetr import RFDETRBase
    ckpt = ckpt_dir / "checkpoint_best_ema.pth"
    if not ckpt.exists():
        ckpt = ckpt_dir / "checkpoint_best.pth"
    model = RFDETRBase(pretrain_weights=str(ckpt))

    def ref(frame):
        import cv2
        from PIL import Image
        det = model.predict(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            threshold=0.3)
        return det.xyxy.tolist()
    score(ref, ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--src", default=str(SRC),
                    help="labelled COCO harvest, or the wide-shot subset "
                         "filter_harvest.py wrote")
    ap.add_argument("--dataset", default=str(DS))
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    args = ap.parse_args()
    src, ds, ckpt_dir = Path(args.src), Path(args.dataset), Path(args.ckpt_dir)
    if args.prepare:
        prepare(src, ds)
    if args.baseline:
        baseline(ds)
    if args.train:
        train(ds, ckpt_dir)
    if args.eval:
        evaluate(ds, ckpt_dir)
    if not any((args.prepare, args.baseline, args.train, args.eval)):
        print("pass --prepare / --baseline / --train / --eval (in that order)")


if __name__ == "__main__":
    main()
