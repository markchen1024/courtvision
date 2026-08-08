"""Train a court keypoint model, so calibration stops needing a human.

The two ready-made models both fail on community-gym footage -- measured at 182
and 268 pixels of geometric inconsistency against a 1px feature-matching ruler --
so the remaining off-the-shelf option is the dataset rather than the model.
reloc2 has 1468 labelled broadcast frames and no trained version, which is why
the tutorial it comes from trains its own.

    python pipeline/train_court.py --prepare      # download and fix the export
    python pipeline/train_court.py --epochs 100

Two things about the export are wrong out of the box and both are silent:

  flip_idx  ships as the identity, so a horizontal flip leaves every keypoint
            labelled as itself. Half of the augmented samples would then teach
            the model that the left baseline is the right one. Recomputed here
            from court_model.py by mirroring x, which is derivation rather than
            transcription -- getting it backwards is invisible in the loss curve
            and obvious only in a wrong homography weeks later.
  licence   the dataset is marked Private, not an open licence, unlike
            basketball-court-detection-2 which is CC BY 4.0. Fine for something
            run on a laptop; worth knowing before it goes anywhere else.

Keypoint work wants resolution more than it wants parameters: a landmark is a
few pixels wide, and halving the input costs more accuracy than dropping a model
size does. Hence 1280 on a small model rather than 640 on a large one.
"""

import argparse
import io
import os
import re
import zipfile
from pathlib import Path

DATASET = "fyp-3bwmg/reloc2-den7l/1"
ROOT = Path("out/reloc2")


def correct_flip_idx():
    """Mirror each landmark across the halfway line and find what it becomes."""
    from court_model import RELOC2, COURT_LENGTH

    idx = []
    for x, y in RELOC2:
        mx, my = COURT_LENGTH - x, y
        partner = next((k for k, (px, py) in enumerate(RELOC2)
                        if abs(px - mx) < 1e-6 and abs(py - my) < 1e-6), None)
        if partner is None:
            raise SystemExit(f"landmark at ({x}, {y}) has no mirror; flip_idx would be wrong")
        idx.append(partner)
    if [idx[i] for i in idx] != list(range(len(RELOC2))):
        raise SystemExit("flip_idx is not an involution, so it is wrong")
    return idx


def prepare():
    import requests
    from config import secret

    key = secret("ROBOFLOW_API_KEY")
    link = requests.get(f"https://api.roboflow.com/{DATASET}/yolov8",
                        params={"api_key": key}, timeout=180).json()["export"]["link"]
    print(f"downloading {DATASET}...")
    data = requests.get(link, timeout=1800).content
    ROOT.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(ROOT)
    print(f"  extracted {len(data) / 1e6:.0f} MB to {ROOT}")

    yaml_path = ROOT / "data.yaml"
    text = yaml_path.read_text()
    flip = correct_flip_idx()
    before = re.search(r"flip_idx: \[[^\]]*\]", text)
    text = re.sub(r"flip_idx: \[[^\]]*\]", "flip_idx: " + str(flip), text)
    # Ultralytics resolves the split paths relative to the yaml, and the export
    # writes them assuming a directory layout one level up.
    text = re.sub(r"^(train|val|test): .*$",
                  lambda m: f"{m.group(1)}: {m.group(1) if m.group(1) != 'val' else 'valid'}/images",
                  text, flags=re.M)
    text = f"path: {ROOT.resolve().as_posix()}\n" + text
    yaml_path.write_text(text)
    print(f"  flip_idx was {before.group(0) if before else '?'}")
    print(f"  flip_idx now flip_idx: {flip}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="fetch and fix the dataset, then stop")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="court")
    args = ap.parse_args()

    if args.prepare:
        prepare()
        return

    if not (ROOT / "data.yaml").exists():
        raise SystemExit(f"no dataset at {ROOT}; run with --prepare first")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str((ROOT / "data.yaml").resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="out/train",
        name=args.name,
        exist_ok=True,
        # The court never appears upside down or rotated, and it never appears
        # mirrored in a way the labels do not already describe, so the geometric
        # augmentations are turned down rather than left at their defaults.
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,       # stitching four courts together teaches nothing here
        fliplr=0.5,       # safe now that flip_idx is right
        flipud=0.0,
    )
    print("\nweights: out/train/" + args.name + "/weights/best.pt")
    print("next: measure it the same way the ready-made models were measured, with")
    print("      pipeline/try_court_model.py, so the numbers are comparable.")


if __name__ == "__main__":
    main()
