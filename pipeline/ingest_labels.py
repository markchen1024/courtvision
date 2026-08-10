"""Bring hand labels back from Roboflow and clean what the labels now reveal.

filter_harvest.py had to judge which frames were wide game shots using the
pseudo-boxes, because that was all there was before labelling. Those boxes
are wrong in exactly the frames that matter: on a close-up the detector
scatters small guesses over the background bench and crowd, which reads as
"many small boxes" -- the signature of a wide shot. So a few close-ups slip
through.

After labelling the boxes are true, so the same judgement can be made again
and made properly. On the NYK @ DET set that recovered 5 close-ups out of
150 frames, missed the first time round.

It also drops boxes whose aspect ratio no number region can have. A number
is roughly 0.4-2.5 wide over tall; a 13x131 box is a drag that overshot the
frame edge, and an 8x38 sliver is a number the frame cut in half. Neither
teaches the model anything except to emit degenerate boxes.

    python pipeline/ingest_labels.py --src out/_rf_export/train \
        --out out/harvest_labelled

Filenames are restored to the harvest's f<frame>.jpg, because everything
downstream sorts by filename to recover game-time order.
"""

import argparse
import json
import shutil
import statistics
from pathlib import Path


def original_name(name):
    """f0104295_jpg.rf.<hash>.jpg -> f0104295.jpg"""
    stem = name.split(".rf.")[0]
    for suffix in (".jpg", "_jpg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return f"{stem}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="the train/ directory of a Roboflow COCO export")
    ap.add_argument("--out", default="out/harvest_labelled")
    ap.add_argument("--max-median-h", type=float, default=45.0,
                    help="px; frames whose median box is taller are close-ups")
    ap.add_argument("--min-aspect", type=float, default=0.25)
    ap.add_argument("--max-aspect", type=float, default=3.0)
    args = ap.parse_args()

    src = Path(args.src)
    coco = json.loads((src / "_annotations.coco.json")
                      .read_text(encoding="utf-8"))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    dropped_boxes = []
    for image_id, anns in by_img.items():
        keep = []
        for a in anns:
            _, _, w, h = a["bbox"]
            ar = w / h if h > 0 else 0
            if args.min_aspect <= ar <= args.max_aspect:
                keep.append(a)
            else:
                dropped_boxes.append((a, ar))
        by_img[image_id] = keep

    kept, close_ups = [], []
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        if anns:
            med_h = statistics.median(a["bbox"][3] for a in anns)
            if med_h >= args.max_median_h:
                close_ups.append((im, med_h))
                continue
        kept.append(im)

    print(f"{len(coco['images'])} labelled frames in")
    print(f"  {len(dropped_boxes)} boxes dropped on aspect ratio")
    for a, ar in dropped_boxes:
        w, h = a["bbox"][2], a["bbox"][3]
        print(f"      {w:.0f}x{h:.0f}  w/h {ar:.2f}")
    print(f"  {len(close_ups)} close-ups dropped (median box >= "
          f"{args.max_median_h:.0f}px)")
    for im, med_h in close_ups:
        print(f"      {original_name(im['file_name'])}  median {med_h:.0f}px")

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for im in kept:
        name = original_name(im["file_name"])
        shutil.copy(src / im["file_name"], img_dir / name)
        images.append({**im, "file_name": name})
        annotations.extend(by_img.get(im["id"], []))

    (out / "annotations.json").write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }), encoding="utf-8")

    empties = sum(1 for i in kept if not by_img.get(i["id"]))
    print(f"\nwrote {len(images)} frames ({empties} hard negatives), "
          f"{len(annotations)} boxes -> {out}")


if __name__ == "__main__":
    main()
