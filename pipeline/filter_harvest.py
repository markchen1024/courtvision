"""Keep the wide game shots out of a harvest, drop the close-ups.

A broadcast cuts constantly between the wide shot that shows a lineup and
close-ups, bench reactions and crowd cutaways. Only the wide shot matches
what identify.py reads downstream, and the two differ by roughly 5x in
scale -- measured on the NYK @ DET harvest, number regions are 24px tall in
the wide shot and 41px in the close-ups.

The close-ups are also where the pseudo-boxes are worst: on a bench shot the
detector boxed a sponsor logo and missed the number filling half the frame.
Labelling them is draw-from-scratch work that then pulls the training scale
distribution away from the frames the model has to serve.

    python pipeline/filter_harvest.py --src out/harvest_numbers \
        --out out/harvest_wide --max-frames 150

Frames are thinned evenly across game time, never randomly, so the
time-based split downstream still sees the whole game.
"""

import argparse
import json
import shutil
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/harvest_numbers")
    ap.add_argument("--out", default="out/harvest_wide")
    ap.add_argument("--min-boxes", type=int, default=3,
                    help="a wide shot shows most of a lineup")
    ap.add_argument("--max-median-h", type=float, default=45.0,
                    help="px; wide-shot number regions sit around 25")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 keeps every wide frame; otherwise thin evenly "
                         "across game time")
    ap.add_argument("--keep-empty", action="store_true", default=True,
                    help="keep the zero-detection frames as hard negatives")
    args = ap.parse_args()

    src = Path(args.src)
    coco = json.loads((src / "annotations.json").read_text(encoding="utf-8"))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    kept, dropped = [], 0
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        anns = by_img.get(im["id"], [])
        if not anns:
            if args.keep_empty:
                kept.append(im)
            else:
                dropped += 1
            continue
        med_h = statistics.median(a["bbox"][3] for a in anns)
        if len(anns) >= args.min_boxes and med_h < args.max_median_h:
            kept.append(im)
        else:
            dropped += 1

    print(f"{len(coco['images'])} frames in, {len(kept)} wide, {dropped} "
          f"close-ups dropped")

    if args.max_frames and len(kept) > args.max_frames:
        stride = len(kept) / args.max_frames
        kept = [kept[int(i * stride)] for i in range(args.max_frames)]
        print(f"thinned evenly across game time to {len(kept)} frames")

    ids = {i["id"] for i in kept}
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for im in kept:
        shutil.copy(src / "images" / im["file_name"], img_dir / im["file_name"])

    anns = [a for a in coco["annotations"] if a["image_id"] in ids]
    (out / "annotations.json").write_text(json.dumps({
        "images": kept,
        "annotations": anns,
        "categories": coco["categories"],
    }), encoding="utf-8")

    empties = sum(1 for i in kept if not by_img.get(i["id"]))
    print(f"wrote {len(kept)} frames ({empties} hard negatives), "
          f"{len(anns)} pseudo-boxes -> {out}")
    print("next: label these, then point train_rfdetr_numbers.py --src at them")


if __name__ == "__main__":
    main()
