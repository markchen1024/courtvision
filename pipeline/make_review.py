"""Build the review queue the identity-correction UI reads.

The industry flow (Veo's Lineup, Hudl's manual identify) is: AI resolves
what it can, everything else lands in a queue a human clears in minutes.
This script builds that queue from the pipeline's own outputs -- every
track with its crop thumbnails, its OCR/cluster evidence, and its current
status -- so the webapp's /review page has everything in one fetch.

    python pipeline/make_review.py

Outputs land in web/data/review/ (manifest.json + crops/) and are
hardlinked into webapp/public/data/review/ so the Next.js dev server
serves them without a copy drifting stale.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2

from progress import Progress

CROPS_PER_TRACK = 4
CROP_HEIGHT = 260


def link_tree(src: Path, dst: Path):
    """Hardlink src's files under dst (same volume); copy when linking fails."""
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="web/data/nba.json")
    ap.add_argument("--boxes", default="out/track_boxes_nba.json")
    ap.add_argument("--identities", default="out/identities_nba.json")
    ap.add_argument("--rosters", default="web/data/rosters_nba.json")
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--out", default="web/data/review")
    ap.add_argument("--mirror", default="webapp/public/data/review")
    ap.add_argument("--no-crops", action="store_true",
                    help="rebuild the manifest only; keep existing crop files")
    args = ap.parse_args()

    doc = json.loads(Path(args.data).read_text(encoding='utf-8'))
    idn = {int(k): v for k, v in
           json.loads(Path(args.identities).read_text(encoding='utf-8'))["identities"].items()}
    frames = {int(k): v for k, v in
              json.loads(Path(args.boxes).read_text(encoding='utf-8'))["frames"].items()}
    rosters = json.loads(Path(args.rosters).read_text(encoding='utf-8'))
    fps = json.loads(Path(args.boxes).read_text(encoding='utf-8')).get("fps", 25.0)

    by_track = {}
    for f in sorted(frames):
        for r in frames[f]:
            by_track.setdefault(r["tid"], []).append((f, r["box"]))

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    tracks = []
    prog = Progress("review-build", total=len(doc["players"]))
    for p in doc["players"]:
        tid = p["id"]
        occ = by_track.get(tid, [])
        crop_files = []
        if occ and args.no_crops:
            crop_files = [f"{tid}_{k}.jpg" for k in range(CROPS_PER_TRACK)
                          if (crops_dir / f"{tid}_{k}.jpg").exists()]
        elif occ:
            step = max(1, len(occ) // CROPS_PER_TRACK)
            for k, (f, box) in enumerate(occ[::step][:CROPS_PER_TRACK]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = box
                mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.08
                cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
                cx2, cy2 = min(w_img, int(x2 + mx)), min(h_img, int(y2 + my))
                crop = frame[cy1:cy2, cx1:cx2].copy()
                if crop.size == 0:
                    continue
                # outline the track's own subject -- crowded crops otherwise
                # leave the reviewer guessing which person is meant
                bx1, by1 = int(x1 - cx1), int(y1 - cy1)
                bx2, by2 = int(x2 - cx1), int(y2 - cy1)
                th = max(2, crop.shape[0] // 90)
                cv2.rectangle(crop, (bx1, by1), (bx2, by2), (49, 240, 200), th)
                scale = CROP_HEIGHT / crop.shape[0]
                crop = cv2.resize(crop, (int(crop.shape[1] * scale), CROP_HEIGHT))
                name = f"{tid}_{k}.jpg"
                cv2.imwrite(str(crops_dir / name), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                crop_files.append(name)

        ident = idn.get(tid, {})
        if p.get("identity") == "human":
            status = "human"
        elif p.get("identity") == "ignored":
            status = "ignored"
        elif p.get("name") and p.get("identity") == "jersey-ocr":
            status = "named"
        elif ident.get("number"):
            status = "number-only"
        else:
            status = "anonymous"
        first = occ[0][0] / fps if occ else None
        last = occ[-1][0] / fps if occ else None
        tracks.append({
            "id": tid,
            "status": status,
            "team": p.get("team"),
            "number": p.get("number"),
            "name": p.get("name"),
            "ocr": {"number": ident.get("number"), "club": ident.get("club"),
                    "votes": ident.get("team_votes", {})},
            "crops": crop_files,
            "samples": len(occ),
            "span": [round(first, 1) if first is not None else None,
                     round(last, 1) if last is not None else None],
            # grid frames this track actually occupies -- the UI intersects
            # these to find same-name-same-frame conflicts live
            "frames": [f for f, _ in occ],
        })
        prog.step(note=f"track {tid}")
    cap.release()

    clubs = {c: rosters[c] for c in rosters if isinstance(rosters[c], list)}
    import time
    (out / "manifest.json").write_text(json.dumps({
        "video": args.video, "data": args.data, "builtAt": int(time.time()),
        "clubs": clubs,
        "tracks": tracks,
    }), encoding="utf-8")
    prog.done()

    link_tree(out, Path(args.mirror))
    counts = {}
    for t in tracks:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    print(f"review queue built: {len(tracks)} tracks -> {out}")
    print("  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
