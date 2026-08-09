"""Name the tracks: jersey OCR plus team clustering, the Roboflow way.

This follows blog.roboflow.com/identify-basketball-players component for
component, WITH ONE SUBSTITUTION to be upfront about: the tutorial's tracker
is SAM2; the tracks here come from CourtTracker in project.py, which is our
own hand-written code, not a ready-made component. Both were run through
this same pipeline on the same window and the comparison lives in
docs/tracking-comparison.md (court-space named 6 players to SAM2's 2 --
SAM2 swaps identities silently across broadcast cuts and can only track
who frame 0 shows). Everything else is theirs:

  detection   basketball-player-detection-3-ycjdo/4 -- one model for players,
              referees, jersey-number regions, rim, ball
  OCR         basketball-jersey-numbers-ocr/3 -- their fine-tuned SmolVLM2,
              prompted "Read the number."
  matching    a number region belongs to the player whose SAM2 mask it sits
              on at IoS >= 0.9 (the notebook matches against masks, not
              boxes -- in a crowded paint a box overlaps everyone nearby,
              a silhouette does not; --match box restores the loose version)
  validation  ConsecutiveValueTracker: sampled every 5 frames, a number is
              confirmed after 3 identical consecutive reads
  teams       sports.TeamClassifier (SigLIP + UMAP + K-means) fit on torso
              crops (boxes scaled to 0.4) sampled about once a second

The one addition is the last step, which their notebook does by hand
(TEAM_ROSTERS dict): clusters map to clubs by counting which club's roster
contains more of the cluster's confirmed numbers, and numbers become names
via the ESPN box score in web/data/rosters_nba.json.

    python pipeline/identify.py --video web/media/nba.mp4 \
        --boxes out/track_boxes_nba.json --apply web/data/nba.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from progress import Progress

DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
DETECTION_CONFIDENCE = 0.4
DETECTION_IOU = 0.9
OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
OCR_PROMPT = "Read the number."
IOS_THRESHOLD = 0.9
N_CONSECUTIVE = 3
TEAM_CROP_SCALE = 0.4


def ios(number_box, player_box):
    """Intersection over the smaller (number) area."""
    x1, y1, x2, y2 = number_box
    a1, b1, a2, b2 = player_box
    iw = max(0.0, min(x2, a2) - max(x1, a1))
    ih = max(0.0, min(y2, b2) - max(y1, b1))
    area = (x2 - x1) * (y2 - y1)
    return (iw * ih) / area if area > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--boxes", default="out/track_boxes_nba.json",
                    help="per-frame (tid, box) rows from project.py --boxes-out")
    ap.add_argument("--rosters", default="web/data/rosters_nba.json")
    ap.add_argument("--out", default="out/identities_nba.json")
    ap.add_argument("--apply", help="viewer JSON whose players get the real names")
    ap.add_argument("--stride", type=int, default=1,
                    help="OCR every Nth grid frame. The sidecar grid is already "
                         "every 5th frame, so 1 matches the notebook's cadence; "
                         "an every-frame tracker output (SAM2) needs 5")
    ap.add_argument("--team-stride", type=int, default=6,
                    help="team crops every Nth OCR sample (6 at 5Hz is ~1.2s, "
                         "the notebook's stride-30 at 25fps)")
    ap.add_argument("--match", choices=["mask", "box"], default="mask",
                    help="attach numbers to players via SAM2 silhouettes "
                         "(the notebook's way; costs ~0.3s per OCR frame) "
                         "or plain box overlap")
    ap.add_argument("--apply-only", action="store_true",
                    help="skip the 13-minute OCR pass and apply an existing --out")
    args = ap.parse_args()

    if args.apply_only:
        if not args.apply:
            raise SystemExit("--apply-only needs --apply <viewer json>")
        stored = json.loads(Path(args.out).read_text())
        apply_identities({int(k): v for k, v in stored["identities"].items()},
                         args.apply)
        return

    config.load_env()
    import os
    os.environ.setdefault("ROBOFLOW_API_KEY", config.secret("ROBOFLOW_API_KEY"))
    import supervision as sv
    from inference import get_model
    from sports import ConsecutiveValueTracker, TeamClassifier

    sidecar = json.loads(Path(args.boxes).read_text())
    if sidecar["video"] != args.video:
        raise SystemExit(f"{args.boxes} belongs to {sidecar['video']}")
    frames = {int(k): v for k, v in sidecar["frames"].items()}
    grid = sorted(frames)[::args.stride]
    all_tids = sorted({t["tid"] for rows in frames.values() for t in rows})
    print(f"{len(grid)} frames on the grid, {len(all_tids)} tracks")

    det_model = get_model(model_id=DETECTION_MODEL_ID)
    ocr = get_model(model_id=OCR_MODEL_ID)
    sam_model = None
    if args.match == "mask":
        from ultralytics import SAM
        sam_model = SAM("sam2.1_b.pt")
    validator = ConsecutiveValueTracker(n_consecutive=N_CONSECUTIVE)
    team_crops, team_crop_tids = [], []
    reads = 0

    cap = cv2.VideoCapture(args.video)
    prog = Progress("identify", total=len(grid))
    for n, frame_idx in enumerate(grid):
        rows = frames[frame_idx]
        if not rows:
            prog.step(note=f"frame {frame_idx}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            prog.step(note=f"frame {frame_idx}")
            continue
        h, w = frame.shape[:2]

        result = det_model.infer(frame, confidence=DETECTION_CONFIDENCE,
                                 iou_threshold=DETECTION_IOU)[0]
        det = sv.Detections.from_inference(result)
        numbers = det[det.data["class_name"] == "number"]

        # each number region goes to the player it sits on: the silhouette
        # when masks are on (a box overlaps every neighbour in a crowded
        # paint, a silhouette does not), the box otherwise
        tids, values = [], []
        track_masks = None
        if sam_model is not None and len(numbers):
            res = sam_model(frame, bboxes=[t["box"] for t in rows],
                            verbose=False)[0]
            if res.masks is not None:
                m = res.masks.data.cpu().numpy() > 0.5
                if len(m) == len(rows):
                    track_masks = m
        for nb in numbers.xyxy:
            best_tid, best = None, IOS_THRESHOLD
            if track_masks is not None:
                nx1, ny1 = max(0, int(nb[0])), max(0, int(nb[1]))
                nx2, ny2 = min(w, int(nb[2])), min(h, int(nb[3]))
                area = max(1, (nx2 - nx1) * (ny2 - ny1))
                for t, m in zip(rows, track_masks):
                    s = m[ny1:ny2, nx1:nx2].sum() / area
                    if s >= best:
                        best_tid, best = t["tid"], s
            else:
                for t in rows:
                    s = ios(nb, t["box"])
                    if s >= best:
                        best_tid, best = t["tid"], s
            if best_tid is None:
                continue
            x1, y1, x2, y2 = sv.clip_boxes(
                sv.pad_boxes(np.array([nb]), px=10, py=10), (w, h))[0]
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            reading = str(ocr.predict(crop, OCR_PROMPT)[0]).strip()
            if reading.isdigit():
                tids.append(best_tid)
                values.append(reading)
                reads += 1
        if tids:
            validator.update(tracker_ids=tids, values=values)

        if n % args.team_stride == 0:
            boxes = np.array([t["box"] for t in rows], np.float32)
            for t, box in zip(rows, sv.scale_boxes(xyxy=boxes, factor=TEAM_CROP_SCALE)):
                crop = sv.crop_image(frame, box)
                if crop.size:
                    team_crops.append(crop)
                    team_crop_tids.append(t["tid"])
        prog.step(note=f"frame {frame_idx}, {reads} reads")
    cap.release()

    confirmed = {tid: v for tid in all_tids
                 if (v := validator.get_validated(tid)) is not None}
    print(f"{reads} OCR reads -> {len(confirmed)} tracks with a confirmed number")

    print(f"fitting team classifier on {len(team_crops)} crops...")
    import torch
    team_classifier = TeamClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    team_classifier.fit(team_crops)
    crop_teams = team_classifier.predict(team_crops)
    votes = defaultdict(Counter)
    for tid, team in zip(team_crop_tids, crop_teams):
        votes[tid][int(team)] += 1
    cluster = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}

    # clusters -> clubs, by which roster the confirmed numbers belong to
    rosters = json.loads(Path(args.rosters).read_text())
    clubs = [k for k in rosters if isinstance(rosters[k], list)]
    club_numbers = {c: {str(p["num"]) for p in rosters[c]} for c in clubs}
    club_names = {c: {str(p["num"]): p["name"] for p in rosters[c]} for c in clubs}
    score = Counter()
    for tid, num in confirmed.items():
        if tid not in cluster:
            continue
        for c in clubs:
            if num in club_numbers[c]:
                score[(cluster[tid], c)] += 1
    straight = score[(0, clubs[0])] + score[(1, clubs[1])]
    crossed = score[(0, clubs[1])] + score[(1, clubs[0])]
    club_of = {0: clubs[0], 1: clubs[1]} if straight >= crossed else \
              {0: clubs[1], 1: clubs[0]}
    print(f"cluster->club: {club_of} (evidence {max(straight, crossed)} vs "
          f"{min(straight, crossed)} confirmed numbers)")

    identities, labels = {}, {}
    for tid in all_tids:
        club = club_of.get(cluster.get(tid))
        num = confirmed.get(tid)
        name = club_names.get(club, {}).get(num) if num else None
        identities[tid] = {"number": num, "club": club, "name": name,
                           "team_votes": dict(votes.get(tid, {}))}
        if num:
            labels[tid] = f"#{num} {name}" if name else f"#{num}"
    named = sum(1 for v in identities.values() if v["name"])
    print(f"{named} tracks carry a full name, "
          f"{len(confirmed) - named} a number the roster does not list")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "boxes": args.boxes,
        "models": {"detection": DETECTION_MODEL_ID, "ocr": OCR_MODEL_ID},
        "policy": f"IoS>={IOS_THRESHOLD}, {N_CONSECUTIVE} consecutive reads",
        "identities": {str(k): v for k, v in identities.items()},
        "labels": {str(k): v for k, v in labels.items()},
    }))
    prog.done(note=f"{named} named")
    print(f"wrote {args.out}")

    if args.apply:
        apply_identities(identities, args.apply)


def apply_identities(identities, viewer_path):
    doc = json.loads(Path(viewer_path).read_text())

    # The viewer's home/away came from shirt-colour clustering; the OCR clubs
    # are better evidence. Keep the majority home/away<->club convention so
    # most colours stay put, then let the club win every disagreement.
    side_votes = Counter()
    for p in doc["players"]:
        club = (identities.get(p["id"]) or {}).get("club")
        if club:
            side_votes[(club, p["team"])] += 1
    clubs = sorted({c for c, _ in side_votes})
    side_of = {}
    if len(clubs) == 2:
        a, b = clubs
        straight = side_votes[(a, "home")] + side_votes[(b, "away")]
        crossed = side_votes[(a, "away")] + side_votes[(b, "home")]
        side_of = {a: "home", b: "away"} if straight >= crossed else \
                  {a: "away", b: "home"}

    changed, resided = 0, 0
    for p in doc["players"]:
        ident = identities.get(p["id"])
        if not ident or not ident["number"]:
            continue
        p["number"] = ident["number"]
        if ident["name"]:
            p["name"] = ident["name"]
        else:
            # an OCR number with no roster match must not keep a stale
            # placeholder name attached to it
            p.pop("name", None)
        p["identity"] = "jersey-ocr"
        side = side_of.get(ident.get("club"))
        if side and p["team"] != side:
            p["team"] = side
            resided += 1
        changed += 1
    Path(viewer_path).write_text(json.dumps(doc), encoding="utf-8")
    print(f"applied {changed} identities to {viewer_path}"
          + (f", moved {resided} tracks to the club's side" if resided else ""))


if __name__ == "__main__":
    main()
