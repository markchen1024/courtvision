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
  validation  majority vote over all of a track's reads (>=2 votes and a
              clear winner). This DEVIATES from the notebook's 3-identical-
              consecutive rule, approved 2026-08-10 on evidence: fragment
              tracks structurally fail the consecutive test -- track 12 read
              ['22','22','45'] and confirmed nothing. --confirm consecutive3
              restores the notebook's rule. Number regions are taken at
              conf 0.2 (notebook: 0.4) -- measured +40-50% more sightings on
              Summer League kits, misreads absorbed by the vote.
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
# The notebook pins v3. v7 replaces it here on a same-conditions comparison
# (docs/tracking-comparison.md): identical 33s segment, identical SAM2 tracks,
# identical roster and aggregation, only the OCR version swapped. Both name
# the same eleven tracks; what changes is how hard the evidence is. Hart's 3
# went from losing 95-50 to a misread 9, to winning 174-5. The two differ by
# more than training data -- v3's base is SmolVLM2-256M, v7's is the 2.2B.
NOTEBOOK_OCR_MODEL = "basketball-jersey-numbers-ocr/3"
OCR_MODEL_ID = "basketball-jersey-numbers-ocr/7"
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
    ap.add_argument("--confirm", choices=["majority", "consecutive3", "roster"],
                    default="majority",
                    help="how a track's number is confirmed; consecutive3 is "
                         "the notebook's original rule")
    ap.add_argument("--number-conf", type=float, default=0.2,
                    help="confidence floor for number regions (notebook: 0.4)")
    ap.add_argument("--match", choices=["mask", "box"], default="mask",
                    help="attach numbers to players via SAM2 silhouettes "
                         "(the notebook's way; costs ~0.3s per OCR frame) "
                         "or plain box overlap")
    ap.add_argument("--detection-model", default=DETECTION_MODEL_ID,
                    help="Roboflow model id for players and number regions")
    ap.add_argument("--ocr-model", default=OCR_MODEL_ID,
                    help=f"Roboflow model id for reading the digits. Defaults "
                         f"to v7 on measured evidence; {NOTEBOOK_OCR_MODEL} is "
                         f"what the notebook pins, and restores it.")
    ap.add_argument("--min-votes", type=int, default=2,
                    help="votes a roster-constrained assignment needs before "
                         "it counts as confirmed; below this it is kept but "
                         "marked needs_review rather than asserted or dropped")
    ap.add_argument("--club-margin", type=int, default=2,
                    help="confirmed numbers by which the cluster->club mapping "
                         "must win before it is trusted; below this the clubs "
                         "are left blank rather than guessed. 0 forces it.")
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
    config.inference_env()   # key, cache path, GPU provider -- before the import
    import supervision as sv
    from inference import get_model
    # sports/__init__.py is empty in the published package; the classes live
    # in the submodules.
    from sports.common.team import TeamClassifier

    sidecar = json.loads(Path(args.boxes).read_text())
    if sidecar["video"] != args.video:
        raise SystemExit(f"{args.boxes} belongs to {sidecar['video']}")
    frames = {int(k): v for k, v in sidecar["frames"].items()}
    grid = sorted(frames)[::args.stride]
    all_tids = sorted({t["tid"] for rows in frames.values() for t in rows})
    print(f"{len(grid)} frames on the grid, {len(all_tids)} tracks")

    det_model = get_model(model_id=args.detection_model)
    ocr = get_model(model_id=args.ocr_model)
    if args.ocr_model != NOTEBOOK_OCR_MODEL:
        print(f"OCR: {args.ocr_model} (the notebook pins "
              f"{NOTEBOOK_OCR_MODEL})")
    sam_model = None
    if args.match == "mask":
        from ultralytics import SAM
        sam_model = SAM("sam2.1_b.pt")
    validator = None
    if args.confirm == "consecutive3":
        # Only the notebook's rule needs this, and the class is not in the
        # published sports package -- it came from the notebook itself. Fail
        # here, where the flag was asked for, rather than at import time on a
        # run that never wanted it.
        try:
            from sports import ConsecutiveValueTracker
        except ImportError:
            raise SystemExit(
                "--confirm consecutive3 needs ConsecutiveValueTracker, which "
                "the installed sports package does not provide.\n"
                "  Use --confirm majority (the default), or vendor the class "
                "from the tutorial notebook.")
        validator = ConsecutiveValueTracker(n_consecutive=N_CONSECUTIVE)
    number_votes = defaultdict(Counter)
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

        result = det_model.infer(frame,
                                 confidence=min(DETECTION_CONFIDENCE, args.number_conf),
                                 iou_threshold=DETECTION_IOU)[0]
        det = sv.Detections.from_inference(result)
        numbers = det[(det.data["class_name"] == "number")
                      & (det.confidence >= args.number_conf)]

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
            if args.confirm in ("majority", "roster"):
                # roster assignment weights its matching by the same votes
                for tid, v in zip(tids, values):
                    number_votes[tid][v] += 1
            else:
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

    if args.confirm in ("majority", "roster"):
        confirmed = {}
        for tid, votes in number_votes.items():
            top = votes.most_common(2)
            # at least two votes, and a strict winner -- a lone read or a tie
            # stays unconfirmed rather than guessed
            if top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]):
                confirmed[tid] = top[0][0]
        # In roster mode these are provisional. Mapping a cluster to a club is
        # scored by how many confirmed numbers land in each roster, so it needs
        # numbers before the roster is known, and the roster-constrained
        # assignment needs the club before it can match. The majority result
        # breaks that circle and is replaced below.
    else:
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
    # GATE. Which cluster is which club is decided by how many confirmed
    # numbers land in each roster, and the two readings are usually close: 9-7
    # on one run of this clip, 8-8 on the next. A tie does not degrade the
    # output, it inverts it -- every colour and every name on screen swaps
    # sides, confidently. So refuse the mapping unless it wins by a margin,
    # and leave the clubs unassigned rather than guess. Numbers still render;
    # names do not.
    lead, second = max(straight, crossed), min(straight, crossed)
    if lead - second < args.club_margin:
        club_of = {}
        print(f"cluster->club: REFUSED -- evidence {lead} vs {second} is "
              f"within the margin of {args.club_margin}.")
        print("  A mapping this close is a coin toss, and getting it wrong "
              "swaps every name and colour.")
        print("  Numbers are kept; names and clubs are left blank. Pass "
              "--club-margin 0 to force it, or name the clubs by hand.")
    else:
        club_of = {0: clubs[0], 1: clubs[1]} if straight >= crossed else \
                  {0: clubs[1], 1: clubs[0]}
        print(f"cluster->club: {club_of} (evidence {lead} vs {second} "
              f"confirmed numbers)")

    tentative = {}
    if args.confirm == "roster":
        # A roster is a constraint, not a lookup table: one number belongs to
        # one player. Deciding each track on its own throws that away, and the
        # two corrections the demo needed by hand were both recoverable from
        # it -- track 7 tied 25 against 8, but 8 was already taken by a track
        # with twice the votes, so 25 was the only reading left. Solve all the
        # tracks of a club at once: maximum-weight matching between tracks and
        # the club's numbers, weights being the votes.
        from scipy.optimize import linear_sum_assignment

        confirmed = {}
        for cluster_id, club in club_of.items():
            tids = [t for t in all_tids if cluster.get(t) == cluster_id]
            numbers = sorted(club_numbers.get(club, []))
            if not tids or not numbers:
                continue
            cost = np.zeros((len(tids), len(numbers)))
            for i, t in enumerate(tids):
                for j, n in enumerate(numbers):
                    cost[i, j] = -number_votes[t].get(n, 0)
            for i, j in zip(*linear_sum_assignment(cost)):
                got = int(-cost[i, j])
                if got >= args.min_votes:
                    confirmed[tids[i]] = numbers[j]
                elif got > 0:
                    # the matching says this is the only number left for this
                    # track, but the reads barely support it. Keep it, marked,
                    # rather than either asserting or discarding it.
                    tentative[tids[i]] = (numbers[j], got)
        print(f"roster-constrained assignment: {len(confirmed)} confirmed, "
              f"{len(tentative)} tentative (under {args.min_votes} votes)")
        for tid, (num, got) in sorted(tentative.items()):
            print(f"  track {tid}: #{num} on {got} vote(s) -- needs review")

    identities, labels = {}, {}
    for tid in all_tids:
        club = club_of.get(cluster.get(tid))
        num = confirmed.get(tid)
        weak = tentative.get(tid)
        if num is None and weak:
            num = weak[0]
        name = club_names.get(club, {}).get(num) if num else None
        identities[tid] = {"number": num, "club": club, "name": name,
                           **({"needs_review": weak[1]} if weak else {}),
                           "team_votes": dict(votes.get(tid, {})),
                           # Keep what the OCR actually saw. Without it a
                           # wrong number is a bare assertion: on this
                           # footage a 3 was read as a 9, and whether that
                           # was unanimous or a one-vote win is the
                           # difference between a hard case and a close one.
                           "number_votes": dict(number_votes.get(tid, {}))}
        if num:
            labels[tid] = f"#{num} {name}" if name else f"#{num}"
    named = sum(1 for v in identities.values() if v["name"])
    print(f"{named} tracks carry a full name, "
          f"{len(confirmed) - named} a number the roster does not list")

    # GATE. Two live tracks confirmed to the same club and number is one player
    # holding two slots -- a hard error, not a judgement call, and it reached
    # the screen unnoticed every time (Duren rendered twice, Towns twice). The
    # weaker claim loses its identity: it keeps its number for review but stops
    # short of a name, so the render draws one label per player.
    by_number = defaultdict(list)
    for tid, v in identities.items():
        if v["number"]:
            by_number[(v["club"], v["number"])].append(tid)
    duplicates = {k: v for k, v in by_number.items() if len(v) > 1}
    for (club, number), tids in duplicates.items():
        strength = {t: sum((identities[t].get("number_votes") or {}).values())
                    for t in tids}
        winner = max(strength, key=strength.get)
        losers = [t for t in tids if t != winner]
        print(f"DUPLICATE: #{number} {club or '?'} claimed by tracks "
              f"{', '.join(str(t) for t in tids)} "
              f"(votes {', '.join(f'{t}:{strength[t]}' for t in tids)})")
        print(f"  keeping track {winner}; {', '.join(str(t) for t in losers)} "
              f"keep the number for review but lose the name")
        for t in losers:
            identities[t]["name"] = None
            identities[t]["duplicate_of"] = winner
    if duplicates:
        print(f"{len(duplicates)} duplicate identities demoted")
        named = sum(1 for v in identities.values() if v["name"])
        print(f"{named} tracks now carry a full name")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "boxes": args.boxes,
        "models": {"detection": args.detection_model, "ocr": args.ocr_model},
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

    changed, resided, kept_human = 0, 0, 0
    for p in doc["players"]:
        # A human decision outranks any model rerun, permanently. Without this
        # guard a reapply would silently undo review-UI corrections.
        if p.get("identity") in ("human", "ignored"):
            kept_human += 1
            continue
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
          + (f", moved {resided} tracks to the club's side" if resided else "")
          + (f", left {kept_human} human decisions untouched" if kept_human else ""))


if __name__ == "__main__":
    main()
