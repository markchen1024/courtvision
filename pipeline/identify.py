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
import overlap
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


def decisive(counter, floor=2, share=0.2):
    """Does this vote actually have an opinion?

    A fragment whose team crops split 11 to 10 has none, and must not be
    allowed to veto a merge on the strength of a coin toss.
    """
    if not counter:
        return False
    top = counter.most_common(2)
    second = top[1][1] if len(top) > 1 else 0
    return top[0][1] - second >= max(floor, share * sum(counter.values()))


def merge_tracklets(life, number_votes, team_votes, min_votes, overlap_frames=15):
    """Tracks that are one player wearing several ids: {tid: canonical}.

    Three conditions, all necessary:

      same number     the top read agrees, on at least min_votes each. This is
                      the evidence; everything else is a veto.
      never together  their lifetimes do not overlap by more than
                      `overlap_frames`. Two ids on the floor at once reading
                      the same number is the duplicate case, a different fault
                      with a different fix -- but a tracker hands over with a
                      few frames of double report, and demanding a clean seam
                      cost Beasley the end of seg_01m10.87s_19s: his fragments
                      ran 0-941 and 935-1152, seven frames of overlap, 0.12s,
                      so the second was refused and he went unlabelled from
                      15.7s. Real coexistence lasts seconds, not frames.
      not clearly
      opposed teams   both team votes decisive and disagreeing means two men
                      who happen to share a number across the two rosters --
                      Cunningham is Pistons 2, McBride is Knicks 2. An
                      undecided vote abstains rather than blocks.

    Chained carefully: a third track joins a group only if it clears the whole
    group's span, so A-B and B-C disjoint but A-C overlapping cannot sneak
    through.
    """
    top = {}
    for tid, counter in number_votes.items():
        if not counter:
            continue
        best = counter.most_common(1)[0]
        if best[1] >= min_votes:
            top[tid] = best[0]

    by_number = defaultdict(list)
    for tid, num in top.items():
        by_number[num].append(tid)

    alias = {}
    for num, tids in sorted(by_number.items()):
        if len(tids) < 2:
            continue
        # strongest first: it becomes the canonical id and the others join it
        tids.sort(key=lambda t: -sum(number_votes[t].values()))
        groups = []           # [(canonical, first, last)]
        for tid in tids:
            a, b = life[tid]
            for g in groups:
                canon, lo, hi = g
                # measured against the whole group's span, not one member, so a
                # third fragment cannot slip in between two that already merged
                if min(b, hi) - max(a, lo) + 1 <= overlap_frames:
                    ca, cb = team_votes.get(canon), team_votes.get(tid)
                    if (decisive(ca) and decisive(cb)
                            and ca.most_common(1)[0][0] != cb.most_common(1)[0][0]):
                        continue              # two clubs, same number
                    alias[tid] = canon
                    g[1], g[2] = min(lo, a), max(hi, b)
                    break
            else:
                groups.append([tid, a, b])
    return alias


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
    ap.add_argument("--overlap-seconds", type=float, default=overlap.MIN_SECONDS,
                    help="how long two tracks must share a position before the "
                         "pair is called broken (see pipeline/overlap.py)")
    ap.add_argument("--max-collapsed", type=float, default=0.5,
                    help="a track collapsed onto another for more than this "
                         "fraction of its life gets no name at all -- there is "
                         "not enough of it left to say whose track it is")
    ap.add_argument("--merge-overlap-frames", type=int, default=15,
                    help="frames two tracklets may coexist and still be judged "
                         "one player -- a tracker hands over with a few frames "
                         "of double report. 0 demands a clean seam.")
    ap.add_argument("--no-split-check", action="store_true",
                    help="keep a name on a track whose reads are split between "
                         "two clubs' numbers -- it has been on two players")
    ap.add_argument("--no-merge", action="store_true",
                    help="skip tracklet association -- treat every track id as "
                         "a different player, even two that never coexist and "
                         "read the same number")
    ap.add_argument("--no-overlap", action="store_true",
                    help="trust every track everywhere, the way this ran before "
                         "the collapse on seg_01m10.87s_19s was found")
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

    # A track is only worth reading where it is on one player. Two tracks
    # sharing a position read the same jersey and both come away confirmed,
    # which is how two labels ended up following Brunson while Beasley went
    # unmarked -- both numbers were correct, so no later check could see it.
    fps = sidecar.get("fps", 59.94)
    overlaps = [] if args.no_overlap else overlap.find_overlaps(
        frames, fps=fps, min_seconds=args.overlap_seconds)
    if overlaps:
        # Geometry alone cannot tell a collapse from one player standing in
        # front of another, and they want opposite treatment. Ask the detector
        # what is inside each shared box before believing the worst.
        overlap.count_players(args.video, frames, overlaps)
    collapse = overlap.collapse_spans(overlaps)
    dupes = overlap.duplicate_pairs(overlaps)
    for o in overlaps:
        a, b = o["pair"]
        seen = f"  {o['players']:.0f} in the box" if "players" in o else ""
        print(f"  {o['kind']:9} tracks {a} and {b}: "
              f"{o['start'] / fps:6.2f}s - {o['end'] / fps:6.2f}s{seen}")

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
    reads, blocked = 0, 0

    cap = cv2.VideoCapture(args.video)
    prog = Progress("identify", total=len(grid), video=args.video)
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
            if overlap.is_contaminated(collapse, best_tid, frame_idx):
                # this track is sharing its position with another right now,
                # so whichever jersey is under it belongs to whoever the pair
                # settled on. Do not read it, let alone vote with it.
                blocked += 1
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
    if blocked:
        print(f"{blocked} number regions ignored inside collapsed spans")

    print(f"fitting team classifier on {len(team_crops)} crops...")
    import torch
    team_classifier = TeamClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    team_classifier.fit(team_crops)
    crop_teams = team_classifier.predict(team_crops)
    votes = defaultdict(Counter)
    for tid, team in zip(team_crop_tids, crop_teams):
        votes[tid][int(team)] += 1

    # Tracklet association. Re-seeding the tracker can hand one player several
    # ids over a clip -- measured on seg_01m10.87s_19s at 3s re-seeds, Harris
    # ran as track 5 until 15s and as track 12 from 18s, both reading '12'. The
    # roster matching gives one number per track, so the weaker was pushed onto
    # the only number left and became '#7 Paul Reed', who did not play. Two
    # tracks that never coexist and read the same number are one player.
    #
    # This is the step GTA and SportsSUSHI perform with ReID embeddings. A
    # jersey number is the stronger feature of the two: teammates in the same
    # kit look alike to a ReID model and never share a number.
    life = overlap.lifetimes(frames)
    alias = {} if args.no_merge else merge_tracklets(
        life, number_votes, votes, args.min_votes, args.merge_overlap_frames)
    for tid, canon in alias.items():
        print(f"MERGED: track {tid} is track {canon} again "
              f"(#{number_votes[tid].most_common(1)[0][0]}, "
              f"{life[tid][0]}-{life[tid][1]} after {life[canon][0]}-"
              f"{life[canon][1]})")
        number_votes[canon].update(number_votes.pop(tid))
        votes[canon].update(votes.pop(tid, Counter()))
        # the merged track owns every frame either half was alive for, and is
        # distrusted wherever either half was
        life[canon] = (min(life[canon][0], life[tid][0]),
                       max(life[canon][1], life[tid][1]))
        if tid in collapse:
            collapse[canon] = sorted(collapse.get(canon, []) + collapse.pop(tid))
    if alias:
        dupes = sorted({tuple(sorted((alias.get(a, a), alias.get(b, b))))
                        for a, b in dupes} - {(a, a) for a in life})
        dupes = [p for p in dupes if p[0] != p[1]]
        all_tids = [t for t in all_tids if t not in alias]
        print(f"{len(alias)} tracklets merged into "
              f"{len(set(alias.values()))} players")
    cluster = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}

    # A duplicate pair is one man wearing two track ids from the first frame
    # on. Unlike a collapse there is nothing to distrust -- the reads are of a
    # real player and they are right -- so the fix is to retire one id. The
    # one with less evidence goes, and its number slot is freed before the
    # roster matching runs, so it cannot take a number off a real track.
    retired = {}
    for a, b in dupes:
        if a in retired or b in retired:
            continue
        strength = {t: sum(number_votes[t].values()) for t in (a, b)}
        winner = max(strength, key=strength.get)
        loser = b if winner == a else a
        retired[loser] = winner
        print(f"DUPLICATE TRACK: {a} and {b} are one player "
              f"(votes {strength[a]} vs {strength[b]}); "
              f"retiring track {loser}, keeping {winner}")
    # GATE. Blocking the contaminated reads is not enough on its own. A track
    # that spends most of its life on someone else's man has only a handful of
    # clean reads left, and the roster matching will still hand it a leftover
    # number on that handful -- measured on seg_00m30.68s_17s, where track 10
    # lost its 22 reads of '3' and was promptly named Cameron Payne on four
    # reads of '1'. Trading a name that was right by accident for one that is
    # wrong on purpose is not an improvement. Below half a life, the pre-
    # contact evidence still stands and the track keeps its name (it simply
    # goes unmarked inside the span); above it, there is no track left to name.
    unvouched = {}
    for tid, spans in collapse.items():
        if tid in retired or tid in alias:
            continue
        span = sum(e - s + 1 for s, e in spans)
        whole = life[tid][1] - life[tid][0] + 1
        frac = span / max(1, whole)
        if frac > args.max_collapsed:
            unvouched[tid] = frac
            print(f"UNVOUCHED: track {tid} is collapsed for {frac:.0%} of its "
                  f"life -- no name, no number")
    live_tids = [t for t in all_tids
                 if t not in retired and t not in unvouched]

    if args.confirm in ("majority", "roster"):
        confirmed = {}
        # not `votes` -- that name holds the team-cluster votes, and shadowing
        # it here emptied every team_votes field in the output
        for tid, counts in number_votes.items():
            if tid in retired or tid in unvouched:
                continue
            # number_votes is a defaultdict, and the duplicate check above
            # reads number_votes[t] for tracks that were never read at all,
            # which quietly creates an empty counter. Harmless until SAM3
            # produced 189 fragments, most of them never read, and one that
            # survived the duplicate check reached most_common() empty.
            if not counts:
                continue
            top = counts.most_common(2)
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
        confirmed = {tid: v for tid in live_tids
                     if (v := validator.get_validated(tid)) is not None}
    print(f"{reads} OCR reads -> {len(confirmed)} tracks with a confirmed number")

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
            tids = [t for t in live_tids if cluster.get(t) == cluster_id]
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
        # A tentative number is one the matching had to give this track because
        # nothing else was left, on reads that barely support it. Keeping the
        # number for review was always the intent; attaching a name to it was
        # not, and that is how a single stray read of '7' put `#7 Paul Reed` on
        # screen -- a man who did not play. Below min-votes: number yes, name
        # no, exactly as the duplicate gate does it.
        name = (club_names.get(club, {}).get(num)
                if num and not weak else None)
        identities[tid] = {"number": num, "club": club, "name": name,
                           # kept in the file for the review queue, kept off
                           # the screen: a number the matching had to invent
                           # from one read is a claim like any other, and '#7'
                           # drawn on Brunson is wrong whether or not Paul
                           # Reed's name is next to it
                           **({"needs_review": weak[1],
                               "ignored": "tentative"} if weak else {}),
                           # the render skips these outright: a retired track
                           # is a second box on a player who already has one
                           **({"ignored": "duplicate-track",
                               "duplicate_of": retired[tid]}
                              if tid in retired else {}),
                           **({"ignored": "mostly-collapsed",
                               "collapsed_fraction": round(unvouched[tid], 3)}
                              if tid in unvouched else {}),
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
    print(f"{named} players carry a full name, "
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

    # GATE. A track that reads two numbers about equally often, where no single
    # roster holds both, has been on two players. Measured on
    # seg_00m30.68s_17s: track 6 read '0' thirty times and '32' thirty times --
    # Duren is Pistons 0, Towns is Knicks 32 -- because it followed a Detroit
    # player for the first seconds and Towns from about 8s. The render put
    # `#32 TOWNS` on a Piston while the real Towns stood unmarked beside him.
    #
    # The majority rule would have refused this outright: it demands a strict
    # winner. Roster mode replaces majority with the matching, which has no
    # such requirement, and that is the hole. Only cross-club ties are refused,
    # never same-club ones -- resolving those is the matching's whole value
    # (track 7 tied 25 against 8, and 8 was already taken, so 25 was the only
    # reading left).
    split = {}
    if not args.no_split_check:
        for tid, v in identities.items():
            if not v.get("name") or v.get("ignored"):
                continue
            counts = number_votes.get(tid) or Counter()
            top = counts.most_common(2)
            if len(top) < 2 or decisive(counts):
                continue
            (a, _), (b, _) = top
            if any(a in club_numbers[c] and b in club_numbers[c] for c in clubs):
                continue          # one roster holds both -- the matching's job
            # A clipped crop drops a digit, so '25' comes back as '5'. Two
            # readings where one is the head or tail of the other are one
            # number read twice, not two men -- measured on seg_02m44.15s_10s,
            # where Bridges read '5' 29 times and '25' 20 times and this gate
            # threw away an assignment the roster matching had already got
            # right (#5 was taken, so #25 was the only reading left). Two
            # genuinely different players do not produce substrings of each
            # other: the case this gate exists for read '0' and '32'.
            short, long = sorted((a, b), key=len)
            if long.startswith(short) or long.endswith(short):
                continue
            split[tid] = (a, b)
            print(f"SPLIT IDENTITY: track {tid} reads #{a} and #{b} about "
                  f"equally ({top[0][1]} vs {top[1][1]}), and no roster holds "
                  f"both -- it has been on two players, so it gets no name")
        for tid in split:
            identities[tid]["name"] = None
            identities[tid]["ignored"] = "split-identity"
            identities[tid]["split_between"] = list(split[tid])
    if split:
        named = sum(1 for v in identities.values() if v["name"])
        print(f"{len(split)} split identities dropped; {named} tracks named")

    # Only now do the merged halves get entries of their own, copied from the
    # canonical, so render_final can go on looking a box up by whatever id the
    # tracker gave it. Any earlier and the duplicate gate above would see the
    # two halves as two players on one number and demote the one it just spent
    # the association step deciding was the same man.
    for tid, canon in alias.items():
        identities[tid] = {**identities[canon], "merged_into": canon}
        if canon in labels:
            labels[tid] = labels[canon]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "boxes": args.boxes,
        "models": {"detection": args.detection_model, "ocr": args.ocr_model},
        "policy": f"IoS>={IOS_THRESHOLD}, {N_CONSECUTIVE} consecutive reads",
        # render_final.py reads this: a track is drawn only outside its
        # collapsed spans, because inside them it is on someone else's man.
        "overlap": {
            "policy": f"IoS>={overlap.IOS_THRESHOLD} sustained "
                      f"{args.overlap_seconds}s",
            "collapse": {str(k): [list(s) for s in v]
                         for k, v in collapse.items()},
            "retired": {str(k): v for k, v in retired.items()},
            "unvouched": {str(k): round(v, 3) for k, v in unvouched.items()},
        },
        "merged": {str(k): v for k, v in alias.items()},
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
