"""Find where two tracks are following the same person, and say which kind.

SAM2 matches by appearance memory. When two players make body contact its
masks merge, and when they separate it has no way to decide which one it was
holding -- so both tracks can walk away on the same man. Measured on
seg_01m10.87s_19s: Brunson and Beasley collide at 11.9s and their boxes stay
at IoU 0.97-1.00 for the remaining seven seconds; Towns and Harris do the same
thing at 14.5s. Neither track is lost, all four carry a correct number from
before the contact, and the render draws two labels on one player.

Nothing downstream can see this. The number vote is taken over the whole clip
and the pre-contact reads settle both numbers; the duplicate check in
identify.py compares numbers, and 11 is not 5.

Two different faults look the same to a geometry test, and they want opposite
treatment:

  duplicate  the pair sits together for essentially the whole of the shorter
             track's life. It was never two players -- frame 0 was prompted
             twice on one man. One track is redundant; drop it and keep the
             other's reads, which are perfectly good.
  collapse   the pair separates for a while and then merges. Both tracks were
             right before the contact and neither can be trusted after it.

Telling them apart is what `coverage` is for: the overlap's length over the
shorter track's lifetime.

The treatment of a collapse follows what sports MOT does about occlusion
rather than trying to undo it. FC-Track and its relatives build a pairwise IoA
matrix over tracklets and, where the overlap is high, *suspend appearance
updates* rather than trusting contaminated features. The same idea applies
here without any new model: while two tracks sit on top of each other,
whatever is read from them belongs to nobody in particular, so it does not
vote and it is not drawn.

Splitting the pair properly -- deciding which man each track was on -- is a
different job. That is GTA (OSNet ReID embeddings, DBSCAN over a tracklet,
+3.8 HOTA on SportsMOT), an offline pass with a model attached. The spans
found here are exactly its input if it ever gets built.

    python pipeline/overlap.py --boxes out/seg_01m10.87s_19s_tracks.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Intersection over the smaller box, not over the union: when one player is
# swallowed by another's mask the smaller box sits inside the larger, and IoU
# understates that badly while IoS does not.
IOS_THRESHOLD = 0.75
# Players cross constantly, and a genuine crossing is over fast. A second of
# sustained overlap is past what a pass-behind produces; half a second was not,
# and flagged ordinary traffic on this footage.
MIN_SECONDS = 1.0
# Masks flicker apart for a frame or two mid-contact. Bridge that, but treat
# half a second of daylight as the players genuinely separating.
GAP_SECONDS = 0.5
# Overlapping for this much of the shorter track's life means it was never a
# second player.
DUPLICATE_COVERAGE = 0.9
# for count_players: the same detector the rest of the pipeline prompts with
DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
DETECTION_IOU = 0.9
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]


def ios(a, b):
    """Intersection over the smaller area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def lifetimes(frames):
    """{tid: (first_frame, last_frame)} over the whole sidecar."""
    seen = {}
    for f, rows in frames.items():
        for r in rows:
            a, b = seen.get(r["tid"], (f, f))
            seen[r["tid"]] = (min(a, f), max(b, f))
    return seen


def find_overlaps(frames, fps=59.94, threshold=IOS_THRESHOLD,
                  min_seconds=MIN_SECONDS, gap_seconds=GAP_SECONDS,
                  duplicate_coverage=DUPLICATE_COVERAGE):
    """Sustained same-position spans, each classed duplicate or collapse.

    frames: {frame_index: [{"tid": int, "box": [x1,y1,x2,y2]}, ...]}
    returns: [{"pair": (a, b), "start": int, "end": int,
               "kind": "duplicate"|"collapse", "coverage": float}, ...]
    """
    min_len = max(1, int(round(fps * min_seconds)))
    max_gap = max(1, int(round(fps * gap_seconds)))
    hits = defaultdict(list)
    for f in sorted(frames):
        rows = frames[f]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if ios(a["box"], b["box"]) >= threshold:
                    hits[tuple(sorted((a["tid"], b["tid"])))].append(f)

    live = lifetimes(frames)
    found = []
    for pair, fs in hits.items():
        fs.sort()
        run_start = prev = fs[0]
        for f in fs[1:] + [None]:
            if f is not None and f - prev <= max_gap:
                prev = f
                continue
            if prev - run_start + 1 >= min_len:
                shortest = min(live[t][1] - live[t][0] + 1 for t in pair)
                cover = (prev - run_start + 1) / max(1, shortest)
                found.append({
                    "pair": pair, "start": run_start, "end": prev,
                    "kind": "duplicate" if cover >= duplicate_coverage
                            else "collapse",
                    "coverage": cover,
                })
            if f is not None:
                run_start = prev = f
    return sorted(found, key=lambda o: (o["start"], o["pair"]))


def collapse_spans(overlaps):
    """{tid: [(first_frame, last_frame), ...]} for the collapses only.

    One track can collapse against several others at once -- Brunson sat on
    both of Duren's ids -- so the same span arrives more than once. Merge them:
    a track is either trustworthy at a frame or it is not, and the caller
    should not have to see the same range three times.
    """
    spans = defaultdict(list)
    for o in overlaps:
        if o["kind"] == "collapse":
            for tid in o["pair"]:
                spans[tid].append((o["start"], o["end"]))
    merged = {}
    for tid, rng in spans.items():
        out = []
        for s, e in sorted(rng):
            if out and s <= out[-1][1] + 1:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        merged[tid] = out
    return merged


def duplicate_pairs(overlaps):
    """[(tid, tid), ...] for the pairs that were one player all along."""
    return sorted({o["pair"] for o in overlaps if o["kind"] == "duplicate"})


def count_players(video, frames, overlaps, sample_every=15, confidence=0.4):
    """Ask the detector how many players are inside each shared box.

    Two boxes on top of each other say nothing about what is under them, and
    the two cases want opposite treatment. Measured at 16.0s of
    seg_01m10.87s_19s, both pairs had identical boxes:

      tracks 6 and 8   one player in the box -- Beasley. Brunson is gone, both
                       tracks are on the same man, and drawing either name is
                       a coin toss.
      tracks 1 and 5   two players in the box -- Towns posting up with Harris
                       behind him. Neither track lost anyone; the boxes
                       overlap because one man is standing in front of the
                       other, which is most of basketball.

    Suppressing the second kind is what left four players unlabelled at 16s.
    So each candidate span is sampled and reclassified: a span whose shared box
    usually holds two or more players is occlusion and is left alone; one that
    usually holds a single player is a collapse.

    Mutates and returns `overlaps`, adding "players" (the median count) and
    turning "kind" into "occlusion" where the pair is merely stacked.
    """
    import cv2
    import numpy as np
    import supervision as sv
    from inference import get_model

    model = get_model(model_id=DETECTION_MODEL_ID)
    cap = cv2.VideoCapture(video)
    try:
        for o in overlaps:
            if o["kind"] != "collapse":
                continue
            a, b = o["pair"]
            counts = []
            for f in range(o["start"], o["end"] + 1, sample_every):
                rows = {r["tid"]: r["box"] for r in frames.get(f, [])}
                if a not in rows or b not in rows:
                    continue
                box = [min(rows[a][0], rows[b][0]), min(rows[a][1], rows[b][1]),
                       max(rows[a][2], rows[b][2]), max(rows[a][3], rows[b][3])]
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    continue
                res = model.infer(frame, confidence=confidence,
                                  iou_threshold=DETECTION_IOU)[0]
                det = sv.Detections.from_inference(res)
                det = det[np.isin(det.class_id, PLAYER_CLASS_IDS)]
                # a detection belongs to the shared box if its centre is inside
                # it -- a neighbour clipping the edge should not count as a
                # second man
                inside = 0
                for x1, y1, x2, y2 in det.xyxy:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                        inside += 1
                counts.append(inside)
            o["players"] = float(np.median(counts)) if counts else 0.0
            # no samples at all leaves it a collapse: unverified is not the
            # same as cleared
            if counts and o["players"] >= 2:
                o["kind"] = "occlusion"
    finally:
        cap.release()
    return overlaps


def is_contaminated(spans, tid, frame):
    for a, b in spans.get(tid, ()):
        if a <= frame <= b:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--threshold", type=float, default=IOS_THRESHOLD)
    ap.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    ap.add_argument("--video", help="run the detector inside each candidate "
                                    "span to separate a collapse from plain "
                                    "occlusion; without it every candidate is "
                                    "reported as a collapse")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    p = Path(args.boxes)
    if not p.is_absolute():
        p = root / p
    sidecar = json.loads(p.read_text(encoding="utf-8"))
    fps = sidecar.get("fps", 59.94)
    frames = {int(k): v for k, v in sidecar["frames"].items()}

    overlaps = find_overlaps(frames, fps, args.threshold, args.min_seconds)
    if not overlaps:
        print(f"no overlapping tracks over {args.min_seconds}s "
              f"at IoS >= {args.threshold}")
        return 0
    if args.video:
        import config
        config.load_env()
        config.inference_env()
        v = Path(args.video)
        count_players(str(v if v.is_absolute() else root / v), frames, overlaps)

    def show(kind, note):
        rows = [o for o in overlaps if o["kind"] == kind]
        if not rows:
            return
        print(f"\n{len(rows)} {kind}: {note}")
        for o in rows:
            a, b = o["pair"]
            s, e = o["start"], o["end"]
            seen = f", {o['players']:.0f} in the box" if "players" in o else ""
            print(f"  tracks {a} and {b}: {s / fps:6.2f}s - {e / fps:6.2f}s "
                  f"({(e - s + 1) / fps:5.1f}s, {o['coverage']:.0%} of the "
                  f"shorter track{seen})")

    print(f"{len(overlaps)} sustained overlaps (IoS >= {args.threshold} for "
          f"{args.min_seconds}s or more)")
    show("duplicate", "one player prompted twice -- the weaker track is dropped, "
                      "the other keeps its reads")
    show("occlusion", "both players still there, one standing in front of the "
                      "other -- left alone")
    show("collapse", "two tracks, one player -- neither is trusted inside the "
                     "span: no votes, nothing drawn")

    spans = collapse_spans(overlaps)
    lost = sum(e - s + 1 for v in spans.values() for s, e in v)
    print(f"\n{len(spans)} tracks lose {lost} track-frames to collapse; "
          f"{len(duplicate_pairs(overlaps))} duplicate pairs to resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
