"""Score a render's labels against a track-level ground truth.

Two numbers, and the first matters far more than the second:

  precision   of the frames where a name was drawn, the share drawn on the
              right man. A wrong name on screen is the failure this project
              has spent the most effort avoiding, and until now the only test
              was cropping a frame and arguing about it.
  coverage    of the ten men on court, the share carrying a correct label,
              averaged over frames. The denominator is ten whatever tracker
              produced the boxes, so trackers stay comparable.

Frames a truth file leaves unlabelled -- an unreadable track, or the handover
inside a track that changed man -- are skipped, not guessed. They are reported
separately as `unknown`, because a metric computed over a shrinking sample is
worth less and the reader should see the sample shrink.

The drawing rule is copied from render_final.py rather than approximated: a
track is drawn when it has a number, is not ignored, is not inside one of its
collapsed spans, and is the only survivor among tracks resolving to the same
player that frame.

    python pipeline/score.py --truth eval/seg19_sam2_truth.json \
        --identities out/seg_01m10.87s_19s_identities.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import overlap

ON_COURT = 10


def truth_at(segments, frame):
    """Who this track was on at this frame, or None when unlabelled."""
    for player, a, b in segments or ():
        if a <= frame <= b:
            return player
    return None


def drawn(frames, idn, collapse, f):
    """Exactly what render_final.py would draw on this frame."""
    rows = [r for r in frames.get(f, [])
            if (v := idn.get(r["tid"])) and v.get("number")
            and not v.get("ignored")
            and not overlap.is_contaminated(collapse, r["tid"], f)]
    one = {}
    for r in rows:
        v = idn.get(r["tid"]) or {}
        who = v.get("merged_into", r["tid"])
        x1, y1, x2, y2 = r["box"]
        area = (x2 - x1) * (y2 - y1)
        if who not in one or area > one[who][0]:
            one[who] = (area, r)
    return [r for _, r in one.values()]


def score(truth_path, identities_path, boxes_path=None):
    truth = json.loads(Path(truth_path).read_text())
    doc = json.loads(Path(identities_path).read_text())
    boxes = boxes_path or truth["boxes"]
    sidecar = json.loads((ROOT / boxes).read_text())
    frames = {int(k): v for k, v in sidecar["frames"].items()}
    idn = {int(k): v for k, v in doc["identities"].items()}
    collapse = {int(k): [tuple(s) for s in v]
                for k, v in (doc.get("overlap") or {}).get("collapse", {}).items()}
    tt = {int(k): v for k, v in truth["tracks"].items()}

    right = wrong = unknown = 0
    covered, n_frames = 0, 0
    culprits = {}
    for f in sorted(frames):
        n_frames += 1
        here = set()
        for r in drawn(frames, idn, collapse, f):
            tid = r["tid"]
            said = (idn.get(tid) or {}).get("number")
            want = truth_at((tt.get(tid) or {}).get("segments"), f)
            if want is None or want in ("?", "not-player"):
                unknown += 1
                continue
            if str(said) == str(want):
                right += 1
                here.add(want)
            else:
                wrong += 1
                culprits[(tid, str(said), str(want))] = \
                    culprits.get((tid, str(said), str(want)), 0) + 1
        covered += len(here)

    judged = right + wrong
    return {
        "precision": right / judged if judged else float("nan"),
        "coverage": covered / (n_frames * ON_COURT) if n_frames else 0.0,
        "right": right, "wrong": wrong, "unknown": unknown,
        "frames": n_frames, "culprits": culprits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--identities", required=True, nargs="+",
                    help="one or more identities JSONs to compare")
    ap.add_argument("--boxes", help="override the tracks file named by --truth")
    args = ap.parse_args()

    print(f"{'identities':<44} {'precision':>10} {'coverage':>9} "
          f"{'right':>7} {'wrong':>6} {'unknown':>8}")
    for ip in args.identities:
        s = score(args.truth, ip, args.boxes)
        print(f"{Path(ip).name:<44} {s['precision']:>9.1%} {s['coverage']:>8.1%} "
              f"{s['right']:>7} {s['wrong']:>6} {s['unknown']:>8}")
        for (tid, said, want), n in sorted(s["culprits"].items(),
                                           key=lambda x: -x[1]):
            print(f"    track {tid}: drew #{said}, truth #{want}  "
                  f"({n} frames, {n / s['frames']:.0%} of the clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
