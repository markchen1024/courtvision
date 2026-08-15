"""Combine a forward and a backward pass into one set of labels.

SAM2's memory is causal, so running the clip backwards is not the same run
twice. Measured on seg_01m10.87s_19s: going forward, Brunson and Beasley merge
at 11.9s and stay merged for the remaining 7.3 seconds; coming the other way,
from a prompt at 19.0s, that merge never happens and the only collapse is 1.2s
long somewhere else. Each direction has blind spots the other does not.

What is fused is labels, not boxes. The two passes number their tracks
independently and matching those id spaces is fiddly and easy to get wrong,
but the question we actually care about is per frame and per player: does this
man carry a correct label right now? So each pass is reduced to
{frame: {(club, number): box}} -- the same drawing rule render_final uses --
and the two are unioned:

    only one pass has him      take it
    both have him              take the box from the pass with more votes
                               behind that identity
    two different players      only when the boxes are all but identical, and
    land on one spot           then the better-supported one wins rather than
                               both being dropped. The first version used
                               IoS >= 0.75 and dropped both, which is the
                               threshold the collapse detector uses for a
                               different question -- two players posting up
                               overlap that much as a matter of course, so
                               every ordinary occlusion was thrown away and
                               the fusion scored below either source alone.

The output is a synthetic pair of sidecars -- one track per player -- that
render_final.py and report.py consume unchanged.

    python pipeline/fuse.py --a out/X_tracks.json out/X_identities.json \
                            --b out/X_rev_tracks.json out/X_rev_identities.json \
                            --out-stem seg19_fused
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import overlap
from progress import Progress
from score import drawn

CONFLICT_IOS = 0.95


def per_frame_players(tracks_path, identities_path):
    """{frame: {(club, number): (box, strength)}} for one pass."""
    tr = json.loads(Path(tracks_path).read_text())
    doc = json.loads(Path(identities_path).read_text())
    frames = {int(k): v for k, v in tr["frames"].items()}
    idn = {int(k): v for k, v in doc["identities"].items()}
    collapse = {int(k): [tuple(s) for s in v]
                for k, v in (doc.get("overlap") or {}).get("collapse", {}).items()}

    out = {}
    names = {}
    for f in sorted(frames):
        here = {}
        for r in drawn(frames, idn, collapse, f):
            v = idn.get(r["tid"]) or {}
            key = (v.get("club"), str(v.get("number")))
            strength = sum((v.get("number_votes") or {}).values())
            names[key] = v.get("name")
            # a pass can legitimately hold one player on two ids at a handover;
            # keep the better-supported box
            if key not in here or strength > here[key][1]:
                here[key] = (r["box"], strength)
        out[f] = here
    return out, names, tr.get("fps", 59.94), len(frames)


def fuse(a, b, conflict_ios=CONFLICT_IOS, on_conflict="stronger"):
    """Union the two, then drop anything that would put two men on one spot."""
    frames = sorted(set(a) | set(b))
    fused, stats = {}, Counter()
    for f in frames:
        left, right = a.get(f, {}), b.get(f, {})
        here = {}
        for key in set(left) | set(right):
            if key in left and key in right:
                here[key] = left[key] if left[key][1] >= right[key][1] else right[key]
                stats["both"] += 1
            elif key in left:
                here[key] = left[key]
                stats["only_a"] += 1
            else:
                here[key] = right[key]
                stats["only_b"] += 1
        # a union can create a clash neither source had: two players whose
        # boxes are the same box
        keys = sorted(here)
        drop = set()
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                if ki in drop or kj in drop:
                    continue
                if overlap.ios(here[ki][0], here[kj][0]) >= conflict_ios:
                    if on_conflict == "drop":
                        drop.update((ki, kj))
                    else:
                        drop.add(ki if here[ki][1] < here[kj][1] else kj)
        for k in drop:
            here.pop(k, None)
        stats["dropped_conflict"] += len(drop)
        fused[f] = here
    return fused, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, required=True, metavar=("TRACKS", "IDENTITIES"))
    ap.add_argument("--b", nargs=2, required=True, metavar=("TRACKS", "IDENTITIES"))
    ap.add_argument("--out-stem", required=True,
                    help="writes out/<stem>_tracks.json and _identities.json")
    ap.add_argument("--video", help="recorded in the sidecar for later stages")
    ap.add_argument("--conflict-ios", type=float, default=CONFLICT_IOS)
    ap.add_argument("--on-conflict", choices=["stronger", "drop"],
                    default="stronger")
    args = ap.parse_args()

    # This stage was invisible on the progress page: it reports in seconds, so
    # nobody missed a bar, but a route with no run of its own cannot be told
    # apart from the pass it was built out of, and the archive is the only
    # place the two-pass route exists as a thing rather than as two things.
    video = args.video or json.loads(Path(args.a[0]).read_text()).get("video")
    prog = Progress("fuse-tracks", total=3, video=video,
                    artifact=str(ROOT / "out" / f"{args.out_stem}_tracks.json"),
                    meta={"a": Path(args.a[0]).name, "b": Path(args.b[0]).name,
                          "conflict_ios": args.conflict_ios,
                          "on_conflict": args.on_conflict})

    A, names_a, fps, n_a = per_frame_players(*args.a)
    prog.set(1, note=f"read {Path(args.a[0]).name}")
    B, names_b, _, n_b = per_frame_players(*args.b)
    prog.set(2, note=f"read {Path(args.b[0]).name}")
    names = {**names_b, **names_a}
    fused, stats = fuse(A, B, args.conflict_ios, args.on_conflict)

    mean_a = sum(len(v) for v in A.values()) / max(1, len(A))
    mean_b = sum(len(v) for v in B.values()) / max(1, len(B))
    mean_f = sum(len(v) for v in fused.values()) / max(1, len(fused))

    # one synthetic track per player, so the rest of the pipeline needs no
    # changes at all
    keys = sorted({k for v in fused.values() for k in v})
    tid_of = {k: i + 1 for i, k in enumerate(keys)}
    out_frames = {}
    for f, here in fused.items():
        out_frames[str(f)] = [{"tid": tid_of[k], "box": box}
                              for k, (box, _) in sorted(here.items())]
    identities = {
        str(tid_of[(club, num)]): {"number": num, "club": club,
                                   "name": names.get((club, num)),
                                   "team_votes": {}, "number_votes": {}}
        for club, num in keys}

    out_dir = ROOT / "out"
    tp = out_dir / f"{args.out_stem}_tracks.json"
    ip = out_dir / f"{args.out_stem}_identities.json"
    tp.write_text(json.dumps({
        "video": args.video or json.loads(Path(args.a[0]).read_text())["video"],
        "every": 1, "fps": fps,
        "tracker": f"fused: {Path(args.a[0]).name} + {Path(args.b[0]).name}",
        "frames": out_frames}))
    ip.write_text(json.dumps({
        "video": args.video or json.loads(Path(args.a[0]).read_text())["video"],
        "boxes": tp.relative_to(ROOT).as_posix(),
        "fused_from": [args.a[1], args.b[1]],
        "identities": identities, "overlap": {"collapse": {}}}))

    print(f"a: {mean_a:.2f} players/frame over {n_a} frames")
    print(f"b: {mean_b:.2f} players/frame over {n_b} frames")
    print(f"fused: {mean_f:.2f} players/frame over {len(fused)} frames "
          f"({len(keys)} distinct players)")
    print(f"  from a only {stats['only_a']}, from b only {stats['only_b']}, "
          f"both {stats['both']}, dropped as conflicts {stats['dropped_conflict']}")
    prog.done(note=f"{mean_f:.2f} players/frame, {len(keys)} players, "
                   f"{stats['dropped_conflict']} conflicts dropped")
    print(f"wrote {tp.name} and {ip.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
