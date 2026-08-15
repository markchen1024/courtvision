"""Take one segment from clip to finished render, in one command.

The four stages have to be chained by hand otherwise, each with paths that
have to agree with the last one's output, and a run that dies halfway leaves
no sign of which stage it reached. This does the chain, names everything off
the clip, and skips a stage whose output already exists so a resumed run does
not repeat an hour of SAM2.

    python pipeline/run_segment.py --video out/segments/seg_01m10.87s_19s.mp4
    python pipeline/run_segment.py --video X --out-dir out/reel --force

Stages, in order:

  1  check_lineup       refuses the run if the prompt frame is short;
                        a headless Claude then audits the same image for the
                        thing counting cannot see -- one player boxed twice
  2  track_sam2_tutorial   SAM2, tutorial-aligned
  3  identify           OCR + teams + roster-constrained assignment
  4  render_final       club-tinted masks, name chips
  5  report             grades the result against the rules of basketball,
                        writes out/report_<stem>.html and refreshes the index

Everything after stage 1 is expensive, which is why stage 1 is there.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(label, cmd, quiet_markers=()):
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.splitlines():
        low = line.lower()
        if any(m in low for m in ("warning", "deprecat", "timm", "loading weights",
                                  "it/s]", "modeldependency", "depthpro",
                                  "out of date", "symlink", "developer mode",
                                  "install --", "inference for", "and bug")):
            continue
        if line.strip():
            print("   ", line, flush=True)
    print(f"    [{label} took {(time.time() - t0) / 60:.1f} min]", flush=True)
    if proc.returncode:
        print(f"    FAILED with exit {proc.returncode}")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--rosters", default="web/data/rosters_det.json")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--need", type=int, default=10)
    ap.add_argument("--force", action="store_true",
                    help="rerun stages whose output already exists")
    ap.add_argument("--skip-gate", action="store_true",
                    help="run even if the prompt frame is short")
    ap.add_argument("--strict-audit", action="store_true",
                    help="stop when the vlm audit says the prompts cover fewer "
                         "than ten distinct players; default is advisory")
    args = ap.parse_args()

    video = Path(args.video)
    stem = video.stem
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    tracks = out / f"{stem}_tracks.json"
    ids = out / f"{stem}_identities.json"
    final = out / f"{stem}_final.mp4"

    started = time.time()
    print(f"segment: {video.name}")

    rc = run("gate: lineup", [PY, "pipeline/check_lineup.py", "--video",
                              str(video), "--need", str(args.need),
                              "--render", str(out / f"{stem}_lineup.jpg")])
    if rc and not args.skip_gate:
        print("\nstopping: the prompt frame is short, and everything after "
              "this costs an hour. --skip-gate overrides.")
        return 1

    # The lineup gate counts boxes; it cannot tell two players from one player
    # boxed twice. That cost seg_g6206_43s its fifth Piston: ten prompts, nine
    # men, and the render was the first place anyone noticed. A headless
    # Claude reads the lineup image and answers what a human answers at a
    # glance. Advisory: a bad verdict is printed, recorded next to the other
    # artifacts, and does not stop the run (--strict-audit makes it fatal).
    rc = run("gate: vlm audit", [PY, "pipeline/vlm_check.py",
                                 "--image", str(out / f"{stem}_lineup.jpg")]
                                + (["--strict"] if args.strict_audit else []))
    if rc and args.strict_audit:
        print("\nstopping: the audit says these prompts do not cover ten "
              "distinct players.")
        return 1

    if tracks.exists() and not args.force:
        print(f"\n=== track: reusing {tracks.name} ===")
    elif run("track: SAM2", [PY, "pipeline/track_sam2_tutorial.py",
                             "--video", str(video), "--out", str(tracks)]):
        return 1

    if ids.exists() and not args.force:
        print(f"\n=== identify: reusing {ids.name} ===")
    elif run("identify: numbers, teams, roster",
             [PY, "pipeline/identify.py", "--video", str(video),
              "--boxes", str(tracks), "--rosters", args.rosters,
              "--out", str(ids), "--stride", str(args.stride),
              "--confirm", "roster"]):
        return 1

    if final.exists() and not args.force:
        print(f"\n=== render: reusing {final.name} ===")
    elif run("render: final", [PY, "pipeline/render_final.py",
                               "--video", str(video), "--boxes", str(tracks),
                               "--identities", str(ids), "--out", str(final)]):
        return 1

    # Always, even on a resumed run: it costs a second and it is the only
    # stage that says whether the twenty minutes above produced something
    # worth watching. Every defect this pipeline shipped was found by a human
    # watching the render instead.
    run("grade: report", [PY, "pipeline/report.py", "--tracks", str(tracks),
                          "--identities", str(ids)])
    run("grade: index", [PY, "pipeline/report.py", "--index"])

    print(f"\ndone in {(time.time() - started) / 60:.1f} min -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
