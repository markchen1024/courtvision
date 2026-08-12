"""Find segments worth running: no cut inside, and a full lineup at the start.

The two gates already exist separately and neither is enough on its own.
detect_cuts.py finds stretches with no camera cut, but a stretch can open on a
close-up -- one 34.9s candidate began on a two-player shot of Tobias Harris.
check_lineup.py counts the players on a frame, but a frame with all ten is
useless if the camera cuts two seconds later.

This runs them together. Cuts first, because they are cheap and rule out most
of the footage. Then, inside each surviving stretch, walk forward until a
frame carries a full lineup: everything from there to the next cut is a
runnable segment, and SAM2 prompted on that frame can reach every player in
it.

    python pipeline/find_segments.py --video web/media/det.mp4 --min-seconds 10
    python pipeline/find_segments.py --video "d:/game.mkv" --start-sec 1740 \
        --max-sec 600 --min-seconds 15 --top 10

Prints ffmpeg commands, longest first. Nothing here is expensive enough to be
worth skipping before an hour of SAM2.
"""

import argparse
from pathlib import Path

import numpy as np

import config
from check_lineup import CONFIDENCE, IOU_THRESHOLD, PLAYER_CLASS_IDS, dedupe
from detect_cuts import find_cuts
from progress import Progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-sec", type=float, default=0.0, help="0 reads to the end")
    ap.add_argument("--need", type=int, default=10, help="players at the start")
    ap.add_argument("--min-seconds", type=float, default=8.0,
                    help="a segment shorter than this is not worth a run")
    ap.add_argument("--probe-every", type=int, default=15,
                    help="frames between lineup checks inside a stretch")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--cut", metavar="DIR",
                    help="write the segments out as mp4 rather than only "
                         "printing where they are")
    ap.add_argument("--cut-max", type=int, default=6,
                    help="how many of the longest to write when --cut is given")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    video = Path(args.video)
    if not video.is_absolute():
        video = root / video

    config.load_env()
    config.inference_env()

    import cv2
    import supervision as sv
    from inference import get_model

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start = int(fps * args.start_sec)
    end = total if not args.max_sec else min(total, start + int(fps * args.max_sec))

    prog = Progress("find-segments-cuts", total=end - start)
    cuts, flashes, last = find_cuts(
        video, args.threshold, 5, start, end,
        lambda i, c, f: prog.step(500, note=f"frame {i}, {c} cuts"))
    prog.done(note=f"{len(cuts)} cuts")
    print(f"{len(cuts)} cuts, {flashes} flashes ignored, over "
          f"{(last - start) / fps:.0f}s")

    bounds = [start] + cuts + [last]
    stretches = [(a, b) for a, b in zip(bounds, bounds[1:])
                 if (b - a) / fps >= args.min_seconds]
    print(f"{len(stretches)} cut-free stretches of at least "
          f"{args.min_seconds:.0f}s\n")
    if not stretches:
        return 1

    model = get_model(model_id="basketball-player-detection-3-ycjdo/4")

    def prompts_at(cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return 0
        res = model.infer(frame, confidence=CONFIDENCE,
                          iou_threshold=IOU_THRESHOLD)[0]
        det = sv.Detections.from_inference(res)
        det = det[np.isin(det.class_id, PLAYER_CLASS_IDS)]
        if len(det) == 0:
            return 0
        return len(dedupe(det.xyxy.tolist(), det.confidence.tolist()))

    cap = cv2.VideoCapture(str(video))
    segments = []
    prog = Progress("find-segments-lineup", total=len(stretches))
    for a, b in stretches:
        # walk forward until the lineup is complete; everything from there to
        # the cut is runnable
        best = None
        f = a
        while f < b - int(fps * args.min_seconds):
            n = prompts_at(cap, f)
            if n >= args.need:
                best = (f, n)
                break
            f += args.probe_every
        if best:
            f, n = best
            segments.append(((b - f) / fps, f, b, n))
        prog.step(note=f"{len(segments)} found")
    cap.release()
    prog.done(note=f"{len(segments)} segments")

    segments.sort(reverse=True)
    print(f"\n{len(segments)} runnable segments "
          f"(no cut, {args.need}+ players at the start):\n")
    print(f"{'length':>8} {'start':>10} {'prompts':>8}   ffmpeg")
    for length, f, b, n in segments[:args.top]:
        print(f"{length:>7.1f}s {f / fps:>9.2f}s {n:>8}   "
              f"ffmpeg -ss {f / fps:.2f} -t {length:.2f}")
    if not segments:
        print("  none -- every cut-free stretch opens without a full lineup")
        return 1

    if args.cut:
        import subprocess

        out_dir = Path(args.cut)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        for length, f, b, n in segments[:args.cut_max]:
            t0 = f / fps
            # named for where it came from, because a segment file with no
            # provenance is unusable an hour later
            name = f"seg_{int(t0 // 60):02d}m{t0 % 60:05.2f}s_{length:.0f}s.mp4"
            dst = out_dir / name
            cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.2f}",
                   "-i", str(video), "-t", f"{length:.2f}",
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   # keep the commentary: it survives to the final render,
                   # and a silent possession does not read as basketball
                   "-c:a", "aac", "-b:a", "192k", str(dst)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(f"  FAILED {name}: {r.stderr.strip()[:200]}")
                continue
            mb = dst.stat().st_size / 1e6
            print(f"  wrote {dst.relative_to(root)}  "
                  f"{length:.1f}s  {n} prompts  {mb:.1f}MB")
            contact_sheet(dst, out_dir / f"{dst.stem}_sheet.jpg")

        print("\nLook at the sheets before running any of these. detect_cuts\n"
              "misses a cut between two shots that share a colour layout, and\n"
              "one got through on seg_00m03.54s_26s: SAM2 lost every object at\n"
              "22.89s and the run was 40 minutes old before it showed.")
    return 0


def contact_sheet(video, out, shots=6):
    """Start, end and four points between, in one image.

    A segment that looks right in a frame count can still open on a close-up
    or contain a cut the detector missed. Both are obvious in six frames and
    invisible in the numbers, and the check has to be in the same command as
    the cut or it does not happen.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    want = [int(total * i / (shots - 1)) - (1 if i == shots - 1 else 0)
            for i in range(shots)]
    tiles, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            t = cv2.resize(frame, (480, 270))
            cv2.putText(t, f"{idx / fps:5.2f}s", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(t)
        idx += 1
    cap.release()
    if not tiles:
        return
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 270, cols * 480, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 270:(r + 1) * 270, c * 480:(c + 1) * 480] = t
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"       sheet -> {out.name}")


if __name__ == "__main__":
    raise SystemExit(main())
