"""Track players with ByteTrack, the way the tutorial this borrows from does.

The court-space tracker written here first holds identity for 7.4s and leaves
129 tracks where ten players stand. Before improving it, the two ready-made
trackers the published pipelines use get their turn: this is the video course's
one -- trackers/player_tracker.py in abdullahtarek/basketball_analysis is
literally `sv.ByteTrack()` over YOLO boxes. SAM2, the Roboflow pipeline's
choice, is the next candidate if this one is not enough.

ByteTrack associates by IoU in image space, which only works when boxes overlap
frame to frame. At the 5Hz the calibration is sampled at, a fast player moves
most of a box width between samples, so detection runs densely here -- every
second frame, 12.5Hz -- which is the regime the tutorial runs it in. The
projection step keeps its 5Hz grid and adopts identities by matching its cached
boxes to the tracked ones.

    python pipeline/track_bytetrack.py --video web/media/nba.mp4

sv.ByteTrack is deprecated in supervision 0.30 (Roboflow moved trackers to
their standalone `trackers` package). Used anyway because it is exactly what
the tutorial uses; the migration is one import if it disappears.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from project import CACHE_THRESHOLD, detect, load_detector
from progress import Progress


def dense_detections(video, cache_path, every):
    """Player boxes on a dense grid, cached. Sequential decode, not seeking:
    scattered cap.set() reads are what made the sparse pass slow."""
    if Path(cache_path).exists():
        c = json.loads(Path(cache_path).read_text())
        if c.get("video") == video and c.get("every") == every:
            print(f"dense detections: reusing {cache_path} ({len(c['frames'])} frames)")
            return {int(k): v for k, v in c["frames"].items()}, c["fps"]

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"detecting every {every} frames of {total} ({total // every} passes)...")

    proc, model = load_detector()
    out = {}
    idx = 0
    prog = Progress("dense-detect", total=total // every, video=video,
                    artifact=cache_path, meta={"every": every})
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            rows = [{"box": box, "conf": score}
                    for name, score, box in detect(proc, model, frame, CACHE_THRESHOLD)
                    if name == "player"]
            out[idx] = rows
            prog.step(note=f"frame {idx}")
            if len(out) % 200 == 0:
                print(f"  {len(out)}/{total // every} frames...")
        idx += 1
    cap.release()

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).write_text(json.dumps({
        "video": video, "every": every, "fps": fps,
        "frames": {str(k): v for k, v in out.items()},
    }))
    prog.done(note=f"{len(out)} frames cached")
    print(f"cached {len(out)} frames to {cache_path}")
    return out, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--cache", default="out/detections_nba_dense.json")
    ap.add_argument("--out", default="out/tracks_nba.json")
    args = ap.parse_args()

    import supervision as sv

    frames, fps = dense_detections(args.video, args.cache, args.every)

    tracker = sv.ByteTrack(frame_rate=fps / args.every)
    tracks = {}
    for idx in sorted(frames):
        rows = frames[idx]
        if rows:
            det = sv.Detections(
                xyxy=np.array([r["box"] for r in rows], np.float32),
                confidence=np.array([r["conf"] for r in rows], np.float32),
                class_id=np.zeros(len(rows), int),
            )
        else:
            det = sv.Detections.empty()
        got = tracker.update_with_detections(det)
        tracks[idx] = [{"tid": int(t), "box": [float(v) for v in b]}
                       for b, t in zip(got.xyxy, got.tracker_id)]

    ids = {t["tid"] for rows in tracks.values() for t in rows}
    lifetimes = {}
    for idx, rows in tracks.items():
        for t in rows:
            a, b = lifetimes.get(t["tid"], (idx, idx))
            lifetimes[t["tid"]] = (min(a, idx), max(b, idx))
    spans = np.array([(b - a) / fps for a, b in lifetimes.values()])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "video": args.video, "every": args.every, "fps": fps,
        "tracker": f"supervision {sv.__version__} ByteTrack",
        "frames": {str(k): v for k, v in tracks.items()},
    }))
    print(f"\nwrote {args.out}")
    print(f"  {len(ids)} track ids over {len(tracks)} frames at {fps / args.every:.1f}Hz")
    print(f"  lifetime median {np.median(spans):.1f}s  p90 {np.percentile(spans, 90):.1f}s  "
          f"max {spans.max():.1f}s")
    print("compare against the court-space tracker on the same clip: 129 tracks,")
    print("identity held 7.4s. Same detections, same footage, different association.")


if __name__ == "__main__":
    main()
