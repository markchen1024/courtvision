"""Is this detection standing on the floor, or sitting in the stands?

Frame 0 of seg_02m27.00s_14s produced ten player detections and the lineup
gate passed. One of them, at confidence 0.64, was a spectator in the front row
wearing a CUNNINGHAM #2 Pistons jersey. It took a prompt, SAM2 tracked it for
the whole clip, the OCR read '2' off its back twenty times, and the roster
matching handed it a name. The real Cade Cunningham was never tracked at all.

The gate exists to guarantee all ten players are prompted, and a crowd
detection defeats it twice over: it fills a slot that a player should have
had, and it makes a short lineup look full.

The test uses the tutorial's own court model rather than anything new.
basketball-court-detection-2/14 gives landmarks, ViewTransformer maps the
image to the NBA court plan out of sports.basketball, and a detection counts
as on the floor when its feet -- bottom-centre, the only part of a standing
player on the court plane -- land inside the court plus a margin. Anyone in
the stands is above and behind the plane, so their feet project far outside;
measured on that frame, the spectator lands 20+ metres past the sideline while
the furthest real player is under 2.

Falls open, not closed, and the reason matters. A first attempt without the
residual check below threw out two real players on frame 0 of
seg_00m30.68s_17s, projecting a man standing in the paint to 44m off the far
sideline. That frame offers nine landmarks over an 817x249px patch and the
homography fitted to them is worthless -- median residual 1.9m against the
0.09m and 0.12m the other two segments manage on thirteen. So the fit is
measured against its own anchors first, and a solve that cannot reproduce
them is not allowed to judge anybody.

A tried and rejected alternative: the convex hull of the landmarks in image
space, which needs no homography. The landmarks cluster near the centre of
the visible court, so the hull put five of nine real players outside on the
14s frame. Not usable.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KEYPOINT_MODEL_ID = "basketball-court-detection-2/14"   # notebook cell 76
KEYPOINT_CONFIDENCE = 0.3
ANCHOR_CONFIDENCE = 0.5
MIN_ANCHORS = 4
# Players step out of bounds, and inbounders stand well behind the baseline.
# Three metres covers that without reaching the front row -- the spectator
# measured at 4.1m past the sideline, the deepest real player at 1.7m inside.
MARGIN_M = 3.0
# Median metres by which the fitted homography misses its own anchors before
# it is treated as unusable. Measured: 0.09 and 0.12 on the two frames whose
# verdicts check out, 1.91 on the one that put a man in the paint 44m away.
MAX_RESIDUAL_M = 0.5


def keypoint_model():
    from inference import get_model
    return get_model(model_id=KEYPOINT_MODEL_ID)


def court_transformer(frame, model=None, max_residual_m=MAX_RESIDUAL_M):
    """ViewTransformer from this frame's image to the NBA court plan, in metres.

    Returns (transformer, length_m, width_m, why). `transformer` is None when
    the frame shows too few landmarks or when the fit cannot reproduce the
    anchors it was built from; `why` says which.
    """
    import supervision as sv
    from sports import MeasurementUnit, ViewTransformer
    from sports.basketball import CourtConfiguration, League

    court = CourtConfiguration(league=League.NBA,
                               measurement_unit=MeasurementUnit.CENTIMETERS)
    vertices_m = np.array(court.vertices, dtype=np.float32) / 100.0
    length_m = court.court_length / 100.0
    width_m = court.court_width / 100.0

    model = model or keypoint_model()
    res = model.infer(frame, confidence=KEYPOINT_CONFIDENCE)[0]
    kp = sv.KeyPoints.from_inference(res)
    conf = kp.confidence[0] if kp.confidence is not None else np.zeros(0)
    mask = conf > ANCHOR_CONFIDENCE
    n = int(np.count_nonzero(mask))
    if n < MIN_ANCHORS:
        return None, length_m, width_m, f"only {n} landmarks above {ANCHOR_CONFIDENCE}"

    src = kp[:, mask].xy[0].astype(np.float32)
    dst = vertices_m[mask]
    transformer = ViewTransformer(source=src, target=dst)
    # A homography fitted to landmarks bunched into a small patch of the image
    # extrapolates wildly outside it. Ask it to reproduce the very points it
    # was built from: if it cannot, nothing it says about a player is worth
    # acting on.
    residual = float(np.median(np.linalg.norm(
        transformer.transform_points(points=src) - dst, axis=1)))
    if residual > max_residual_m:
        return (None, length_m, width_m,
                f"{n} landmarks but the fit misses them by {residual:.2f}m "
                f"(limit {max_residual_m}m)")
    return transformer, length_m, width_m, f"{n} landmarks, fit {residual:.2f}m"


def filter_tracks(video, frames, every=15, margin_m=MARGIN_M, progress=None):
    """Drop boxes that are not standing on the floor, frame by frame.

    For SAM2 this only matters on the prompt frame. A concept tracker detects
    afresh every frame, so it keeps re-admitting the bench and the courtside
    seats: measured on seg_01m10.87s_19s, SAM3 returns a median of 17 boxes per
    frame where ten players are on court, and the team crops that come off
    those boxes collapse the SigLIP clustering into a single cluster -- after
    which cluster-to-club is a structural 11-11 tie and nobody gets named.

    The court is solved every `every` frames and the nearest solve reused;
    broadcast cameras pan slowly enough for that, and a per-frame solve would
    cost more than the tracking did. Frames whose solve fails keep every box,
    the same falling-open rule as everywhere else here.

    Returns (filtered_frames, stats).
    """
    import cv2

    model = keypoint_model()
    cap = cv2.VideoCapture(str(video))
    out, kept_n, dropped_n, unsolved = {}, 0, 0, 0
    transformer = length_m = width_m = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = frames.get(idx, [])
        if idx % every == 0:
            transformer, length_m, width_m, _ = court_transformer(frame, model)
        if rows and transformer is not None:
            boxes = np.asarray([r["box"] for r in rows], dtype=np.float32)
            feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
            xy = transformer.transform_points(points=feet.astype(np.float32))
            inside = (np.isfinite(xy).all(axis=1)
                      & (xy[:, 0] >= -margin_m) & (xy[:, 0] <= length_m + margin_m)
                      & (xy[:, 1] >= -margin_m) & (xy[:, 1] <= width_m + margin_m))
            keep = [r for r, v in zip(rows, inside) if v]
            dropped_n += len(rows) - len(keep)
            kept_n += len(keep)
            out[idx] = keep
        else:
            if rows and transformer is None:
                unsolved += 1
            kept_n += len(rows)
            out[idx] = rows
        idx += 1
        if progress:
            progress.step(note=f"frame {idx}, {dropped_n} dropped")
    cap.release()
    return out, {"frames": idx, "kept": kept_n, "dropped": dropped_n,
                 "unsolved_frames": unsolved}


def feet_on_court(frame, boxes, margin_m=MARGIN_M, model=None):
    """Which boxes are standing on the floor.

    boxes: array-like of xyxy
    returns (mask, positions, note) -- mask all True with note explaining
    itself when the court could not be solved
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0, bool), np.zeros((0, 2), np.float32), "no boxes"

    transformer, length_m, width_m, why = court_transformer(frame, model)
    if transformer is None:
        return (np.ones(len(boxes), bool), np.zeros((len(boxes), 2), np.float32),
                f"court not solved ({why}); every detection kept")

    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
    xy = transformer.transform_points(points=feet.astype(np.float32))
    inside = (np.isfinite(xy).all(axis=1)
              & (xy[:, 0] >= -margin_m) & (xy[:, 0] <= length_m + margin_m)
              & (xy[:, 1] >= -margin_m) & (xy[:, 1] <= width_m + margin_m))
    off = len(boxes) - int(inside.sum())
    note = (f"{off} of {len(boxes)} detections are off the court "
            f"(NBA {length_m:.2f}x{width_m:.2f}m plus {margin_m:.0f}m; {why})"
            if off else f"all {len(boxes)} detections are on the court ({why})")
    return inside, xy, note


def main():
    ap = argparse.ArgumentParser(
        description="Strip off-court boxes from a tracker's output.")
    ap.add_argument("--boxes", required=True, help="tracks JSON to filter")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=15,
                    help="frames between court solves")
    ap.add_argument("--margin", type=float, default=MARGIN_M)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from progress import Progress
    config.load_env()
    config.inference_env()

    root = Path(__file__).resolve().parent.parent

    def path(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    sidecar = json.loads(path(args.boxes).read_text(encoding="utf-8"))
    frames = {int(k): v for k, v in sidecar["frames"].items()}
    before = sum(len(v) for v in frames.values())
    ids_before = len({r["tid"] for v in frames.values() for r in v})

    prog = Progress("oncourt-filter", total=len(frames), video=args.video)
    kept, stats = filter_tracks(path(args.video), frames, every=args.every,
                                margin_m=args.margin, progress=prog)
    prog.done(note=f"{stats['dropped']} dropped")

    ids_after = len({r["tid"] for v in kept.values() for r in v})
    sidecar["frames"] = {str(k): v for k, v in kept.items()}
    sidecar["oncourt_filter"] = {**stats, "every": args.every,
                                 "margin_m": args.margin}
    path(args.out).write_text(json.dumps(sidecar))
    print(f"wrote {args.out}")
    print(f"  boxes  {before} -> {stats['kept']}  ({stats['dropped']} off court)")
    print(f"  ids    {ids_before} -> {ids_after}")
    print(f"  per frame  {before/max(1,len(frames)):.1f} -> "
          f"{stats['kept']/max(1,len(frames)):.1f}")
    if stats["unsolved_frames"]:
        print(f"  {stats['unsolved_frames']} frames kept everything -- "
              f"court not solved there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
