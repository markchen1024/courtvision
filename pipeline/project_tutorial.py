"""Project tracked players onto the court plan, the way the notebook does it.

project.py solves the court by hand-clicked landmarks and its own FIBA-ish
coordinates, because court_model.py recorded the off-the-shelf keypoint model
finding "0-2 of its 33 points on our footage". That was measured on the Summer
League clip. On the NBA broadcast the notebook's own model and thresholds --
basketball-court-detection-2/14 at confidence 0.3, anchors above 0.5 (cell 76)
-- return 10 to 12 usable landmarks on every frame sampled, comfortably past
the four a homography needs.

So this takes the notebook's route end to end (cells 83 and 85):

    keypoints -> ViewTransformer -> court coordinates

with the court itself coming from sports.basketball.CourtConfiguration at
League.NBA -- 94 x 50 feet, 33 vertices, the same table the keypoint model was
labelled against. No hand calibration, and no home-made court model.

    python pipeline/project_tutorial.py \
        --video web/media/nba.mp4 \
        --boxes out/tracks_sam2tut_detfinal.json \
        --identities out/identities_sam2tut.json \
        --out web/data/nba.json

Writes the viewer's schema, so webapp reads it unchanged.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import config
from progress import Progress

KEYPOINT_MODEL_ID = "basketball-court-detection-2/14"   # notebook cell 76
KEYPOINT_CONFIDENCE = 0.3
ANCHOR_CONFIDENCE = 0.5
# Detroit host this game, so the Pistons are the home side.
HOME_CLUB = "pistons"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--identities")
    ap.add_argument("--out", default="web/data/nba.json")
    ap.add_argument("--hz", type=float, default=5.0,
                    help="court positions per second; the keypoint model is the "
                         "expensive part and the viewer interpolates between samples")
    ap.add_argument("--home-club", default=HOME_CLUB)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    config.load_env()
    config.inference_env()   # key, cache path, GPU provider -- before the import

    import cv2
    import supervision as sv
    from inference import get_model
    from sports import MeasurementUnit, ViewTransformer
    from sports.basketball import CourtConfiguration, League

    court = CourtConfiguration(league=League.NBA,
                               measurement_unit=MeasurementUnit.CENTIMETERS)
    vertices_m = np.array(court.vertices, dtype=np.float32) / 100.0
    length_m = round(court.court_length / 100.0, 2)
    width_m = round(court.court_width / 100.0, 2)

    def path(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    sidecar = json.loads(path(args.boxes).read_text(encoding="utf-8"))
    frames = {int(k): v for k, v in sidecar["frames"].items()}

    identities = {}
    if args.identities:
        identities = {int(k): v for k, v in json.loads(
            path(args.identities).read_text(encoding="utf-8"))["identities"].items()}

    model = get_model(model_id=KEYPOINT_MODEL_ID)
    cap = cv2.VideoCapture(str(path(args.video)))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / args.hz)))

    frames_out = []
    seen = set()
    unsolved = 0
    idx = 0
    prog = Progress("project-tutorial", total=total // step)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rows = frames.get(idx, [])
            if rows:
                res = model.infer(frame, confidence=KEYPOINT_CONFIDENCE)[0]
                kp = sv.KeyPoints.from_inference(res)
                conf = kp.confidence[0] if kp.confidence is not None else np.zeros(0)
                mask = conf > ANCHOR_CONFIDENCE
                if np.count_nonzero(mask) >= 4:
                    transformer = ViewTransformer(
                        source=kp[:, mask].xy[0].astype(np.float32),
                        target=vertices_m[mask],
                    )
                    # feet, not centre of mass: the only point of a standing
                    # player that lies on the court plane
                    feet = np.array([[(r["box"][0] + r["box"][2]) / 2.0, r["box"][3]]
                                     for r in rows], dtype=np.float32)
                    xy = transformer.transform_points(points=feet)
                    positions = []
                    for r, (x, y) in zip(rows, xy):
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue
                        positions.append({"id": int(r["tid"]),
                                          "x": round(float(x), 2),
                                          "y": round(float(y), 2)})
                        seen.add(int(r["tid"]))
                    frames_out.append({"t": round(idx / fps, 2),
                                       "positions": positions})
                else:
                    unsolved += 1
            prog.step(note=f"frame {idx}, {len(frames_out)} solved")
        idx += 1
    cap.release()
    prog.done(note=f"{len(frames_out)} frames")

    players = []
    for tid in sorted(seen):
        ident = identities.get(tid) or {}
        club = ident.get("club")
        players.append({
            "id": tid,
            "team": "home" if club == args.home_club else "away",
            "number": str(ident.get("number")) if ident.get("number") else f"T{tid}",
            **({"name": ident["name"]} if ident.get("name") else {}),
        })

    doc = {
        "source": f"{KEYPOINT_MODEL_ID} landmarks, NBA court plan, "
                  f"{sidecar.get('tracker', 'tracked')}",
        "video": {"duration": round(idx / fps, 2), "hz": args.hz},
        "court": {"length_m": length_m, "width_m": width_m},
        "players": players,
        "frames": frames_out,
        "events": [],
    }
    out = path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")

    per_frame = [len(f["positions"]) for f in frames_out]
    named = sum(1 for p in players if p.get("name"))
    print(f"\nwrote {out}")
    print(f"  court {length_m} x {width_m} m (NBA, from sports.basketball)")
    print(f"  {len(frames_out)} solved frames at {args.hz} Hz, "
          f"{unsolved} skipped for too few landmarks")
    print(f"  {len(players)} players, {named} carrying a name")
    if per_frame:
        print(f"  positions per frame: median {sorted(per_frame)[len(per_frame)//2]}, "
              f"min {min(per_frame)}, max {max(per_frame)}")


if __name__ == "__main__":
    main()
