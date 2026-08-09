"""Render the tutorial's top-down court map from any tracks file.

The last visual from blog.roboflow.com/identify-basketball-players: dots on
a drawn court, moving as the players do. Their notebook projects through a
per-frame keypoint homography; we already solve those homographies in
auto_calibrate.py, so this reuses ours -- one H per grid frame, nearest
grid entry for frames between. Both trackers' outputs go through this same
code, so the two map videos differ only by the tracks fed in.

    python pipeline/render_map.py --tracks out/track_boxes_nba_30swin.json \
        --identities out/identities_court_30s.json --out out/map_court_30s.mp4

    python pipeline/render_map.py --tracks out/tracks_sam2_30s_L.json \
        --identities out/identities_sam2_30s.json --offset 1375 \
        --out out/map_sam2_30s.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from calibrate import COURT_LENGTH, COURT_WIDTH

CLUB_COLOURS = {   # BGR, roughly the strips: warriors royal blue, grizzlies white
    "warriors": (255, 140, 40),
    "grizzlies": (235, 235, 235),
}
UNKNOWN_COLOUR = (128, 128, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="per-frame (tid, box) json")
    ap.add_argument("--calibration", default="out/auto_calibration.json")
    ap.add_argument("--identities", help="identify.py output, for club colours")
    ap.add_argument("--offset", type=int, default=0,
                    help="tracks frame 0 equals calibration frame OFFSET "
                         "(a cut clip against the full video's calibration)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sports import MeasurementUnit
    from sports.basketball import CourtConfiguration, League, draw_court, \
        draw_points_on_court

    tracks = json.loads(Path(args.tracks).read_text())
    frames = {int(k): v for k, v in tracks["frames"].items()}
    fps = tracks.get("fps", 25.0)

    cal = json.loads(Path(args.calibration).read_text())
    H_by_frame = {r["frame"]: np.array(r["H_court_to_image"], np.float64)
                  for r in cal["frames"] if r.get("solved") and not r.get("suspect")}
    cal_grid = sorted(H_by_frame)

    club_of = {}
    if args.identities:
        ids = json.loads(Path(args.identities).read_text())["identities"]
        club_of = {int(t): v.get("club") for t, v in ids.items()}

    config = CourtConfiguration(league=League.NBA,
                                measurement_unit=MeasurementUnit.FEET)
    court_w_ft = max(v[0] for v in config.vertices)
    court_h_ft = max(v[1] for v in config.vertices)
    base = draw_court(config=config)
    H, W = base.shape[:2]

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-crf", "22", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)

    grid = sorted(frames)
    span = grid[-1] - grid[0] + 1
    import bisect
    n = 0
    last_rows = []
    for f in range(grid[0], grid[-1] + 1):
        if f in frames:
            last_rows = frames[f]
        full_frame = f + args.offset
        i = bisect.bisect_left(cal_grid, full_frame)
        best = min((c for c in cal_grid[max(0, i - 1):i + 1]),
                   key=lambda c: abs(c - full_frame), default=None)
        court = base.copy()
        if best is not None and abs(best - full_frame) <= 6 and last_rows:
            Hinv = np.linalg.inv(H_by_frame[best])
            feet = np.array([[[(b["box"][0] + b["box"][2]) / 2, b["box"][3]]]
                             for b in last_rows], np.float32)
            metres = cv2.perspectiveTransform(feet.reshape(1, -1, 2), Hinv)[0]
            for row, (mx, my) in zip(last_rows, metres):
                if not (-1 <= mx <= COURT_LENGTH + 1 and -1 <= my <= COURT_WIDTH + 1):
                    continue
                xy = np.array([[mx / COURT_LENGTH * court_w_ft,
                                my / COURT_WIDTH * court_h_ft]])
                colour = CLUB_COLOURS.get(club_of.get(row["tid"]), UNKNOWN_COLOUR)
                import supervision as sv
                court = draw_points_on_court(
                    config=config, xy=xy, court=court,
                    fill_color=sv.Color(r=colour[2], g=colour[1], b=colour[0]))
        cv2.putText(court, f"t={f / fps:.1f}s", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
        ff.stdin.write(court.tobytes())
        n += 1
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out}: {n} frames ({span} track frames, "
          f"{len(cal_grid)} calibrated)")


if __name__ == "__main__":
    main()
