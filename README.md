# courtvision

Turning basketball footage into a live stat line.

A weekend experiment: take an ordinary video of a game, detect and track the players,
map the court from camera perspective into a top-down plan, and update a stat panel as
the video plays.

## The interesting part

The detection and tracking are off-the-shelf. The part that actually turns pixels into
basketball is the **homography** — solving the perspective transform from the four court
corners in frame to a standard court plan, so a player's position on screen becomes a
position on the floor. Everything downstream (heat maps, shot charts, spacing) falls out
of having real court coordinates.

## Stack

- Python + OpenCV
- YOLO for player detection
- ByteTrack for tracking across frames
- Homography via `cv2.findHomography` / `getPerspectiveTransform`
- Simple web front end to play the video alongside the top-down view

## Run it

```bash
python pipeline/make_sample_data.py --seconds 180   # writes web/data/sample.json
cd web && python -m http.server 8765                # open http://localhost:8765
```

Drop a clip at `web/media/game.mp4` and it plays automatically. With no clip the
page falls back to an internal clock so there is always something moving.

## Status

Experimental. Deliberately scoped: no ball tracking (small, fast, heavily occluded — a
different problem), no claimed accuracy numbers.
