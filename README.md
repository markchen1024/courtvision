# courtvision

Turning basketball footage into a live stat line.

A weekend experiment: take an ordinary video of a game, detect and track the players,
map the court from camera perspective into a top-down plan, and update a stat panel as
the video plays.

## The interesting part

The detection and tracking are off-the-shelf. The part that actually turns pixels into
basketball is the **homography** — the perspective transform from court markings in the
frame to a standard court plan, so a player's position on screen becomes a position on
the floor. Everything downstream (heat maps, shot charts, spacing) falls out of having
real court coordinates.

The footage here comes from a camera that pans left and right but never moves. That
matters more than it sounds: a camera that only rotates about its optical centre maps
between any two of its frames by a *pure homography*. So the court only has to be
calibrated once, on a reference frame; every other frame is registered back to that
reference by feature matching, and the two homographies compose. No per-frame court-line
detection required.

## Stack

- Python + OpenCV
- **RF-DETR** for player detection — first real-time detector past 60 AP on COCO, and
  it transfers to custom domains better than the YOLO line. YOLO26 is the pick instead
  if this ever has to run on CPU or an edge device.
- **Deep-EIoU** for tracking, not ByteTrack. Pedestrian trackers underperform badly on
  sport: on SportsMOT, ByteTrack scores 62.8 HOTA, BoT-SORT 68.7, Deep-EIoU 77.2. Fast
  erratic motion, near-identical uniforms and constant occlusion are exactly the cases
  the pedestrian benchmarks do not cover.
- Camera-motion homography chain (ORB/SIFT + RANSAC against a reference frame)
- Court homography via `cv2.findHomography`
- Team assignment by clustering appearance embeddings — no labelling needed
- Static web front end that plays the clip beside the top-down view

## Run it

The models need a GPU-sized environment, kept outside the repo:

```bash
C:/Users/Mark/.venvs/courtvision/Scripts/python.exe pipeline/try_models.py --every 30
```

The front end itself needs nothing but Python:

```bash
python pipeline/make_sample_data.py --seconds 180   # writes web/data/sample.json
python pipeline/serve.py                            # open http://localhost:8765
```

Two pages, sharing `web/assets/theme.css` and one court renderer in
`web/assets/court.js`:

- `/` — the product page. Mock SaaS: real copy, real numbers off the tracking data, and a
  hero that plays the actual clip beside a live top-down court. Every link but the demo
  goes nowhere. The stat tabs — box score, team comparison, shot chart, timeline, minutes
  — run on one invented game that is internally consistent: the zone splits add back to
  the box score's field goals, the shot chart is generated from those zones, and the
  plus-minus column sums to five times the final margin.
- `/app.html` — the viewer. Footage, top-down court, box score, play by play.

Drop a clip at `web/media/game.mp4` and it plays automatically. With no clip both pages
fall back to an internal clock so there is always something moving.

The clip has to be written **faststart**, or the browser cannot begin playback until it
has pulled the entire file — an exported mp4 usually has its `moov` atom at the end:

```bash
ffmpeg -i raw.mp4 -c copy -movflags +faststart web/media/game.mp4   # remux, no re-encode
```

`serve.py` rather than `python -m http.server` because the stdlib server ignores Range
requests — the browser then has to fetch the whole clip before it plays and the scrubber
does nothing, which defeats the point.

## Status

Experimental. Deliberately scoped: no ball tracking (small, fast, heavily occluded — a
different problem), no claimed accuracy numbers.
