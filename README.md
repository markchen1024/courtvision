# courtvision

Give it a broadcast of a basketball game. It returns who is on the floor, where
they are standing in metres, and which club they play for — from the television
picture alone, with no fixed camera and no instrumented arena.

Currently: an eight-second possession from NYK @ DET, game 4 of the 2025 East
first round. Ten players on court, ten identified by jersey number, ten named
against the roster, no duplicates.

## What it actually does, and what that cost

Every number below was measured on this footage by the scripts in `pipeline/`,
not quoted from a paper or a vendor page. Where something has not been
measured, it says so.

| stage | measured |
|---|---|
| Player detection, per frame | 10 of 10 players on **52%** of frames — the rest of the time somebody is genuinely off camera |
| Jersey-number regions | **precision 95.7%, recall 82.2%** against 30 hand-labelled frames (163 boxes) |
| Court projection | **88 of 88** sampled frames solved, 10–12 landmarks found where 4 are needed |
| Identity, 8-second clip | 10 of 10, no duplicates |
| Identity, 17-second clip | 8–9 of 10, one duplicate |

That last row is the honest shape of this: the result degrades with clip
length, for a reason given below.

## The interesting part

Not the models. All five are off the shelf. What took the time was finding out
which of the pipeline's own assumptions were wrong.

**The tracker only asks once.** SAM2 takes its prompts on frame 0 and
propagates from there, so the set of players it can ever identify is fixed by a
single detection call. On this broadcast only about half of all frames show all
ten players, and the camera cuts every 8.8 seconds — so a longer clip is not
more data, it is more chances to be holding the wrong ten people. Eight-second
single-shot segments are not a demo shortcut; they are the regime the
architecture actually supports.

**The roster is a constraint, not a lookup table.** One number belongs to one
player. Deciding each track independently throws that away and produces two
players wearing #0. Solving a club's tracks together — maximum-weight matching
between tracks and the club's numbers, weighted by OCR votes — recovers
readings that independent voting refuses: a track tied 13–13 between #25 and #8
resolves to #25 once #8 is taken by a track with twice the evidence.

**Gates belong at the front.** The pipeline's checks were all at the end, where
a bad result is cheap. Three runs of roughly an hour each were decided by a
detection call in minute two and only discovered at the final render. The fix
was not a model: count the lineup on the prompt frame and refuse to continue
when it is short. Its first run repaid itself by revealing that the segment had
never been the problem — the prompts were coming from the wrong detector.

## Stack

| | |
|---|---|
| Detection | RF-DETR (`basketball-player-detection-3-ycjdo/4`) — players, jersey-number regions, referees, rim, ball |
| Tracking | SAM2 (`sam2.1_hiera_large`) through `segment-anything-2-real-time` |
| Teams | SigLIP embeddings → UMAP → K-means, no labelling |
| Numbers | SmolVLM2 LoRA (`basketball-jersey-numbers-ocr/7`), aggregated over a track. The notebook pins v3; v7 replaces it on a same-conditions comparison — its base is the 2.2B rather than the 256M, and the one systematic misread on this footage disappears |
| Court | `basketball-court-detection-2/14` keypoints → `ViewTransformer` → NBA court plan from `sports.basketball` |
| Front end | Next.js on port 3100, playing the clip beside a live top-down court |

This follows [Roboflow's basketball notebook](https://blog.roboflow.com/identify-basketball-players/)
component for component. Where this repo deviates, the deviation was measured
against the original on the same footage — see `docs/tracking-comparison.md`.

## What was tried and rejected

Kept here because a negative result that is not written down gets paid for
twice.

- **Fine-tuning the number detector on our own labels.** 150 frames labelled by
  hand, split by game time, scored before and after. It got worse: precision
  95.7% → 68.0%, false positives 6 → 57. 101 training frames against the 13
  games behind the base weights. The base detector stayed.
- **Filtering unreadable crops with signals we already had** — box size,
  detector confidence, Laplacian sharpness. None separates a correct read from
  a wrong one; a threshold on confidence drops 25 bad reads and loses 26 good
  ones. Doing this properly needs a trained legibility classifier.
- **Deep-EIoU**, which an earlier version of this file recommended on SportsMOT
  numbers. Never implemented. The tutorial designates SAM2 and this repo
  follows the designated baseline, so the comparison was never run and the
  claim has been removed rather than left standing unearned.

## Known limits

- **Eight seconds, not a game.** Identity does not survive a camera cut, and
  this broadcast cuts every 8.8 seconds. The route to a full game is in
  `docs/tracking-comparison.md`: segment at cut boundaries, re-prompt each
  segment, stitch identities across them by jersey number. Not built.
- **No event detection.** No shots, rebounds, assists or made/missed.
  `ShotEventTracker` ships in `sports@feat/basketball` and has never been run
  here. The event list on the homepage is placeholder and labelled as such.
- **No ball tracking.** Small, fast, heavily occluded — a different problem.
- **No tracking accuracy benchmark.** No HOTA, no IDF1. "10 of 10" is counted
  on one clip, not measured against ground truth.
- **One game, one broadcaster.** Nothing here has been tried on another arena,
  another camera crew, or another league.

`docs/gap-analysis.md` holds the full version of this, stage by stage.

## Run it

The models need a GPU-sized environment, kept outside the repo:

```bash
python -m venv ~/.venvs/courtvision
~/.venvs/courtvision/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
~/.venvs/courtvision/Scripts/pip install rfdetr supervision inference onnxruntime-gpu ultralytics \
    "git+https://github.com/roboflow/sports.git@feat/basketball" "supervision==0.27.0"
```

Then, gate first — it costs two minutes and decides the ceiling of everything
after it:

```bash
python pipeline/check_lineup.py --video web/media/demo.mp4 --render out/lineup.jpg
python pipeline/detect_cuts.py  --video web/media/demo.mp4
python pipeline/track_sam2_tutorial.py --video web/media/demo.mp4 --out out/tracks.json
python pipeline/identify.py --video web/media/demo.mp4 --boxes out/tracks.json \
    --rosters web/data/rosters_det.json --out out/identities.json --confirm roster
python pipeline/render_final.py --video web/media/demo.mp4 --boxes out/tracks.json \
    --identities out/identities.json --out out/final.mp4
python pipeline/project_tutorial.py --video web/media/demo.mp4 --boxes out/tracks.json \
    --identities out/identities.json --out web/data/nba.json
```

Long runs report to files rather than a terminal buffer —
`python pipeline/progress.py --serve`, then http://localhost:8799. A two-and-a-
half hour SAM2 run once produced no visible output at all because ultralytics
buffered every result; progress belongs in files.

The front end:

```bash
cd webapp && npx next dev -p 3100
```

`web/media/` and `web/data/` are the sources of truth; `webapp/public/{media,data}`
are junctions to them.

### Three Windows traps

Each reports something a long way from its cause.

- `rfdetr[train]` pulls in `opencv-python-headless`, which ships the same `cv2`
  module as `opencv-python` and wins by installing second. Everything keeps
  working until `calibrate.py` opens a window and OpenCV says it was built
  without GUI support. Uninstall the headless build and reinstall
  `opencv-python` — a superset, so training dependencies are satisfied either way.
- `inference` defaults its model cache to `/tmp/cache`, which on Windows
  produces `/tmp/cache\lora-bases/...` and a complaint from transformers about
  a malformed HuggingFace repo id. `config.inference_env()` sets
  `MODEL_CACHE_DIR` before the import — along with `ONNXRUNTIME_EXECUTION_PROVIDERS`,
  without which every model runs on CPU. Setting it after the import is
  silently ignored, and it only works with `onnxruntime-gpu` installed in place
  of `onnxruntime`. Measured: identify goes from 30 minutes to 9.5.
- The same package unpacks the OCR model's LoRA base into a *flattened*
  directory — `~lora-bases-smolvlm2-...-<hash>` — then reads it back from a
  nested path. The weights download fine and the load still fails, again as a
  repo-id error. The path differs by base: v3 wants
  `lora-bases/smolvlm2/smolvlm-256m/main`, v7 wants `lora-bases/smolvlm2/main`.
  Link the flattened directory to where it is looked for — a junction rather
  than a copy, since the 2.2B base is 12.6GB:

  ```powershell
  $c = "$HOME\.cache\inference"
  # v7 (2.2B, the default)
  $src = (Get-ChildItem $c -Directory -Filter "~lora-bases-smolvlm2-main-*")[0].FullName
  $dst = "$c\lora-bases\smolvlm2\main"
  # v3 (256M) would be ~lora-bases-smolvlm2-smolvlm-256m-main-* into
  #   $c\lora-bases\smolvlm2\smolvlm-256m\main
  New-Item -ItemType Directory -Force (Split-Path $dst)
  New-Item -ItemType Junction -Path $dst -Target $src
  ```

Clips must be written **faststart** or the browser cannot begin playback until
it has the whole file:

```bash
ffmpeg -i raw.mp4 -c copy -movflags +faststart web/media/nba.mp4
```

## Status

Experimental, and scoped on purpose. The box score on the homepage is the real
ESPN line for this game; the shot-zone split is illustrative and adds back to
the real field-goal totals; the event list is placeholder. Anything not
measured says so.
