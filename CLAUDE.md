# courtvision — working agreements for Claude sessions

Private demo repo (keep it private). The full private brief, when present,
lives in `CLAUDE.local.md` (excluded via `.git/info/exclude`, does not travel
with clones); the Chinese working notes live in `notes/` (also excluded).
This file carries the portable, non-sensitive operating rules so any machine's
session starts aligned.

## Communication

- Chat replies to Mark in **Chinese**. Everything that lands in the repo —
  code, comments, commit messages, README, web UI copy — stays **English**
  (the repo is shown to English-speaking interviewers).

## Git

- Commit each finished piece of work directly on `main`; no batching.
- Stage explicit paths only, never `git add -A` — Mark often has his own
  in-progress edits in the same tree.
- End commit messages with the Co-Authored-By line the harness specifies.

## Engineering discipline (hard rules, earned the hard way)

- When a baseline is designated, **read its source first** — the notebook, the
  repo, the paper. This file and the docstrings in `pipeline/` describe that
  baseline second-hand and have been wrong about it. Check the component ids,
  the package branch and version, the thresholds, the class ids, the weight
  filenames, and write the differences down before touching anything. Cost of
  skipping it, measured: SAM2 was prompted for days by
  `koppolusameer/rfdetr-...` out of `project.py` while the tutorial prompts
  with `basketball-player-detection-3-ycjdo/4`, and every diagnosis built on
  top of that was wrong. **If the same material is supplied twice, that is the
  signal it has not actually been read.**
- When a baseline is designated (a tutorial, a model, a config), **no
  component or parameter substitution without either (a) prior approval or
  (b) running both and presenting a same-conditions comparison**. Evidence of
  the substitute being better does not waive this.
- When proposing any new library/model, **state its provenance first**: who
  built it, where it came from (tutorial? user's material? my own knowledge?),
  why it is credible — then wait for a nod.
- Vendor-published benchmarks (a company measuring its own model) are
  marketing until reproduced; never quote a number the pipeline has not
  measured itself.
- Every per-frame processing stage must emit a review render
  (`out/review_<stage>_<source>.mp4`); extract and eyeball frames before
  reporting success. Metrics have lied here before; rendered frames caught it.

## Layout

- Frontend work happens **only in `webapp/`** (Next.js, port 3100).
  `web/index.html` (port 8765) is a frozen archive — do not update it.
  `web/media/` and `web/data/` remain the media/data sources of truth
  (webapp references them via hardlinks; write in place to keep inodes).
- Pipeline scripts live in `pipeline/`; long runs report to
  `pipeline/progress.py --serve` → http://localhost:8799.
- `out/` is the workbench; a render that is measured, watched and shipped
  gets **moved to `finals/`** (see `finals/README.md` for the three bars).
  The reel builds from `finals/`, never from `out/`.
- Secrets live in `.env` (gitignored): ROBOFLOW_API_KEY, HF_TOKEN. Never
  print values; verify presence by length only.

## Where the knowledge lives

- `docs/tracking-comparison.md` — every measured comparison (EN, shareable).
- `notes/pipeline-map.md` — the full pipeline map and decision tables (CN,
  local-only).
- `notes/tutorial-notebook.md` — the reference tutorial dissected, with our
  deviations and their evidence (CN, local-only).
- `notes/tutorial/*.ipynb` — **the tutorial's own source**, and the thing to
  read before trusting anything above about it. Local-only: 47MB with its
  outputs. Re-fetch on a new machine with
  `curl -L -o notes/tutorial/basketball.ipynb https://raw.githubusercontent.com/roboflow-ai/notebooks/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb`
- `notes/memory.md` — snapshot of session memory (CN, local-only).

## Current state (2026-08-10)

Summer League closes first, then Big V community footage. SL checklist:
Mark labels `out/harvest_numbers` (Roboflow) and fine-tunes RF-DETR himself
on the 4080 (`pipeline/train_rfdetr_numbers.py`, prepare → baseline → train →
eval); weights come back into `identify.py`; the conflicts queue in
`/review` reaches zero; the final full-clip render replaces
`web/media/nba_ai.mp4`. Big V then runs the fixed-camera route: one manual
calibration (`pipeline/calibrate.py`) instead of per-frame keypoints, same
pipeline downstream.
