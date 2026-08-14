# The tutorial, as source material

The walkthrough this project reproduces, and every resource it links, sorted by
whether we have it and what it answers. Kept because the same links have been
handed over three times and the repo has twice described this baseline wrongly
from memory — see `docs/tracking-comparison.md`, "What the tutorial actually
does about occlusion".

Video: https://www.youtube.com/watch?v=yGQb9KkvQ1Q
Blog: https://blog.roboflow.com/identify-basketball-players/

## Chapters, mapped onto our stages

| Video | Stage here | State |
|---|---|---|
| 02:04 Detect players and numbers with RF-DETR | `check_lineup.py`, prompt frame | reproduced; plus an on-court filter the tutorial has no equivalent of |
| 07:12 Track players with SAM2 | `track_sam2_tutorial.py` | reproduced line by line; the notebook prompts frame 0 only and never re-prompts |
| 11:59 Team clustering with SigLIP, UMAP, K-means | `identify.py` via `sports.TeamClassifier` | reproduced |
| 16:31 Fine-tuning SmolVLM2 | `identify.py`, OCR v7 | reproduced, then improved on measured evidence |
| 21:56 Map positions to court coordinates | `project_tutorial.py`, `oncourt.py` | reproduced |
| 31:51 Detect shot event and classify result | `shot_events.py` | **written without reading their version** |
| 35:54 Conclusions | | |

## Already on disk — do not download

Installed in the venv, full source readable at
`C:\Users\fqche\.venvs\courtvision\Lib\site-packages\`:

- **`sports`** — the highest-value read of the four. `common/team.py` holds the
  TeamClassifier (`KMeans(n_clusters=2)` hardcoded, UMAP n_components=3), which
  is what explained the cluster collapse on unfiltered SAM3 tracks: with the
  bench in frame the two clusters become on-court vs bench rather than one club
  vs the other. `common/temporal.py` holds `ConsecutiveValueTracker`, exported
  from the package top level — `identify.py` still claims it is unavailable,
  which is wrong and should be corrected. Also `common/view.py`
  (ViewTransformer) and `basketball/config.py` (CourtConfiguration).
- **`supervision`** — annotators, `mask_to_xyxy`, `filter_segments_by_distance`.
- **`rfdetr`**, **`inference`**.

The main notebook is at
`notes/tutorial/basketball_ai_how_to_detect_track_and_identify_basketball_players.ipynb`
(47MB with outputs, local-only). Read: its clips are 8s and 5s, which is why
the merge failure measured here never appears in it.

## Worth fetching, in order

1. **Make or Miss — jumpshot detection notebook.** The one resource in this
   list nobody here has read, and it covers the one stage written without
   reading theirs.
   https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-make-or-miss-jumpshot-detection.ipynb

2. **Court keypoint dataset.** `seg_00m30.68s_17s` cannot be court-solved --
   nine landmarks in an 817x249px patch, median residual 1.91m against 0.09m
   elsewhere -- which disables the on-court filter on exactly the segment that
   needs it. The training data would say whether that camera angle is out of
   distribution.
   https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2

3. **Jersey number OCR dataset.** ResNet-32 measured 94.6% on its test split
   against SmolVLM2's 82.1%, and the ranking inverts on our footage. The
   dataset's fonts and kit colours are the stated reason.
   https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr

4. **Player detection dataset** — the detector we prompt with, already used as
   a hosted model.
   https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo

## Low priority

Both jobs are done and recorded: RF-DETR fine-tuning was measured *worse* than
the base model (95.7%/82.2% -> 68.0%/74.2%), and SAM2 video segmentation is
what `track_sam2_tutorial.py` already does.

- Fine-tune RF-DETR notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb
- Segment video with SAM2 notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-videos-with-sam-2.ipynb
- RF-DETR: https://github.com/roboflow/rf-detr
- Supervision: https://github.com/roboflow/supervision
- Sports: https://github.com/roboflow/sports (the installed copy already carries
  the `@feat/basketball` API; fetch only for a newer branch)

Author: https://github.com/SkalskiP
