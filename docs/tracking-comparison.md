# Three trackers, one clip: what actually holds identity

Same footage throughout: the GSW–MEM Summer League broadcast clip
(`web/media/nba.mp4`, 178s, 25fps, hard camera cuts at replays). All numbers
below were computed from the output files in `out/`, not quoted from papers.

## Full clip (178s)

| tracker | ids where ~10 players stand | lifetime median | p90 | max |
|---|---|---|---|---|
| court-space (ours: associate after homography) | 129 | **12.2s** | 29.2s | 76.2s |
| ByteTrack (supervision, tutorial settings, 12.5Hz) | 379 | 1.6s | 12.0s | 44.4s |

ByteTrack associates by IoU in image space, so every camera cut and every
fast pan births a new id. Projecting to court coordinates first divides the
camera motion out, which is worth a ~7x longer median identity on this
footage. Neither survives a hard cut: a replay is a teleport in any
geometry.

## The SAM2 experiment (30s test window, 55–85s, cut at 64.4s)

SAM2 was the one candidate with a *mechanism* for cuts — appearance memory
instead of motion. `sam2.1_t` on a 750-frame test clip, prompted with the 9
confident player boxes on frame 0:

- **By the metric: perfect.** 9/9 prompts alive at the end, every lifetime
  spans the full 30s. In the same window ByteTrack burns through 88 ids
  (median 1.3s) and not one crosses the cut.
- **By the eyeballs: the ids shuffle.** Rendering the boxes and reading
  jerseys (`img/sam2_f*.jpg`):
  - tid 5: white #10 → reattaches to **blue #1** after the cut → later
    back to a white player
  - tid 4: blue #1 → a white player → ends on **blue #3**, which was
    tid 7's player at frame 0
  - tid 1: blue ballhandler → white #22 → back to a blue player
  - tid 7: blue #3 → **blue #45**

  At least 4 of 9 identities are provably on the wrong player at some
  point after the cut. During the close-up segment before the cut SAM2
  correctly output nothing (frame 215: zero boxes) — it fails by
  *reattaching wrong*, not by hallucinating.

**It also just misses people** (review finding, from watching
`review_sam2_*.mp4`): coverage is structurally capped by the wrapper's
first-frame-only prompting — whoever isn't a confident detection on frame
0 (a sub, a bench player standing up, the 10th man at 0.32 conf) never
gets an id — and objects lost to a close-up or occlusion only partially
reacquire (3 of 9 right after the cut, 8 of 9 by the end). A
detection-per-frame pipeline cannot miss a visible player for long;
prompt-based tracking can miss them forever.

**The real finding: SAM2 changes the failure mode from loud to silent.**
ByteTrack's failure is visible in the data — a new id appears, and
downstream code can see the seam. SAM2 keeps the same id on a different
human, which poisons per-player stats with no signal that anything
happened. For a stats pipeline that is strictly worse.

**Replicated with the stack's default model.** The tiny-model run above
was a substitution and could have taken the blame, so the experiment was
rerun with `sam2.1_b` — the script's and the Roboflow pipeline's default —
on a 10s window (58–68s, same cut). Same verdict, jersey-verified: tid 0
starts on blue #1 and ends on white #23. During the close-up it correctly
tracks nothing, and it reattaches most ids after the cut — to the wrong
players. Review footage: `out/review_sam2_10s.mp4` /
`out/review_sam2_30s.mp4` (rendered by `pipeline/render_tracks.py`).

Also measured, for the record:

- SAM2's VRAM grows with clip length (it banks memory features per
  frame). `sam2.1_b` on the 178s clip overflows the 10GB card
  (9.97/10.24GB), spills to system RAM, and crawls at 0.3–0.4 fps; the
  same model on a 250-frame clip fits in 4.5GB and runs 0.88 fps.
  Before the `stream=True` fix it also buffered every result invisibly
  until the end — 2.5h of GPU produced nothing.
- `sam2.1_t` fits in 4.3GB at any tested length and runs 0.91 fps — the
  30s clip took 13.7 minutes.

## Identity head-to-head: the tutorial's tracker vs ours, same everything else

The jersey-OCR pipeline (identify.py) was run twice on the same 30s window
(55–85s, hard cut at 64.4s) with the same detector, OCR model, voting
policy, and roster — the only variable is the tracker underneath. The
SAM2 side uses `sam2.1_l`, matching the notebook's hiera-large (via
ultralytics' predictor rather than the notebook's
segment-anything-2-real-time fork, which needs a CUDA source build).

| same window, same OCR | SAM2 large (tutorial) | court-space (ours) |
|---|---|---|
| tracks in window | 9 (only frame-0 prompts exist) | 41 fragments |
| players boxed at t=20s | 5 | 11 |
| OCR reads fed to voting | 146 | 220 |
| tracks with a confirmed number | 5 | 10 |
| full names resolved | 3 rows, **2 unique players** (two SAM2 slots converged on the same #45) | 7 rows, **6 unique players** |
| tracker runtime | 21 min, 5.4GB VRAM | comes free with the projection pass |

Same-instant frames (`out/review_id_sam2_30s.mp4` vs
`out/review_id_court_30s.mp4`, t=20s): the court-space side names Boozer,
Prosper, Richard, Coward, Cryer and Ike on the correct jerseys; the SAM2
side names Ike and Coward, misses half the players on screen, and its two
other numbered tracks are a 33→35 misread and a #0 not in the roster.
Both sides share the OCR failure modes (33→35 happens on both); the gap
is coverage: more tracked boxes means more OCR chances, and fragments
that break at the cut vote independently instead of letting one
identity ride across a player swap.

Worth saying plainly: fragmentation, the court-space tracker's weakness
in the tracking comparison above, is mostly harmless here — the numbers
stitch fragments back to the same player. Continuity, SAM2's apparent
strength, is exactly what hurts: a track that silently changes player
mid-life carries its confirmed number onto the wrong human.

## Decision

The demo ships the court-space tracker. Cuts stay track boundaries on
purpose: a visible seam is honest, an invisible identity swap is not.
SAM2 stays in the repo as a measured experiment (`pipeline/track_sam2.py`,
evidence frames in `img/`) — it is the "here's where it fell apart" part
of the story, with numbers.
