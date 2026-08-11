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

## Team clustering, verified

The tutorial's SigLIP + UMAP + K-means (packaged as sports.TeamClassifier)
runs inside identify.py. Verified against jersey-number ground truth — the
44 tracks whose confirmed number exists in exactly one club's roster:

- **TeamClassifier: 36/44 = 82%** track-level accuracy; per-track crop
  votes have median purity 75%, and only 27% of tracks are unanimous.
- Our previous method (Fashion-CLIP zero-shot colour words) on the same
  ground truth: **35/44 = 80%**. A tie.
- The per-cluster crop grid (`img/team_clusters.jpg`, tutorial-style
  eyeball check) shows why: the blue/white split is right, but crowded
  boxes crop the wrong torso and tracks that switch players mid-life
  poison their own vote.

Two unrelated methods hitting the same ceiling says the bottleneck is
track purity, not the classifier. Improving teams means improving
tracking (or cropping smarter), not swapping embedding models.

The same conclusion arrived from a second direction: re-reading the
notebook showed it matches number regions to SAM2 *silhouettes*, not
boxes, so identify.py was made faithful (--match mask) and A/B'd on the
full clip. Box vs mask: 60 vs 60 confirmed numbers, 48 vs 48 named, 9
unique players both, misreads 4 vs 3, cross-club conflicts 8 vs 9,
cluster-owner agreement 82% vs 80% — statistically identical. The number
attachments were not the error source; tracks that switch players
mid-life are. Every accuracy signal in this document now points at the
same place: **track purity is the ceiling.** The levers that could move
it: better calibration (finer keypoints → tighter association gate),
appearance-based track splitting, or a ready-made tracker on court
coordinates (the untried Norfair experiment).

## ResNet-32 vs SmolVLM2: the blog's conclusion reproduces, then inverts

The blog's other number reader — "ResNet-32 ... reached 93% test
accuracy, outperforming the fine-tuned SmolVLM2" (86%) — publishes the
dataset and the claim but no training code, so
`pipeline/train_resnet_ocr.py` fills the recipe in (CIFAR-style
resnet32 from torch.hub, pretrained CIFAR-100, choices documented in the
script) and measures both sides on the same held-out test split:

| basketball-jersey-numbers-ocr/3 test set (312 crops) | blog says | measured |
|---|---|---|
| ResNet-32 (ours, 60 epochs) | 93% | **94.6%** |
| SmolVLM2 (their hosted model) | 86% | **82.1%** |

Reproduced — in domain. On *our* footage's number crops (eyeball-read
ground truth, small n=4 of clear cases) the ranking inverts: SmolVLM2
reads 3 of 4, ResNet-32 reads 1 of 4. The training data is Celtics/
Knicks/Magic playoff jerseys; Summer League fonts and colours are out of
distribution, and the 32x32 CNN degrades where the OCR-pretrained VLM
holds up. The pipeline therefore keeps SmolVLM2. The blog's conclusion
is not wrong — it is in-domain, and the fix it implies is fine-tuning
on crops from the target footage, which is exactly the hand-labelling
session already planned.

## The tutorial's shot events vs the hand tags

The tutorial's event route needs no training: the same detector
classifies shot poses (player-jump-shot, player-layup-dunk) and the
made-basket moment (ball-in-basket), and sports.basketball's
ShotEventTracker debounces those flags into events. Run over the full
178s clip (`pipeline/shot_events.py`, out/shot_events_auto.json) and
scored against the 12 hand-tagged events:

- **Attempt detection is real**: 5 of the 8 hand-tagged shots inside the
  tagged region are found within ±2s, plus one genuine attempt at 25s the
  hand pass skipped (its rebound was tagged — a rebound implies the miss).
  The 72.4s miss → 75.1s putback pair is reproduced exactly.
- **Made/missed is not to be trusted from this angle**: 3 of 5 matched
  outcomes correct. Two made baskets (6.2s, 36.2s) are called MISSED —
  ball-in-basket needs the ball visibly in the hoop, which a distant
  side broadcast angle rarely shows. The failure is systematic
  (biased toward MISSED), not random.

So the honest split stands, now with numbers: attempt candidates could be
auto-proposed, outcomes still need the hand pass (or a rim-focused
camera). The viewer keeps its hand-tagged events.

## Tutorial-style top-down maps, both trackers

`pipeline/render_map.py` renders the notebook's final visual — club-
coloured dots on a drawn NBA court — from any tracks file, through our
per-frame homographies, so both trackers go through identical code:
out/map_sam2_30s.mp4 vs out/map_court_30s.mp4 (stacked:
out/compare_maps_30s.mp4, boxes side: out/compare_trackers_30s.mp4).
SAM2's structural miss shows up directly as a sparser map.

## SAM3 concept tracking, measured (2026-08-09)

SAM3 (Meta, gated weights; used for player tracking by arXiv 2607.21267)
replaces box prompts with a text concept and keeps admitting new
instances mid-video — on paper, both of SAM2's structural limits. Run on
the same 18s ESPN segment with the paper's exact prompt ("basketball
player on the court"), via ultralytics' SAM3VideoSemanticPredictor:

| same 18s ESPN segment | SAM2 large (box prompts) | SAM3 (text prompt) |
|---|---|---|
| players covered | 9 of 10 (frame-0 miss stays missed) | all 10, incl. mid-video pickups — LJ Cryer gets named where SAM2 never saw him |
| track ids over 18s | 9, lifetime = whole clip | **117**, median lifetime **0.6s** |
| unique players named (same OCR/voting) | 6 | 7 |
| artifacts | one player invisible | the bench masked as one blob despite the spatial qualifier |
| speed / VRAM | 0.75 fps, 5.4GB | 0.25 fps, 9.1GB of 10.24 |

Reading: SAM3's recall claim is real and its identity persistence is
not — with these defaults it behaves like a segmenting detector, not a
tracker, and the jersey-number voting is what stitches its 117 fragments
back into 7 named players. Fragment churn is what our number-voting
already absorbs, so SAM3's coverage may still win on longer footage —
but at 3x SAM2's cost and the VRAM ceiling, it is an experiment, not the
default. Script: pipeline/track_sam3.py; side-by-side:
out/compare_sam2_sam3_espn.mp4.

## Decision (revised 2026-08-09, by the user, with new evidence)

The regime decides the tracker. On the tutorial's own single-shot clip,
the fully notebook-faithful chain — their detector prompting sam2.1_large,
their OCR, their annotators — scores **10 tracks, 10 full names, 0
unresolved**, with per-frame boxes eliminating the stale-chip artifacts
the 5Hz court-space render suffered (out/tut_final_sam2.mp4 vs
out/tut_final.mp4). The one visible error is a duplicated big-man
identity, the same two-slots-one-player mode measured before.

So: **SAM2 is the tracker for single-shot segments** — the tutorial
regime, where every earlier objection (cuts, VRAM growth, frame-0-only
prompts) is void by construction. Everything measured above about
broadcast footage with cuts still stands; the full-game path implied by
the blog's own caveat ("a component that monitors detections and
re-prompts") is shot segmentation: split at cut boundaries, run SAM2 per
segment with fresh prompts, stitch identities across segments by jersey
number. Court-space association remains in the repo as the measured
alternative and the source of the court-coordinate viewer data.

## Fine-tuning the number detector on our own labels (2026-08-10)

The premise going in was that the number-region detector
(`basketball-player-detection-3-ycjdo/4`) was trained on 2025 playoff kits
and had never seen ours — measured earlier at 6.5 regions/frame on its home
domain against 2.2 on Summer League. This run tested the fix on **playoff
footage**, which is the detector's home domain, so it measures the labelling
pipeline rather than the domain gap. Source: NYK @ DET 2025-04-27 (ABC,
1080p60, 136.7 min).

145 labelled frames, sampled every 8s across the full game from tip-off,
close-ups filtered out, split by game time — train 101 / valid 14 / test 30
(163 boxes). RF-DETR Base, 50 epochs, batch 8, grad accum 2, lr 1e-4, 11
minutes on a 4080 SUPER.

| on the same 30-frame test split | precision | recall | tp | fp | fn |
|---|---|---|---|---|---|
| base detector, untouched | **95.7%** | **82.2%** | 134 | 6 | 29 |
| fine-tuned on our 101 frames | 68.0% | 74.2% | 121 | 57 | 42 |

**The fine-tune is worse, and not marginally**: precision falls 27.7 points
and false positives go from 6 to 57. On f0468814 the base detector is
perfect — 7 numbers, no misses, no false alarms — while the fine-tune finds
3, misses 4, and puts boxes on **spectators in the stands**
(`img/numbers_baseline_f0468814.jpg` vs `img/numbers_finetune_f0468814.jpg`).
Crowd numbers were deliberately excluded during labelling; 101 frames were
not enough to teach the distinction, and were enough to damage what the base
weights already knew.

That is the result, and it is a measurement rather than a failure: 101 frames
of ours against 13 games of theirs, on their home domain. It says nothing
about the Summer League gap, which is the case the fine-tune was proposed
for and which this footage cannot test. **Decision: `identify.py` keeps the
base detector.** The chain — harvest, shot-type filter, label, ingest,
time-based split, baseline, train, eval, render — is in the repo and
reproducible, which is what makes the next attempt on Summer League footage
cheap.

## The tutorial chain, end to end on NBA playoff footage (2026-08-10)

blog.roboflow.com/identify-basketball-players run as specified -- RF-DETR
detection, SAM2 tracking, SigLIP/UMAP/K-means teams, SmolVLM2 numbers -- on
NYK @ DET 2025-04-27. No homography, no court-space association: the tutorial
does not use them.

**Segment choice decided the result, and it took three attempts to see that.**

A 3-minute window picked for *wide-shot density* (73%) still contained a cut
every 8.8s. SAM2 lost two thirds of its objects inside 7.5s and had moved
track 0 from Brunson onto a referee by 15s -- while reporting every one of
its ten prompts alive for the full 30s, because `--conf 0.05` deliberately
keeps lost objects in the output to protect the slot mapping. The metric
cannot see this; the render can.

De-duplicating the prompts (two detections on one player, plus the net boxed
as a player at 0.83) changed nothing measurable. The bottleneck was the cuts.

On a 17-second segment verified single-shot -- detect_cuts.py reporting zero
cuts, confirmed by eye -- the same chain behaves as the tutorial describes:

| | 30s, cut every 8.8s | 17s, single shot |
|---|---|---|
| prompts | 9 | 8 |
| still tracked at ~8s | 3 | **8 of 8** |
| identity drift | track 0 onto a referee | none seen |

and the identification stage then works:

- 691 OCR reads over 204 sampled frames
- **8 of 8 tracks with a confirmed number** (majority vote, >=2 and a clear winner)
- **7 carrying a full name**: Brunson, Towns, Bridges, Anunoby, Duren,
  Cunningham, Thompson
- clusters mapped to clubs correctly on 7-vs-5 roster evidence
- the eighth read 9 for a Knick, and no Knick wears 9. It is rendered as a
  bare `#9` rather than given a name, which is the intended behaviour.

Two limits are structural rather than faults. SAM2's prompts are taken on
frame 0 only, so players who enter later are never identified -- several
appear untinted in the final render. And a broadcast cuts every 8.8s here,
so this chain covers a possession, not a game; the full-game path remains
shot segmentation with re-prompting, stitched by jersey number.

Artifacts: out/final_det3006.mp4, out/review_sam2_det3006.mp4,
out/identities_det3006.json.

## Jersey OCR: the notebook's v3 against v7 (2026-08-12)

The notebook pins `basketball-jersey-numbers-ocr/3`. Roboflow has published up
to v7, and the difference is larger than the version numbers suggest: v3's
LoRA base is SmolVLM2-**256M** (884MB), v7's is the **2.2B** (12.6GB). Training
crops also grew, 3136 to 3615.

Same 33-second segment, same SAM2 tracks, same roster, same
roster-constrained assignment. Only the OCR version changed.

**Both name the same eleven tracks.** What changes is how hard the evidence is:

| track | v3 (256M) | v7 (2.2B) |
|---|---|---|
| #3 Josh Hart | `9:95, 3:50` — **the misread wins** | `3:174, 2:5` |
| #8 OG Anunoby | `8:139, 0:103` | `8:169, 0:65` |
| #25 Mikal Bridges | `25:121, 8:32, 45:21` | `25:142, 5:19, 8:12` |
| cluster→club evidence | 9 vs 7 | 10 vs 6 |

Hart's 3 read as a 9 in every earlier run on this footage, on both segments
and under both aggregation rules — the one systematic error this pipeline
had. On the 2.2B base it disappears.

The identical outcome is not an argument for staying on v3. It only holds
because the roster constraint is strong enough to overturn a 95-50 misread,
and that safety net has a hole in it: track 7 reads `8:139, 0:103`, and 0 is
a number the Knicks roster does contain (Delon Wright). A read that is wrong
*and* plausible cannot be caught by the roster, and v3 was 36 votes from
producing one. v7 widens that to 104.

v7 is the default from here; `--ocr-model basketball-jersey-numbers-ocr/3`
restores the notebook's.

Cost: the 2.2B base is a 12.6GB download, and `inference` unpacks it to a
flattened cache directory it then cannot read — the same trap as the 256M
base but at a different path (`lora-bases/smolvlm2/main`, without the
`smolvlm-256m` level). A junction avoids storing it twice.
