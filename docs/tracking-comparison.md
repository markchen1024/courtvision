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

## A roster is only as strong as what it excludes (2026-08-12)

On the 17-second segment the pipeline named a Knick who never left the bench.

Duren's 0 was read correctly. The team classifier put that track on the wrong
side. The roster then found a Knicks #0 -- Delon Wright, DNP in this game --
and the constrained assignment accepted it, because a number that exists on
the roster is exactly what it checks for.

This is the hole named earlier in this file: a read that is wrong *and*
plausible cannot be caught by a roster. What was missed is that we had widened
the hole ourselves. The roster was built from the full ESPN team list,
including eleven players who did not play, and every one of them is a legal
answer waiting for a misassigned track to find it.

Dropping the DNPs takes the Knicks from fifteen numbers to nine. The 0 is gone,
so that track can no longer be given a name at all -- it goes blank, which is
the honest outcome, and blank tracks are neither labelled nor tinted.

Worth stating plainly: the bug was present for every segment. The first four
did not hit it only because the numbers they misread happened not to exist on
the other team's bench.

## Two tracks on one player, and how to tell the two kinds apart (2026-08-12)

On the 19-second segment Brunson went unmarked from about twelve seconds in.
He was not lost. SAM2 held both his track and Beasley's, and after their
contact both were on Beasley: IoU 0.06 at 11.0s, 0.29 at 12.0s, 0.97 at 12.5s,
1.00 from 14.0s to the end of the clip. Brunson's box area jumped 4472 to
24738 at the moment of contact.

Nothing downstream could see it. Both tracks had read their own jersey for
twelve clean seconds before the collision, so both numbers were right; the
duplicate gate compares numbers, and 11 is not 5. The vote is taken over the
whole clip, so the contaminated reads were outvoted rather than excluded. The
render then drew exactly what it was told.

`pipeline/overlap.py` tests the tracks for sustained shared position --
intersection over the *smaller* box, since a swallowed player sits inside the
larger one and IoU understates that badly. Two faults look identical to that
test and want opposite treatment, so they are separated by coverage, the
overlap's length over the shorter track's lifetime:

| | coverage | what it is | treatment |
|---|---|---|---|
| duplicate | >= 90% | frame 0 prompted twice on one man | retire the track with fewer votes, keep the other's reads |
| collapse | < 90% | two players merged mid-clip | no votes and no drawing inside the span |

Both thresholds were set from measurements on this footage: IoS 0.75 sustained
1.0s. At 0.5s the test flagged ordinary crossings -- players pass behind each
other constantly and are apart again inside half a second.

Treating a collapse by suspending trust rather than repairing it follows what
sports MOT does about occlusion. FC-Track and relatives build a pairwise IoA
matrix over tracklets and suspend appearance updates where overlap is high.
Repairing it -- deciding which man each track was on -- is GTA's job (OSNet
ReID embeddings, DBSCAN over a tracklet, +3.83 HOTA on SportsMOT), an offline
pass with a model attached. The spans found here are its input if it is ever
built.

Measured on the 19s segment after the change: 104 number regions ignored
inside collapsed spans, one duplicate pair retired (track 9 had 0 votes
against track 10's 109 -- it was Duren's shadow from frame 0), 1025 reads,
still all ten players named, cluster-to-club still 10 against 3. Frames pulled
from the render confirm it: at 8s all ten carry chips; at 13s and at 18s the
two collapsed pairs carry neither chip nor tint while the other six are
correct. Two labels on one player is replaced by no label on either, which is
the honest output when the tracker cannot say which is which.

The same test run over every segment already rendered:

| segment | duplicates | collapses | track-frames distrusted |
|---|---|---|---|
| tut_sample | 0 | 0 | 0 |
| seg_02m27.00s_14s | 0 | 1 | 182 |
| demo | 0 | 2 | 304 |
| seg_00m30.68s_17s | 0 | 1 | 1236 |
| seg_01m10.87s_19s | 1 | 5 | 1886 |
| seg26 | 2 | 7 | 2249 |
| seg33 | 0 | 5 | 4531 |

seg33 is the 33-second clip on the home page, and its number is the least
alarming of the set: 33.4s of it belong to track 1, which never got a number
and was therefore never drawn. The three collapses between tracks that *were*
drawn last 1.3s, 1.5s and 1.6s. seg26 is the one that most needs the rerun.

## Blocking bad reads is not enough on its own (2026-08-12)

Rerunning the other segments through the overlap gate changed exactly one
label, and changed it for the worse. On `seg_00m30.68s_17s`, track 10 went
from `#3 Josh Hart` to `#1 Cameron Payne`:

    before the gate   '3' x 22,  '1' x 6,  '11' x 2, ...
    after the gate    '1' x 4,   '11' x 1, '7' x 1,  '17' x 1

All twenty-two reads of 3 fell inside the span where track 10 was collapsed
onto track 8, so they were blocked -- correctly. What was not anticipated is
what the roster-constrained assignment does with what is left. It is a
maximum-weight matching, so it always finds something: four stray reads of 1,
and Cameron Payne was still unclaimed. A name that was right by accident was
traded for one that is wrong on purpose.

The gate added for this is a fraction, not another threshold on reads: a track
collapsed for more than half its life gets no number and no name. Below half,
the pre-contact evidence still stands and the track keeps its identity -- it
simply goes unmarked inside the span, which is what the 19s segment does and
what was verified there. Above half there is not enough track left to say
whose it is. Measured: the 19s segment's collapsed tracks sit at 34-44% and
are unaffected; this segment's tracks 8 and 10 sit at 60% and both drop out.
Named tracks went from 10 to 8, and the two that left were `#2 McBride` and
`#1 Payne`, neither of whom was on the floor.

What that segment then exposed is a separate fault the overlap test cannot
see. Cutting each track's box out of the raw video, at 3s:

| track | named | actually on |
|---|---|---|
| 8 | `#2 McBride` | Towns -- its own top read is `32` x 26, but track 6 had taken 32 |
| 6 | `#32 Towns` | a Pistons player in white; its votes are `'0' x 30, '32' x 30` |
| 10 | `#1 Payne` | collapsed onto track 8 for 60% of the clip |

Track 6 is on a Piston at 3s and on Towns from about 8s -- an identity switch,
not a collapse. The two players never sat on top of each other long enough for
the IoS test to fire, so nothing flags it, and the render puts `#32 TOWNS` on
a Detroit player while the real Towns stands unmarked beside him.

Its signature is visible in the votes and nowhere else: a dead tie between
`0` and `32`, which are numbers belonging to *different clubs*. The majority
rule would have refused it -- it requires a strict winner -- but roster mode
replaces majority with the matching, and the matching has no such rule. That
is worth fixing separately, and carefully: the whole value of the matching is
that it resolves ties like track 7's 25-against-8, where both numbers belonged
to the same club and one was already taken. A tie *across* clubs is a
different animal and means the track is on two players.

Until that is settled, `seg_00m30.68s_17s` carries a wrong label in its first
five seconds and should not go in the reel.

## Two boxes on top of each other say nothing about what is under them (2026-08-12)

The first version of the overlap gate suppressed four of ten players for the
last five seconds of the 19s segment, and two more at 5s and 11s. Mark caught
it on the render. Cutting the boxes out of the raw video at 16.0s, where both
pairs had effectively identical boxes:

| pair | shared box | what is in it |
|---|---|---|
| tracks 6 and 8 | 1138,297,1221,478 (identical) | **one player** -- Beasley. Brunson is gone. |
| tracks 1 and 5 | 907,525,1062,781 vs 899,525,1062,781 | **two players** -- Towns posting up, Harris behind him. |

The second is not a failure. Neither track lost anyone; the boxes overlap
because one man is standing in front of another, which is most of basketball.
Geometry cannot tell the two apart -- both give IoS near 1.0 -- and treating
them the same is what stripped the labels off players who were tracked
perfectly well.

The discriminator is the detector the pipeline already runs. Sample each
candidate span, count player detections whose centre falls inside the shared
box, take the median: two or more is occlusion and is left alone, one is a
collapse. Cost is about thirty extra detector calls per span.

Reclassified with it, the 19s segment:

    duplicate  9 and 10    0.00s - 19.22s
    occlusion  1 and 5     4.04s -  5.82s   2 in the box
    occlusion  8 and 9    10.68s - 11.88s   2 in the box
    occlusion  8 and 10   10.68s - 11.88s   2 in the box
    collapse   6 and 8    11.88s - 19.22s   1 in the box
    occlusion  1 and 5    14.48s - 19.22s   2 in the box

One collapse out of five candidates. Distrusted track-frames fell from 1960 to
882, and the render checks out: ten of ten labelled at 5s and at 11s, eight of
ten at 16s, the two missing being exactly the pair that merged.

The other segments, reclassified: `seg_02m27.00s_14s`'s only candidate is
occlusion, so it loses nothing at all. `seg_00m30.68s_17s`'s remains a genuine
collapse -- one player in the box -- so tracks 8 and 10 stay unnamed there.

## A spectator took a prompt, and the gate called it a full lineup (2026-08-12)

Mark: "活塞二号从始至终都没有 cover 住." Cade Cunningham is never marked in
`seg_02m27.00s_14s`, and the reason is on frame 0. The gate reported *10
detections, 10 prompts, 10 needed -- PASS*. One of those ten, at confidence
0.64, was a spectator in the front row wearing a CUNNINGHAM #2 Pistons jersey.

It took a prompt. SAM2 tracked it for the whole clip. The OCR read `2` off its
back twenty times -- the only number that track ever produced. The team
classifier put it in the Knicks cluster on four crops, and since 2 exists on
both rosters the constrained assignment named it `#2 Miles McBride`. The real
Cunningham, off camera at frame 0, was never tracked at all.

The gate exists to promise that a run from this frame can reach every player.
A crowd detection breaks that promise twice: it fills a slot a player should
have had, and it makes a nine-player frame look like ten.

`pipeline/oncourt.py` tests where a detection is standing, using the
tutorial's own court model rather than anything new: landmarks from
basketball-court-detection-2/14, ViewTransformer onto the NBA plan from
sports.basketball, and the feet -- bottom-centre, the only part of a standing
player on the court plane. On that frame the spectator lands 4.1m past the
sideline; the deepest real player is 1.7m inside. A 3m margin separates them
and still allows for inbounders standing behind the baseline.

Two things were needed to make it safe:

**It must check its own fit.** The first version threw out two real players on
frame 0 of `seg_00m30.68s_17s`, projecting a man standing in the paint to 44m
off the far sideline. That frame yields nine landmarks bunched into an
817x249px patch, and the homography fitted to them misses its own anchors by a
median of 1.91m, against 0.09m and 0.12m on the two frames whose verdicts are
correct. So the transform is asked to reproduce the points it was built from,
and a solve above 0.5m is not allowed to judge anybody -- that segment now
falls open and prompts with everything, as before.

**The alternative was worse.** An image-space convex hull of the landmarks
needs no homography and cannot blow up like one, but the landmarks cluster
near the middle of the visible court: the hull put five of nine real players
outside on the 14s frame. Rejected.

With the filter on, `seg_02m27.00s_14s` fails its gate at 9 prompts -- which
is the honest verdict, since the tenth player is off camera. Scanning the clip
finds ten on-court players from frame 60 onward, so the segment simply starts
one second too early.

This is a deviation from the notebook, which prompts with every player-class
detection on frame 0. `--no-court-filter` restores that on both
`check_lineup.py` and `track_sam2_tutorial.py`.

Re-cut one second later as `seg_02m28.00s_13s` and run end to end: the gate
passes at 11 detections, one struck out as off the court, 10 prompts. SAM2
keeps all ten for the full 12.5s. Ten tracks carry a full name, and the split
is five a side for the first time on this segment -- Knicks Brunson, Towns,
Hart, Bridges, Anunoby; Pistons Harris, Beasley, Duren, Hardaway Jr., and
**Cunningham**, track 10, `2` read twenty times, verified against the raw
frames where the jersey is legible. `#2 Miles McBride` is gone, because
nothing was left over to give his number to. Both spectators in CUNNINGHAM
jerseys sit in the crowd untinted and unlabelled.

## Re-seeding SAM2 fixes the tracking and breaks the naming (2026-08-12)

Mark, on the 19s render: Beasley and Brunson just disappear at 12s. They do,
and by design -- after their contact both tracks sit on Beasley, so there is
no box on Brunson at all and no rendering choice brings him back. The only
real fix is upstream, so the tracker was changed to re-detect and re-seed
every three seconds, inheriting ids by a one-to-one assignment. One-to-one is
the whole trick: when two tracks have merged onto one man, only one of them
can be given that man's box and the other is forced onto what is left, which
is the player it lost.

It works, at the level it was aimed at:

| | prompt once | re-seed every 3s |
|---|---|---|
| merge lasts | 7.3s, to the end of the clip | 3.0s |
| Brunson | lost from 11.9s | re-acquired at 15.0s, verified on the crop |
| Duren's duplicate | ids 9 and 10 | gone -- prompts are de-duplicated now |
| tracks named | 10, all correct | 11, one of them invented |

Two things it did not fix, and one it broke.

It did not un-merge them at 12.0s. The re-seed lands while the two are still
in contact, the detector returns two boxes on top of each other, and SAM2 has
them merged again within a fifth of a second. Only the 15.0s re-seed, after
they separate, splits the pair. Re-seeding during contact is wasted.

Ids drift. Track 11 was born at 9.0s on a man standing behind the basket --
inside the court plus 3m, so the on-court filter allowed him -- and became
Brunson at 15.0s. Its crops therefore span two people and the team clustering
split 11 votes to 10, putting it on the Pistons, whose roster has no 11. The
recovered track went unnamed.

And it invented a player. Harris ended up on two ids, 5 until 15s and 12 from
18s, both reading `12`. The roster-constrained assignment gives one number per
track, so the stronger took #12 and the other was pushed onto the only number
left: `#7 Paul Reed`, who did not play in this segment. Trading two players
unmarked for seven seconds against a name on screen belonging to a man who is
not on the floor is a bad trade by the standard the rest of this file keeps.

The missing piece for both is tracklet association: tracks that never coexist
and read the same number are one player, and must be merged before the
assignment runs. That is GTA's job done with jersey numbers instead of ReID
embeddings, which is the stronger evidence of the two. Until it exists,
`--reprompt-seconds` defaults to 0 and the renders keep the single prompt.

## What the tutorial actually does about occlusion: nothing (2026-08-12)

The blog's model list says of SAM2: "re-identifies players after occlusion and
keeps target IDs stable through body contact." Read against the notebook, that
is a claim about SAM2's memory, not a description of code. `SAM2Tracker`
(cell 40) has two methods:

    prompt_first_frame(frame, detections)   load_first_frame + add_new_prompt
    propagate(frame)                        predictor.track(frame)

and a `reset()` that sets `self._prompted = False` and nothing else. All six
call sites (cells 41, 55, 65, 67, 71, 85) prompt once at the start of a clip
and propagate to the end. There is no re-detection, no re-prompt, no recovery
of a lost object, no conflict handling.

Why it never shows: their clips are eight seconds and five seconds.

    cell 18   ...game-1-q1-04.28-04.20.mp4    game clock 4:28 -> 4:20
    cell 97   ...game-1-q1-03.16-03.11.mp4    3:16 -> 3:11

The saved progress bars confirm it -- 237 frames, 7.9s at 30fps. The merge
measured here takes twelve seconds to develop (IoU 0.06 at 11.0s, 0.97 at
12.5s). An eight-second clip does not give it time to happen. The tutorial's
result is not evidence the problem is solved; it is evidence the problem is
out of frame.

Two more things the list gets wrong about the notebook:

**ResNet-32 is not in it.** Zero matches for `resnet` across code and
markdown. The only number model is `NUMBER_RECOGNITION_MODEL_ID =
"basketball-jersey-numbers-ocr/3"`. The 93%-beats-SmolVLM2 claim has no
implementation to read.

**The roster is a dict lookup**, not an assignment:

    f"#{number} {TEAM_ROSTERS[TEAM_NAMES[team]].get(number)}"

No de-duplication and no conflict resolution, so two tracks reading the same
number get the same name drawn twice, and a misread that lands on a bench
player's number gets that player's name. Those are the two holes this repo
added gates for; the notebook does not detect them, let alone fix them.

The conclusion for tracklet association: there is nothing to copy. It is work
beyond the baseline, not a step of it we skipped.

## Tracklet association by jersey number (2026-08-13)

Re-seeding gave one player several ids and the roster matching turned that into
an invented name. The association step closes it: two tracks that never coexist
and read the same number are one player, merged before the matching runs. This
is what GTA and SportsSUSHI do with ReID embeddings; a jersey number is the
stronger feature of the two, because teammates in one kit look alike to a ReID
model and never share a number.

Three conditions, all necessary. Same top number on at least `--min-votes`
each, the evidence. Lifetimes that do not overlap -- two ids on the floor at
once reading one number is the duplicate fault, which has its own fix. And not
two decisively opposed team clusters, since Cunningham is Pistons 2 and
McBride is Knicks 2; a cluster vote that splits 11-10 has no opinion and
abstains rather than blocks.

Measured on identical re-seeded tracks for seg_01m10.87s_19s, so the only
variable is the association step:

| | prompt once | re-seed 3s | re-seed 3s + merge |
|---|---|---|---|
| names | 10, all correct | 11, one invented | **10, all correct** |
| Harris | one track | split 5 and 12 | **12 merged back into 5, 40 votes** |
| Beasley | undrawn from 11.9s | back at 15.0s | **back at 15.0s** |
| Brunson | undrawn from 11.9s | undrawn from 11.9s | undrawn from 11.9s |

`#7 Paul Reed` -- a man who did not play -- was blocked twice over, by
different gates, because it arose two different ways. Without merging it went
to track 5 on four reads of `7`, which is a *confirmed* assignment; merging
gives track 5 its real number and the fabrication moves to track 11 on a
*single* read of `7`, which the maximum-weight matching hands out because
something is always left over. So a second gate: an assignment below
`--min-votes` keeps its number for the review queue and gets no name, and is
marked `ignored` so the render draws neither. A number on one read is a claim
like any other, and `#7` on Brunson is wrong whether or not a name sits beside
it.

Brunson is the case association cannot reach, and the reason is exact. His
recovered track was born at 9.0s, while his original was still alive until
11.9s, so the two overlap and merging them would be the duplicate fault. Track
11 is two people -- a man behind the basket until the 15.0s re-seed, Brunson
after it -- which is also why its team vote split 11-10. Splitting it at the
re-seed boundaries would leave a tail that is disjoint from track 8 and reads
the same number, and Brunson would keep his name through the clip. That is
GTA's other half, the splitter, and the boundaries are already recorded in the
tracks sidecar as `reseeds`.

Caveat on the record: the `ignored: "tentative"` flag is committed but its
verification run was stopped before it wrote, so it is unverified end to end.
Everything above it in this section was measured.

## The first measurement reverses the choice (2026-08-14)

`pipeline/score.py` replays render_final.py's drawing rule against a per-track
ground truth and reports precision (labels drawn on the right man) and
coverage (of the ten on court, how many carry a correct label). The 19s
segment, both routes:

| | precision | coverage | right | wrong | unknown |
|---|---|---|---|---|---|
| SAM2, prompt once | 95.5% | 74.2% | 9706 | 462 | 480 |
| SAM3 + on-court filter + merge | **100.0%** | **81.7%** | 10574 | **0** | 180 |

Every wrong frame in the SAM2 run is one track: number 5 is Harris for four
seconds and Towns after, and the render calls it Harris throughout, so two
fifths of that clip carries Harris's name on Towns. It survived every frame
cropped by hand over two days, including the check that answered "are Towns
and Harris both labelled at 15s?" -- they were, and one of the labels was on
the wrong man. Counting labels cannot see that; only checking them can.

Worth recording because it was the wrong call, made twice: before the metric
existed, SAM2 looked better on "share of frames showing all ten labels", 62%
against 50%. That proxy rewards drawing a label whether or not it is right,
and it inverted the answer. The eleven-label frame it also failed to flag was
found by hand; the 462 wrong frames were not.

Caveats on the sample. The two truths are separate files because track ids
differ between trackers, and the SAM3 one covers only the fifteen tracks that
run actually draws -- enough for both metrics, since a track with no label
changes neither. Both were read off crop strips by eye; unreadable stretches
are marked unknown and skipped rather than guessed, which is where the 480 and
180 come from.

## A segment that measures 100% (2026-08-14)

`seg_02m28.00s_13s` -- the re-cut that put Cunningham back on the floor --
scores perfectly:

| segment | precision | coverage | right | wrong | unknown |
|---|---|---|---|---|---|
| **seg_02m28.00s_13s** | **100.0%** | **100.0%** | 7500 | 0 | 0 |
| seg_01m10.87s_19s, SAM3 route | 100.0% | 91.7% | 10574 | 0 | 180 |
| seg_01m10.87s_19s, SAM2 route | 95.5% | 84.2% | 9706 | 462 | 480 |

Ten players, 750 frames, 7500 labels, every one on the right man and none
missing. All ten of its tracks hold a single identity for the whole clip --
no switch, no collapse, nothing for the gates to catch.

Found while fixing the scorer, which had been keying identity on the jersey
number alone. #8 is Anunoby for the Knicks and Hardaway Jr. for the Pistons,
so the two of them counted as one man and the 13s segment reported 90%
coverage while labelling all ten correctly on every frame. Identity is (club,
number); the truth files now record the kit colour and the scorer checks both.
The 19s figures moved with it, 74.2% to 84.2% and 81.7% to 91.7%.

What separates this clip from the 19s one is not the pipeline, it is the
footage: no two players make sustained contact, so no track ever has to be
re-acquired. The 19s segment loses Brunson for 2.6s to an occlusion where the
detector returns nine boxes for ten men -- nothing downstream can recover a
player who has no box.
