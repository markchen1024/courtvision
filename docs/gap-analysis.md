# The pipeline has no gates

Every stage hands its output to the next one unconditionally. Not one stage
stops because its own result is unusable. So a run that was already doomed at
minute two keeps consuming the GPU for another hour, and the failure is only
visible in the final render — at which point the only diagnosis available is
"something upstream was wrong".

That is the expensive defect. Not the accuracy of any single model.

## What it cost, measured

Four attempts at this clip. Each one was decided at stage 2 and discovered at
stage 6.

| Attempt | Frame-0 prompts | Detectable at | Discovered at | Wasted |
|---|---|---|---|---|
| 30:06 | 9 (of 10 players) | **2 min** | render, ~58 min | ~56 min |
| 30:16 | 9 | **2 min** | render, ~58 min | ~56 min |
| 30:05.5, ours | 9 | **2 min** | render, ~58 min | ~56 min |
| 30:05.5, tutorial | 11 | **2 min** | — (best run) | — |

Roughly **three hours of GPU spent on runs whose ceiling was already fixed
before SAM2 started**. The check that would have caught all three is counting
the players on frame 0 — two minutes, and we had the number every time without
looking at it.

And the gate would have caught more than a short lineup. Re-running the count
afterwards with the tutorial's detector, which is what should have been
prompting SAM2 all along:

| Segment | correct detector | what we ran with |
|---|---|---|
| 30:05.5 | **10, passes** | 9 |
| 30:06 | **10, passes** | 9 |
| 30:16 | 9, fails | 9 |
| 30:05 | **0, fails** — no players at all on frame 0 | — |

So 30:06, the attempt that identified only three Pistons, had a full lineup
available. It was never a bad segment. It was the wrong detector
(`koppolusameer/rfdetr-...` out of `project.py` instead of
`basketball-player-detection-3-ycjdo/4`), and three hours were spent looking
for the fault in the footage.

That is the real argument for the gate. Its value is not that it picks a
better segment — it is that "9 where 10 are standing on court" is a question
worth asking on minute two, and asking it leads to the detector. Without it,
the same number scrolled past four times and was read as a property of the
footage.

The same shape repeated inside a run: a track carrying a number no roster
contains still gets its mask cut, its colour chosen, and its label drawn.

## What each stage should refuse to pass on

Gate = a condition that must hold for the stage's output to be worth the next
stage's compute. "Have" means it exists today.

| # | Stage | Gate it should enforce | Have | Cost of not having it |
|---|---|---|---|---|
| 1 | Segment selection | no cut inside the window | **yes** (`detect_cuts.py`, verified by eye) | — |
| 2 | **Detection, frame 0** | **players detected ≥ players on court; report which are missing** | **no** | the whole run's identity set is fixed here. 3 × ~56 min. |
| 3 | Prompt construction | no two prompts on one player; no non-player prompts | **partly** — NMS added; the net was still prompted as a player at 0.83 | a wasted slot, and a phantom track through the render |
| 4 | Tracking | no two live tracks on one player | **no** | Duren and Towns each rendered twice; nothing noticed |
| 5 | Team clustering | cluster→club mapping must win by a margin | **no** | 8–8 tie flips every colour and every name, silently and confidently |
| 6 | Number OCR | drop reads from frames where the number is not legible | **no** | blurred frames vote as loudly as clear ones; Hart's 3 → 9 in every version |
| 7 | Read aggregation | confirm only on enough votes and a clear winner | **yes** (majority: ≥2 and a strict winner) | — |
| 8 | Roster lookup | never invent a name for a number no roster has | **yes** — renders a bare `#9` | — |
| 9 | Court projection | ≥4 landmarks or skip the frame | **yes** | — |
| 10 | Render | do not draw a label the pipeline does not stand behind | **no** | wrong numbers reach the screen with full confidence |

Five of ten stages will pass anything.

## The pattern

The gates that exist are all at the **end** of the pipeline (7, 8, 9) — the
places where a bad result is cheap, because there is nothing expensive left to
run. The stages with no gate are at the **front** (2, 3, 4), where a bad result
costs an hour.

That is exactly backwards. A gate is worth the most where the compute
downstream of it is the most expensive.

## What this changes about priorities

The gap table from the previous version of this file listed model accuracies.
Those were the wrong thing to lead with. In order:

1. **Stage 2 gate — count the lineup on frame 0 before anything else runs.**
   Two minutes of detection, and it decides the ceiling of the entire run.
   Should print which players are present and refuse to continue when the
   lineup is short unless explicitly overridden.
2. **Stage 4 gate — refuse duplicate identities.** Two tracks confirmed to the
   same number is a detectable, hard error, not a judgement call.
3. **Stage 5 gate — refuse a cluster→club mapping without a margin.** This one
   does not degrade, it inverts. Every name and colour on screen depends on it.
4. **Stage 10 gate — do not render what is not confirmed.** A bare number is
   honest; a wrong name is worse than no name.
5. **Stage 6 legibility filter.** The published state of the art (Koshkina et
   al., CVPRW 2024) classifies whether a number is readable *before* voting.
   This is the only item on the list that needs a new model.

Items 1 through 4 are conditions on data that is already computed. None of
them requires a model, more footage, or more GPU. They are cheap, and their
absence is what made the last three hours expensive.

## What is already right

Worth stating so it does not get rebuilt: the aggregation refuses to confirm
on a single read, the roster lookup refuses to invent a name, the projection
skips frames it cannot solve, and `detect_cuts.py` verifies the segment before
anything runs. Those four are the pattern the front of the pipeline is
missing.
