# Labelling jersey numbers: the rules we label by

One class, `number`. This is the standard every labelling session has to hold
to, because with a few hundred frames consistency is the only thing we have.
Written while labelling the NYK @ DET harvest; it applies unchanged to any
future footage, Big V included.

## The one rule

**Box it if you can see that the region *is* a number. You do not have to be
able to read which number it is.**

What we are training is region detection, single class. Reading the digits is
a separate job that happens downstream in `identify.py`, on the crop this
model hands it. Labelling only the numbers you can personally read teaches
the model to skip exactly the small, distant, half-turned regions that make
up most of a broadcast frame — and recall is what is short already. On the
first harvested frame the detector found 3 number regions with 10 players on
court.

Everything below follows from that rule.

| | |
|---|---|
| Back of the jersey | **Box it.** NBA kits carry a number front and back. |
| Turned side-on, digits skewed | **Box it.** |
| Half occluded by an arm or a defender, the rest still reads as a number | **Box the visible part.** |
| Blurred to a colour smear — number, logo or crease is a coin flip | **Leave it.** A wrong box teaches the model that fabric texture is a number. |
| Crowd, bench in warm-up shirts, fans in replica jerseys | **Leave it.** The stands are half the frame; a single box there is worth many false positives at inference. |
| Sponsor patch, team wordmark, player name | **Leave it.** Not a number. |
| Referee numbers | **Leave it.** Not a player, and the roster match downstream has nothing to do with it. |

Box the digits only — not the jersey around them, not the name above them.

Frames that arrive with zero boxes are deliberate: they are the hard
negatives `harvest_numbers.py` keeps, one in every twelve empty frames. Leave
them empty.

## Where the frames come from

Pseudo-labels, not blank canvases. `harvest_numbers.py` runs the existing
ten-class detector (`basketball-player-detection-3-ycjdo/4`) at confidence
0.10 and keeps its `number` guesses, so the session is confirm-and-fix.

They are guesses, and the low-confidence ones are guesses in the literal
sense — 29% of the boxes in the NYK @ DET harvest sit below 0.5. Seen in the
rendered frames: a box on a StockX sponsor patch while the number filling
half the same frame went unboxed; a box on a player's ear; a box on a fan in
the stands. Confirm every one.

`filter_harvest.py` then drops the close-ups. A broadcast cuts constantly
between the wide shot that shows a lineup and reaction shots, and the two are
different problems:

| | frames | boxes | median box height |
|---|---|---|---|
| wide game shots | 409 | 2405 | 24 px |
| close-ups, bench, crowd cutaways | 162 | 277 | 41 px |

Only the wide shot matches what `identify.py` reads downstream, the scales
differ by roughly 5x, and the close-ups are where the pseudo-boxes are worst
— the most expensive frames to label and the least useful once labelled.

## Roboflow settings, and why each one matters

Three defaults will quietly damage the dataset. All three have been hit.

**Split: "Add All Images to Training Set".** The default splits
train/valid/test randomly. Our frames are sampled every 8 seconds from one
game, so a random split lands near-duplicate frames on both sides of the
train/test line and reports a score that is partly memorisation. The split is
done downstream by `train_rfdetr_numbers.py --prepare`, in game-time order:
the last 20% of the game is the test set, minutes the model has never seen.

**Preprocessing: auto-orient only. Remove the resize.** The default stretches
to 512×512 — non-uniform, from 1920×1080. That distorts the numbers relative
to what `identify.py` feeds at inference, and it drops a 24 px number region
to about 11 px vertically. Let the local rfdetr handle resizing, so training
and inference go through the same path.

**Augmentation: none.** Generated variants of one frame scatter across
splits, which is the random-split leak again wearing a different hat. If
augmentation is wanted it belongs in training, where it is not baked into the
data.

**Visibility: private.** It is broadcast footage.

A version is a moment-in-time snapshot — label first, generate the version
after. Labels edited later do not flow back into a version already created.

## Exporting back

Export COCO, drop it in as the `--src` of the training script, and let the
time-based split happen locally:

```
python pipeline/train_rfdetr_numbers.py --prepare  --src out/harvest_labelled --dataset out/<name>_ds
python pipeline/train_rfdetr_numbers.py --baseline --dataset out/<name>_ds
python pipeline/train_rfdetr_numbers.py --train    --dataset out/<name>_ds
python pipeline/train_rfdetr_numbers.py --eval     --dataset out/<name>_ds
```

`--baseline` scores the current detector against the labelled test split and
has to run *before* `--train`. Afterwards the weights are gone and there is
nothing left to compare the fine-tune against.
