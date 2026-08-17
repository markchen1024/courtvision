# finals/ — the shipped renders, nothing else

`out/` is the workbench: twenty-odd `*_final.mp4` files live there because
every pipeline run produces one, most of them superseded the same day. The
renders that actually shipped kept getting lost in that pile, so they are
promoted here instead.

A file earns this directory by clearing all three bars:

1. **Measured** — scored by `pipeline/score.py` against a hand ground truth
   in `eval/`, at 100% label precision.
2. **Watched** — a person eyeballed the full render, not just the numbers.
3. **Shipped** — it is on the site (homepage clip or reel) or in the reel
   master that built it.

Current contents:

| file | length | coverage |
|---|---|---|
| seg43c_final.mp4 | 42.6 s | 98.8% — the homepage possession |
| seg_g6149_36s_final.mp4 | 35.5 s | 96.1% |
| seg19_sam3_final.mp4 | 19.2 s | 91.7% |
| seg_02m28.00s_13s_final.mp4 | 12.5 s | 100.0% |
| seg_02m44.15s_10s_final_sound.mp4 | 10.0 s | 99.8% |
| reel_final.mp4 | 2 m 07 s | all five, with title cards (CRF 18 master) |

All at 100% precision; every number from `eval/`. The web encodes derived
from these live in `web/media/` (the sources of truth the site serves) and
are not duplicated here. Videos in this directory stay out of git like all
footage in this repo — only this README is tracked.
