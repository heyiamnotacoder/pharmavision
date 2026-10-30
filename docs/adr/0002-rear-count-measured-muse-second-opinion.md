---
status: accepted; point 1 superseded by ADR 0003 (no rear counts)
---

# Rear counts come from a measured scorer; Muse Spark is a second opinion

Muse Spark was to classify each rear-candidate clip (FR-6, and the application's "Muse Spark
classifies rearing"). Measured on Oct 30 (ARCHITECTURE open decision 3), its labels are not fit
to count rears. Zero-shot, it is near chance on hard negatives (κ 0.04 on 80 held-out train
clips) and over-calls rears (precision 6/16 on the smoke clip). Few-shot, with 4 examples (the
API limit is 5 videos per request), it reaches κ 0.30 on one ask and 0.13 on the next, and does
worse on the smoke clip (3/10). A re-ask changes about 1 label in 4. The mask features the
measure stage already computes separate better: a logistic fit gives κ 0.37 and precision 0.71,
leave-one-video-out on the train split.

We decided (Oct 30):

1. **The rear count comes from a measured rear scorer** in the measures layer: a deterministic
   score per rear candidate from its mask features, with coefficients fitted on the train split
   and versioned in the settings. A candidate counts as a rear when its score clears the
   threshold.
2. **Muse Spark stays as a second opinion.** Stage 5 still judges every candidate clip. Its label
   and rationale stay on the event, are shown in the UI, and the eval reports how often Muse
   Spark agrees with the raters and with the scorer. Its label never sets the count.
3. **Few-shot stays off by default.** The committed set (`judge/few_shot/`) remains available.
4. **Track the 14 eval-split videos now** (SAM 3.1, about $25 of the $150 budget), so the scorer
   is checked on the eval split in its own PR. The eval harness (Nov 2) reuses the cached tracks.

## Considered options

| Muse Spark's role for rearing | Why not chosen |
|---|---|
| **Second opinion, measured count (chosen)** | Keeps the application's claim honest: Muse Spark's agreement is measured and reported, not assumed |
| Asked only when the score is uncertain | Muse Spark is at chance on exactly those hard cases |
| Dropped for rearing | Simplest and cheapest, but drops an application promise without measuring it in the eval |

## Consequences

- FR-6 changes from "Muse Spark classifies clips" to "a measured scorer counts rears; Muse Spark
  gives a second opinion". The acceptance criterion is unchanged: per-video count agreement with
  the raters, against agreement between raters.
- The scorer's coefficients are numbers fitted on the GPL-3.0 rater labels. The label file and
  its rows are still never committed.
- Until the scorer lands, the report abstains on rear counts.
- Muse Spark spend on rearing stays at about $3 per video (every candidate is still judged).
