---
status: accepted
---

# No rear counts: the measured score is evidence per candidate

ADR 0002 moved the rear count from Muse Spark to a measured scorer. The scorer was built (measure
settings v4) and checked on the 14 eval videos on Oct 31 (ARCHITECTURE open decision 3). It
judges single candidates about as well on unseen videos as on the train split. Its per-video
counts do not agree with the raters:

| | Train, leave-one-video-out | Eval split (14 videos) |
|---|---|---|
| Per candidate | AUC 0.75, κ 0.37, precision 0.74, recall 0.67 | AUC 0.71, κ 0.29, precision 0.66, recall 0.74 |
| Per-video count against the raters (Pearson r) | 0.53 | **0.07** (raters against each other: 0.90) |
| The number of candidates alone | 0.35 | 0.41 |
| The sum of scores | 0.64 (in-sample) | 0.22 |

The errors are video-specific: some animals make many compact, still postures that look like
rears in the mask (OFT_23: 149 counted, the raters about 47). Muse Spark, the other judge, was
at κ 0.04–0.30 per clip and changed 1 label in 4 on a re-ask.

We decided (Oct 31):

1. **The pipeline makes no rear count.** `Metrics` has no rear count, and the report abstains on
   rearing counts and says why, citing the eval numbers. This replaces ADR 0002's first point.
2. **The score stays as evidence.** Each rear candidate keeps its features and its score
   (`MeasureSettings.rear_scorer`), and Muse Spark's label and rationale stay on its event as a
   second opinion. Both are shown per window, never summed into a count.
3. **FR-6 is reported as not met**, with the numbers above, in the eval report. The harness
   keeps measuring it, so a better signal can be judged against the same bar.

## Considered options

| Option | Why not chosen |
|---|---|
| **Score as evidence, no count (chosen)** | Honest under cite-or-abstain, fits the build window (Nov 4) |
| Report the thresholded count with a caveat | A number we know is wrong would pass the number audit and reach the results section |
| Spike a new signal first (e.g. a SAM 3.1 concept prompt for a rearing mouse) | About $1.80 per video and hours of work with no evidence yet that it helps; it can still follow, judged by the same harness |

## Consequences

- The scorer has no threshold: a count threshold (0.55) exists only in the spike, to measure
  count agreement.
- The rear-candidate stage keeps its recall-first settings (ADR 0001): the windows remain the
  places worth showing a reviewer.
- A lab using PharmaVision sees each candidate window with its score and Muse Spark's opinion,
  and counts rears by hand if it needs the number.
