---
status: accepted
---

# Rear candidates favour recall over clip count

Stage 5 can only label a rear that some candidate window's clip shows, so candidate recall caps
rearing recall. Measured on the 6 Sturman train videos (Oct 30), the v2 defaults found only 0.59
of the 503 consensus rears. We chose measure settings v3 (area drop 0.03, elongation change 0.05,
15 cm/s, min 0.1 s): **consensus recall 0.91**, with about 180 windows over 41% of the video time,
about $2.90 of Muse Spark per video (about $58 for the 20-video eval).

## Considered options

| Setting | Consensus recall | Time | Muse Spark / video |
|---|---|---|---|
| v2 defaults (0.1, 0.1, 10 cm/s, 0.2 s) | 0.59 | 14.7% | $1.70 |
| 0.05, 0.05, 15 cm/s, 0.1 s | 0.85 | 33.4% | $2.80 |
| **0.03, 0.05, 15 cm/s, 0.1 s (v3)** | **0.91** | **40.7%** | **$2.90** |

The 0.85 option saves only about $0.10 per video. The full table and the method are in
ARCHITECTURE open decision 2.

## Consequences

- The candidates miss the "well under half the time" goal: they are a coarse filter, and stage 5
  carries the precision (45–51% of windows overlap a consensus rear).
- The area drop is the limit, not the thresholds: 22% of consensus rears never shrink the mask by
  10%, and at a 3% drop the windows cover 41% of the time. Better recall at less time needs a different
  signal, not further tuning.
