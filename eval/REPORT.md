# PharmaVision eval report

Made by `python -m eval report` from stored results: no model is called, so it costs nothing to make again. Sturman et al. 2020 open-field videos (Zenodo 3608658, CC BY 4.0), the 14 eval videos of the fixed split (`eval/split.json`, seed 20261029); human labels from DLCAnalyzer (GPL-3.0, not in this repo). Measure settings v4.

## Validation table

| Measure | vs Furkan | vs Jin | vs Oliver | Raters against each other | Bar | Met |
|---|---|---|---|---|---|---|
| Rear count, ICC(2,1): candidates scoring ≥ 0.55 | 0.02 | 0.09 | 0.04 | 0.87 to 0.91 | within the raters' range | no |
| Rear events, F1 (±0.5 s): candidates scoring ≥ 0.55 | 0.60 | 0.61 | 0.62 | 0.87 to 0.89 | within the raters' range | no |
| Rear count, ICC(2,1): every candidate | 0.02 | 0.02 | 0.01 | 0.87 to 0.91 | report only | — |
| Rear events, F1 (±0.5 s): every candidate | 0.55 | 0.56 | 0.56 | 0.87 to 0.89 | report only | — |

Over 14 eval videos. The pipeline makes no rear count (ADR 0003): the eval counts the candidates scoring at least 0.55 (a threshold set on the train split's counts, in the eval only) to measure how far such a count would be from the raters, and every candidate beside it. A window and a rear match when they overlap or the gap between them is at most 0.5 s (not their onsets: a long window can match a rear anywhere in it), each in one pair at most, pairing as many as possible.

| Report | Pass rate | Bar | Met |
|---|---|---|---|
| Number-audit pass rate | 0 of 0 study reports (—) | 100% | — |

## Rearing per video

| Video | Scoring ≥ 0.55 | Candidates | Furkan | Jin | Oliver |
|---|---|---|---|---|---|
| OFT_11 | 66 | 177 | 74 | 72 | 63 |
| OFT_12 | 140 | 196 | 98 | 102 | 105 |
| OFT_14 | 113 | 188 | 106 | 100 | 101 |
| OFT_15 | 85 | 173 | 88 | 79 | 93 |
| OFT_23 | 149 | 184 | 51 | 43 | 46 |
| OFT_24 | 86 | 188 | 117 | 104 | 130 |
| OFT_39 | 129 | 197 | 102 | 93 | 102 |
| OFT_41 | 120 | 181 | 91 | 101 | 83 |
| OFT_43 | 137 | 169 | 69 | 77 | 85 |
| OFT_49 | 67 | 182 | 66 | 65 | 68 |
| OFT_5 | 88 | 173 | 91 | 86 | 93 |
| OFT_51 | 82 | 173 | 58 | 58 | 57 |
| OFT_52 | 120 | 191 | 92 | 110 | 102 |
| OFT_6 | 119 | 196 | 80 | 75 | 65 |

## Rear events against each rater

| Windows vs rears | Matched | Windows | Rears | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| scoring ≥ 0.55 vs Furkan | 806 | 1501 | 1183 | 0.54 | 0.68 | 0.60 |
| scoring ≥ 0.55 vs Jin | 807 | 1501 | 1165 | 0.54 | 0.69 | 0.61 |
| scoring ≥ 0.55 vs Oliver | 830 | 1501 | 1193 | 0.55 | 0.70 | 0.62 |
| every candidate vs Furkan | 1036 | 2568 | 1183 | 0.40 | 0.88 | 0.55 |
| every candidate vs Jin | 1040 | 2568 | 1165 | 0.40 | 0.89 | 0.56 |
| every candidate vs Oliver | 1061 | 2568 | 1193 | 0.41 | 0.89 | 0.56 |
| Furkan vs Jin | 1020 | 1183 | 1165 | 0.86 | 0.88 | 0.87 |
| Furkan vs Oliver | 1051 | 1183 | 1193 | 0.89 | 0.88 | 0.88 |
| Jin vs Oliver | 1049 | 1165 | 1193 | 0.90 | 0.88 | 0.89 |

Between two raters, the first rater's rears are the windows.

## Number audit

No study report found: the pass rate abstains.

## Tracking against DeepLabCut

| Video | Tracked % | Frames compared | Median cm | p95 cm | ≥ 5 cm % | Best lag (frames) |
|---|---|---|---|---|---|---|
| OFT_11 | 98.9 | 9017 | 1.72 | 2.95 | 0.0 | -2 |
| OFT_12 | 99.8 | 9086 | 1.42 | 2.48 | 0.1 | -1 |
| OFT_14 | 99.3 | 9050 | 1.53 | 2.69 | 0.2 | -2 |
| OFT_15 | 98.9 | 9042 | 1.65 | 2.86 | 0.0 | -2 |
| OFT_23 | 99.1 | 9027 | 1.31 | 2.37 | 0.1 | -1 |
| OFT_24 | 98.1 | 8922 | 1.64 | 2.77 | 0.1 | -2 |
| OFT_39 | 99.4 | 9032 | 1.62 | 2.77 | 0.2 | -2 |
| OFT_41 | 99.4 | 9038 | 1.62 | 2.58 | 0.0 | -2 |
| OFT_43 | 98.9 | 9005 | 1.12 | 2.72 | 0.0 | -1 |
| OFT_49 | 99.0 | 9012 | 1.48 | 2.66 | 0.1 | -2 |
| OFT_5 | 98.6 | 8991 | 1.71 | 2.80 | 0.1 | -2 |
| OFT_51 | 98.4 | 8977 | 1.71 | 2.65 | 0.1 | -2 |
| OFT_52 | 99.2 | 9029 | 1.66 | 2.59 | 0.1 | -2 |
| OFT_6 | 98.1 | 8945 | 1.46 | 2.55 | 0.1 | -2 |

DeepLabCut is a reference tracker, not ground truth: no pass bar. Distance is from our mask centroid to DLC's body centre.

Model API spend logged for these videos: $0.00.

## Not measured here

- **Supported vs unsupported rears:** the label set is binary (rearing / none), since Muse Spark could not tell them apart (ARCHITECTURE, open decision 3).
- **Muse Spark's rear labels against the raters and the score** on the eval split, which ADR 0003 (point 3) asks the harness to keep measuring: it needs the behaviours stage run on the eval videos (new Muse Spark spend), so it is not in this report yet. Its train-split agreement is in ARCHITECTURE, open decision 3.
