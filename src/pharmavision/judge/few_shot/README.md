# Few-shot clips

Four 2–4 s clips for `MuseSparkSettings.few_shot`, cut by `python -m spike.muse_few_shot build`
from the train-split videos the way stage 5 cuts clips (square crop, tracked mask outlined).
`examples.json` gives each clip's video and time window. The set is off by default
(ARCHITECTURE open decision 3).

- **Source videos:** Sturman et al. 2020, open field recordings, Zenodo record 3608658,
  licensed CC BY 4.0. The clips are cropped, re-encoded excerpts with the tracked outline drawn.
- **Labels and rationales are ours.** Each clip was picked where all three human raters agreed
  (or none marked a rear within 1 s), then checked by eye; the rationales describe what the clip
  shows. No rows of the GPL-3.0 DLCAnalyzer label file are included.
