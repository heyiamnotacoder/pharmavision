# PharmaVision

Open-source AI lab for behavioural pharmacology: rodent test video in → scored, statistically
analysed, ARRIVE-compliant results section out.

**SAM 3.1 sees, Python measures, Muse Spark judges.** See `docs/ARCHITECTURE.md`.

Research tool only; not validated for regulatory GLP studies.

## Develop

```bash
uv sync
scripts/check.sh
```

## Licence and credits

- Code: MIT (see `LICENSE`).
- Validation videos: von Ziegler, Sturman & Bohacek, open field / plus maze / forced swim
  recordings, Zenodo record [3608658](https://zenodo.org/records/3608658), CC BY 4.0
  (Sturman et al. 2020, *Neuropsychopharmacology*).
- Human behaviour labels: [ETHZ-INS/DLCAnalyzer](https://github.com/ETHZ-INS/DLCAnalyzer),
  GPL-3.0. Downloaded at eval time into `eval/data/` and never redistributed here.
- Segmentation and tracking: Meta's SAM 3.1, used under the SAM License, which asks that
  publications acknowledge SAM.
