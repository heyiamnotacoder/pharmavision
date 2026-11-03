# PharmaVision

Open-source AI lab for behavioural pharmacology: rodent test video in → scored, statistically
analysed, ARRIVE-compliant results section out.

**SAM 3.1 sees, Python measures, Muse Spark judges.** No model writes a number into the report:
every number in the generated prose must match a computed value, or the report fails.

> **Research tool only.** Not validated for regulatory GLP studies. Rearing is not counted
> (see [Limitations](#limitations)).

Built for the Meta Global AI Developer Hackathon (Indie track, "Push the Frontier"), Oct 26 –
Nov 4, 2026.

## How it works

```
video ─▶ 1 ingest ─▶ 2 arena ─▶ 3 track ─▶ 4 measure ─▶ 5 behaviours ─▶ 6 QC ─▶ 7 stats ─▶ 8 report
                      (sees)     (sees)    (measures)     (judges)      (both)  (measures)  (judges)
                         ▲                                                                    │
                   human confirms zones                                          number audit ┘
```

| Layer | Who | Does | Never does |
|---|---|---|---|
| **Sees** | SAM 3.1 | Segments the arena floor from a text prompt; tracks the animal in every frame from the prompt "mouse", with no training data; looks for hands in flagged windows | Produce a number for the report |
| **Measures** | Deterministic Python | Every metric (distance, velocity, centre time and entries, thigmotaxis, immobility), every QC rule, every statistic, with the test chosen by a fixed rule | Call a model |
| **Judges** | Muse Spark | Gives a second opinion on each rear-candidate clip, reviews flagged QC windows, writes the ARRIVE 2.0 methods and results through read-only tools | Compute anything |

- **The arena gate.** The scientist confirms or redraws the floor in the zone editor. Nothing
  downstream runs on an unconfirmed arena.
- **The number audit.** Every number in the report agent's prose is checked against the tables
  its tools returned. An unmatched number sends the draft back with the offending sentences;
  after a second failure those sentences ship struck out and flagged.
- **Cite or abstain.** What the system can't know (housing, randomisation, strain) comes out as
  `[TO FILL: …]`, never invented. Where the data can't support a claim, the report says so.
- **Auditable.** Every stage writes typed artifacts (Pydantic) to a run folder. Every exclusion
  has a frame range and a reason, and every test records why it was chosen. Model responses are
  cached by input hash, so a re-run is free and gives the same result.

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Scope and acceptance criteria:
[`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md). Decisions: [`docs/adr/`](docs/adr/).

## Validation

On the Sturman et al. 2020 open field videos: a fixed split of 6 train and 14 eval videos
(`eval/split.json`), three human raters, and DeepLabCut output as a reference tracker. Full
tables, per video: [`eval/REPORT.md`](eval/REPORT.md), made by `python -m eval report`.

| What | Result on the 14 eval videos | Reference | Verdict |
|---|---|---|---|
| Frames tracked (SAM 3.1, zero-shot) | 98.1–99.8% per video | target ≥ 95% | **met** |
| Centroid vs DeepLabCut body centre | median 1.1–1.7 cm, p95 2.4–3.0 cm, ≤ 0.2% of frames ≥ 5 cm apart | DeepLabCut is a reference, not ground truth | reported |
| Rear count agreement, ICC(2,1) | 0.02–0.09 against each rater | raters against each other: 0.87–0.91 | **not met** |
| Rear events, F1 (± 0.5 s) | 0.60–0.62 against each rater | raters against each other: 0.87–0.89 | **not met** |
| Rear candidate recall (every candidate) | 0.88–0.89 against each rater | — | the windows catch most rears |

Rearing failed its bar, so **the pipeline makes no rear count**
([ADR 0003](docs/adr/0003-no-rear-counts-score-as-evidence.md)). Each candidate window keeps its
measured score and Muse Spark's opinion, shown per window, for a person to check.

## Limitations

- **No rear count**, as above. The report says it abstains on rearing and why.
- **Muse Spark's rear labels are not trusted.** Per clip it agreed with the raters at κ
  0.04–0.30 and changed 1 label in 4 when asked again. It is a second opinion beside the score.
- **Open field only.** Elevated plus maze, forced swim and two-animal tests are not built.
- **One animal, a fixed overhead camera, a square arena of known size.**
- **QC window exclusions are listed but don't change the metrics yet**: the metrics are computed
  before QC. A whole-trial exclusion does leave the trial out of the stats.
- **The overlay shows the raw track**, not the filtered path the metrics come from, so it can
  show jitter the metrics never saw.
- **New videos need the Meta Model API** (`MODEL_API_KEY`). The validation used the Contributor
  tier on public data, where Meta may train on prompts and outputs; a lab's own videos should use
  the standard tier, the default.

## Run it

Needs [uv](https://docs.astral.sh/uv/), Node 22 and ffmpeg.

```bash
uv sync
scripts/check.sh            # ruff, pyright, pytest, and the frontend's typecheck, lint, tests and build
```

The API, with any finished run folders served read-only (needs `MODEL_API_KEY` in `.env` for
anything not already in `runs/model_cache`):

```bash
uv run python -m pharmavision serve --precomputed path/to/finished-runs
```

The frontend beside it (Vite on `localhost:5173`, which proxies `/api` to the API on `:8000`):

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

One video from the command line, without the UI:

```bash
uv run python -m pharmavision start VIDEO.mp4     # ingest + arena; stops at the arena gate
uv run python -m pharmavision finish RUN_DIR      # confirm the arena, then every later stage
```

The eval harness (`fetch` downloads the Sturman data into the git-ignored `eval/data/`):

```bash
uv run python -m eval fetch
uv run python -m eval track
uv run python -m eval report
```

## Repository

| Path | What |
|---|---|
| `src/pharmavision/stages/` | One module per pipeline stage |
| `src/pharmavision/measure/`, `stats/`, `audit/` | The deterministic core: metrics, QC rules, statistics, the number audit (pyright strict, no I/O) |
| `src/pharmavision/segmenter/`, `judge/` | The SAM 3.1 and Muse Spark boundaries |
| `src/pharmavision/api/` | FastAPI: runs, background jobs, the arena gate, study reports |
| `frontend/` | React + TypeScript: run viewer with overlay, zone editor, timeline, QC list, report view |
| `eval/` | The validation harness and `REPORT.md` |
| `spike/` | The Oct 29–31 measurement scripts behind the decisions in `docs/` |

## Licence and credits

- Code: MIT (see `LICENSE`).
- Validation videos: von Ziegler, Sturman & Bohacek, open field / plus maze / forced swim
  recordings, Zenodo record [3608658](https://zenodo.org/records/3608658), CC BY 4.0
  (Sturman et al. 2020, *Neuropsychopharmacology*).
- Human behaviour labels: [ETHZ-INS/DLCAnalyzer](https://github.com/ETHZ-INS/DLCAnalyzer),
  GPL-3.0. Downloaded at eval time into `eval/data/` and never redistributed here.
- Segmentation and tracking: Meta's SAM 3.1, used under the SAM License, which asks that
  publications acknowledge SAM.
- Clip review, QC review and report writing: Meta's Muse Spark, through the Meta Model API.
