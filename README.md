# PharmaVision

Open-source AI lab for behavioural pharmacology: rodent test video in → scored, statistically
analysed, ARRIVE-compliant results section out.

**SAM 3.1 sees, Python measures, Muse Spark judges.** Every number in model-written prose must
match a computed value, or that prose fails the number audit.

> **Research tool only.** Not validated for regulatory GLP studies. Rearing is not counted
> (see [Limitations](#limitations)).

**Live demo: <https://pharmavision-gh6a.onrender.com>.** It's read-only: 14 precomputed runs and
one study report (see [The demo](#the-demo)). It runs on a free instance that sleeps when idle,
so the first visit can take about a minute to wake it.

Built for the Meta Global AI Developer Hackathon (Indie track, "Push the Frontier"), Oct 26 –
Nov 4, 2026.

## How it works

```
video ─▶ 1 ingest ─▶ 2 arena ─▶ 3 track ─▶ 4 measure ─▶ 5 behaviours ─▶ 6 QC ─▶ 7 stats ─▶ 8 report
                         ▲                                                                    │
                   human confirms zones                                          number audit ┘
```

| Layer | Who | Does | Never does |
|---|---|---|---|
| **Sees** | SAM 3.1 | Segments the arena floor from a text prompt. Tracks the animal in every frame from a text prompt, with no training data. Looks for hands in flagged windows. | Produce a number for the report |
| **Measures** | Deterministic Python | Every metric (distance, velocity, centre time and entries, thigmotaxis, immobility), every QC rule, every statistic, with the test chosen by a fixed rule | Call a model |
| **Judges** | Muse Spark | Gives a second opinion on each rear-candidate clip, reviews flagged QC windows, writes the ARRIVE 2.0 methods and results through read-only tools | Compute anything |

A stage can use more than one layer, but each piece of it belongs to one. The arena is found by
SAM 3.1, and its centre zone and px/cm scale are derived in Python. QC runs deterministic rules,
then SAM 3.1 and Muse Spark review the flagged windows. The report is written by Muse Spark and
then gated by the deterministic number audit.

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
tables, per video: [`eval/REPORT.md`](eval/REPORT.md), made by `python -m eval report`. The
figures below are from measure settings v4, Nov 3, 2026; the report is the source if they differ.

| What | Result on the 14 eval videos | Reference | Verdict |
|---|---|---|---|
| Frames tracked (SAM 3.1, zero-shot) | 98.1–99.8% per video | target ≥ 95% | **met** |
| Centroid vs DeepLabCut body centre | median 1.1–1.7 cm, p95 2.4–3.0 cm, ≤ 0.2% of frames ≥ 5 cm apart | DeepLabCut is a reference, not ground truth | report only |
| Rear count, ICC(2,1): candidates scoring ≥ 0.55 (an eval-only count) | 0.02–0.09 against each rater | raters against each other: 0.87–0.91 | **not met** |
| Rear events, F1 (± 0.5 s): candidates scoring ≥ 0.55 | 0.60–0.62 against each rater | raters against each other: 0.87–0.89 | **not met** |
| Rear events: every candidate window | recall 0.88–0.89, precision 0.40–0.41 against each rater | — | report only |
| Number-audit pass rate | 1 of 1 study reports, on the first draft (216 numbers, every one matched) | 100% | **met** |

The application promised rear counts that agree with the raters as well as they agree with each
other. That was measured and is **not met**, so **the pipeline makes no rear count**
([ADR 0003](docs/adr/0003-no-rear-counts-score-as-evidence.md)). Each candidate window keeps its
measured score and Muse Spark's opinion, shown per window, for a person to check.

## The demo

The live site shows the 14 eval videos, each run through every stage, and one study made from
them, `sturman-null-split`.

- **The study is a demonstration, not an experiment.** The Sturman open field videos have no
  treatment groups, so the 14 untreated mice are split 7/7 at random (fixed seed) into "Cohort A"
  and "Cohort B". No treatment was given and no difference is expected. The report finds none
  (Student's t, p 0.42–0.78 across the six metrics). It exists to exercise the stats, the ARRIVE report and the
  number audit end to end. `study.yaml` and the report both state this.
- **What to look at:** a run's video with the floor, centre zone, mask and path overlaid; its
  metrics, behaviour events and QC exclusions (each with frames and a reason); then the study's
  report, with every `[TO FILL]` it abstained on.
- **QC on the demo runs:** the hand placing the mouse at the start and lifting it out at the end
  is excluded as a window, never the whole trial (the QC stage measures where each flag lies).
  All 14 trials are analysed.
- **Cost:** the Muse Spark clip and QC review for all 14 runs cost $2.36 on the Contributor tier
  (about $0.12–0.27 a video). SAM 3.1 tracking costs about $1.83 per 10-minute video. Every
  response is cached, so rebuilding the demo costs nothing.
- **Rebuilding it:** `scripts/pack-demo.sh sturman-null-split` packs the runs and the study into a
  tarball. The site's Docker build downloads it from the
  [`demo-data-v1`](https://github.com/heyiamnotacoder/pharmavision/releases/tag/demo-data-v1)
  release (`DEMO_URL`) and checks its checksum (`DEMO_SHA256`). Deploy: `Dockerfile` and
  `render.yaml`.

## Limitations

- **No rear count**, as above. The report says it abstains on rearing and why.
- **Muse Spark's rear labels are not trusted.** On 80 held-out clips in the spike it agreed
  with the raters at κ 0.04 zero-shot and 0.30 few-shot, and changed about 1 label in 4 when
  asked again. It is a second opinion beside the score. Its agreement on the eval split is not
  measured yet.
- **Open field only.** Elevated plus maze, forced swim and two-animal tests are not built.
- **One animal, a fixed overhead camera, a square arena of known size.**
- **QC window exclusions are listed but don't change the metrics yet**: the metrics are computed
  before QC. A whole-trial exclusion does leave the trial out of the stats.
- **The overlay shows the raw track**, not the filtered path the metrics come from, so it can
  show jitter the metrics never saw.
- **New videos need the Meta Model API** (`MODEL_API_KEY`). The spike and development on the
  public Sturman clips used the Contributor tier, where Meta may train on prompts and outputs. A
  lab's own videos should use the standard tier, the default.

## Run it

Needs [uv](https://docs.astral.sh/uv/), Node 22.22.2 or later, and ffmpeg.

```bash
uv sync
scripts/check.sh            # the same gates CI runs
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

The eval harness. `fetch` downloads the Sturman data into the git-ignored `eval/data/`. `plan`
prints the projected SAM 3.1 spend. `track` calls SAM 3.1 (it needs `MODEL_API_KEY` and asks
before it spends). `report` calls no model.

```bash
uv run python -m eval fetch
uv run python -m eval plan
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
