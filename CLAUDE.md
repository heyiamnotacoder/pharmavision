# PharmaVision

Open-source AI lab for behavioural pharmacology: rodent test video in → scored, statistically
analysed, ARRIVE-compliant results section out. Built for the Meta Global AI Developer Hackathon
(Indie track, theme "Push the Frontier"). Build window **Oct 26 – Nov 4, 2026, 9:00 PM ET**.

## The invariant: sees → measures → judges

Every feature belongs to exactly one layer. Place it before writing it.

- **Sees: SAM 3.1.** Pixels to masks. Animal tracks, arena zones, intruding hands. Produces geometry only.
- **Measures: deterministic Python.** Masks to numbers. Every metric, every statistic, every unit
  conversion. Pure functions, unit-tested against synthetic ground truth.
- **Judges: Muse Spark.** Classifies behaviour clips, reviews QC flags, writes prose. Reads numbers
  through tools; produces labels and text.

Muse Spark output reaches the report only after the **number audit**: every number in generated
prose must match a value in the computed tables, or the report fails. This is PaedScale's
cite-or-abstain rule carried over: when data is missing, the system abstains and says so.

## Where things are

- `docs/REQUIREMENTS.md`: scope, MVP vs stretch, acceptance criteria. Check before adding a feature.
- `docs/ARCHITECTURE.md`: pipeline stages, artifacts, interfaces, open decisions. Read before
  touching a stage boundary or a model call.
- `docs/application.md`: the hackathon application as submitted. The promises judges will check.
- `spike/`: the T2 measurement scripts (laptop only).
- `eval/`: the eval harness (`python -m eval`, laptop only). `eval/split.json` is the fixed 6/14
  split: the harness checks it and never rewrites it.

## Gotchas

- SAM 3.1 and Muse Spark APIs are new (2026) and change often. Re-check docs
  (dev.meta.ai/docs) before writing a call. `docs/ARCHITECTURE.md` "Verified facts" records what
  was confirmed on Oct 21, 2026.
- SAM 3.1 output frames can skip and arrive out of order: key records by the `<Nf>` frame index,
  never by line position.
- Hackathon budget: $150 Meta Model API credits for the whole build. Muse Spark sees only short
  candidate clips, never full videos; cache every model response by input hash.
- Validation data is the Sturman et al. 2020 public dataset (Zenodo record 3608658). No new animal
  experiments are part of this project. Its human labels are GPL-3.0: fetch them into a
  git-ignored folder, never commit them.

## Workflow

- **Where work runs.** Building happens in cloud sessions (claude.ai/code). The SAM 3.1 / Muse Spark
  spike and `eval/` run on the laptop: they need `MODEL_API_KEY` (in `.env`, never committed) and
  the Sturman data. Cloud sessions can't reach `api.meta.ai`, so tests use recorded responses.
  Cloud setup is in `docs/agents/cloud.md`.
- **One branch and one PR per stage or feature** (`feat/<stage>`, `fix/…`, `chore/…`). Never
  commit to `main`.
- **Gates.** `scripts/check.sh` (ruff lint, ruff format, pyright, pytest, and the frontend checks
  once `frontend/` exists) must pass before every push. CI runs the same script on every PR and on
  `main`.
- **Review before merge.** Run the `code-review` skill with `origin/main` as the fixed point, post
  its findings as a PR comment, fix them, push, wait for green CI, then squash-merge. Follow the
  checklist in `.github/pull_request_template.md`.
- **Cloud GitHub access.** `gh pr …` and `gh issue …` fail in cloud sessions (the proxy blocks
  GraphQL). Use the built-in GitHub tools or the REST calls in `docs/agents/issue-tracker.md`.
- **Skills.** `.claude/skills/` holds Matt Pocock's skills, vendored at a pinned commit. Refresh
  them with `scripts/sync-matt-pocock-skills.sh`; don't edit them by hand.

## Agent skills

### Issue tracker

GitHub Issues on `heyiamnotacoder/pharmavision`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
