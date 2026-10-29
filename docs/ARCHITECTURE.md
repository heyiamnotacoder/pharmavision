# PharmaVision: proposed architecture

Status: **proposal**, written Oct 21, 2026, before the build. API shapes, limits and dataset
contents were checked against Meta's docs, PyPI, Zenodo and GitHub on Oct 21 (see
[Verified facts](#verified-facts-oct-21-2026)). The open decisions at the end must be settled in
the Day 1 spike.

## Shape

A linear pipeline of stages. Each stage reads typed artifacts from a run folder and writes new
ones, so any stage can be re-run, cached or tested on its own.

```
video ─▶ 1 ingest ─▶ 2 arena ─▶ 3 track ─▶ 4 measure ─▶ 5 behaviours ─▶ 6 QC ─▶ 7 stats ─▶ 8 report
                      (sees)     (sees)    (measures)     (judges)      (both)  (measures)  (judges)
                         ▲
                   human confirms zones
```

Run folder layout, one per video:

```
runs/<run_id>/
  manifest.json      # input hash, fps, frame size, test type, arena dims, code + model versions
  video.mp4          # normalised video
  arena.json         # zone polygons (px) + px_per_cm, confirmed flag
  track.parquet      # one row per frame: centroid, area, bbox, axes, orientation, mask RLE, gap flag
  metrics.json       # per-animal metric table (measures)
  events.parquet     # behaviour events: start, end, label, confidence, source (rule | model)
  qc.json            # flags with frame ranges and severity
  llm_cache/         # model responses keyed by input hash
```

A study is a set of runs plus `study.yaml` (groups, treatment, dose, blinding codes, arena template).
Stats and report run at study level.

All artifacts are Pydantic models serialised to JSON or Parquet. The type is the stage contract.

## Stages

### 1. Ingest (measures)
`ffmpeg` normalises the video to **H.264 MP4** at a fixed fps (default 15) and a max resolution,
and strips audio. It records the source hash, duration and frame size in `manifest.json`. It
rejects videos shorter than the protocol duration.

Why 15 fps: SAM 3.1 accepts **at most 15,000 frames per request**. The Sturman videos are 10 min
at 25 fps, which is exactly 15,000 frames and leaves no margin. At 15 fps a 10-min video is 9,000
frames. Videos longer than the frame limit are split into chunks (see stage 3). MP4 is the only
video format both SAM 3.1 and Muse Spark accept.

### 2. Arena (sees, with human confirmation)
- Take a **background frame**: the per-pixel median over sampled frames, so the animal disappears.
- SAM 3.1 image segmentation with **text prompts from the test template**:
  - open field: "arena floor"
  - elevated plus maze: "open arm", "closed arm", "center square"
  - forced swim: "water cylinder"
  SAM 3.1 takes **one concept per request**, so each zone prompt is a separate call.
- **Derived zones are geometry, not prompts.** The open-field centre zone is the inner X% of the
  floor polygon, computed in Python.
- Calibration: `px_per_cm` comes from the floor polygon and the arena's real dimensions in the template.
- **The UI shows the zones and the user confirms or drags vertices.** Nothing downstream runs until
  `arena.json.confirmed == true`. This is the one human-in-the-loop gate. It is cheap and removes
  the largest source of systematic error.

### 3. Track (sees)
- SAM 3.1 video tracking with a species prompt ("mouse" or "rat"). It expects N animals from the
  template (1 by default, 2 for social tests via multi-object tracking).
- Keep the mask per frame as RLE, plus derived shape features: centroid, area, bbox, major and
  minor axis, orientation and elongation.
- Frames with no mask, or more masks than expected, are marked as gaps. Interpolation happens in
  stage 4, not here.

**The call** (confirmed in Meta's docs):
- Upload the video once with `POST /v1/files` (`purpose=user_data`, up to 1 GiB, ready on a `200`).
  Then call `POST https://api.meta.ai/v1/responses` with `model: "sam-3.1"`, an `input_text` noun
  phrase and an `input_video` with the `file_id`. Use the OpenAI SDK with `base_url="https://api.meta.ai/v1"`.
- Set `stream: true`. Each frame arrives as its own `response.output_text.delta`.
- Set `metadata`: `mask_encoding: "one_bit"` (smaller masks, better for video) and
  `include_confidence: "true"`.
- Parse the output with `meta-sam-parser` ≥ 0.1.0 (Python, typed, no dependencies). Store masks
  as COCO RLE via `decode_mask_to_rle`.

**Output gotchas from the docs:**
- Each line starts with a frame marker `<Nf>`. Frames **can skip and are not always in ascending
  order**, so key every record by its frame index, never by line position.
- No match gives an empty stream, not an error. Treat it as "animal not found".
- Object ids are stable within one response but **not reliable across exit, full occlusion or very
  long clips**. For a single animal, take the highest-confidence instance per frame. For videos
  near the frame limit, split into overlapping chunks and relink tracks by mask IoU in the overlap.
- At most 16 objects per frame. That's enough for every planned test.

**Segmenter interface.** One protocol and two backends, so hosting is a config switch:

```python
class Segmenter(Protocol):
    def segment_image(self, frame: Frame, prompt: str) -> list[Mask]: ...
    def track_video(self, video: Path, prompt: str, max_objects: int) -> Iterator[FrameMasks]: ...
```

- `MetaApiSegmenter`: **primary**. The Model API supports SAM 3.1 video tracking.
- `SelfHostedSegmenter`: fallback only, if API price or latency rules it out. Uses open weights
  (`facebook/sam3.1`, gated on Hugging Face) on a Modal GPU.

### 4. Measure (measures, the core engine)
Pure functions from `track.parquet` + `arena.json` to `metrics.json`. No I/O and no model calls.
This is the most heavily tested module.

| Metric | Method |
|---|---|
| Trajectory | Median filter on centroids. Linear interpolation for gaps ≤ `max_gap` frames; longer gaps become QC flags. |
| Distance, velocity | Sum of frame-to-frame displacement ÷ `px_per_cm`. Movement below the jitter threshold is zeroed. |
| Zone time and entries | Point-in-polygon on the centroid, with an optional mask-overlap fraction ≥ threshold (EPM convention). Entries use hysteresis and a minimum dwell so boundary jitter isn't counted. |
| Thigmotaxis | Fraction of time within d cm of the floor-polygon edge. |
| Immobility (FST) | Fraction of mask pixels changed between consecutive frames below a threshold, sustained for ≥ the minimum duration. |
| Freezing | Velocity and mask change both below threshold for ≥ the minimum duration. |
| Rear candidates | Rule-based windows: mask area drop + elongation change + low centroid velocity. These are *candidates* for stage 5, not results. |

Every threshold is a named parameter in a versioned `MeasureConfig` and recorded in the manifest.

### 5. Behaviours (judges)
- Cut 2–4 s clips around each **candidate window** from stage 4, with the mask overlay drawn
  in. Never send the full video.
- `muse-spark-1.3` (Responses API, `input_video` as base64 MP4 under 50 MB or a `file_id`,
  structured output) returns
  `{label: rearing_supported | rearing_unsupported | grooming | head_dip | none, confidence, rationale}`.
- Calibration: few-shot examples come from the training split of the Sturman dataset only.
  The evaluation split is never shown to the model.
- Events are written with `source = model`. Low-confidence events go to the QC review list.

### 6. QC (measures, then judges)
- Deterministic flags: tracking-loss %, out-of-arena frames, centroid jumps above physical maximum
  speed, a second large mask (experimenter's hand). SAM 3.1 is also prompted with "hand" and "glove"
  on flagged windows.
- Muse Spark reviews only the flagged windows and returns `{verdict: exclude_window | keep | exclude_trial, reason}`.
- Every exclusion is listed in the report. Nothing is dropped silently.
- Verdicts apply automatically. Zone confirmation stays the only human gate; low-confidence events
  are reported as such rather than queued for review. A human override after the fact is stretch
  goal FR-S6.

### 7. Stats (measures)
- Input: per-animal metrics × `study.yaml` groups.
- **The test is chosen by a deterministic rule, not by the model:**
  normality (Shapiro–Wilk) and equal variance (Levene) → t-test / Welch / Mann–Whitney for 2 groups;
  one-way ANOVA + Tukey, Welch ANOVA + Games-Howell, or Kruskal–Wallis + Dunn for 3 or more.
- Output: test used and why, statistic, p, effect size with 95% CI, n per group after exclusions.
- Libraries: `scipy` and `statsmodels`, with `pingouin` if it saves time. Pin versions.

### 8. Report (judges, then the number audit)
- A Muse Spark agent with read-only tools: `get_protocol`, `get_metrics_table`, `get_stats`,
  `get_qc`, `get_events_summary`.
- It writes a methods and results section covering the **ARRIVE 2.0 Essential 10**. Items the system
  can't know (housing, randomisation method) come out as explicit `[TO FILL]` placeholders, never
  invented.
- **Number audit** (deterministic, blocking): extract every number from the prose and check it
  against the tool outputs, with a rounding tolerance. Any unmatched number fails the report and
  the agent is re-prompted with the list of offending numbers. After 2 failures the report ships
  with those sentences struck out and flagged.

## Evaluation harness

`eval/` is separate from the app. It runs the pipeline on the Sturman et al. 2020 dataset.

**Data sources:**
- **Videos:** Zenodo record 3608658, CC BY 4.0, 2.5 GB, **top-view** camera. It holds 20 open
  field, 24 plus maze and 29 forced swim MP4s. DLCAnalyzer states 25 fps.
- **Human labels:** GitHub `ETHZ-INS/DLCAnalyzer`, files under `data/*/Labels/`. Each row is an
  interval: `file;Experimenter;from;to;type;ID`, with `from` and `to` in seconds. Map labels to
  videos by `ID` (e.g. `OFT_11`), not by the DLC filename.

| Test | Videos × raters | Labelled behaviours (row counts) |
|---|---|---|
| Open field | 20 × 3 (Oliver, Furkan, Jin) | Supported rear 3003, Unsupported rear 2090, Grooming 275, Jumping 16 |
| Plus maze | 5 × 3 (Jin, Sian, Oliver) | Head dip 573, Grooming 376, Protected stretch 183, Rearing 160, Unprotected stretch 71 |
| Forced swim | 10 videos, 4 raters in total | Floating 804 |

- **No human ground truth for position.** The repo's `Output_DLC` CSVs are DeepLabCut *model*
  output. Use them as a reference tracker to compare against, never as ground truth.
- **Licence:** the repo is GPL-3.0. Download the label CSVs at eval time into a git-ignored folder,
  never commit them into the MIT repo. Credit the Zenodo videos (CC BY 4.0) in the README.

| What | Reference | Metric | Pass bar |
|---|---|---|---|
| Tracking | DeepLabCut output (reference, not truth) | Centroid distance to DLC body centre (cm), % frames tracked | Report it; check large disagreements by eye |
| Rearing (OFT) | 3 human raters, 20 videos | Per-video count ICC vs each rater; event-level F1 with ± tolerance; supported vs unsupported | Agreement with humans within the human–human range |
| Floating (FST, stretch) | Human raters, 10 videos | Immobility time vs each rater | Within the human–human range |
| Head dips (EPM, stretch) | 3 raters, 5 videos | Count agreement | Report only, n is too small for a claim |
| Report | Computed tables | Number-audit pass rate | 100% |

Use a fixed train/eval split of the 20 open-field videos, **6 train / 14 eval**, drawn at random
with a fixed seed and recorded in `eval/split.json`. Few-shot examples come
only from the train split. The eval report is generated as `eval/REPORT.md` and its numbers feed
the demo and write-up, the same way PaedScale published its validation table.

## Application layer

- **Backend:** FastAPI. `POST /runs` (upload video + template), `POST /runs/{id}/arena/confirm`,
  `GET /runs/{id}` (status + artifacts), `GET /runs/{id}/stream` (SSE progress),
  `POST /studies/{id}/report`. Long stages run as background jobs with status in SQLite.
- **Frontend:** video player with a canvas overlay (masks, trajectory, zones), a zone editor, an
  event timeline, a metrics table and the rendered report. This screen is the demo.
- **Deploy:** API on Render or Modal (CPU). SAM on Modal GPU if self-hosted. Static frontend.
  Precomputed demo runs ship with the app so the demo never waits on a cold GPU.

## Cost control

- Muse Spark sees candidate clips and flagged windows only. A 10-minute video should need tens of
  calls, not thousands.
- `llm_cache/` is keyed by (model id, prompt version, clip hash). Re-runs are free.
- `eval/` runs log tokens per video so total spend can be projected against the $150 budget
  before running the full eval.
- Prices (third-party trackers, confirm on Meta's pricing page): `muse-spark-1.3` about
  $1.25 input / $4.25 output per million tokens. `muse-wonder-1.0` (preview) about $4 / $20, so
  only consider it for the report step. **SAM 3.1 pricing was not found online**; measure it on Day 1.
- The **Contributor tier** (`muse-spark-1.3-contributor`, about $0.10 / $0.20) lets Meta train on
  prompts and outputs, with lower rate limits (reported 100 RPM vs 3,000 RPM). It's acceptable for
  the public-dataset eval only, never for a lab's own videos. **Decided:** use it for the Sturman
  eval and for development on Sturman clips; the tier is a config value, standard tier otherwise.

## Open decisions (settle in the Day 1 spike)

1. **SAM 3.1 cost and latency.** Video tracking on the API is confirmed. Still unknown: price, and
   wall-clock time for a 9,000-frame video. Measure one Sturman video end to end. **Rule:** stay on
   the API at ≤ $3.30 per 10-min video (about 30 videos ≈ $100); above that, self-host on Modal or
   process fewer eval videos. Wall-clock time doesn't rule the API out: the demo uses precomputed
   runs plus one short live clip.
2. **Rearing from a top-down camera.** The Sturman videos are confirmed top view, so rearing shows
   up only as mask shrink and elongation change. Measure how well the rule-based candidates find
   human-labelled rears before building stage 5 on them.
3. **Muse Spark on short clips.** Can it tell supported from unsupported rears at 15 fps in 2–4 s
   clips? Run 20 labelled clips before committing to the label set. Fall back to a binary
   rearing/none if needed.
4. **Frontend stack. Decided Oct 29:** Vite + React + TypeScript (strict), because the overlay and
   zone editor are the demo.
5. **Python environment. Decided Oct 29:** `uv`, Python 3.12, `ruff` everywhere, `pyright` strict on
   `measure/` and `stats/`.

## Verified facts (Oct 21, 2026)

| Fact | Source |
|---|---|
| Model API models: `muse-spark-1.3` (+ `-contributor`), `1.2`, `1.1`; `muse-wonder-1.0` (preview); `muse-image-1.0`; `sam-3.1`. All Muse Spark versions take text, image, video, audio and PDF, with 1M context. | [dev.meta.ai/docs/models](https://dev.meta.ai/docs/models.md) |
| `sam-3.1`: text, box or point prompts on image or video → boxes + masks, tracked across frames. Up to 15,000 frames per request, 16 objects per frame, MP4 or MOV (H.264, H.265, MPEG-4). | [SAM segmenting](https://dev.meta.ai/docs/sam/segmenting.md), [SAM overview](https://dev.meta.ai/docs/sam/overview.md) |
| SAM output grammar: `<Nf>` frame markers, base85 masks, sparse ids, optional confidence `c`. | [Reading segmentation](https://dev.meta.ai/docs/sam/reading-segmentation.md) |
| `meta-sam-parser` 0.1.0 on PyPI, Python ≥ 3.10, SAM License (publications must acknowledge SAM). | [PyPI](https://pypi.org/project/meta-sam-parser/), [client libraries](https://dev.meta.ai/docs/sam/client-libraries.md) |
| Files API: 1 GiB per upload, 50 MB inline, 100 GiB per team, no expiry by default. Muse Spark video input is MP4 only. | [File handling](https://dev.meta.ai/docs/file-handling.md), [Video understanding](https://dev.meta.ai/docs/video-understanding.md) |
| Sturman videos: CC BY 4.0, top view, 2.5 GB; labels as in the eval table above. | [Zenodo 3608658](https://zenodo.org/records/3608658), [DLCAnalyzer](https://github.com/ETHZ-INS/DLCAnalyzer) |
| Not found online: SAM 3.1 pricing, Muse Spark video duration limit, frame sampling rate for video understanding. | Measure in the Day 1 spike |
