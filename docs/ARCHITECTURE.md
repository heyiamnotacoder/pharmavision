# PharmaVision: proposed architecture

Status: **proposal**, written Oct 21, 2026, before the build. API shapes, limits and dataset
contents were checked against Meta's docs, PyPI, Zenodo and GitHub on Oct 21 (see
[Verified facts](#verified-facts-oct-21-2026)). The open decisions at the end must be settled in
the spike (Oct 29).

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
  manifest.json      # input hash, fps, frame size, test type, arena dims, code + model versions, measure settings
  video.mp4          # normalised video
  arena.json         # zone polygons (px) + px_per_cm, confirmed flag
  track.parquet      # one row per frame: centroid, area, bbox, axes, orientation, mask RLE, gap flag
  metrics.json       # per-animal metric table (measures)
  events.parquet     # behaviour events: start, end, label, confidence, source (rule | model)
  qc.json            # flags with frame ranges and severity
```

Model responses live outside any one run, in `runs/model_cache/<model>/<prompt version>/<input
hash>.jsonl.gz` (the raw event stream, the same format as the test fixtures), so re-running a
video in a new run folder costs nothing.

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
  - open field: "square floor" ("arena floor", "floor", "box" and "open field arena" found
    nothing on the Sturman background frame in the spike, Oct 30)
  - elevated plus maze: "open arm", "closed arm", "center square"
  - forced swim: "water cylinder"
  SAM 3.1 takes **one concept per request**, so each zone prompt is a separate call.
- **Derived zones are geometry, not prompts.** The open-field centre zone is the inner X% of the
  floor polygon **by area** (the floor shrunk about its centroid), computed in Python. The open
  field template uses 50%, as Sturman et al. 2020 did (45 × 45 cm arena, 10 min).
- Calibration: `px_per_cm` comes from the floor polygon and the arena's real dimensions in the template.
  The Sturman camera has a wide-angle lens and a tilted view: the square floor spans 387 × 313 px,
  so calibrate x and y separately (or fit a 4-corner homography), not with one `px_per_cm`
  (found in the spike, Oct 30).
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
    model_id: str  # recorded in the manifest per stage

    def segment_image(self, image: NDArray[np.uint8], prompt: str) -> list[Mask]: ...
    def track_video(self, video: Path, prompt: str, max_objects: int) -> Iterator[FrameMasks]: ...
```

`SyntheticSegmenter` stands in for SAM 3.1 in tests: it draws a scripted floor and trajectory,
and can skip frames or deliver them out of order like the real stream.

- `MetaApiSegmenter` (`segmenter/meta_api.py`, T4): **primary**. The Model API supports SAM 3.1
  video tracking. Chosen by `make_segmenter(MetaApiSettings(...))` in `segmenter/config.py`.
  It keeps the `max_objects` most confident instances per frame. Videos over
  `max_frames_per_request` go in chunks overlapping by `chunk_overlap_frames` (150), each cut
  frame-exact with ffmpeg and uploaded on its own; a chunk's objects take the previous chunk's
  ids by mask IoU ≥ 0.5, averaged over the overlap frames where either object appears, and the
  overlap is handed over at its middle. Frame counts are exact (ffprobe packet count).
  The cache key is the video's hash, the frame span, the prompt and the request metadata, so a
  cached re-run uploads nothing and needs no API key. Cost: `meta-sam-parser` is pure Python,
  about 15 ms per frame to parse and as much again to decode, so a 9,000-frame video takes
  about 4–5 min of CPU after the API returns.
- `SelfHostedSegmenter`: fallback only, if API price or latency rules it out. Uses open weights
  (`facebook/sam3.1`, gated on Hugging Face) on a Modal GPU.

### 4. Measure (measures, the core engine)
Pure functions from `track.parquet` + `arena.json` to `metrics.json`. No I/O and no model calls.
This is the most heavily tested module.

| Metric | Method |
|---|---|
| Trajectory | Linear interpolation across gaps of at most `max_gap_s` with a detection on both sides; longer gaps, and gaps at either end, stay gaps and are listed in `metrics.json` for QC. Then a median filter (`median_window_frames`) on the centroids of each stretch without a gap. |
| Distance, velocity | Sum of steps between consecutive frames that both have a position, ÷ `px_per_cm`. A kept gap is not bridged. Mean velocity is that distance over the time its steps took. The median filter replaces the earlier idea of zeroing movement below a jitter threshold, so there is one smoothing step, not two. |
| Zone time and entries | Signed distance of the centroid to the zone edge. Entering takes `zone_hysteresis_cm` inside the edge and leaving takes that far outside; while a visit or absence is shorter than `min_dwell_s`, the shortest flips and merges with its neighbours, so a visit broken by flickers survives and flickers alone vanish. Both guards restart after a kept gap. Entries count moves from outside to inside (across a gap too), so starting in the zone is not an entry. Mask-overlap fraction (EPM convention) comes with the EPM template. |
| Thigmotaxis | Fraction of frames with a known position within `thigmotaxis_wall_cm` of the floor-polygon edge (or beyond it). |
| Immobility | Mask change between consecutive frames, 1 − IoU on the stored RLE. A bout is a stretch of changes below `immobility_change_max`; k of them join k + 1 frames, and the bout counts if those frames last at least `immobility_min_s`. |
| Freezing | Velocity and mask change both below threshold for ≥ the minimum duration. |
| Rear candidates | Rule-based windows: mask area drop + elongation change + low centroid velocity. These are *candidates* for stage 5, not results. |

Every threshold is a named parameter in a versioned `MeasureSettings`, recorded in the manifest
when the run starts, with distances in cm and times in seconds (the filter window is in frames).
A metric that can't be computed (no position, no mask) is `null`, never zero. `measure/` has its
own `ruff.toml` that bans I/O, network and model imports, so `scripts/check.sh` enforces its
purity.

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
- `runs/model_cache/` is keyed by (model id, prompt version, input hash). Re-runs are free.
- `eval/` runs log tokens per video so total spend can be projected against the $150 budget
  before running the full eval.
- Prices (third-party trackers, confirm on Meta's pricing page): `muse-spark-1.3` about
  $1.25 input / $4.25 output per million tokens. `muse-wonder-1.0` (preview) about $4 / $20, so
  only consider it for the report step. **SAM 3.1 pricing was not found online**; measure it in the spike (Oct 29).
- The **Contributor tier** (`muse-spark-1.3-contributor`, about $0.10 / $0.20) lets Meta train on
  prompts and outputs, with lower rate limits (150 RPM vs 3,000 RPM, Meta's pricing page, Oct 30). It's acceptable for
  the public-dataset eval only, never for a lab's own videos. **Decided:** use it for the Sturman
  eval and for development on Sturman clips; the tier is a config value, standard tier otherwise.

## Open decisions (settle in the spike, Oct 29)

1. **SAM 3.1 cost and latency. Measured Oct 30; decided: stay on the API.** One Sturman video
   (OFT_16, 609 s) through `python -m spike.sam_run`, ingested at 15 fps:

   | Measure | Result |
   |---|---|
   | Frames billed (`usage.video_frames_processed`) | 9,142, exactly the frames sent |
   | Cost | $1.83 for the video (+ $0.0025 for the arena image) = **$1.80 per 10 min**, under $3.30 |
   | Wall-clock | upload 20 s, first frame 3.5 s, whole track 328 s (about 1.9× faster than real time) |
   | Track "mouse" | mask on 9,062 / 9,142 frames (99.1%), always one object id, never two masks, frames arrived in order this time, mean confidence 0.71, masks box-local (about 99 × 86 px) |
   | Arena "square floor" | one mask, confidence 0.66; "arena floor" returned an empty stream |

   Prices: $0.20 per 1,000 video frames, $2.50 per 1,000 images, at least 50 frames billed per
   video request. At the native 25 fps a Sturman video is 15,234 frames, over the 15,000-frame
   limit, which confirms ingest at 15 fps. The full eval (20 videos) costs about $37. The raw
   responses are committed as replayable fixtures for T4 (`tests/fixtures/sam/`, track 3.7 MB,
   plus the arena and an empty "no match" stream). **Rule:** stay on the API at ≤ $3.30 per
   10-min video; above that, self-host on Modal or process fewer eval videos.
2. **Rearing from a top-down camera.** The Sturman videos are confirmed top view, so rearing shows
   up only as mask shrink and elongation change. Measure how well the rule-based candidates find
   human-labelled rears before building stage 5 on them.
3. **Muse Spark on short clips. Measured Oct 30; label set decided: binary rearing / none.**
   `python -m spike.muse_clips [fps]` asked `muse-spark-1.3-contributor` (zero-shot, whole-frame
   928×576 clips, no mask overlay) about 20 train-split clips: 8 supported and 8 unsupported rears
   that at least two raters agree on (time IoU ≥ 0.5), plus 4 windows with no rear as a control.

   | `input_video` sampling | Tokens / clip | 3-way acc (κ) | Supported vs unsupported, true rears (κ) | Rear vs none (κ) | Cost |
   |---|---|---|---|---|---|
   | default | 731 | 0.40 (0.13) | 0.38 (0.13), n = 16 | 0.55 (0.04) | $0.008 |
   | `fps: 15` | 5,170 | 0.50 (0.25) | 0.50 (0.24), n = 16 | 0.65 (0.15) | $0.016 |

   **Rule:** keep the supported/unsupported split only if κ on the true rears is ≥ 0.4. It is 0.24,
   so the label set is binary. Even rear vs none is near chance, so **zero-shot Muse Spark on whole
   frames cannot carry rear detection**: the mouse fills about 2% of the frame. Stage 5 (T5b) must
   send clips cropped to the SAM mask, with the overlay drawn in, at `fps: 15`, with few-shot
   examples from the train split, and re-measure on the same 20 clips before trusting model
   labels. The rule-based rear candidates from stage 4 (open decision 2) carry more weight.
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
| SAM 3.1: $0.20 per 1,000 video frames, $2.50 per 1,000 images, 50-frame minimum per video request. Muse Spark 1.3: $1.25 / $4.25 per M tokens; contributor $0.10 / $0.20, 150 RPM. Usage reports `video_frames_processed` and `images_processed`. (Checked Oct 30.) | [Pricing and rate limits](https://dev.meta.ai/docs/pricing-rate-limits.md), [Responses schemas](https://dev.meta.ai/docs/api-reference/responses/schemas.md) |
| `input_video` takes `video_url` (public or base64 data URL) or `file_id`, plus an optional `fps`. Structured output is `text.format` with `type: "json_schema"`. (Checked Oct 30.) | [Responses schemas](https://dev.meta.ai/docs/api-reference/responses/schemas.md) |
| Sturman open-field videos: 928×576, 25 fps, about 609 s (15,234 frames), wide-angle lens, burned-in timestamp. (Checked Oct 30.) | `ffprobe` on the downloaded files |
| Muse Spark video tokens, measured: a 2–4 s 928×576 clip costs about 731 input tokens at the default sampling and 5,170 with `fps: 15`. Not found online: a video duration limit. | `spike.muse_clips`, Oct 30 |
