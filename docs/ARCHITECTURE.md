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
  arena.json         # zone polygons (px) + px_per_cm, confirmed flag; empty if no floor was found
  background.png     # the background frame the floor was looked for on; the zone editor draws on it
  track.parquet      # one row per frame: centroid, area, bbox, axes, orientation, mask RLE, gap flag, second-mask area
  metrics.json       # per-animal metric table (measures)
  events.parquet     # behaviour events: first/last frame, label, confidence, low_confidence, rationale, source (rule | model)
  qc.json            # flags with frame ranges, severity and reason; the model's reviews; the exclusions
```

Model responses live outside any one run, in `runs/model_cache/<model>/<prompt version>/<input
hash>.jsonl.gz` (the raw event stream, the same format as the test fixtures), so re-running a
video in a new run folder costs nothing.

A study is a set of runs plus `study.yaml` (groups, treatment, dose, blinding codes). Stats and
report run at study level, in a study folder:

```
studies/<study>/
  study.yaml         # the design, written by the scientist: groups in order (the first is the reference), and per trial its run id, blinding code and group
  trials.json        # per-trial metrics table: values, group, blinding code and every QC exclusion; excluded trials stay in it
  stats.json         # per metric: n per group after exclusions, the checks, the test and why, statistic, df, p, effect size + CI, post-hoc pairs
  report.json        # every draft of the report agent with its number audit, and the report as shipped
  report.md          # the report as shipped: ARRIVE 2.0 Essential 10 methods and results
```

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
- **The prompt sees the background equalised** (CLAHE on lightness, `ArenaSettings.contrast_clip_limit`,
  default 2.0). SAM 3.1 returns no mask it scores under about 0.5. On the ingested background of
  OFT_44 (CAM01) and OFT_54 (CAM03), "square floor" found nothing; equalised, it scored 0.62 and
  0.53, the same floor (21.7% of the frame) as on the CAM04 videos, which also scored a little
  higher (0.66 → 0.69 on OFT_50). Found by the rear-recall run, Oct 30. OFT_54 still sits near the
  cutoff, so a floor that is never found falls back to the user drawing it (T10): the run waits
  at the gate with an **empty arena** (no floor, no centre, `px_per_cm` 0), and only a floor can
  be confirmed. A floor is usable with at least 3 corners, all on the frame, edges that don't
  cross, and some area; a detected outline that isn't is treated as no floor.
- **Derived zones are geometry, not prompts.** The open-field centre zone is the inner X% of the
  floor polygon **by area** (the floor shrunk about its centroid), computed in Python. The open
  field template uses 50%, as Sturman et al. 2020 did (45 × 45 cm arena, 10 min).
- Calibration: `px_per_cm` comes from the floor polygon and the arena's real dimensions in the template.
  The Sturman camera has a wide-angle lens and a tilted view: the square floor spans 387 × 313 px,
  so calibrate x and y separately (or fit a 4-corner homography), not with one `px_per_cm`
  (found in the spike, Oct 30).
- **The UI shows the zones and the user confirms or drags vertices.** Nothing downstream runs until
  `arena.json.confirmed == true`. This is the one human-in-the-loop gate. It is cheap and removes
  the largest source of systematic error. The editor sends only the floor
  (`confirm_arena(run, floor_px)`): the centre zone and `px_per_cm` are derived from it again,
  exactly as from a detected floor, so the centre zone is never drawn by hand (T10).

### 3. Track (sees)
- SAM 3.1 video tracking with a species prompt ("mouse" or "rat"). It expects N animals from the
  template (1 by default, 2 for social tests via multi-object tracking).
- Keep the mask per frame as RLE, plus derived shape features: centroid, area, bbox, major and
  minor axis, orientation and elongation.
- Frames with no mask are marked as gaps. Interpolation happens in stage 4, not here.
- The stage asks for one object more than the template expects and keeps the largest other
  mask's area per frame (`second_area_px`), so stage 6 can flag an intruding hand. The extra
  object doesn't change the SAM cache key: the backend filters after the response.

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
| Rear candidates | Rule-based windows, listed in `metrics.json` as `rear_candidates` (first and last frame, with their features and score: next row). A candidate frame has a mask area at least `rear_area_drop` below its baseline, an elongation (major over minor axis, from the mask's second moments) at least `rear_elongation_change` away from its baseline either way, and a centroid speed of at most `rear_max_speed_cm_s`. Each baseline is a running median over `rear_baseline_s` centred on the frame, because the wide-angle lens changes the apparent size of the animal across the arena. A break of less than `rear_merge_s` between candidate frames joins them into one window unless the position is unknown somewhere in the break; windows shorter than `rear_min_s` are dropped. These are *candidates* for stage 5, not results. |
| Rear score | Each candidate carries its features (`RearFeatures`: duration, deepest and mean area drop below baseline, largest elongation change, mean speed, mean mask area in cm²) and a score: the logistic of `MeasureSettings.rear_scorer`'s intercept plus each feature times its weight, fitted on the train split by `spike.rear_scorer` and versioned with the settings (v4). Evidence per window only: **no rear count is made**, because no count from it agrees with the raters ([ADR 0003](adr/0003-no-rear-counts-score-as-evidence.md)). |

Every threshold is a named parameter in a versioned `MeasureSettings`, recorded in the manifest
when the run starts, with distances in cm and times in seconds (the filter window is in frames).
A metric that can't be computed (no position, no mask) is `null`, never zero. `measure/` has its
own `ruff.toml` that bans I/O, network and model imports, so `scripts/check.sh` enforces its
purity.

### 5. Behaviours (judges)
- Cut a clip around each **candidate window** from stage 4 (`stages/behaviours.py`, T5b): the
  window padded by `clip_pad_s` (0.5 s) each side, widened to `clip_min_s` (2 s) or cut to
  `clip_max_s` (4 s) about its middle. The clip is a square crop around the animal's masks over
  the clip, plus `crop_margin_cm` (4 cm), scaled to `clip_side_px` (256), with the tracked mask
  outlined in green, encoded bit-exact as H.264 so the same frames always give the same bytes.
  The video is read once for all clips. Never send the full video.
- **Judge interface** (`judge/__init__.py`), the one boundary to the judges layer, like the
  Segmenter for the sees layer:

  ```python
  class Judge(Protocol):
      model_id: str  # recorded in the manifest per stage

      def classify_clip(self, clip: Clip) -> ClipJudgement: ...  # label, confidence, rationale
      def review_window(self, window: FlaggedWindow) -> WindowVerdict: ...  # verdict, reason
      def write_report(self, brief: ReportBrief) -> str: ...  # Markdown, one heading per section
  ```

  The QC review joined it in T6b (stage 6); the report agent in T8b (stage 8).
- `MuseSparkJudge` (`judge/muse_spark.py`): `muse-spark-1.3` on the Responses API, streamed,
  `input_video` as a base64 MP4 with `fps: 15`, structured output
  `{label: rearing | none, confidence, rationale}`: binary, as open decision 3 decided. The
  tier (`standard` or `contributor`) is `MuseSparkSettings.tier`, and the manifest records the
  resulting model id. Each response is cached under `<model>/clip-v1/<hash>`, the hash covering
  the clip's frames (not the MP4's bytes, which depend on the ffmpeg build), the prompt, the
  schema, the sampling rate and the few-shot examples.
  `RecordedJudge` replays that cache by the same key and refuses a clip it has no response for,
  so a response captured on the laptop is a test fixture as is.
- Calibration: few-shot examples (`MuseSparkSettings.few_shot`) go ahead of the clip as earlier
  turns, the example clip in a user turn and its label replayed in an assistant turn. They come
  from the training split of the Sturman dataset only: the judge refuses an example whose video
  is not in `eval/split.json`'s train list. The evaluation split is never shown to the model.
  The API takes at most 5 videos per request, so the settings refuse more than 4 examples. The
  committed set (`judge/few_shot/`, 2 rears and 2 hard negatives, `few_shot_examples()`) is
  **off by default**: it did not make the labels trustworthy (open decision 3).
- Every candidate becomes one event with `source = model`, so each verdict is auditable. An
  event under `BehaviourSettings.min_confidence`
  (0.6) is marked `low_confidence` and reported as such. The settings are versioned in the
  manifest.
- **Muse Spark is a second opinion (open decision 3, ADRs 0002 and 0003).** Its labels are not
  trusted, zero-shot or few-shot. Each event keeps its label and rationale beside the
  candidate's measured score, shown per window. **The pipeline makes no rear count**, and the
  report abstains on rearing counts, citing the eval.

### 6. QC (measures, then judges)
- Deterministic flags (`measure/qc.py`, T6a), on the raw track, with every threshold in a
  versioned `QcSettings` recorded in the manifest:

  | Flag | Rule | Severity |
  |---|---|---|
  | `tracking_loss` | each gap too long to interpolate | warning |
  | `tracking_loss`, whole trial | animal missing on more than `max_tracking_loss_fraction` (5%) of frames | critical: the trial is excluded |
  | `out_of_arena` | centroid at least `out_of_arena_margin_cm` (2 cm) outside the floor | warning |
  | `speed_jump` | a step between consecutive detections faster than `max_speed_cm_s` (100 cm/s); jumps within `speed_jump_join_s` (1 s) join one flag, so a multi-frame mis-detection is covered whole | warning |
  | `second_mask` | another mask of at least `second_mask_min_area_cm2` (10 cm²) in the frame | warning |

  Only critical flags exclude anything; warnings are listed and go to the model review below.
- **Known limits of `second_mask`** (measure on the laptop before trusting it): the second mask
  is the next instance of the *animal* prompt, so real SAM 3.1 may rarely give one for a hand,
  and may give one for a reflection. A hand seen while the animal is not is tracked *as* the
  animal and shows up as a `speed_jump`, not a `second_mask`. The "hand" / "glove" prompts
  (T6b) are the real hand detector. With two animals per video (FR-S3) the rule must look at
  the third mask instead.
- **Model review** (`stages/qc.py`, T6b), only of warnings, with `QcSettings.review`
  (`QcReviewSettings`) in the manifest; `review: None` turns it off and leaves the deterministic
  QC exactly as T6a wrote it. Nothing goes to a model when nothing is flagged, or when a rule has
  already excluded the trial (no verdict could change it). For each warning:
  - **Sees:** SAM 3.1 `segment_image` with each of `hand_prompts` ("hand", "glove") on
    `hand_frames` (3) frames evenly spaced over the flag, read front to back from the
    normalised video so the cache key (the frame's bytes) is the same on every machine. Each
    prompt's largest mask becomes a `HandSighting` (prompt, frame, area in cm²).
  - **Judges:** Muse Spark sees a clip of the **whole frame** (a hand comes in from the edge, so
    no crop around the animal), the flag padded by `clip_pad_s` (1 s), 2–6 s, scaled to fit
    `clip_max_side_px` (480), mask outlined in green, cut by the same `stages/clips.py` as the
    behaviours clips. The prompt states the flag's kind, frames and reason and the hand
    sightings, and asks for `{verdict: keep | exclude_window | exclude_trial, reason}`, cached
    under `<model>/qc-v1/<hash>`. It is told not to state numbers in its reason, but nothing
    enforces that yet: the report stage must run the number audit over every model-written
    reason it shows, not only over the agent's prose (the audit never takes a number inside a
    string as known, T8a). A flag
    longer than `clip_max_s` is judged on its middle, and `exclude_window` still leaves out all
    of its frames (the safer error).
  - Each review (flag, sightings, verdict, reason) is listed in `qc.json` `reviews`, kept or
    not. `exclude_window` adds a window exclusion over the flag's frames and `exclude_trial` a
    trial exclusion, both `source: model` with the model's reason. The manifest records the
    models as `qc` (judge) and `qc_hands` (segmenter) when the review called them.
- Every exclusion is listed in the report. Nothing is dropped silently.
- Verdicts apply automatically. Zone confirmation stays the only human gate; low-confidence events
  are reported as such rather than queued for review. A human override after the fact is stretch
  goal FR-S6.

### 7. Stats (measures)
- Input: per-trial metrics × `study.yaml` groups. The stage refuses a study while any run lacks
  `qc.json`, or when its runs were measured or QC'd with different settings.
- Trials with a trial exclusion are left out, and so is a trial whose metric is unknown (None);
  each group reports `n` analysed, `n_excluded` and `n_missing` per metric. Window exclusions are
  listed on the trial's row but do not yet change its metrics (they were computed before QC).
- **The test is chosen by a deterministic rule, not by the model** (`stats/rule.py`, thresholds
  in the versioned `StatsSettings`):
  1. Any group with fewer than 3 trials left, or every value the same: no test, and the reason
     says why (abstain).
  2. Shapiro–Wilk on each group (a group with no spread counts as not normal); Levene across
     them, median-centred.
  3. Any group not normal → Mann–Whitney U (2 groups) or Kruskal–Wallis + Dunn, Holm-adjusted
     (3 or more). All normal, Levene p > 0.05 → Student's t or one-way ANOVA + Tukey's HSD.
     Otherwise Welch's t or Welch's ANOVA + Games-Howell.
- Effect sizes, each with a 95% CI:

  | Test | Effect size | Interval |
  |---|---|---|
  | Student's / Welch's t | difference in means (later group minus the reference), metric units | from the same t distribution |
  | Mann–Whitney U | rank-biserial r | Fisher's z; zero width when the groups do not overlap |
  | ANOVA | η² (SSB / SST) | noncentral F, ends as nc / (nc + df₂) so the estimate stays inside |
  | Welch's ANOVA | η² (SSB / SST) | percentile bootstrap within groups, fixed seed |
  | Kruskal–Wallis | ε² = H / (n − 1) | percentile bootstrap within groups, fixed seed |

- Post-hoc pairs in design order: the difference, t (or z for Dunn), adjusted p, and an interval
  for Tukey and Games-Howell.
- Libraries: `scipy` only, typed by `scipy-stubs` (dev) so `stats/` passes pyright strict. The tests
  check against `statsmodels` and literals from `pingouin` 0.7.0 and `scikit-posthocs` 0.17.1
  (not dependencies) for the tests scipy lacks.

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
  - Built in T8a as the pure `audit/` package (no I/O, no model calls, pyright strict).
    `known_values` flattens the tool outputs into numbers sourced by field path
    (`stats.metrics[0].comparison.p`), plus each list's length (`len(qc.exclusions)`) for the
    counts. Numbers inside strings are never known: a string may be model-written.
  - `audit_numbers` finds each number in the prose (decimals, thousands commas, percents,
    `1.6e-3` and `1.6 × 10⁻³`), skipping identifiers (`B07`, `OFT_16`) and the
    `AuditSettings.allowed_phrases` ("ARRIVE 2.0", "Essential 10", "SAM 3.1"). A number
    matches when a known value rounds to it: within half a unit of its last written digit,
    magnitudes only (a sign is as often a word), a percent against the value × 100. The "95"
    of "95% CI" matches `ci_level`, not an allow-list, so a study at another level fails. On a
    tie the source is a field before a list's length.
  - The result, `NumberAudit`, lists every number with its matched source or none, and the
    sentence it is in with its offsets, for the re-prompt and the strike-out.
  - Known strictness (false fails, the safe side): thresholds such as "p < 0.001" and derived
    numbers ("an 18% increase") match only a known value that rounds to them, so the agent is
    to state exact values from the tools; methods tokens ("Python 3.12", "30 min") need a
    table or an allowed phrase; "F(1,100)" reads as 1100 (write "F(1, 100)"); trailing zeros
    count ("3,400" is not 3412.6). Numbers that exist only inside a deterministic reason string
    (a QC flag's "17% of frames (34 of 200)") must reach the agent as fields.
  - Known leniency (false passes): the audit checks that a number exists in the tables, not
    that it is attached to the right label. Small integers and round percents nearly always
    match something (any value from 0 to 1 gives some percent), so the report stage should hand
    the agent only the tables it cites, without settings it does not report. Numbers written
    as words ("twelve mice") are not found: the agent is to write numbers as digits.
- Rearing counts: the agent states that the pipeline does not count rears and why (ADR 0003);
  `get_events_summary` gives the number of candidate windows, never a rear count.
- **Built in T8b** (`stages/report.py`, `arrive.py`, `MuseSparkJudge.write_report`), entered
  through `pipeline.study_report`, which refuses a study whose stats have not run. A report
  quotes the stats it was written from, so the stats stage removes `report.json` and
  `report.md` before it writes new stats: a report that then fails leaves no stale one on show.
  - **Tools** take no arguments and return a stored table as JSON, built once per report
    (`stages/report.py` `TOOLS`): `get_protocol` (design by blinding code, test, arena, video
    fps, models, the measure and QC thresholds the methods cite), `get_metrics_table`
    (`trials.json` by blinding code, no run ids), `get_stats` (the rule's thresholds and every
    metric's result; no version or bootstrap seed), `get_qc` (each trial's flags, reviews and
    exclusions) and `get_events_summary` (per trial the number of rear candidate windows and
    each low-confidence event with the model's rationale, the `min_confidence` threshold, and
    why there is no rear count). They select and rename fields and compute nothing: a test
    checks that every number they return is a stored field, a stored list's length, or a
    number in a rule's reason. The audit's known values are these outputs, so a number reaches
    the report only if a tool gave it to the agent.
  - **Rule reasons:** a flag's or rule exclusion's reason ("not found on 17% of frames (limit
    5%)") is written by deterministic code, so its numbers go out as `reason_values` (a percent
    as a fraction, `audit.rule_reason_values`) and the agent may quote the reason (FR-7). A
    model's reason or rationale never gets them, and the model's own confidence is not handed
    out at all (NFR-1).
  - The stage refuses a study whose trials differ in template, video fps, behaviour settings
    or models (the stats stage already refuses differing measure or QC settings), since the
    methods describe one protocol.
  - **Agent loop:** each Responses request offers the five tools as strict `function` tools;
    each `function_call` (read from its `response.output_item.done` event) is answered with a
    `function_call_output` in the next request, until a response writes text. At most
    `MuseSparkSettings.report_max_turns` (8) requests per draft. Each request is cached under
    `<model>/report-v1/<hash of the whole request>`; the tools' outputs decide the
    conversation, so a re-run replays every turn and `RecordedJudge` needs no API.
  - **ARRIVE shape** (`arrive.py`): the agent writes one `## <heading>` per Essential 10 item.
    The stage keeps its text under the ten headings only, and writes `[TO FILL: <fact>]` itself
    for the facts the system cannot know: sample-size reasoning, pre-test criteria,
    randomisation (the whole section: whatever the agent writes there is dropped), who knew
    the allocation, strain/sex/age/weight/source, housing and husbandry, drug administration,
    time and place of testing. An item the agent left out reads `[TO FILL: nothing was written
    for this item]`. Headings carry no numbers, so the audit reads only the agent's text.
  - **Gate** (`ReportSettings`, versioned in `report.json`): the assembled draft is audited; on
    a failure the agent is re-prompted (its draft as an assistant turn, then each unmatched
    number with its sentence). `max_drafts` is 2: when the second draft fails too, each
    sentence holding an unmatched number ships as `~~sentence~~ **[failed the number
    audit]**`, and `Report.struck` lists the numbers. A heading's, quote's or list item's
    marker stays outside the strike; a table row (one sentence to the audit) is struck cell by
    cell with the flag in its last cell, since GFM splits a row into cells before it reads
    `~~`. Every draft and its audit stay in `report.json`.
  - **Live check:** the function-calling event shape and whether Muse Spark accepts a
    `function_call` item without the reasoning item that preceded it. **Checked live Oct 31**
    (PR #35): Muse Spark sent `reasoning`, a `message`, then 5 parallel `function_call` items,
    and accepted them back without the reasoning item; the draft passed the audit first time.
    Its `message` came in `phase: "commentary"` beside the tool calls, where a turn's text is
    not read anyway; the answer text leaves out every commentary message in case one comes
    beside a final draft too (not seen yet). About 22k reasoning tokens per draft are the
    report's main cost. A re-prompt starts
    a fresh conversation (prompt, rejected draft, failures), so the agent calls the tools again:
    a few more cached requests per report.

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
- **Licence:** the DLCAnalyzer repo is GPL-3.0. Download the label CSVs at eval time into a git-ignored folder,
  never commit them into the MIT repo. Credit the Zenodo videos (CC BY 4.0) in the README.

| What | Reference | Metric | Pass bar |
|---|---|---|---|
| Tracking | DeepLabCut output (reference, not truth) | Centroid distance to DLC body centre (cm), % frames tracked | Report it; check large disagreements by eye |
| Rearing (OFT) | 3 human raters, 20 videos | Per-video count ICC vs each rater; event-level F1 with ± tolerance; supported vs unsupported | Agreement with humans within the human–human range. **Not met (Oct 31):** count ICC 0.05 on the eval split against 0.90 between raters, so the pipeline makes no count ([ADR 0003](adr/0003-no-rear-counts-score-as-evidence.md)) |
| Floating (FST, stretch) | Human raters, 10 videos | Immobility time vs each rater | Within the human–human range |
| Head dips (EPM, stretch) | 3 raters, 5 videos | Count agreement | Report only, n is too small for a claim |
| Report | Computed tables | Number-audit pass rate | 100% |

Use a fixed train/eval split of the 20 open-field videos, **6 train / 14 eval**, drawn at random
with a fixed seed and recorded in `eval/split.json`. Few-shot examples come
only from the train split. The eval report is generated as `eval/REPORT.md` and its numbers feed
the demo and write-up, the same way PaedScale published its validation table.

**The harness (T12a, tracking).** `python -m eval fetch | plan | track [--yes]`, laptop only.

```
fetch:  labels CSV + DLC CSVs (DLCAnalyzer @ pinned commit) + eval videos (Zenodo) → eval/data/
plan:   eval/split.json ✓ → per-video projected SAM 3.1 spend, total
track:  plan printed → ask → per eval video:
          ingest → arena (floor prompts in turn, auto-confirmed) → track      [sees]
          our centroid vs DLC body centre → tracking/<ID>.json               [measures]
          live calls' usage and $ → spend.jsonl (also when no floor is found)
        → tracking/summary.json; exits non-zero if any eval video has no result
```

- **Guards, every command:** `eval.guard` refuses to run unless git ignores `eval/data/` and
  tracks nothing there or named like a DLCAnalyzer file (`AllLabData…`, `…DeepCut…`,
  `OFT_<n>_<rater>…`); a test runs the same check in CI. `eval.split` recomputes the seeded draw
  from the label file's IDs and refuses if `split.json` differs; nothing rewrites it.
- **No floor:** the arena stage leaves an empty arena for the scientist to draw (T10); the
  eval can't draw, so it deletes that attempt's run folder and tries the next floor prompt.
  When none finds a floor, `track` prints the video as skipped and it reads as missing.
- **Videos by ID:** a DLC file is named by its recording (`3_01_A_…DeepCut….csv` holds OFT_11),
  so it is found through the label rows' `DLCFile` column, never by the number in a name.
- **Matching frames:** our frame i meets DLC frame floor((i + 0.5) × 25 / 15), the source
  frame ingest's ffmpeg `fps` filter keeps (checked against ffmpeg on a frame-numbered video,
  Nov 3: round(i × 25 / 15) is one source frame off on a third of the frames). DLC pixels are scaled to the normalised video
  (1:1 for Sturman) and converted with our arena's px/cm. Frames where DLC's `bodycentre`
  likelihood is below 0.9 are left out (it is near 0 before the animal is placed). Assumes DLC
  ran on the 928×576 source; check its arena corners on a frame if distances look scaled.
- **Reported per video:** % of frames tracked, % with a confident DLC frame, frames compared,
  centroid distance in cm (mean, median, p95, max), % of compared frames ≥ 5 cm apart, the
  10 largest disagreements (frame, time) to check by eye, and an alignment check: DLC's row
  count beside the source's frame count, and the constant shift (±50 source frames) that
  brings the tracks closest. OFT_11's DLC has 15,253 rows for a ~609.5 s video, so a start
  offset is possible; a best shift far from 0 says the distances at 0 overstate. No pass
  bar: DLC is a reference.
- **Spend:** a metered transport wraps the Model API and logs each live response's usage
  (tokens, video frames, images) and price (`eval.spend.PRICES`); a cached response never
  reaches it, so it costs nothing. A model with no price is refused before it is called; a
  response with no usage, or a stream that broke off, is logged as an unreported call, not as
  free. The projection prices SAM 3.1's floor image and every frame
  at 15 fps in the chunks the segmenter would send (about $1.83 per video, $25.63 for the 14).
  Less if `runs/model_cache` (shared with the spike) already holds a video's responses; $0.0025
  more per extra floor prompt tried. Checked Nov 3 on the real
  label file and one DLC file: the split matches, OFT_11's DLC is 15,253 frames at 25 fps, its
  body centre confident on 99.2% of them.

**The report (T12b).** `python -m eval report`, laptop only (it reads the labels). It builds no
segmenter and no judge, so it costs nothing and can be run again whenever the code changes.

```
report: eval/split.json ✓ → per eval video with a tracked run (eval/data/runs/<ID>):
          track + confirmed arena → open_field_metrics (current MeasureSettings)   [measures]
          rear candidates (s, score) vs each rater's rears → eval.rearing           [measures]
        studies/*/report.json → shipped draft audited again vs tool outputs now     [measures]
        tracking/<ID>.json + spend.jsonl
        → eval/REPORT.md (committed) + eval/data/report.json (every number);
          exits non-zero unless a study report exists and every one passes again
```

- **Rearing, per rater, beside the raters against each other:** per-video count ICC(2,1)
  (Shrout & Fleiss) and event-level precision, recall and F1, pooled over the videos. Events
  match one-to-one (the most pairs, by `linear_sum_assignment`), a candidate window and a rear
  matching when they overlap or lie at most 0.5 s apart (`RearingSettings.tolerance_s`; the
  gap between the intervals, not between onsets, so a long window can match a rear inside it). The bar: every rater's
  agreement at least the lowest between two raters. Supported and unsupported rears both count
  as rears (the label set is binary).
- **Which count:** the pipeline makes none (ADR 0003). The eval counts the candidates scoring
  ≥ 0.55 (`RearingSettings.count_threshold`, set on the train split's counts by
  `spike.rear_scorer fit`), and every candidate beside it (report only).
- **Number-audit pass rate:** each study's stored report, its last draft audited again against
  the tables its tools return now, so a report gone stale fails. A report that can't be read,
  or whose tables can't (no design, a run gone), counts as not passing; with no report, the
  rate abstains and the command fails. Bar: 100%.
- **Not yet measured:** Muse Spark's rear labels against the raters and the score on the eval
  split (ADR 0003 point 3): it needs the behaviours stage on the eval videos, which is new
  spend. Supported vs unsupported: the label set is binary.
- **Abstain:** an eval video with no tracked run is left out and listed; a value that is
  undefined (no spread, no events) shows as "—".

## Application layer

- **Backend:** FastAPI (`api/`, T9a), served by `python -m pharmavision serve`.

  | Endpoint | Does |
  |---|---|
  | `POST /runs` | upload a video (multipart `video`) and a `template` name; 202, then ingest + arena run in the background |
  | `GET /runs`, `GET /runs/{id}` | status, the stage running (or the one that failed), the error, `stages_done`, and a URL per artifact present |
  | `GET /runs/{id}/artifacts/{name}` | `manifest`, `arena`, `track`, `metrics`, `events` or `qc` as JSON, validated on read; `GET /runs/{id}/video` the normalised MP4 |
  | `POST /runs/{id}/arena/confirm` | the gate, with an optional JSON body `{"floor_px": [[x, y], …]}`, the floor as edited or drawn: 202, then track → QC in the background; 409 unless the run is `awaiting zone confirmation`; 422, leaving it there, for a floor the zones can't come from, or none on an empty arena. The floor is checked before the gate is claimed, and the confirmed arena written before the response |
  | `GET /runs/{id}/background` | the background frame (PNG) the zone editor draws on |
  | `GET /runs/{id}/stream` | SSE: one `stage` event per finished stage (its `id` is the resume point for `Last-Event-ID`), then one `status` event once the run rests, and the stream ends |
  | `GET /studies` | every study folder with a `study.yaml`, as `GET /studies/{id}` shows it |
  | `POST /studies/{id}/report`, `GET /studies/{id}`, `GET /studies/{id}/artifacts/{name}` | the study's report job over `studies/<id>/study.yaml`: stage 7 (stats), then stage 8 (the report agent), `report` served as an artifact |

  - **Jobs:** FastAPI background tasks in the API's process, status in SQLite
    (`runs/jobs.sqlite3`): `queued`, `running`, `awaiting zone confirmation`, `done` or
    `failed` with the error message (a too-short video). One process
    only: on start-up, jobs a previous process left queued or running are marked failed. A
    failed run is not retried; processing the video again is a new run.
  - **Uploads** wait in `runs/.uploads/<run_id>/` until ingest has read them, then are
    deleted. A body over `max_upload_bytes` (1 GiB, the Files API limit) is refused (413) on
    its `Content-Length`, before it is spooled.
  - **The stream covers the run, not one job:** after the confirm, a stream opened without
    `Last-Event-ID` replays `ingest` and `arena` before the new stages.
  - **Precomputed runs** (`--precomputed DIR`) are listed and served beside the live ones
    with `read_only: true`. Their stages and status are read from the artifacts present (a
    folder stopped at the gate reads as `awaiting zone confirmation`). Confirming one is
    refused, and nothing calls a model. A study report reads its runs from `runs_dir` only,
    so a demo study over precomputed runs needs them there (T13).
- **Frontend:** video player with a canvas overlay (masks, trajectory, zones), a zone editor, an
  event timeline, a metrics table and the rendered report. This screen is the demo.
  - `frontend/`: Vite + React + TypeScript (strict), checked by `scripts/check.sh`
    (typecheck, ESLint, Vitest, build). It reads the API at `VITE_API_BASE` (default `/api`,
    which the dev server proxies to `:8000`). Three screens, by URL hash: the study and run
    lists, `#/runs/<id>`, the run viewer (T9b, T11), and `#/studies/<id>`, the report (T11).
  - **Zone editor (T10):** on a live run at the gate, the background frame with the floor as an
    SVG polygon over it. Its corners drag (or move with the arrow keys, Shift for 10 px); where
    no floor was found, four clicks draw it. The centre zone is a dashed preview that follows
    the floor (`zones.ts` mirrors `shrink_to_area_fraction`) and has no corners of its own.
    Confirm sends the floor; the backend's refusal reason is shown and the editor stays open.
    Once confirmed, the viewer follows the run's progress stream and reloads as stages finish.
  - **Frames are shown by index**, as the artifacts and the report write them (frame 0 is the
    first), with the time they start at. Seeking to frame i sets the video to the middle of
    its interval, `(i + 0.5) / fps`, so the player shows frame i and not the one before.
  - **Timeline (T11):** under the player, one bar per lane over the trial's frames: each
    behaviour event (green rearing, grey none, hatched when low-confidence), and each QC
    exclusion (red; a whole-trial exclusion spans the bar, hatched). A click seeks the video to the window's first frame. Below the
    metrics, the QC list has every exclusion in `qc.json` with its frames, reason and
    whether a rule or the model decided it; its frames seek too.
  - **Report view (T11):** `report.json`'s shipped Markdown, rendered by `react-markdown` with
    GFM (model-written text: no raw HTML is rendered). A remark plugin marks each
    `[TO FILL: …]` (`mark.to-fill`, dashed amber), each struck sentence (`del.struck`, red)
    and its `[failed the number audit]` flag, so neither reads as audited prose. A line above
    says how the last draft's audit went and how many items are left to fill.
  - **Overlay sync:** the canvas has the normalised video's pixel size, so artifact pixels are
    canvas pixels. Each frame is drawn inside `requestVideoFrameCallback`, keyed by the frame
    index of the presented frame (`floor(mediaTime × fps)`), and again on `seeked`. Checked on
    Nov 3 against a synthetic run whose video shows a disc on the tracked path: within 1 px on
    every presented frame of playback.
  - The overlay decodes each frame's COCO RLE mask in the browser and draws the zones and the
    last 10 s of the path; a gap breaks the path, never bridged. The path is the raw tracked
    centroids, not the median-filtered, gap-bridged path the measure stage computes metrics
    from, so it can show jitter and breaks the metrics never saw.
  - Missing data stays visible: a run's later artifacts load only once the run view lists
    them, and the viewer says which are missing ("no zones yet", "not tracked yet", "no
    metrics yet") instead of failing. The metric units copy `METRIC_UNITS` by hand until the
    API serves them.
- **Deploy (T13):** one Docker web service on Render (`Dockerfile`, `render.yaml`), one
  process. `serve --frontend frontend/dist` mounts the API under `/api`, where the built
  frontend's default `VITE_API_BASE` reaches it, and serves the frontend at `/`
  (`api.with_frontend`). SAM 3.1 and Muse Spark stay on the Meta API (`MODEL_API_KEY`, set on
  the host). Precomputed demo runs and their study ship in the image: `scripts/pack-demo.sh
  STUDY_ID` packs a finished study and its runs into a tarball, and the build fetches it from
  `DEMO_URL` into `demo/runs` (`--precomputed`) and `demo/studies` (`--studies-dir`), so the
  demo never waits on a model. Live runs, their jobs database and the model cache sit in the
  container's `runs/` and are lost on a redeploy.

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
   human-labelled rears before building stage 5 on them. **Candidates built (T5a, measure
   settings v2); recall measured Oct 30; decided: measure settings v3 favour recall, consensus
   recall 0.91 over 41% of the time ([ADR 0001](adr/0001-rear-candidates-favour-recall.md)).**
   `python -m spike.rear_recall [setting=value ...]`
   tracks the 6 train videos once (about $11 of SAM 3.1, cached), then counts the labelled rears
   that some candidate window overlaps, per rater, pooled and for consensus rears, and how many
   windows overlap a rear. Overlap, not containment, because stage 5 judges a 2–4 s clip cut
   around each window. Re-running with other thresholds costs nothing.

   Count check without labels (cloud, Oct 30), on the committed OFT_16 track (9,142 frames;
   the measure stage takes 21 s): the first defaults (area drop 0.2, elongation change 0.15,
   5 cm/s) gave 21 windows over 2% of the video, against about 85 rears per rater per video in
   the labels (5,093 rear rows / 20 videos / 3 raters). Recall-first defaults (0.1, 0.1,
   10 cm/s) give 110 windows over 16% of the time, mean 0.9 s. At the `fps: 15` clip price
   (about $0.016, open decision 3) that is about $1.80 of Muse Spark per video.

   | Area drop | Elongation change | Max speed | Windows | Time |
   |---|---|---|---|---|
   | 0.2 | 0.15 | 5 cm/s | 21 | 2.0% |
   | 0.15 | 0.15 | 10 cm/s | 61 | 6.8% |
   | 0.1 | 0.15 | 10 cm/s | 100 | 12.4% |
   | 0.1 | 0.1 | 10 cm/s | 110 | 15.8% (v2) |

   The goal set before measuring: stage 5 can only label rears that some candidate's clip
   shows, so candidate recall bounds rearing recall. Aim for consensus recall ≥ 0.9 with the
   candidates covering well under half of the video time.

   **Recall on the train split (laptop, Oct 30).** 6 videos, 503 consensus rears (at least two
   raters), equalised arenas (#27); SAM 3.1 tracking cost about $11. The three raters give
   recalls within 0.02 of each other at every setting, so only consensus is shown. Clip cost
   is windows × $0.016.

   | Area drop | Elongation change | Max speed | Min length | Consensus recall | Per video | Windows / video | Time | Muse Spark / video |
   |---|---|---|---|---|---|---|---|---|
   | 0.1 | 0.1 | 10 cm/s | 0.2 s | 0.59 (299 / 503) | 0.51–0.67 | 105 | 14.7% | $1.70 |
   | 0.05 | 0.1 | 10 cm/s | 0.2 s | 0.76 | 0.70–0.80 | 150 | 25.5% | $2.40 |
   | 0.05 | 0.05 | 15 cm/s | 0.1 s | 0.85 (427 / 503) | 0.79–0.90 | 175 | 33.4% | $2.80 |
   | **0.03** | **0.05** | **15 cm/s** | **0.1 s** | **0.91 (456 / 503), v3** | 0.85–0.96 | 182 | 40.7% | $2.90 |

   - **The area drop is the limit.** Only 78% of consensus rears have any frame with the mask
     10% below its baseline. Elongation (99.6%) and speed (100%) almost never exclude a rear.
     Reaching 0.9 takes a 3% drop, and then the windows cover 41% of the time.
   - 45–51% of windows overlap a consensus rear at every setting above.
   - A longer baseline (30, 60, 120 s in place of 10 s) lowers recall at every area drop.
   - The rows above come from `spike.rear_recall`. The per-rule reach and the baseline sweep
     came from a one-off sweep over the same stored tracks, not kept in the repo.

   **Rule:** measure settings v3 use the 0.91 row. Recall comes first; the candidates are a
   coarse filter and stage 5 carries the precision. "Well under half the time" is dropped
   (41%). More recall at less time needs a signal other than mask area, not more tuning.
3. **Muse Spark on short clips. Measured Oct 30; label set decided: binary rearing / none; model
   rear labels not trusted, few-shot or not.**
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

   **Cropped clips, with and without few-shot (laptop, Oct 30).** `python -m spike.muse_few_shot
   pool | build | score | again` cuts every clip the way stage 5 does (`cut_clips`: square crop,
   mask outlined, 256 px, 15 fps) from the train videos' run folders. The few-shot pool is 84
   rear-candidate clips from the 6 train videos: rears all three raters marked (≥ 0.6 s, mostly
   inside the clip) and hard negatives (no rater's rear within 1 s). It keeps 2 s away from the
   20 T2 clips and from OFT_16 540–600 s (the smoke run). From it, 4 examples, checked by eye:
   a supported and an unsupported rear, a turn at the wall and a turn on the spot (no still or
   sitting negative: the 5-video limit leaves room for two; one turn ends 2.4 s before a T2
   clip, just outside the margin). The other
   80 clips (34 rears, 46 hard negatives) are a held-out test of the same kind as the false
   positives. Contributor tier; the smoke run is standard, like its baseline.

   | Test | Zero-shot | Few-shot (4) |
   |---|---|---|
   | 20 T2 clips, cropped (16 rears) | acc 0.70, κ 0.06 | acc 0.70, κ 0.06 (6 labels flip) |
   | 80 held-out pool clips | acc 0.47, κ 0.04, precision 0.44, recall 0.85 | acc 0.62, κ 0.30, precision 0.53, recall 0.91 |
   | The same 80, asked again | acc 0.44, κ −0.03; 76% same label | acc 0.54, κ 0.13; 79% same label |
   | Smoke clip, OFT_16 540–600 s (20 candidates) | precision 6/16, consensus rears 7/7 | precision 3/10, consensus rears 3/7 |
   | Input tokens / clip | 2,300 | 9,900 |
   | Cost / 20 clips, standard (contributor) | $0.18 ($0.01) | $0.37 ($0.03) |

   - **Muse Spark is near chance on hard negatives and unstable:** a re-ask changes about 1
     label in 4, which is as large as the few-shot gain. Few-shot helps on the held-out pool,
     not on T2, and is worse on the smoke clip.
   - **The mask features already separate better.** On all 900 train-split candidates that are
     a consensus rear or a hard negative, a logistic fit on duration, area drop, elongation
     change and speed, scored leave-one-video-out, gives AUC 0.75, κ 0.37, precision 0.71,
     recall 0.74 (one-off check; the scorer's own spike will replace it with a kept script).

   **Rule ([ADR 0002](adr/0002-rear-count-measured-muse-second-opinion.md)):** few-shot stays
   off by default and the model's rear labels are not trusted. The rear count moves to a
   measured scorer on the mask features (measures layer), fitted on the train split and checked
   on the eval split (tracked now, about $25). Muse Spark stays as a second opinion: its label
   and rationale stay on each event and the eval reports its agreement, but it never sets the
   count. (The count part is superseded by ADR 0003, below.)

   **The measured scorer (laptop, Oct 31).** `python -m spike.rear_scorer track | fit | eval`.
   Every candidate (settings v4) is labelled from the raters: a rear when it overlaps a
   consensus rear, not a rear when no rater marked a rear within 1 s of it, and left out
   otherwise (so the negatives are clean ones, which flatters κ and precision); a ridge logistic regression over its 6 features is fitted on the 6 train
   videos, scored leave-one-video-out, then the fitted defaults run on the 14 eval videos
   (tracked for this, about $25; OFT_12 and OFT_52 took the floor prompt "white floor", OFT_39
   "floor", and px/cm stayed within 7.26–7.45 on all 20 videos). Counts use a threshold set on
   the train split's counts (0.55).

   | | Train, leave-one-video-out (900 candidates) | Eval split (2,056 candidates) |
   |---|---|---|
   | Per candidate | AUC 0.75, κ 0.37, precision 0.74, recall 0.67 | AUC 0.71, κ 0.29, precision 0.66, recall 0.74 |
   | Per-video count against the raters: ICC(2,1), Pearson r, mean abs. difference | 0.57, 0.53, 21% | **0.05, 0.07, 29%** |
   | Raters against each other | 0.97, 0.98, 6% | 0.90, 0.90, 9% |
   | Counting all candidates instead: ICC, r | 0.02, 0.34 | 0.02, 0.41 |
   | Summing the scores instead: ICC, r | 0.33, 0.44 | 0.11, 0.22 |

   Adding the mean mask area in cm² lifted the leave-one-video-out AUC from 0.70 to 0.75; a log
   duration added nothing. Windows rarely hold two consensus rears (2–11 per video), so the
   counts don't fail on merging: the errors are video-specific (OFT_23: 149 counted, the raters
   about 47).

   **Rule ([ADR 0003](adr/0003-no-rear-counts-score-as-evidence.md)):** the pipeline makes no
   rear count; the score and Muse Spark's label stay as per-window evidence, and FR-6 is
   reported as not met.
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
| Muse Spark takes at most 5 videos in one request (a 400 above it); assistant turns replaying earlier output work as few-shot turns. A repeated few-shot prefix is not cached (`cached_tokens` 0). A cropped 256 px, 2–4 s clip at `fps: 15` is about 2,300 input tokens, plus about 1,500 output tokens, mostly reasoning. | Live requests, `spike.muse_few_shot`, Oct 30 |
| Muse Spark function calling: strict `function` tools with no arguments are called in parallel (5 `function_call` items, `arguments` `"{}"`, read from `response.output_item.done`), and are accepted back without their reasoning item. A turn may carry a `message` in `phase: "commentary"`. One report draft took 2 requests: 1,532 in / 237 out, then 6,077 in / 23,189 out (22,439 reasoning), about 93 s, roughly $0.11 at standard prices. | Live study report, PR #35, Oct 31 |
