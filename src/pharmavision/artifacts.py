"""The typed run-folder contract. Every artifact is a Pydantic model; the type is the contract
between stages, and every read validates against it."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal, get_args

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .judge import HandSighting, QcVerdict, RearLabel
from .measure.geometry import Point
from .templates import TestTemplate


class ArenaNotConfirmedError(RuntimeError):
    pass


class Artifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VideoInfo(Artifact):
    width: int
    height: int
    fps: float
    duration_s: float


class SourceVideo(VideoInfo):
    filename: str
    sha256: str


class NormalisedVideo(VideoInfo):
    n_frames: int


class IngestSettings(Artifact):
    fps: int = 15
    """SAM 3.1 takes at most 15,000 frames per request; at 15 fps a 10-min video is 9,000."""
    max_side_px: int = 1280
    """Downscale (never upscale) so the longer side fits."""
    crf: int = 20
    """x264 quality: low enough loss that masks on the normalised video match the source."""
    preset: str = "veryfast"


class ArenaSettings(Artifact):
    background_samples: int = 25
    """Frames sampled for the per-pixel median background."""
    outline_tolerance: float = 0.01
    """Floor outline simplification, as a fraction of its perimeter."""
    contrast_clip_limit: float | None = Field(default=2.0, gt=0)
    """CLAHE clip limit for equalising the background frame the floor prompt sees; None sends
    it as is. SAM 3.1 drops a floor it scores under about 0.5, and the plain frame from two of
    the four Sturman cameras scored just under (Oct 30)."""
    contrast_tiles: int = Field(default=8, ge=1)
    """CLAHE equalises the frame as a grid of this many by this many tiles."""


class RearFeatures(Artifact):
    """What the mask did over a rear candidate's window, relative to the animal's usual size
    and shape there (the `rear_baseline_s` running medians)."""

    duration_s: float
    area_drop: float
    """The deepest drop of the mask area below its baseline, as a fraction (0.4: 40% smaller)."""
    mean_area_drop: float
    """The mean drop over the window."""
    elongation_change: float
    """The largest change of the elongation from its baseline, either way, as a fraction."""
    mean_speed_cm_s: float
    mean_area_cm2: float
    """The mean mask area: a rearing animal seen from above looks smaller."""


class RearScorer(Artifact):
    """A logistic score over a candidate's features, fitted on the train split's rater labels:
    1 / (1 + exp(-(intercept + sum of weight x feature))). Evidence per candidate only: summed
    or thresholded, it does not give a per-video rear count that agrees with the raters, so no
    count is made from it (ADR 0003)."""

    intercept: float
    weights: RearFeatures
    """One weight per feature, in that feature's units."""


class MeasureSettings(Artifact):
    """Every threshold the measure stage uses. Times are in seconds and distances in cm, so a
    setting means the same at any fps or frame size."""

    version: int = 4
    """Bump when a default or a metric's definition changes, so metrics from different versions
    are never compared as if alike."""
    median_window_frames: int = Field(default=5, ge=1)
    """Median filter on the centroid trajectory, in frames at the ingest fps. Odd."""
    max_gap_s: float = Field(default=0.5, ge=0)
    """Gaps up to this long are linearly interpolated; longer gaps stay gaps."""
    zone_hysteresis_cm: float = Field(default=1.0, ge=0)
    """Entering a zone takes this far inside its edge, leaving it this far outside."""
    min_dwell_s: float = Field(default=0.5, ge=0)
    """A zone visit (or an absence) shorter than this doesn't change the zone state."""
    thigmotaxis_wall_cm: float = Field(default=5.0, gt=0)
    """Near the wall: within this distance of the floor edge, about two mouse body widths."""
    immobility_change_max: float = Field(default=0.05, ge=0, le=1)
    """Immobile frame: below this fraction of mask pixels changed since the previous frame
    (symmetric difference over union)."""
    immobility_min_s: float = Field(default=1.0, ge=0)
    """Immobile frames count only in bouts at least this long."""
    rear_baseline_s: float = Field(default=10.0, gt=0)
    """The animal's usual mask area and elongation at each frame: their running median over
    this window, centred on the frame. Local, not one value per trial, because the wide-angle
    lens changes the animal's apparent size across the arena."""
    rear_area_drop: float = Field(default=0.03, gt=0, lt=1)
    """Rear candidate frame: the mask area is at least this fraction below its baseline, ..."""
    rear_elongation_change: float = Field(default=0.05, gt=0)
    """... the elongation (major over minor axis) differs from its baseline by at least this
    fraction, either way, ..."""
    rear_max_speed_cm_s: float = Field(default=15.0, ge=0)
    """... and the centroid moves at most this fast. The defaults (v3) favour recall over clip
    count, as measured in ARCHITECTURE open decision 2 and decided in ADR 0001: stage 5 judges
    every candidate, but can never find a rear that no candidate holds."""
    rear_merge_s: float = Field(default=0.5, ge=0)
    """A break of less than this between candidate frames joins them into one window, unless
    the position is unknown somewhere in the break, so a flickering rear is one candidate."""
    rear_min_s: float = Field(default=0.1, ge=0)
    """A candidate window lasts at least this long."""
    rear_scorer: RearScorer = RearScorer(
        intercept=3.74,
        weights=RearFeatures(
            duration_s=0.4837,
            area_drop=5.129,
            mean_area_drop=-7.986,
            elongation_change=0.5375,
            mean_speed_cm_s=-0.06692,
            mean_area_cm2=-0.1449,
        ),
    )
    """Fitted on the 6 train videos by `python -m spike.rear_scorer fit` (ADR 0003): longer
    windows with a deeper but brief area drop and a smaller mask score higher."""

    @model_validator(mode="after")
    def _median_window_is_odd(self) -> "MeasureSettings":
        if self.median_window_frames % 2 == 0:
            raise ValueError(f"median_window_frames must be odd, not {self.median_window_frames}")
        return self


class QcReviewSettings(Artifact):
    """How each flagged window is shown to the models: SAM 3.1 looks for hands on a few of its
    frames, and the judge sees a short whole-frame clip around it."""

    hand_prompts: tuple[str, ...] = ("hand", "glove")
    """SAM 3.1 takes one concept per request, so each prompt is its own call per frame."""
    hand_frames: int = Field(default=3, ge=1)
    """Frames of each window the hand prompts see, evenly spaced from its first to its last."""
    clip_pad_s: float = Field(default=1.0, ge=0)
    """Context before and after the window, so the model sees what came before the flag."""
    clip_min_s: float = Field(default=2.0, gt=0)
    """A clip is widened about the window to at least this long, ..."""
    clip_max_s: float = Field(default=6.0, gt=0)
    """... and a longer one is cut to this long, centred on the window: the model never sees the
    full video (cost control)."""
    clip_max_side_px: int = Field(default=480, ge=16)
    """The whole frame, not a crop around the animal (a hand comes in from the edge), scaled to
    fit this many pixels on its longer side."""

    @model_validator(mode="after")
    def _clip_lengths_are_ordered(self) -> "QcReviewSettings":
        if self.clip_min_s > self.clip_max_s:
            raise ValueError("clip_min_s must not exceed clip_max_s")
        return self


class QcSettings(Artifact):
    """Every threshold the deterministic QC flags use, in cm and seconds like `MeasureSettings`,
    and how the model reviews the flags."""

    version: int = 2
    """Bump when a default or a flag's definition changes."""
    max_tracking_loss_fraction: float = Field(default=0.05, ge=0, le=1)
    """A trial whose animal is missing on more than this fraction of frames is excluded
    (FR-4: at least 95% of frames tracked)."""
    out_of_arena_margin_cm: float = Field(default=2.0, ge=0)
    """Outside the arena: the centroid at least this far beyond the floor edge. The margin
    absorbs the floor outline's error at the walls."""
    second_mask_min_area_cm2: float = Field(default=10.0, gt=0)
    """A second object at least this large beside the animal is flagged: a hand or a second
    animal is far larger; droppings and reflections are smaller."""
    max_speed_cm_s: float = Field(default=100.0, gt=0)
    """Faster than a mouse runs in an open field, so a faster step between two detections is a
    mis-detection, not movement. Walking is a few cm/s."""
    speed_jump_join_s: float = Field(default=1.0, ge=0)
    """Jumps this close together join one flag, so a mis-detection that lasts several frames
    (a jump out, then one back) is flagged over all of its frames."""
    review: QcReviewSettings | None = QcReviewSettings()
    """None turns the model review off (the first item on the cut list): the deterministic flags
    and exclusions then stand alone, as they are with the review on and nothing flagged."""


class BehaviourSettings(Artifact):
    """How the behaviours stage cuts the clip it shows the judge for each rear candidate, and
    which judgements it marks as uncertain. Times in seconds and sizes in cm, like
    `MeasureSettings`."""

    version: int = 1
    """Bump when a default or the way a clip is cut changes."""
    clip_pad_s: float = Field(default=0.5, ge=0)
    """Context before and after the candidate window, so the model sees the animal go up and
    come down."""
    clip_min_s: float = Field(default=2.0, gt=0)
    """A clip is widened about the candidate to at least this long, ..."""
    clip_max_s: float = Field(default=4.0, gt=0)
    """... and a longer one is cut to this long, centred on the candidate. Muse Spark sees short
    clips only, never the full video (cost control)."""
    crop_margin_cm: float = Field(default=4.0, ge=0)
    """The clip is cropped to a square around the animal's masks over the clip, with this
    margin: on the whole frame the mouse fills about 2%, and zero-shot Muse Spark was near
    chance (T2)."""
    clip_side_px: int = Field(default=256, ge=16)
    """The square crop is scaled to this many pixels a side. Even, for H.264's 4:2:0 chroma."""
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    """A judgement below this confidence makes a low-confidence event, which the report lists
    as such."""

    @model_validator(mode="after")
    def _clip_lengths_are_ordered(self) -> "BehaviourSettings":
        if self.clip_min_s > self.clip_max_s:
            raise ValueError("clip_min_s must not exceed clip_max_s")
        if self.clip_side_px % 2:
            raise ValueError(f"clip_side_px must be even, not {self.clip_side_px}")
        return self


class Manifest(Artifact):
    run_id: str
    created_at: datetime
    code_version: str
    template: TestTemplate
    ingest: IngestSettings
    arena: ArenaSettings
    measure: MeasureSettings
    behaviours: BehaviourSettings = BehaviourSettings()
    """Defaulted so a run folder made before the behaviours stage existed still reads."""
    qc: QcSettings
    source: SourceVideo
    video: NormalisedVideo
    models: dict[str, str] = {}
    """Model id per stage that called one, e.g. {"arena": "sam-3.1"}."""


class Arena(Artifact):
    """Zones in pixels of the normalised video. Downstream stages refuse to run until a human
    has confirmed them. Empty (no floor, no centre, `px_per_cm` 0) when SAM 3.1 found no floor:
    the scientist draws it, and only a floor can be confirmed."""

    floor_px: list[Point]
    centre_px: list[Point]
    """Always derived from the floor in code, never drawn."""
    px_per_cm: float = Field(ge=0)
    """From the floor's area and the arena's real size: 0 with no floor."""
    confirmed: bool

    @model_validator(mode="after")
    def _whole_or_empty_and_only_a_floor_confirmed(self) -> "Arena":
        empty = (not self.floor_px, not self.centre_px, self.px_per_cm == 0)
        if any(empty) and not all(empty):
            raise ValueError("an arena has a floor, a centre and a scale, or none of them")
        if self.confirmed and all(empty):
            raise ValueError("an arena without a floor cannot be confirmed")
        return self


class TrackFrame(Artifact):
    """The animal in one frame. A gap (`detected` false) means no mask: missing data stays
    visible instead of being guessed here."""

    frame_index: int
    detected: bool
    x_px: float | None = None
    y_px: float | None = None
    area_px: int = 0
    confidence: float | None = None
    mask_rle: str | None = None
    second_area_px: int = 0
    """Area of the largest other mask in the frame (0 for none): one animal is expected, so a
    second object is something QC must look at, such as a hand."""

    @classmethod
    def gap(cls, frame_index: int) -> "TrackFrame":
        return cls(frame_index=frame_index, detected=False)

    @property
    def position(self) -> Point | None:
        if self.x_px is None or self.y_px is None:
            return None
        return (self.x_px, self.y_px)

    @model_validator(mode="after")
    def _detected_frames_have_a_mask(self) -> "TrackFrame":
        has_mask = None not in (self.x_px, self.y_px, self.mask_rle)
        if self.detected != has_mask:
            raise ValueError(
                f"frame {self.frame_index}: detected={self.detected} but mask={has_mask}"
            )
        return self


TRACK_SCHEMA = pa.schema(
    [
        ("frame_index", pa.int32()),
        ("detected", pa.bool_()),
        ("x_px", pa.float64()),
        ("y_px", pa.float64()),
        ("area_px", pa.int32()),
        ("confidence", pa.float64()),
        ("mask_rle", pa.string()),
        ("second_area_px", pa.int32()),
    ]
)


class Track(Artifact):
    """One record per frame of the normalised video, in frame-index order."""

    frames: list[TrackFrame]

    @model_validator(mode="after")
    def _one_record_per_frame(self) -> "Track":
        for position, frame in enumerate(self.frames):
            if frame.frame_index != position:
                raise ValueError(f"record {position} holds frame {frame.frame_index}")
        return self


class Gap(Artifact):
    """A run of frames where the animal was not found. Interpolated gaps were short enough to
    bridge with a straight line; the rest stay missing in every metric and are left for QC."""

    first_frame: int
    last_frame: int
    interpolated: bool


class RearCandidate(Artifact):
    """A window of frames where the animal may be rearing: the mask shrank and changed shape
    while the animal stayed in place. An input for the behaviours stage to judge, never a
    result."""

    first_frame: int
    last_frame: int
    features: RearFeatures
    score: float = Field(ge=0, le=1)
    """`MeasureSettings.rear_scorer` on the features: how likely the window holds a rear."""


class Metrics(Artifact):
    """Per-trial metrics. None means the metric could not be computed, never zero."""

    distance_travelled_cm: float | None
    mean_velocity_cm_s: float | None
    centre_time_s: float | None
    centre_entries: int | None
    thigmotaxis: float | None
    """Fraction of the time with a known position spent near the wall, 0 to 1."""
    immobility_s: float | None
    gaps: list[Gap]
    rear_candidates: list[RearCandidate]


class BehaviourEvent(Artifact):
    """One judged rear candidate over frames `first_frame` to `last_frame` (inclusive): the
    candidate's own window, though the model saw a clip around it. Every candidate gets one, so
    each verdict is auditable; it is a second opinion, never counted (ADR 0003). For a window
    longer than `BehaviourSettings.clip_max_s` the model saw only its middle: rears last about
    a second, so the label stands for the window."""

    first_frame: int
    last_frame: int
    label: RearLabel
    confidence: float
    low_confidence: bool
    """The confidence is under `BehaviourSettings.min_confidence`: reported as uncertain."""
    rationale: str
    source: Literal["rule", "model"]


EVENTS_SCHEMA = pa.schema(
    [
        ("first_frame", pa.int32()),
        ("last_frame", pa.int32()),
        ("label", pa.string()),
        ("confidence", pa.float64()),
        ("low_confidence", pa.bool_()),
        ("rationale", pa.string()),
        ("source", pa.string()),
    ]
)


class Events(Artifact):
    """Every behaviour event in the trial, in frame order."""

    events: list[BehaviourEvent]


QcFlagKind = Literal["tracking_loss", "out_of_arena", "speed_jump", "second_mask"]
Severity = Literal["warning", "critical"]


class QcFlag(Artifact):
    """Something in the trial that may make its numbers wrong, over frames `first_frame` to
    `last_frame` (inclusive). A critical flag leads to an exclusion; a warning is listed."""

    kind: QcFlagKind
    first_frame: int
    last_frame: int
    severity: Severity
    reason: str


class Exclusion(Artifact):
    """Frames (or the whole trial) left out of the statistics, and why."""

    scope: Literal["trial", "window"]
    first_frame: int
    last_frame: int
    reason: str
    source: Literal["rule", "model"]
    """A deterministic QC rule, or the model's review of a flagged window."""


class QcReview(Artifact):
    """The model review of one flagged window: what SAM 3.1's hand prompts found on its sampled
    frames, and the judge's verdict with its reason. Every verdict is listed, kept or not. For a
    flag longer than `QcReviewSettings.clip_max_s` the judge saw only its middle, and an
    `exclude_window` verdict still leaves out all of the flag's frames: the safer error."""

    flag: QcFlag
    hands: list[HandSighting]
    verdict: QcVerdict
    reason: str


class Qc(Artifact):
    """Every QC flag and every exclusion it led to, so nothing is dropped silently."""

    flags: list[QcFlag]
    exclusions: list[Exclusion]
    reviews: list[QcReview] = []
    """One per warning the models reviewed, in flag order; empty with the review off."""

    @property
    def trial_excluded(self) -> bool:
        return any(e.scope == "trial" for e in self.exclusions)


MetricName = Literal[
    "distance_travelled_cm",
    "mean_velocity_cm_s",
    "centre_time_s",
    "centre_entries",
    "thigmotaxis",
    "immobility_s",
]
"""The `Metrics` fields the stats compare between groups. No rear count: there is none to
compare (ADR 0003)."""
METRICS: tuple[MetricName, ...] = get_args(MetricName)
METRIC_UNITS: dict[MetricName, str] = {
    "distance_travelled_cm": "cm",
    "mean_velocity_cm_s": "cm/s",
    "centre_time_s": "s",
    "centre_entries": "entries",
    "thigmotaxis": "fraction",
    "immobility_s": "s",
}


class DoseGroup(Artifact):
    """One group of a study: the animals given one treatment at one dose."""

    name: str
    treatment: str
    dose: float = Field(ge=0)
    dose_unit: str


class StudyTrial(Artifact):
    """One trial of a study: the run that scored it, the code the scorer saw instead of the
    group, and the group it was in."""

    run_id: str
    blinding_code: str
    group: str


class StudyDesign(Artifact):
    """`study.yaml`: the groups, the trials and which group each trial was in."""

    name: str
    groups: list[DoseGroup] = Field(min_length=2)
    """In order: the first is the reference (usually the vehicle) that differences are taken
    from, and pairs are compared in this order."""
    trials: list[StudyTrial]

    @model_validator(mode="after")
    def _names_unique_and_groups_known(self) -> "StudyDesign":
        names = [g.name for g in self.groups]
        for kind, values in (
            ("group name", names),
            ("run", [t.run_id for t in self.trials]),
            ("blinding code", [t.blinding_code for t in self.trials]),
        ):
            repeated = sorted({v for v in values if values.count(v) > 1})
            if repeated:
                raise ValueError(f"repeated {kind}: {', '.join(repeated)}")
        unknown = sorted({t.group for t in self.trials} - set(names))
        if unknown:
            raise ValueError(f"trials in unknown groups: {', '.join(unknown)}")
        return self


class TrialRow(Artifact):
    """One trial's metrics, its group, and every exclusion QC made in it."""

    run_id: str
    blinding_code: str
    group: str
    values: dict[MetricName, float | None]
    """None: the metric could not be computed, never zero."""
    exclusions: list[Exclusion]

    @property
    def excluded(self) -> bool:
        return any(e.scope == "trial" for e in self.exclusions)

    def analysed_value(self, metric: MetricName) -> float | None:
        """The value the stats use: None for an excluded trial, or a metric not computed."""
        return None if self.excluded else self.values[metric]


class TrialTable(Artifact):
    """`trials.json`: the per-trial metrics table, one row per trial of the design, in its
    order. Excluded trials stay in it with their reasons; the stats leave them out."""

    trials: list[TrialRow]


class StatsSettings(Artifact):
    """Every threshold the test-selection rule uses."""

    version: int = 1
    """Bump when a default, the rule or an effect size's definition changes."""
    normality_alpha: float = Field(default=0.05, gt=0, lt=1)
    """A group is taken as normal when its Shapiro-Wilk p is above this."""
    variance_alpha: float = Field(default=0.05, gt=0, lt=1)
    """Variances are taken as equal when Levene's p (median-centred) is above this."""
    ci_level: float = Field(default=0.95, gt=0, lt=1)
    """The confidence level of every effect size's interval and the post-hoc intervals."""
    min_group_n: int = Field(default=3, ge=3)
    """With fewer trials left in any group the rule abstains: Shapiro-Wilk needs 3."""
    bootstrap_resamples: int = Field(default=9999, ge=100)
    """For the effect sizes whose interval is bootstrapped: eta² after Welch's ANOVA and
    epsilon² after Kruskal-Wallis."""
    bootstrap_seed: int = 0
    """Fixed, so the same table gives the same interval."""


TestName = Literal[
    "student_t", "welch_t", "mann_whitney_u", "anova", "welch_anova", "kruskal_wallis"
]
PosthocName = Literal["tukey_hsd", "games_howell", "dunn"]
EffectName = Literal["mean_difference", "rank_biserial", "eta_squared", "epsilon_squared"]


class GroupSummary(Artifact):
    """One group's values of one metric, over the trials the test used."""

    group: str
    n: int
    """Trials analysed: not excluded, and with the metric known."""
    n_excluded: int
    """Trials left out by a trial exclusion."""
    n_missing: int
    """Trials kept whose metric could not be computed."""
    mean: float | None
    sd: float | None
    """Sample standard deviation (n - 1); None under 2 trials."""
    median: float | None


class NormalityCheck(Artifact):
    """Shapiro-Wilk on one group's values."""

    group: str
    w: float
    p: float


class VarianceCheck(Artifact):
    """Levene's test across the groups, median-centred (Brown-Forsythe)."""

    statistic: float
    p: float


class EffectSize(Artifact):
    """How large the difference is, with its confidence interval at `StatsSettings.ci_level`.
    `mean_difference` is in the metric's unit, later group minus the reference;
    `rank_biserial` is positive when the later group's values tend to be larger."""

    name: EffectName
    estimate: float
    ci_low: float
    ci_high: float


class PairwiseComparison(Artifact):
    """One post-hoc pair after a test of 3 or more groups: `difference` is `group_b` minus
    `group_a` (in mean ranks for Dunn's test, which gives no interval), and `p_adjusted` is
    adjusted for the number of pairs."""

    group_a: str
    group_b: str
    difference: float
    statistic: float
    p_adjusted: float
    ci_low: float | None
    ci_high: float | None


class Comparison(Artifact):
    """The test the rule chose for one metric and what it gave."""

    test: TestName
    statistic: float
    """t, U (the later group's), F or H, following the test."""
    df: tuple[float, ...]
    """Degrees of freedom: one for t and H, two for F, none for U."""
    p: float
    effect: EffectSize
    posthoc: PosthocName | None = None
    pairwise: list[PairwiseComparison] = []


class MetricStats(Artifact):
    """One metric compared between the groups: the checks the rule read, why it chose the test
    it did, and the result. `comparison` None: the rule abstained, and `reason` says why."""

    metric: MetricName
    unit: str
    groups: list[GroupSummary]
    normality: list[NormalityCheck]
    variance: VarianceCheck | None
    reason: str
    comparison: Comparison | None


class StudyStats(Artifact):
    """`stats.json`: every metric compared, and the settings the rule used."""

    settings: StatsSettings
    metrics: list[MetricStats]


class AuditSettings(Artifact):
    """What the number audit lets through without a match in the computed tables."""

    version: int = 1
    """Bump when a default or the matching rule changes."""
    allowed_phrases: tuple[str, ...] = ("ARRIVE 2.0", "Essential 10", "SAM 3.1")
    """Fixed names that hold a number but state no result; their numbers are not audited.
    Matched whole and ignoring case. No "95% CI" here: its 95 must match the stats'
    `ci_level`, so a study run at another level fails the audit instead of passing."""


class KnownValue(Artifact):
    """One number the report agent's tools returned, and where it came from: a field path such
    as `stats.metrics[0].comparison.p`, or `len(qc.exclusions)` for a count of list items."""

    source: str
    value: float


class AuditedNumber(Artifact):
    """One number written in the prose, and the known value it matched. It matches when the
    known value, rounded to the written precision, gives the written number: within half a unit
    of its last written digit (3412.6 written as 3413). Magnitudes are compared, so "decreased
    by 1.9" matches a difference of -1.9; a percent matches a known value times 100."""

    text: str
    """As written, such as "38%" or "1.6e-3"."""
    value: float
    """Its magnitude, in the written units (38 for "38%")."""
    percent: bool
    span: tuple[int, int]
    """Its start and end offsets in the prose."""
    sentence: str
    sentence_span: tuple[int, int]
    """The offsets of the sentence it appears in, so the report can strike that sentence."""
    source: str | None
    """The matched known value's source, the closest one; None when nothing matched."""


class NumberAudit(Artifact):
    """The number audit of one piece of report prose: every number found in it, matched or
    not. Blocking: prose that has not passed never reaches the report (NFR-1)."""

    settings: AuditSettings
    numbers: list[AuditedNumber]
    """In the order they appear."""

    @property
    def unmatched(self) -> list[AuditedNumber]:
        """The numbers no computed table holds: the ones the report agent is re-prompted with."""
        return [n for n in self.numbers if n.source is None]

    @property
    def passed(self) -> bool:
        return not self.unmatched


class ReportSettings(Artifact):
    """How the report stage gates the agent's drafts."""

    version: int = 1
    """Bump when a default, a section or a tool's output changes."""
    max_drafts: int = Field(default=2, ge=1)
    """Drafts the audit sees: the first, then one re-prompt with its failing numbers. When the
    last fails too, the report ships with its failing sentences struck out and flagged."""
    audit: AuditSettings = AuditSettings()


class ReportDraft(Artifact):
    """One draft of the report: the agent's text, the report assembled from it, and its audit."""

    text: str
    """The agent's Markdown as written."""
    markdown: str
    """The report: the agent's text under each section's heading, with the stage's `[TO FILL]`
    placeholders. This is what the audit read."""
    audit: NumberAudit


class Report(Artifact):
    """`report.json`: every draft and its audit, and the report as shipped (`report.md`)."""

    settings: ReportSettings
    model_id: str
    drafts: list[ReportDraft] = Field(min_length=1)
    markdown: str
    """The last draft's report; when its audit failed, with each failing sentence struck out
    and flagged."""

    @property
    def passed(self) -> bool:
        return self.drafts[-1].audit.passed

    @property
    def struck(self) -> list[AuditedNumber]:
        """The numbers whose sentences were struck out: none when the last draft passed."""
        return self.drafts[-1].audit.unmatched


REPORT_MARKDOWN = "report.md"
"""The report as shipped, beside `report.json`: a rendering of it, not an artifact of its own."""

RunArtifactName = Literal["manifest", "arena", "track", "metrics", "events", "qc"]
StudyArtifactName = Literal["design", "trials", "stats", "report"]


@dataclass(frozen=True)
class _ArtifactFolder:
    """A folder of artifacts; every read validates against the artifact's type, and every
    write replaces the file whole, so a reader never sees half of one."""

    path: Path

    FILES: ClassVar[Mapping[str, str]] = {}
    """Each artifact's file in the folder, by artifact name: the folder layout."""

    def has(self, name: str) -> bool:
        """Whether the artifact has been written."""
        return (self.path / self.FILES[name]).is_file()

    def _read[A: Artifact](self, name: str, kind: type[A]) -> A:
        return kind.model_validate_json((self.path / self.FILES[name]).read_text())

    def _write(self, name: str, artifact: Artifact) -> None:
        text = artifact.model_dump_json(indent=2) + "\n"
        self._replace(name, lambda tmp: tmp.write_text(text))

    def _replace(self, name: str, write: Callable[[Path], object]) -> None:
        self._replace_file(self.FILES[name], write)

    def _replace_file(self, filename: str, write: Callable[[Path], object]) -> None:
        path = self.path / filename
        tmp = path.with_name(f".{path.name}.tmp")
        write(tmp)
        os.replace(tmp, path)


@dataclass(frozen=True)
class RunFolder(_ArtifactFolder):
    """One run's folder on disk, with typed reads and writes for each artifact."""

    FILES: ClassVar[Mapping[str, str]] = {
        "manifest": "manifest.json",
        "video": "video.mp4",
        "arena": "arena.json",
        "background": "background.png",
        "track": "track.parquet",
        "metrics": "metrics.json",
        "events": "events.parquet",
        "qc": "qc.json",
    }

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def video_path(self) -> Path:
        return self.path / self.FILES["video"]

    @property
    def background_path(self) -> Path:
        """The background frame the floor was looked for on, which the zone editor draws on:
        a PNG beside the artifacts, written by the arena stage."""
        return self.path / self.FILES["background"]

    def read(self, name: RunArtifactName) -> Artifact:
        """Any artifact by its name."""
        readers: dict[RunArtifactName, Callable[[], Artifact]] = {
            "manifest": self.manifest,
            "arena": self.arena,
            "track": self.track,
            "metrics": self.metrics,
            "events": self.events,
            "qc": self.qc,
        }
        return readers[name]()

    def manifest(self) -> Manifest:
        return self._read("manifest", Manifest)

    def write_manifest(self, manifest: Manifest) -> None:
        self._write("manifest", manifest)

    def arena(self) -> Arena:
        return self._read("arena", Arena)

    def confirmed_arena(self) -> Arena:
        """The arena gate. Every stage after arena reads the arena through this."""
        arena = self.arena()
        if not arena.confirmed:
            raise ArenaNotConfirmedError(f"run {self.run_id}: confirm the arena zones first")
        return arena

    def write_arena(self, arena: Arena) -> None:
        self._write("arena", arena)

    def track(self) -> Track:
        path = self.path / self.FILES["track"]
        rows = pq.read_table(path, schema=TRACK_SCHEMA).to_pylist()
        return Track.model_validate({"frames": rows})

    def write_track(self, track: Track) -> None:
        rows = [frame.model_dump() for frame in track.frames]
        table = pa.Table.from_pylist(rows, schema=TRACK_SCHEMA)
        self._replace("track", lambda tmp: pq.write_table(table, tmp))

    def metrics(self) -> Metrics:
        return self._read("metrics", Metrics)

    def write_metrics(self, metrics: Metrics) -> None:
        self._write("metrics", metrics)

    def events(self) -> Events:
        path = self.path / self.FILES["events"]
        rows = pq.read_table(path, schema=EVENTS_SCHEMA).to_pylist()
        return Events.model_validate({"events": rows})

    def write_events(self, events: Events) -> None:
        rows = [event.model_dump() for event in events.events]
        table = pa.Table.from_pylist(rows, schema=EVENTS_SCHEMA)
        self._replace("events", lambda tmp: pq.write_table(table, tmp))

    def qc(self) -> Qc:
        return self._read("qc", Qc)

    def write_qc(self, qc: Qc) -> None:
        self._write("qc", qc)


class StudyNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class StudyFolder(_ArtifactFolder):
    """One study's folder: its design (`study.yaml`, written by the scientist), the per-trial
    metrics table, the stats and the report."""

    FILES: ClassVar[Mapping[str, str]] = {
        "design": "study.yaml",
        "trials": "trials.json",
        "stats": "stats.json",
        "report": "report.json",
    }

    @property
    def study_id(self) -> str:
        return self.path.name

    def read(self, name: StudyArtifactName) -> Artifact:
        """Any artifact by its name."""
        readers: dict[StudyArtifactName, Callable[[], Artifact]] = {
            "design": self.design,
            "trials": self.trials,
            "stats": self.stats,
            "report": self.report,
        }
        return readers[name]()

    def design(self) -> StudyDesign:
        text = (self.path / self.FILES["design"]).read_text()
        return StudyDesign.model_validate(yaml.safe_load(text))

    def trials(self) -> TrialTable:
        return self._read("trials", TrialTable)

    def write_trials(self, table: TrialTable) -> None:
        self._write("trials", table)

    def stats(self) -> StudyStats:
        return self._read("stats", StudyStats)

    def write_stats(self, stats: StudyStats) -> None:
        self._write("stats", stats)

    def report(self) -> Report:
        return self._read("report", Report)

    def write_report(self, report: Report) -> None:
        """`report.json` with every draft, then `report.md`, the report as shipped."""
        self._write("report", report)
        self._replace_file(REPORT_MARKDOWN, lambda tmp: tmp.write_text(report.markdown))

    def remove_report(self) -> None:
        """Remove `report.md`, then `report.json`: a report quotes the stats it was written
        from, so new stats leave none behind."""
        (self.path / REPORT_MARKDOWN).unlink(missing_ok=True)
        (self.path / self.FILES["report"]).unlink(missing_ok=True)
