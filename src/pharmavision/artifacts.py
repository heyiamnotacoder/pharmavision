"""The typed run-folder contract. Every artifact is a Pydantic model; the type is the contract
between stages, and every read validates against it."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class MeasureSettings(Artifact):
    """Every threshold the measure stage uses. Times are in seconds and distances in cm, so a
    setting means the same at any fps or frame size."""

    version: int = 2
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
    rear_area_drop: float = Field(default=0.1, gt=0, lt=1)
    """Rear candidate frame: the mask area is at least this fraction below its baseline, ..."""
    rear_elongation_change: float = Field(default=0.1, gt=0)
    """... the elongation (major over minor axis) differs from its baseline by at least this
    fraction, either way, ..."""
    rear_max_speed_cm_s: float = Field(default=10.0, ge=0)
    """... and the centroid moves at most this fast. The defaults favour recall (about 110
    windows over 16% of the time on Sturman OFT_16): stage 5 judges every candidate, but can
    never find a rear that no candidate holds."""
    rear_merge_s: float = Field(default=0.5, ge=0)
    """A break of less than this between candidate frames joins them into one window, unless
    the position is unknown somewhere in the break, so a flickering rear is one candidate."""
    rear_min_s: float = Field(default=0.2, ge=0)
    """A candidate window lasts at least this long."""

    @model_validator(mode="after")
    def _median_window_is_odd(self) -> "MeasureSettings":
        if self.median_window_frames % 2 == 0:
            raise ValueError(f"median_window_frames must be odd, not {self.median_window_frames}")
        return self


class QcSettings(Artifact):
    """Every threshold the deterministic QC flags use, in cm and seconds like `MeasureSettings`."""

    version: int = 1
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


class Manifest(Artifact):
    run_id: str
    created_at: datetime
    code_version: str
    template: TestTemplate
    ingest: IngestSettings
    arena: ArenaSettings
    measure: MeasureSettings
    qc: QcSettings
    source: SourceVideo
    video: NormalisedVideo
    models: dict[str, str] = {}
    """Model id per stage that called one, e.g. {"arena": "sam-3.1"}."""


class Arena(Artifact):
    """Zones in pixels of the normalised video. Downstream stages refuse to run until a human
    has confirmed them."""

    floor_px: list[Point]
    centre_px: list[Point]
    px_per_cm: float
    confirmed: bool


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


class Qc(Artifact):
    """Every QC flag and every exclusion it led to, so nothing is dropped silently."""

    flags: list[QcFlag]
    exclusions: list[Exclusion]

    @property
    def trial_excluded(self) -> bool:
        return any(e.scope == "trial" for e in self.exclusions)


@dataclass(frozen=True)
class RunFolder:
    """One run's folder on disk, with typed reads and writes for each artifact."""

    path: Path

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def video_path(self) -> Path:
        return self.path / "video.mp4"

    def manifest(self) -> Manifest:
        return self._read("manifest.json", Manifest)

    def write_manifest(self, manifest: Manifest) -> None:
        self._write("manifest.json", manifest)

    def arena(self) -> Arena:
        return self._read("arena.json", Arena)

    def confirmed_arena(self) -> Arena:
        """The arena gate. Every stage after arena reads the arena through this."""
        arena = self.arena()
        if not arena.confirmed:
            raise ArenaNotConfirmedError(f"run {self.run_id}: confirm the arena zones first")
        return arena

    def write_arena(self, arena: Arena) -> None:
        self._write("arena.json", arena)

    def track(self) -> Track:
        rows = pq.read_table(self.path / "track.parquet", schema=TRACK_SCHEMA).to_pylist()
        return Track.model_validate({"frames": rows})

    def write_track(self, track: Track) -> None:
        rows = [frame.model_dump() for frame in track.frames]
        pq.write_table(pa.Table.from_pylist(rows, schema=TRACK_SCHEMA), self.path / "track.parquet")

    def metrics(self) -> Metrics:
        return self._read("metrics.json", Metrics)

    def write_metrics(self, metrics: Metrics) -> None:
        self._write("metrics.json", metrics)

    def qc(self) -> Qc:
        return self._read("qc.json", Qc)

    def write_qc(self, qc: Qc) -> None:
        self._write("qc.json", qc)

    def _read[A: Artifact](self, name: str, kind: type[A]) -> A:
        return kind.model_validate_json((self.path / name).read_text())

    def _write(self, name: str, artifact: Artifact) -> None:
        (self.path / name).write_text(artifact.model_dump_json(indent=2) + "\n")
