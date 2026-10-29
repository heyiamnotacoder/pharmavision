"""The typed run-folder contract. Every artifact is a Pydantic model; the type is the contract
between stages, and every read validates against it."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, model_validator

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


class Manifest(Artifact):
    run_id: str
    created_at: datetime
    code_version: str
    template: TestTemplate
    ingest: IngestSettings
    arena: ArenaSettings
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


class Metrics(Artifact):
    """Per-trial metrics. None means the metric could not be computed, never zero."""

    distance_travelled_cm: float | None


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

    def _read[A: Artifact](self, name: str, kind: type[A]) -> A:
        return kind.model_validate_json((self.path / name).read_text())

    def _write(self, name: str, artifact: Artifact) -> None:
        (self.path / name).write_text(artifact.model_dump_json(indent=2) + "\n")
