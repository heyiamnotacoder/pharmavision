"""Tracking against DeepLabCut (measures): our mask centroid per frame against DLC's body
centre, in cm. DLC is a reference tracker, not ground truth, so this reports agreement and
lists the largest disagreements to check by eye; it sets no pass bar."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from pharmavision.artifacts import Arena, Manifest, Track

from .dlc import DlcTrack


class TrackingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_likelihood: float = Field(default=0.9, ge=0, le=1)
    """DLC frames below this likelihood are left out: DLC was unsure where the body centre was
    (before the animal is placed in the arena, it is near 0)."""
    disagreement_cm: float = Field(default=5.0, gt=0)
    """A frame this far from DLC counts as a disagreement: about half a mouse's body length."""
    largest_n: int = Field(default=10, ge=0)
    """How many of the largest disagreements to list for checking by eye."""
    max_lag_frames: int = Field(default=50, ge=0)
    """The alignment check shifts DLC up to this many source frames either way (2 s at 25 fps)."""


class VideoMapping(BaseModel):
    """How a run's normalised frames and pixels relate to the source video DLC ran on."""

    model_config = ConfigDict(frozen=True)

    fps: float
    source_fps: float
    source_frames: int
    """Source frames by the container's duration: what DLC's row count should be."""
    scale_x: float
    """Normalised pixels per source pixel (1 for Sturman: ingest keeps 928 x 576)."""
    scale_y: float
    px_per_cm: float
    """From the run's confirmed arena, in normalised pixels."""

    @classmethod
    def of(cls, manifest: Manifest, arena: Arena) -> "VideoMapping":
        source, video = manifest.source, manifest.video
        return cls(
            fps=video.fps,
            source_fps=source.fps,
            source_frames=round(source.duration_s * source.fps),
            scale_x=video.width / source.width,
            scale_y=video.height / source.height,
            px_per_cm=arena.px_per_cm,
        )


class DistanceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: float
    median: float
    p95: float
    max: float


class Disagreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_index: int
    """A frame of the normalised video."""
    time_s: float
    distance_cm: float


class LagCheck(BaseModel):
    """The constant shift of DLC, in source frames, that brings the two tracks closest. Far
    from 0, the two disagree on when the video starts, and the distances at 0 overstate."""

    model_config = ConfigDict(frozen=True)

    lag_frames: int
    median_cm: float


class TrackingComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    frames: int
    frames_tracked: int
    tracked_pct: float
    """Frames with the animal's mask, of all frames of the normalised video."""
    dlc_confident_pct: float
    """Frames whose DLC frame has a body centre at `min_likelihood` or above, of all frames."""
    frames_compared: int
    """Frames both trackers placed the animal in."""
    distance_cm: DistanceSummary | None
    """None when no frame could be compared: the comparison abstains."""
    over_disagreement_pct: float | None
    """Compared frames at least `disagreement_cm` apart."""
    largest: list[Disagreement]
    dlc_frames: int
    source_frames: int
    """DLC's row count beside the source's frame count: a gap may mean a start offset."""
    best_lag: LagCheck | None
    """None when no frame could be compared."""
    settings: TrackingSettings


def source_frame(
    frame_index: NDArray[np.int64], fps: float, source_fps: float
) -> NDArray[np.int64]:
    """The source frame ingest's ffmpeg `fps` filter kept as each output frame: the last one
    starting at or before the middle of the output frame, floor((i + 0.5) * source / fps).
    Checked against ffmpeg itself (tests/test_eval_tracking.py); round(i * source / fps) is one
    source frame off on a third of the frames at 25 -> 15 fps."""
    return np.floor((frame_index + 0.5) * source_fps / fps).astype(np.int64)


def compare_tracking(
    video_id: str,
    track: Track,
    dlc: DlcTrack,
    mapping: VideoMapping,
    settings: TrackingSettings,
) -> TrackingComparison:
    """Our frame i meets DLC frame `source_frame(i, …)`, the source frame ingest kept for it;
    DLC's source pixels are scaled to the normalised video and converted with our px/cm."""
    n = len(track.frames)
    frames = np.arange(n)
    detected = np.array([f.detected for f in track.frames], dtype=bool)
    x = np.array([np.nan if f.x_px is None else f.x_px for f in track.frames], dtype=np.float64)
    y = np.array([np.nan if f.y_px is None else f.y_px for f in track.frames], dtype=np.float64)
    kept = source_frame(frames, mapping.fps, mapping.source_fps)

    def distances_at(lag: int) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Frames both trackers placed the animal in with DLC shifted `lag` source frames, and
        their distances in cm."""
        k = kept + lag
        usable = detected & (k >= 0) & (k < len(dlc.likelihood))
        usable[usable] = dlc.likelihood[k[usable]] >= settings.min_likelihood
        ids, k = frames[usable], k[usable]
        dx = x[ids] - dlc.x[k] * mapping.scale_x
        dy = y[ids] - dlc.y[k] * mapping.scale_y
        return ids, np.hypot(dx, dy) / mapping.px_per_cm

    in_dlc = kept < len(dlc.likelihood)
    confident = np.zeros(n, dtype=bool)
    confident[in_dlc] = dlc.likelihood[kept[in_dlc]] >= settings.min_likelihood
    ids, d = distances_at(0)

    summary: DistanceSummary | None = None
    over: float | None = None
    largest: list[Disagreement] = []
    best_lag: LagCheck | None = None
    if len(d):
        summary = DistanceSummary(
            mean=float(d.mean()),
            median=float(np.median(d)),
            p95=float(np.percentile(d, 95)),
            max=float(d.max()),
        )
        over = 100 * float((d >= settings.disagreement_cm).mean())
        order = np.lexsort((ids, -d))[: settings.largest_n]
        largest = [
            Disagreement(
                frame_index=int(ids[j]),
                time_s=int(ids[j]) / mapping.fps,
                distance_cm=float(d[j]),
            )
            for j in order
            if d[j] >= settings.disagreement_cm
        ]
        # Nearest-to-zero first, so a tie keeps the smaller shift; lag 0 always has distances.
        lags = sorted(range(-settings.max_lag_frames, settings.max_lag_frames + 1), key=abs)
        shifted = ((lag, distances_at(lag)[1]) for lag in lags)
        best_lag = min(
            (
                LagCheck(lag_frames=lag, median_cm=float(np.median(dl)))
                for lag, dl in shifted
                if len(dl)
            ),
            key=lambda check: check.median_cm,
        )
    return TrackingComparison(
        video_id=video_id,
        frames=n,
        frames_tracked=int(detected.sum()),
        tracked_pct=100 * float(detected.mean()) if n else 0.0,
        dlc_confident_pct=100 * float(confident.mean()) if n else 0.0,
        frames_compared=len(d),
        distance_cm=summary,
        over_disagreement_pct=over,
        largest=largest,
        dlc_frames=len(dlc.likelihood),
        source_frames=mapping.source_frames,
        best_lag=best_lag,
        settings=settings,
    )
