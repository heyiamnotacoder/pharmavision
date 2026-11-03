"""Tracking against DeepLabCut (measures): our mask centroid per frame against DLC's body
centre, in cm. DLC is a reference tracker, not ground truth, so this reports agreement and
lists the largest disagreements to check by eye; it sets no pass bar."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pharmavision.artifacts import Track

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
    settings: TrackingSettings


def compare_tracking(
    video_id: str,
    track: Track,
    dlc: DlcTrack,
    fps: float,
    dlc_fps: float,
    scale: tuple[float, float],
    px_per_cm: float,
    settings: TrackingSettings,
) -> TrackingComparison:
    """`track` is at `fps` in normalised-video pixels; `dlc` at `dlc_fps` (the source rate) in
    source pixels, which `scale` (x, y) takes to normalised pixels. Our frame i, at i / fps,
    meets DLC frame round(i * dlc_fps / fps), the source frame ingest's fps filter kept."""
    n = len(track.frames)
    tracked = [f for f in track.frames if f.detected]
    dlc_index = np.rint(np.arange(n) * dlc_fps / fps).astype(np.int64)
    in_dlc = dlc_index < len(dlc.likelihood)
    confident = np.zeros(n, dtype=bool)
    confident[in_dlc] = dlc.likelihood[dlc_index[in_dlc]] >= settings.min_likelihood

    frame_ids: list[int] = []
    distances: list[float] = []
    for frame in tracked:
        i = frame.frame_index
        if not confident[i] or frame.x_px is None or frame.y_px is None:
            continue
        k = dlc_index[i]
        dx = frame.x_px - dlc.x[k] * scale[0]
        dy = frame.y_px - dlc.y[k] * scale[1]
        frame_ids.append(i)
        distances.append(float(np.hypot(dx, dy)) / px_per_cm)

    summary: DistanceSummary | None = None
    over: float | None = None
    largest: list[Disagreement] = []
    if distances:
        d = np.array(distances)
        summary = DistanceSummary(
            mean=float(d.mean()),
            median=float(np.median(d)),
            p95=float(np.percentile(d, 95)),
            max=float(d.max()),
        )
        over = 100 * float((d >= settings.disagreement_cm).mean())
        ranked = sorted(zip(distances, frame_ids, strict=True), key=lambda p: (-p[0], p[1]))
        largest = [
            Disagreement(frame_index=i, time_s=i / fps, distance_cm=dist)
            for dist, i in ranked[: settings.largest_n]
            if dist >= settings.disagreement_cm
        ]
    return TrackingComparison(
        video_id=video_id,
        frames=n,
        frames_tracked=len(tracked),
        tracked_pct=100 * len(tracked) / n if n else 0.0,
        dlc_confident_pct=100 * float(confident.mean()) if n else 0.0,
        frames_compared=len(distances),
        distance_cm=summary,
        over_disagreement_pct=over,
        largest=largest,
        settings=settings,
    )
