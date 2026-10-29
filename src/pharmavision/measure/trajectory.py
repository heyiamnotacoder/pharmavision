"""The animal's trajectory: centroids with short gaps interpolated and long gaps kept."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..artifacts import Gap
from .geometry import Point
from .stretches import stretches


class Trajectory(NamedTuple):
    points: list[Point | None]
    """One per frame; None where the animal was not found and the gap was too long to bridge."""
    gaps: list[Gap]


def clean_trajectory(
    positions: Sequence[Point | None], fps: float, max_gap_s: float, median_window: int
) -> Trajectory:
    """Linear interpolation across each gap that lasts at most `max_gap_s` and has a detection
    on both sides (a gap at the start or end of the track has nothing to interpolate from), then
    a median filter over each stretch without a gap, so one-frame mis-detections don't count as
    movement. The filter never reaches across a gap that was kept."""
    points = list(positions)
    gaps: list[Gap] = []
    for first, last in _gap_spans(positions):
        before = positions[first - 1] if first > 0 else None
        after = positions[last + 1] if last + 1 < len(positions) else None
        n = last - first + 1
        if before is None or after is None or n / fps > max_gap_s:
            gaps.append(Gap(first_frame=first, last_frame=last, interpolated=False))
            continue
        (x0, y0), (x1, y1) = before, after
        for k in range(1, n + 1):
            t = k / (n + 1)
            points[first + k - 1] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        gaps.append(Gap(first_frame=first, last_frame=last, interpolated=True))
    return Trajectory(_median_filtered(points, median_window), gaps)


def _median_filtered(points: list[Point | None], window: int) -> list[Point | None]:
    """Per-axis running median over each stretch of positions; the ends repeat their edge
    value, so a stretch keeps its length and its endpoints aren't pulled toward zero."""
    filtered = list(points)
    for stretch in stretches(p is None for p in points):
        if stretch.value:
            continue
        end = stretch.start + stretch.length
        xy = np.array(points[stretch.start : end], dtype=np.float64)
        padded = np.pad(xy, ((window // 2,) * 2, (0, 0)), "edge")
        medians: list[list[float]] = np.median(
            sliding_window_view(padded, window, axis=0), axis=-1
        ).tolist()
        filtered[stretch.start : end] = [(x, y) for x, y in medians]
    return filtered


def _gap_spans(positions: Sequence[Point | None]) -> list[tuple[int, int]]:
    """First and last frame of each stretch of missing positions."""
    return [
        (s.start, s.start + s.length - 1)
        for s in stretches(p is None for p in positions)
        if s.value
    ]
