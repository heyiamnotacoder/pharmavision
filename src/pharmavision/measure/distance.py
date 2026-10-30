"""Distance travelled along a trajectory."""

import itertools
import math
from collections.abc import Sequence

from .geometry import Point


def distance_travelled_cm(trajectory: Sequence[Point | None], px_per_cm: float) -> float | None:
    """Sum of straight-line steps between consecutive frames that both have a position, in cm.
    A gap left in the trajectory is not bridged: what happened there is unknown. None when there
    is no step at all, so a missing animal gives an unknown distance, not zero."""
    steps = _steps_px(trajectory)
    return sum(steps) / px_per_cm if steps else None


def mean_velocity_cm_s(
    trajectory: Sequence[Point | None], px_per_cm: float, fps: float
) -> float | None:
    """Distance over the time its steps took. None when there is no step."""
    steps = _steps_px(trajectory)
    return sum(steps) / px_per_cm / (len(steps) / fps) if steps else None


def speeds_cm_s(
    trajectory: Sequence[Point | None], px_per_cm: float, fps: float
) -> list[float | None]:
    """Each frame's centroid speed: the mean of its steps in from the frame before and out to
    the frame after, of those that exist. None where the frame has neither."""
    steps: list[float | None] = [
        math.dist(a, b) / px_per_cm * fps if a is not None and b is not None else None
        for a, b in itertools.pairwise(trajectory)
    ]
    speeds: list[float | None] = []
    for before, after in zip([None, *steps], [*steps, None], strict=True):
        known = [s for s in (before, after) if s is not None]
        speeds.append(sum(known) / len(known) if known else None)
    return speeds


def _steps_px(trajectory: Sequence[Point | None]) -> list[float]:
    return [
        math.dist(a, b)
        for a, b in itertools.pairwise(trajectory)
        if a is not None and b is not None
    ]
