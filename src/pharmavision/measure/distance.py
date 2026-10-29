"""Distance travelled along a trajectory."""

import itertools
import math
from collections.abc import Sequence

from .geometry import Point


def distance_travelled_cm(trajectory: Sequence[Point | None], px_per_cm: float) -> float | None:
    """Sum of straight-line steps between consecutive detected positions, in cm. Gaps (None) are
    bridged by a straight line. None when fewer than two positions were detected: the animal was
    not found, so the distance is unknown, not zero."""
    points = [p for p in trajectory if p is not None]
    if len(points) < 2:
        return None
    steps = (math.dist(a, b) for a, b in itertools.pairwise(points))
    return sum(steps) / px_per_cm
