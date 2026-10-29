"""Polygon geometry for zones and calibration. Pure functions on pixel coordinates."""

import math
from collections.abc import Sequence

Point = tuple[float, float]


def polygon_area(polygon: Sequence[Point]) -> float:
    """Unsigned area by the shoelace formula."""
    return abs(_signed_area(polygon))


def polygon_centroid(polygon: Sequence[Point]) -> Point:
    """Area centroid (not the vertex mean, which drifts toward densely sampled edges)."""
    area = _signed_area(polygon)
    cx = cy = 0.0
    for (x0, y0), (x1, y1) in _edges(polygon):
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return (cx / (6 * area), cy / (6 * area))


def shrink_to_area_fraction(polygon: Sequence[Point], fraction: float) -> list[Point]:
    """The polygon scaled about its centroid so its area is `fraction` of the original."""
    cx, cy = polygon_centroid(polygon)
    k = math.sqrt(fraction)
    return [(cx + k * (x - cx), cy + k * (y - cy)) for x, y in polygon]


def px_per_cm(floor: Sequence[Point], width_cm: float, height_cm: float) -> float:
    """Scale from the floor's pixel area and its real size. Using area, not edge length, keeps
    it independent of rotation and of how many vertices the polygon has."""
    return math.sqrt(polygon_area(floor) / (width_cm * height_cm))


def _signed_area(polygon: Sequence[Point]) -> float:
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in _edges(polygon)) / 2


def _edges(polygon: Sequence[Point]) -> list[tuple[Point, Point]]:
    return list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))
