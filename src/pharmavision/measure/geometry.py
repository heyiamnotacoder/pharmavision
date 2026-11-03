"""Polygon geometry for zones and calibration. Pure functions on pixel coordinates."""

import math
from collections.abc import Sequence

import numpy as np
from shapely.geometry import Polygon

Point = tuple[float, float]


class InvalidFloorError(ValueError):
    pass


def check_floor(floor: Sequence[Point], width: int, height: int) -> None:
    """Refuse a floor the zones can't be derived from: fewer than 3 corners, a corner off the
    `width` x `height` frame, edges that cross, or no area."""
    if len(floor) < 3:
        raise InvalidFloorError(f"a floor needs at least 3 corners, not {len(floor)}")
    off = [p for p in floor if not (0 <= p[0] <= width and 0 <= p[1] <= height)]
    if off:
        raise InvalidFloorError(f"corner {off[0]} is outside the {width} x {height} px frame")
    polygon = Polygon(floor)
    if polygon.convex_hull.area == 0:
        raise InvalidFloorError("the floor has no area: its corners are on one line")
    if not polygon.is_valid:
        raise InvalidFloorError("the floor's edges cross")


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


def signed_distance(points: Sequence[Point], polygon: Sequence[Point]) -> list[float]:
    """Distance from each point to the polygon's edge: positive inside, negative outside."""
    if not points:
        return []
    p = np.asarray(points, dtype=np.float64)[:, None, :]  # (points, 1, xy)
    a = np.asarray(polygon, dtype=np.float64)[None, :, :]  # (1, edges, xy)
    ab = np.roll(a, -1, axis=1) - a
    length_sq = (ab**2).sum(axis=-1)
    ap = p - a
    # Closest point on each edge: the projection, clamped to the segment.
    t = np.clip((ap * ab).sum(axis=-1) / np.where(length_sq > 0, length_sq, 1.0), 0.0, 1.0)
    distance = np.sqrt(((ap - t[..., None] * ab) ** 2).sum(axis=-1)).min(axis=1)
    # Even-odd rule: a ray to the right of the point crosses the edge.
    ay, by, ax = a[..., 1], a[..., 1] + ab[..., 1], a[..., 0]
    py, px = p[..., 1], p[..., 0]
    straddles = (ay > py) != (by > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing_x = ax + (py - ay) * ab[..., 0] / ab[..., 1]
    inside = (straddles & (px < crossing_x)).sum(axis=1) % 2 == 1
    signed: list[float] = np.where(inside, distance, -distance).tolist()
    return signed


def signed_distance_per_frame(
    points: Sequence[Point | None], polygon: Sequence[Point]
) -> list[float | None]:
    """`signed_distance` for each frame's point, in frame order, None where it is unknown."""
    known = iter(signed_distance([p for p in points if p is not None], polygon))
    return [None if p is None else next(known) for p in points]


def _signed_area(polygon: Sequence[Point]) -> float:
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in _edges(polygon)) / 2


def _edges(polygon: Sequence[Point]) -> list[tuple[Point, Point]]:
    return list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))
