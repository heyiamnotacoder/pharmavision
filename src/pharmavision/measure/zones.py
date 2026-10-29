"""Time in a zone and entries into it, robust to jitter on the zone edge."""

import itertools
from collections.abc import Sequence
from typing import NamedTuple


class ZoneOccupancy(NamedTuple):
    time_s: float
    entries: int


def zone_occupancy(
    depths_px: Sequence[float], fps: float, hysteresis_px: float, min_dwell_s: float
) -> ZoneOccupancy | None:
    """Occupancy from each frame's depth inside the zone edge (negative outside), over the
    frames where the animal's position is known. None when it is known in no frame.

    Two guards keep jitter on the edge from counting: the animal enters only once it is
    `hysteresis_px` inside and leaves only once it is that far outside, and a visit or an absence
    shorter than `min_dwell_s` leaves the state as it was. Entries count the moves from outside to
    inside, so starting in the zone is not an entry."""
    if not depths_px:
        return None
    states = _debounced(_with_hysteresis(depths_px, hysteresis_px), fps, min_dwell_s)
    entries = sum(1 for was, now in itertools.pairwise(states) if now and not was)
    return ZoneOccupancy(time_s=sum(states) / fps, entries=entries)


def _with_hysteresis(depths_px: Sequence[float], hysteresis_px: float) -> list[bool]:
    inside = depths_px[0] >= 0
    states: list[bool] = []
    for depth in depths_px:
        if depth >= hysteresis_px:
            inside = True
        elif depth <= -hysteresis_px:
            inside = False
        states.append(inside)
    return states


def _debounced(states: list[bool], fps: float, min_dwell_s: float) -> list[bool]:
    """Each run of one state shorter than the minimum dwell takes the state before it; the
    first run stands, as there is nothing before it."""
    debounced: list[bool] = []
    for state, run in itertools.groupby(states):
        n = len(list(run))
        if not debounced or n / fps >= min_dwell_s:
            debounced.extend([state] * n)
        else:
            debounced.extend([debounced[-1]] * n)
    return debounced
