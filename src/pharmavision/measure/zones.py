"""Time in a zone and entries into it, robust to jitter on the zone edge."""

import heapq
import itertools
from collections.abc import Sequence
from typing import NamedTuple

from .stretches import stretches


class ZoneOccupancy(NamedTuple):
    time_s: float
    entries: int


def zone_occupancy(
    signed_distances_px: Sequence[float | None],
    fps: float,
    hysteresis_px: float,
    min_dwell_s: float,
) -> ZoneOccupancy | None:
    """Occupancy from each frame's signed distance to the zone edge (positive inside), None
    where the position is unknown. None when it is known in no frame.

    Two guards keep jitter on the edge from counting: the animal enters only once it is
    `hysteresis_px` inside and leaves only once it is that far outside, and a visit or an absence
    shorter than `min_dwell_s` takes the state around it. Both restart after an unknown stretch,
    so a gap never joins what came before it to what comes after. Entries count the moves from
    outside to inside, across a gap too, so starting in the zone is not an entry."""
    states: list[bool] = []
    for known, stretch in itertools.groupby(signed_distances_px, key=lambda d: d is not None):
        if known:
            seen = [d for d in stretch if d is not None]
            states += _debounced(_with_hysteresis(seen, hysteresis_px), fps, min_dwell_s)
    if not states:
        return None
    entries = sum(1 for was, now in itertools.pairwise(states) if now and not was)
    return ZoneOccupancy(time_s=sum(states) / fps, entries=entries)


def _with_hysteresis(signed_distances_px: Sequence[float], hysteresis_px: float) -> list[bool]:
    inside = signed_distances_px[0] >= 0
    states: list[bool] = []
    for distance in signed_distances_px:
        if distance >= hysteresis_px:
            inside = True
        elif distance <= -hysteresis_px:
            inside = False
        states.append(inside)
    return states


def _debounced(states: list[bool], fps: float, min_dwell_s: float) -> list[bool]:
    """While some stretch of one state is shorter than the minimum dwell, the shortest (the
    earliest on a tie) flips and merges with its neighbours. Shortest first, so a visit broken
    by brief flickers survives while flickers alone vanish. A lone stretch stays as it is.

    A heap of stretch lengths over a linked list of stretches, so a track that flickers on
    every frame still takes O(n log n)."""
    found = stretches(states)
    value = [s.value for s in found]
    length = [s.length for s in found]
    before = [i - 1 for i in range(len(found))]
    after = [i + 1 if i + 1 < len(found) else -1 for i in range(len(found))]
    # Index order is time order: a merged stretch keeps its leftmost index.
    heap = [(n, i) for i, n in enumerate(length)]
    heapq.heapify(heap)
    alive = len(found)
    while heap and alive > 1:
        n, i = heapq.heappop(heap)
        if n != length[i]:
            continue  # merged away, or grown since this entry was pushed
        if n / fps >= min_dwell_s:
            break
        left, right = before[i], after[i]
        keep = left if left != -1 else i
        for gone in {i, right} - {keep, -1}:
            length[keep] += length[gone]
            length[gone] = 0
            alive -= 1
        value[keep] = not value[i]  # the flipped state, which its neighbours already had
        after[keep] = after[right] if right != -1 else -1
        if after[keep] != -1:
            before[after[keep]] = keep
        heapq.heappush(heap, (length[keep], keep))
    debounced: list[bool] = []
    i = 0
    while i != -1:
        debounced += [value[i]] * length[i]
        i = after[i]
    return debounced
