"""Immobility: sustained stretches where the animal's mask barely changes."""

import itertools
from collections.abc import Sequence


def immobility_s(
    changes: Sequence[float | None], fps: float, change_max: float, min_bout_s: float
) -> float | None:
    """Time in bouts of immobile frames that last at least `min_bout_s`. `changes[i]` is the
    mask change from frame i to frame i + 1, None where either mask is missing. A frame is
    immobile when its change since the previous frame is below `change_max`; frame 0, which has
    no previous frame, takes the change to frame 1. None when no change is known at all."""
    if all(c is None for c in changes):
        return None
    immobile = [c is not None and c < change_max for c in [changes[0], *changes]]
    bouts = (len(list(run)) for still, run in itertools.groupby(immobile) if still)
    return sum(n for n in bouts if n / fps >= min_bout_s) / fps
