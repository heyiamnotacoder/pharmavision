"""Immobility: sustained stretches where the animal's mask barely changes."""

from collections.abc import Sequence

from .stretches import stretches


def immobility_s(
    changes: Sequence[float | None], fps: float, change_max: float, min_bout_s: float
) -> float | None:
    """Time in immobile bouts that last at least `min_bout_s`. `changes[i]` is the mask change
    from frame i to frame i + 1, None where either mask is missing. A bout is a stretch of
    consecutive changes below `change_max`; k such changes join k + 1 frames, and the bout lasts
    those frames. None when no change is known at all."""
    if all(c is None for c in changes):
        return None
    still = (c is not None and c < change_max for c in changes)
    bouts = (s.length + 1 for s in stretches(still) if s.value)
    return sum(n for n in bouts if n / fps >= min_bout_s) / fps
