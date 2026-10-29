"""Videos over SAM 3.1's frame limit go in overlapping chunks. Object ids are only stable within
one response, so each chunk's objects are relinked to the previous chunk's by mask IoU over the
frames both chunks saw."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

# frame index -> object id -> full-frame mask
ObjectMasks = Mapping[int, Mapping[str, NDArray[np.bool_]]]


def plan_chunks(n_frames: int, max_frames: int, overlap: int) -> list[range]:
    """Frame ranges of at most `max_frames`, each starting `overlap` frames before the previous
    one ends, together covering 0..n_frames-1. One range when the video fits."""
    if not 0 <= overlap < max_frames:
        raise ValueError(f"overlap {overlap} must be below the chunk size {max_frames}")
    chunks = [range(0, min(max_frames, n_frames))]
    while chunks[-1].stop < n_frames:
        start = chunks[-1].stop - overlap
        chunks.append(range(start, min(start + max_frames, n_frames)))
    return chunks


def handover_frame(previous: range, current: range) -> int:
    """The first frame taken from `current`: the middle of the overlap, the frames furthest
    from either chunk's edge."""
    return current.start + (previous.stop - current.start) // 2


def relink(previous: ObjectMasks, current: ObjectMasks, min_iou: float) -> dict[str, str]:
    """Map each object id in `current` to the `previous` id it overlaps best, by mask IoU
    averaged over the frames where either appears (a frame with only one of them counts as 0,
    so a brief coincidence can't outscore a pair seen together throughout). Pairs are taken
    greedily, best first, and only at `min_iou` or above; an object with no partner is absent
    from the map."""
    old_frames, new_frames = _frames_of(previous), _frames_of(current)
    totals: dict[tuple[str, str], float] = {}
    for frame, current_objects in current.items():
        for old_id, old in previous.get(frame, {}).items():
            for new_id, new in current_objects.items():
                totals[old_id, new_id] = totals.get((old_id, new_id), 0.0) + _iou(old, new)
    scores = sorted(
        (
            (total / len(old_frames[old_id] | new_frames[new_id]), old_id, new_id)
            for (old_id, new_id), total in totals.items()
        ),
        reverse=True,
    )
    linked: dict[str, str] = {}
    for score, old_id, new_id in scores:
        if score >= min_iou and new_id not in linked and old_id not in linked.values():
            linked[new_id] = old_id
    return linked


def _frames_of(masks: ObjectMasks) -> dict[str, set[int]]:
    """The frames each object appears in."""
    frames: dict[str, set[int]] = {}
    for frame, objects in masks.items():
        for object_id in objects:
            frames.setdefault(object_id, set()).add(frame)
    return frames


def _iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
