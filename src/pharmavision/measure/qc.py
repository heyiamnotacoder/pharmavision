"""Deterministic QC flags: where the track may be wrong, and which frames that excludes. Flags
read the raw track, not the cleaned trajectory, so nothing the filters smooth away is missed."""

import itertools
import math
from collections.abc import Sequence

from ..artifacts import (
    Arena,
    Exclusion,
    Gap,
    NormalisedVideo,
    Qc,
    QcFlag,
    QcFlagKind,
    QcSettings,
    Severity,
    Track,
)
from .geometry import signed_distance_per_frame
from .stretches import stretches


def trial_qc(
    track: Track,
    arena: Arena,
    gaps: Sequence[Gap],
    video: NormalisedVideo,
    settings: QcSettings,
) -> Qc:
    """Every flag for one trial, in frame order per kind, and the exclusions the critical ones
    lead to. `gaps` are the measure stage's: only gaps it could not interpolate are flagged."""
    n_frames = len(track.frames)
    flags = [_kept_gap(gap, video.fps, n_frames) for gap in gaps if not gap.interpolated]
    flags += _out_of_arena(track, arena, settings.out_of_arena_margin_cm)
    flags += _speed_jumps(
        track, arena.px_per_cm, video.fps, settings.max_speed_cm_s, settings.speed_jump_join_s
    )
    flags += _second_masks(track, arena.px_per_cm, settings.second_mask_min_area_cm2)
    exclusions: list[Exclusion] = []
    loss = _trial_loss(track, settings.max_tracking_loss_fraction)
    if loss is not None:
        flags.append(loss)
        exclusions.append(
            Exclusion(
                scope="trial",
                first_frame=loss.first_frame,
                last_frame=loss.last_frame,
                reason=loss.reason,
                source="rule",
            )
        )
    return Qc(flags=flags, exclusions=exclusions)


def _kept_gap(gap: Gap, fps: float, n_frames: int) -> QcFlag:
    seconds = (gap.last_frame - gap.first_frame + 1) / fps
    if gap.first_frame == 0:
        why = "at the start of the video, with no frame before it to interpolate from"
    elif gap.last_frame == n_frames - 1:
        why = "at the end of the video, with no frame after it to interpolate from"
    else:
        why = "too long to interpolate"
    reason = f"animal not found for {seconds:.1f} s, {why}"
    return _flag("tracking_loss", gap.first_frame, gap.last_frame, reason)


def _trial_loss(track: Track, max_fraction: float) -> QcFlag | None:
    """A critical flag over the whole trial when the animal is missing on more than
    `max_fraction` of its frames; None otherwise."""
    n_frames = len(track.frames)
    lost = sum(not f.detected for f in track.frames)
    if not n_frames or lost / n_frames <= max_fraction:
        return None
    reason = (
        f"animal not found on {lost / n_frames:.0%} of frames ({lost} of {n_frames}), "
        f"above the {max_fraction:.0%} limit"
    )
    return _flag("tracking_loss", 0, n_frames - 1, reason, "critical")


def _out_of_arena(track: Track, arena: Arena, margin_cm: float) -> list[QcFlag]:
    """Each stretch of frames whose centroid is `margin_cm` or more outside the floor."""
    distances = signed_distance_per_frame([f.position for f in track.frames], arena.floor_px)
    outside_cm = [None if d is None else -d / arena.px_per_cm for d in distances]
    return [
        _flag(
            "out_of_arena",
            first,
            last,
            f"centroid up to {worst:.1f} cm outside the floor, past the {margin_cm:.1f} cm margin",
        )
        for first, last, worst in _runs_at_least(outside_cm, margin_cm)
    ]


def _speed_jumps(
    track: Track, px_per_cm: float, fps: float, max_cm_s: float, join_s: float
) -> list[QcFlag]:
    """Steps between consecutive detections (across a gap too) faster than `max_cm_s`. Jumps
    at most `join_s` apart join one flag over the frames between them, so a one-frame
    mis-detection is one flag over 3 frames, and a longer one is one flag over all of it."""
    detected = [(f.frame_index, f.position) for f in track.frames if f.position is not None]
    jumps: list[tuple[int, int, float]] = []
    for (a, pa), (b, pb) in itertools.pairwise(detected):
        speed = math.dist(pa, pb) / px_per_cm / ((b - a) / fps)
        if speed <= max_cm_s:
            continue
        if jumps and (a - jumps[-1][1]) / fps <= join_s:
            first, _, fastest = jumps.pop()
            jumps.append((first, b, max(fastest, speed)))
        else:
            jumps.append((a, b, speed))
    return [
        _flag(
            "speed_jump",
            first,
            last,
            f"centroid moved at up to {fastest:.0f} cm/s, above the {max_cm_s:.0f} cm/s maximum",
        )
        for first, last, fastest in jumps
    ]


def _second_masks(track: Track, px_per_cm: float, min_area_cm2: float) -> list[QcFlag]:
    """Each stretch of frames with a second object of at least `min_area_cm2` beside the
    animal."""
    area_cm2 = [f.second_area_px / px_per_cm**2 for f in track.frames]
    return [
        _flag(
            "second_mask",
            first,
            last,
            f"a second object of up to {largest:.0f} cm² beside the animal, "
            f"at or above the {min_area_cm2:.0f} cm² limit",
        )
        for first, last, largest in _runs_at_least(area_cm2, min_area_cm2)
    ]


def _runs_at_least(values: Sequence[float | None], limit: float) -> list[tuple[int, int, float]]:
    """First frame, last frame and largest value of each stretch of values at `limit` or
    above."""
    above = [v is not None and v >= limit for v in values]
    return [
        (
            s.start,
            s.start + s.length - 1,
            max(v for v in values[s.start : s.start + s.length] if v is not None),
        )
        for s in stretches(above)
        if s.value
    ]


def _flag(
    kind: QcFlagKind, first: int, last: int, reason: str, severity: Severity = "warning"
) -> QcFlag:
    return QcFlag(kind=kind, first_frame=first, last_frame=last, severity=severity, reason=reason)
