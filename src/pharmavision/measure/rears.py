"""Rear candidates: windows where the animal may be rearing, for the behaviours stage to judge.

From a top-view camera a rear shows only in the mask: it shrinks as the body rises toward the
lens and changes shape, while the animal stays in place. The windows are inputs for
classification, never results."""

import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..artifacts import MeasureSettings, RearCandidate, RearFeatures, RearScorer
from .stretches import stretches


def rear_candidates(
    areas_px: Sequence[float | None],
    elongations: Sequence[float | None],
    speeds_cm_s: Sequence[float | None],
    fps: float,
    px_per_cm: float,
    settings: MeasureSettings,
) -> list[RearCandidate]:
    """One value per frame in each sequence, None where it is unknown; an unknown speed means
    the position is unknown (a kept gap). A candidate frame has its area at least
    `rear_area_drop` below its baseline, its elongation at least `rear_elongation_change` away
    from its baseline either way, and a speed of at most `rear_max_speed_cm_s`. A break of less
    than `rear_merge_s` between candidate frames joins them into one window unless a speed in
    the break is unknown, and windows shorter than `rear_min_s` are dropped. Each window carries
    its features over the frames where they are known, and their score by
    `settings.rear_scorer`."""
    baseline_frames = max(1, round(settings.rear_baseline_s * fps)) | 1  # odd: centred on a frame
    area_baseline = _running_median(areas_px, baseline_frames)
    elongation_baseline = _running_median(elongations, baseline_frames)
    candidate = [
        area is not None
        and elongation is not None
        and speed is not None
        and area <= (1 - settings.rear_area_drop) * area_base
        and abs(elongation / elongation_base - 1) >= settings.rear_elongation_change
        and speed <= settings.rear_max_speed_cm_s
        for area, area_base, elongation, elongation_base, speed in zip(
            areas_px, area_baseline, elongations, elongation_baseline, speeds_cm_s, strict=True
        )
    ]
    found = stretches(candidate)
    for before, between, after in zip(found, found[1:], found[2:], strict=False):
        if (
            before.value
            and after.value
            and between.length / fps < settings.rear_merge_s
            and all(s is not None for s in speeds_cm_s[between.start : after.start])
        ):
            candidate[between.start : after.start] = [True] * between.length
    area = np.array([np.nan if a is None else a for a in areas_px], dtype=np.float64)
    area_ratio = area / area_baseline
    elongation_ratio = np.array([np.nan if e is None else e for e in elongations]) / np.array(
        elongation_baseline
    )
    speed = np.array([np.nan if v is None else v for v in speeds_cm_s], dtype=np.float64)
    # A window starts and ends on a candidate frame, where everything is known, so no slice is
    # all NaN; merged breaks may hold unknown masks, which the nan-functions skip.
    windows: list[RearCandidate] = []
    for s in stretches(candidate):
        if not s.value or s.length / fps < settings.rear_min_s:
            continue
        frames = slice(s.start, s.start + s.length)
        features = RearFeatures(
            duration_s=s.length / fps,
            area_drop=float(1 - np.nanmin(area_ratio[frames])),
            mean_area_drop=float(1 - np.nanmean(area_ratio[frames])),
            elongation_change=float(np.nanmax(np.abs(elongation_ratio[frames] - 1))),
            mean_speed_cm_s=float(np.nanmean(speed[frames])),
            mean_area_cm2=float(np.nanmean(area[frames])) / px_per_cm**2,
        )
        windows.append(
            RearCandidate(
                first_frame=s.start,
                last_frame=s.start + s.length - 1,
                features=features,
                score=rear_score(features, settings.rear_scorer),
            )
        )
    return windows


def rear_score(features: RearFeatures, scorer: RearScorer) -> float:
    """The logistic of the intercept plus each feature times its weight, computed so that no
    feature value, however extreme, overflows."""
    z = scorer.intercept + sum(
        getattr(scorer.weights, name) * getattr(features, name)
        for name in RearFeatures.model_fields
    )
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    return math.exp(z) / (1 + math.exp(z))


def _running_median(values: Sequence[float | None], window: int) -> list[float]:
    """The median of the known values within `window // 2` frames either side of each frame;
    NaN where none is known, which fails every comparison. Unlike the trajectory's filter it
    reaches across gaps: it estimates the animal's usual size, not its path. Rearing must fill
    less than half the window, or the baseline becomes the rear."""
    x = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    padded = np.pad(x, window // 2, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a window with nothing known
        medians: list[float] = np.nanmedian(sliding_window_view(padded, window), axis=-1).tolist()
    return medians
