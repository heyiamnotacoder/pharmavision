"""Rear candidates on per-frame features with known answers: the rules the pipeline tests can't
reach in a 4 s video (merging, kept gaps, the minimum duration, the local baseline)."""

import math

import numpy as np
import pytest
from pycocotools import mask as coco_mask

from pharmavision.artifacts import MeasureSettings, RearCandidate, RearFeatures, RearScorer
from pharmavision.measure.rears import rear_candidates
from pharmavision.measure.shape import mask_elongations

FPS = 10.0
PX_PER_CM = 10.0
SETTINGS = MeasureSettings(
    rear_baseline_s=10.0,
    rear_area_drop=0.1,
    rear_elongation_change=0.1,
    rear_max_speed_cm_s=10.0,
    rear_merge_s=0.5,
    rear_min_s=0.2,
)


def window(first: int, last: int) -> tuple[int, int]:
    return (first, last)


def frames(found: list[RearCandidate]) -> list[tuple[int, int]]:
    return [(c.first_frame, c.last_frame) for c in found]


def windows(
    n: int,
    rearing: set[int],
    area: float = 100.0,
    elongation: float = 2.0,
    rear: tuple[float, float] = (50.0, 3.0),
    unknown: frozenset[int] = frozenset(),
) -> list[tuple[int, int]]:
    """A still animal of `area` and `elongation`, with the `rear` (area, elongation) in the
    `rearing` frames and no position or mask in the `unknown` ones."""
    areas: list[float | None] = [None if i in unknown else rear[0] if i in rearing else area
                                 for i in range(n)]  # fmt: skip
    elongations: list[float | None] = [
        None if i in unknown else rear[1] if i in rearing else elongation for i in range(n)
    ]
    speeds: list[float | None] = [None if i in unknown else 0.0 for i in range(n)]
    return frames(rear_candidates(areas, elongations, speeds, FPS, PX_PER_CM, SETTINGS))


def test_a_body_turning_rounder_counts_as_an_elongation_change() -> None:
    assert windows(100, set(range(40, 50)), rear=(50.0, 1.5)) == [window(40, 49)]


def test_a_break_shorter_than_the_merge_time_joins_two_windows() -> None:
    # Breaks of 0.2 s (frames 45-46) and 0.6 s (frames 52-57).
    rearing = {*range(40, 45), *range(47, 52), *range(58, 63)}

    assert windows(100, rearing) == [window(40, 51), window(58, 62)]


def test_a_kept_gap_keeps_two_windows_apart() -> None:
    rearing = {*range(40, 45), *range(47, 52)}

    assert windows(100, rearing, unknown=frozenset({45, 46})) == [
        window(40, 44),
        window(47, 51),
    ]


def test_a_window_shorter_than_the_minimum_is_dropped() -> None:
    assert windows(100, {40, 60, 61}) == [window(60, 61)]


def test_the_baseline_follows_the_animal_across_the_arena() -> None:
    # The animal looks twice as large in the second half of the video (nearer the lens), where
    # it rears to 150 px: below its local size of 200, though above its first-half size.
    areas = [100.0] * 100 + [200.0] * 100
    elongations = [2.0] * 200
    for i in range(150, 160):
        areas[i], elongations[i] = 150.0, 3.0
    speeds = [0.0] * 200

    local = SETTINGS.model_copy(update={"rear_baseline_s": 2.0})

    assert frames(rear_candidates(areas, elongations, speeds, FPS, PX_PER_CM, local)) == [
        window(150, 159)
    ]
    # Against one baseline for the whole trial (a 40 s window), the rear is missed.
    trial = SETTINGS.model_copy(update={"rear_baseline_s": 40.0})
    assert frames(rear_candidates(areas, elongations, speeds, FPS, PX_PER_CM, trial)) == []


def test_a_moving_animal_is_no_candidate() -> None:
    areas = [100.0] * 40 + [50.0] * 10 + [100.0] * 50
    elongations = [2.0] * 40 + [3.0] * 10 + [2.0] * 50
    speeds = [12.0] * 100

    assert frames(rear_candidates(areas, elongations, speeds, FPS, PX_PER_CM, SETTINGS)) == []


def rle(pixels: np.ndarray) -> str:
    counts = coco_mask.encode(np.asfortranarray(pixels.astype(np.uint8)))["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def test_elongation_is_the_axis_ratio_and_unknown_for_a_line() -> None:
    rectangle = np.zeros((50, 50), dtype=bool)
    rectangle[10:20, 5:45] = True  # 40 x 10 px
    diagonal = np.eye(50, dtype=bool)

    elongations = mask_elongations([rle(rectangle), rle(diagonal), None], 50, 50)

    # A uniform rectangle's axes go as its sides: sqrt((40^2 - 1) / (10^2 - 1)) = 4.01.
    assert elongations[0] == pytest.approx(4.01, abs=0.01)
    assert elongations[1:] == [None, None]


def test_each_candidate_carries_the_features_it_is_scored_on() -> None:
    # A still animal of 100 px and elongation 2 rears to 50 px for 1 s, rounder at first
    # (elongation 1.5) and then longer (3.0), creeping at 2 cm/s.
    areas = [100.0] * 40 + [50.0] * 10 + [100.0] * 50
    elongations = [2.0] * 40 + [1.5] * 5 + [3.0] * 5 + [2.0] * 50
    speeds = [0.0] * 40 + [2.0] * 10 + [0.0] * 50

    [found] = rear_candidates(areas, elongations, speeds, FPS, PX_PER_CM, SETTINGS)

    assert (found.first_frame, found.last_frame) == (40, 49)
    assert found.features == RearFeatures(
        duration_s=1.0,
        area_drop=0.5,
        mean_area_drop=0.5,
        elongation_change=0.5,
        mean_speed_cm_s=2.0,
        mean_area_cm2=0.5,  # 50 px at 10 px/cm
    )


def test_the_score_is_the_logistic_of_the_weighted_features() -> None:
    weights = RearFeatures(
        duration_s=1.0,
        area_drop=2.0,
        mean_area_drop=0.0,
        elongation_change=-1.0,
        mean_speed_cm_s=-0.5,
        mean_area_cm2=0.0,
    )
    scorer = RearScorer(intercept=-1.0, weights=weights)
    settings = SETTINGS.model_copy(update={"rear_scorer": scorer})
    areas = [100.0] * 40 + [50.0] * 10 + [100.0] * 50
    elongations = [2.0] * 40 + [3.0] * 10 + [2.0] * 50
    speeds = [0.0] * 40 + [2.0] * 10 + [0.0] * 50

    [found] = rear_candidates(areas, elongations, speeds, FPS, PX_PER_CM, settings)

    # -1 + 1.0 * 1.0 + 2.0 * 0.5 - 1.0 * 0.5 - 0.5 * 2.0 = -0.5
    assert found.score == pytest.approx(1 / (1 + math.exp(0.5)))
