"""Rearing agreement (eval, measures): per-video count ICC and event-level F1 against each
rater, beside the raters against each other. Rears and windows here are made up, never the
Sturman labels."""

import pytest

from eval.rearing import Candidate, RearingSettings, VideoRears, icc_a1, rearing_agreement


def test_icc_matches_shrout_and_fleiss_worked_example() -> None:
    # Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges; ICC(2,1) = 0.29.
    ratings = [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ]

    assert icc_a1(ratings) == pytest.approx(0.29, abs=0.005)


def test_icc_is_one_when_two_raters_give_the_same_counts() -> None:
    assert icc_a1([[3, 3], [10, 10], [7, 7]]) == pytest.approx(1.0)


def test_icc_abstains_with_fewer_than_two_videos_or_no_spread() -> None:
    assert icc_a1([[3, 4]]) is None
    assert icc_a1([[5, 5], [5, 5], [5, 5]]) is None


def scored(*windows: tuple[float, float, float]) -> list[Candidate]:
    return [Candidate(start_s=s, end_s=e, score=p) for s, e, p in windows]


def video(
    video_id: str, candidates: list[Candidate], **raters: list[tuple[float, float]]
) -> VideoRears:
    return VideoRears(video_id=video_id, candidates=candidates, raters=raters)


SETTINGS = RearingSettings(count_threshold=0.5, tolerance_s=0.5)


def test_windows_on_every_rear_of_a_rater_agree_with_that_rater_fully() -> None:
    videos = [
        video("V1", scored((1, 2, 0.9), (5, 6, 0.9)), A=[(1, 2), (5, 6)], B=[(1, 2)]),
        video("V2", scored((3, 4, 0.9)), A=[(3, 4)], B=[(3, 4), (8, 9)]),
        video("V3", scored(), A=[], B=[(2, 3)]),
    ]

    agreement = rearing_agreement(videos, SETTINGS)

    against_a = agreement.scored["A"]
    assert against_a.events.f1 == pytest.approx(1.0)
    assert against_a.count_icc == pytest.approx(1.0)
    against_b = agreement.scored["B"]
    # 2 of B's 4 rears are found, and 2 of the 3 windows are B's rears.
    assert (against_b.events.matched, against_b.events.windows, against_b.events.rears) == (2, 3, 4)
    assert against_b.events.precision == pytest.approx(2 / 3)
    assert against_b.events.recall == pytest.approx(0.5)


def test_a_window_matches_a_rear_up_to_the_tolerance_away() -> None:
    videos = [
        video("V1", scored((2.4, 3, 0.9)), A=[(1, 2)]),
        video("V2", scored((2.6, 3, 0.9)), A=[(1, 2)]),
    ]

    agreement = rearing_agreement(videos, SETTINGS)

    assert agreement.scored["A"].events.matched == 1


def test_one_window_spanning_two_rears_matches_only_one_of_them() -> None:
    videos = [video("V1", scored((1, 6, 0.9)), A=[(1, 2), (4, 5)])]

    events = rearing_agreement(videos, SETTINGS).scored["A"].events

    assert (events.matched, events.windows, events.rears) == (1, 1, 2)


def test_matching_finds_the_most_pairs_not_the_first_ones() -> None:
    # Taking windows in time, each with its first rear in reach, gives the long first window
    # the first rear and leaves the short second one, which reaches only that rear, unmatched.
    videos = [video("V1", scored((0.0, 2.0, 0.9), (0.2, 0.4, 0.9)), A=[(0.5, 0.8), (1.9, 2.0)])]

    assert rearing_agreement(videos, SETTINGS).scored["A"].events.matched == 2


def test_windows_under_the_threshold_are_left_out_and_every_candidate_keeps_them() -> None:
    videos = [
        video("V1", scored((1, 2, 0.9), (5, 6, 0.2)), A=[(1, 2)]),
        video("V2", scored((1, 2, 0.9), (5, 6, 0.9), (8, 9, 0.1)), A=[(1, 2), (5, 6)]),
    ]

    agreement = rearing_agreement(videos, SETTINGS)

    assert [v.scored for v in agreement.per_video] == [1, 2]
    assert [v.candidates for v in agreement.per_video] == [2, 3]
    assert [v.raters for v in agreement.per_video] == [{"A": 1}, {"A": 2}]
    assert agreement.scored["A"].events.f1 == pytest.approx(1.0)
    assert agreement.every_candidate["A"].events.precision == pytest.approx(3 / 5)


def test_the_raters_are_compared_pair_by_pair_and_their_range_is_given() -> None:
    videos = [
        video("V1", scored(), A=[(1, 2), (4, 5)], B=[(1, 2), (4, 5)], C=[(1, 2)]),
        video("V2", scored(), A=[(1, 2)], B=[(1, 2)], C=[(1, 2), (7, 8)]),
    ]

    agreement = rearing_agreement(videos, SETTINGS)

    assert agreement.raters == ["A", "B", "C"]
    assert list(agreement.rater_pairs) == ["A~B", "A~C", "B~C"]
    assert agreement.rater_pairs["A~B"].events.f1 == pytest.approx(1.0)
    assert agreement.rater_pairs["A~C"].events.f1 == pytest.approx(2 * 2 / (3 + 3))
    assert agreement.human_range("f1") == (pytest.approx(2 / 3), pytest.approx(1.0))


def test_with_no_windows_and_no_rears_event_agreement_abstains() -> None:
    videos = [video("V1", scored((1, 2, 0.1)), A=[]), video("V2", scored(), A=[])]

    events = rearing_agreement(videos, SETTINGS).scored["A"].events

    assert (events.precision, events.recall, events.f1) == (None, None, None)
