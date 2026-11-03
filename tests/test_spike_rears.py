"""Clip picking and agreement for the T2 spike. Label rows are synthetic, never Sturman data."""

from pathlib import Path

from eval.split import train_eval_split
from spike.rears import (
    Rear,
    RearKind,
    agreement,
    clip_window,
    consensus_rears,
    load_rears,
    pick_clips,
    rear_recall,
)

HEADER = '"file";"Experimenter";"from";"to";"type";"ID";"DLCFile"\n'


def write_labels(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + "".join(f"{row}\n" for row in rows))
    return path


def test_load_rears_keeps_only_rears_keyed_by_video_id(tmp_path: Path) -> None:
    path = write_labels(
        tmp_path,
        [
            '"OFT_5_Jin.csv";"Jin";2.6;3.7;"Start/End";"OFT_5";"x.csv"',
            '"OFT_5_Jin.csv";"Jin";4.8;6.5;"Supported";"OFT_5";"x.csv"',
            '"OFT_5_Jin.csv";"Jin";9.0;9.9;"Grooming";"OFT_5";"x.csv"',
            '"OFT_6_Oli.csv";"Oliver";1.0;2.0;"Unsupported";"OFT_6";"x.csv"',
            '"OFT_6_Oli.csv";"Oliver";NA;NA;NA;"OFT_6";"x.csv"',
        ],
    )

    assert load_rears(path) == [
        Rear(video_id="OFT_5", rater="Jin", start=4.8, end=6.5, kind="supported"),
        Rear(video_id="OFT_6", rater="Oliver", start=1.0, end=2.0, kind="unsupported"),
    ]


def rear(rater: str, start: float, end: float, kind: RearKind = "supported") -> Rear:
    return Rear(video_id="OFT_5", rater=rater, start=start, end=end, kind=kind)


def test_consensus_keeps_a_rear_two_raters_agree_on_once() -> None:
    rears = [
        rear("Furkan", 10.0, 12.0),
        rear("Jin", 10.4, 12.2),  # IoU with Furkan 1.6/2.2 = 0.73
        rear("Oliver", 10.2, 11.8),  # third rater on the same rear: still one rear
        rear("Jin", 30.0, 31.0),  # only one rater saw it
        rear("Furkan", 50.0, 51.0, "supported"),
        rear("Oliver", 50.0, 51.0, "unsupported"),  # raters disagree on the kind
    ]

    assert consensus_rears(rears) == [
        Rear(video_id="OFT_5", rater="consensus", start=10.4, end=11.8, kind="supported")
    ]


def test_consensus_needs_half_the_union_to_overlap() -> None:
    rears = [rear("Furkan", 10.0, 12.0), rear("Jin", 11.0, 13.5)]  # IoU 1.0/3.5 = 0.29

    assert consensus_rears(rears) == []


def test_clip_window_pads_the_rear_and_stays_between_2_and_4_s() -> None:
    assert clip_window(rear("c", 100.0, 101.0), video_s=600.0) == (99.5, 101.5)  # 1 s + 1 s pad
    assert clip_window(rear("c", 100.0, 100.4), video_s=600.0) == (99.2, 101.2)  # at least 2 s
    assert clip_window(rear("c", 100.0, 106.0), video_s=600.0) == (101.0, 105.0)  # at most 4 s


def test_clip_window_shifts_inside_the_video() -> None:
    assert clip_window(rear("c", 0.2, 1.2), video_s=600.0) == (0.0, 2.0)
    assert clip_window(rear("c", 599.0, 599.8), video_s=600.0) == (598.0, 600.0)


def test_split_is_6_train_14_eval_and_ignores_input_order() -> None:
    ids = [f"OFT_{n}" for n in range(20)]

    train, held_out = train_eval_split(ids)

    assert (len(train), len(held_out)) == (6, 14)
    assert sorted(train + held_out) == sorted(ids)
    assert train_eval_split(list(reversed(ids))) == (train, held_out)


def test_pick_clips_uses_train_videos_only_and_balances_labels() -> None:
    def video_rears(video_id: str) -> list[Rear]:
        kinds: list[RearKind] = ["unsupported", "supported"] * 4
        return [
            Rear(video_id, rater, 20.0 * k, 20.0 * k + 1.0, kind)
            for k, kind in enumerate(kinds, start=1)
            for rater in ("Furkan", "Jin")
        ]

    rears = video_rears("OFT_1") + video_rears("OFT_2")

    clips = pick_clips(
        rears,
        train_ids=["OFT_1"],
        video_s=200.0,
        n_per_kind={"supported": 2, "unsupported": 2, "none": 3},
    )

    assert {c.video_id for c in clips} == {"OFT_1"}
    assert sorted(c.truth for c in clips) == ["none"] * 3 + ["supported"] * 2 + ["unsupported"] * 2
    for clip in clips:
        assert 2.0 <= clip.end - clip.start <= 4.0
        if clip.truth == "none":  # no rater marked a rear within 1 s of the clip
            assert all(clip.end + 1.0 <= r.start or r.end <= clip.start - 1.0 for r in rears)
    assert (
        pick_clips(rears, ["OFT_1"], 200.0, {"supported": 2, "unsupported": 2, "none": 3}) == clips
    )  # fixed seed


def test_agreement_scores_accuracy_and_kappa_against_a_worked_example() -> None:
    # po = 3/4; truth is 50/50 s/u, predictions 25/75, so pe = .5*.25 + .5*.75 = .5; kappa = .5
    score = agreement(
        truth=["supported", "supported", "unsupported", "unsupported"],
        predicted=["supported", "unsupported", "unsupported", "unsupported"],
    )

    assert score.n == 4
    assert score.accuracy == 0.75
    assert score.kappa == 0.5


def test_agreement_kappa_is_undefined_when_chance_explains_everything() -> None:
    score = agreement(truth=["none", "none"], predicted=["none", "none"])

    assert (score.accuracy, score.kappa) == (1.0, None)


def test_rear_recall_counts_the_rears_a_candidate_window_overlaps() -> None:
    windows = [(10.0, 11.0), (20.0, 22.0), (40.0, 41.0)]
    rears = [
        rear("Jin", 10.5, 12.0),  # overlaps the first window
        rear("Jin", 11.0, 12.0),  # only touches it
        rear("Jin", 21.0, 21.5),  # inside the second
        rear("Jin", 30.0, 31.0),  # in no window
    ]

    recall = rear_recall(rears, windows)

    assert (recall.rears, recall.found, recall.recall) == (4, 2, 0.5)
    # The third window holds no rear: 2 of 3 windows overlap one.
    assert (recall.windows, recall.windows_on_a_rear) == (3, 2)


def test_rear_recall_is_undefined_without_rears() -> None:
    assert rear_recall([], [(1.0, 2.0)]).recall is None
