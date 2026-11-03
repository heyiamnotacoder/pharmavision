"""The harness on one video, through the real pipeline with a synthetic SAM 3.1: run, compare
with DeepLabCut, log the spend. Videos and DLC files are synthetic, never Sturman data."""

import json
import math
from pathlib import Path

import pytest

from eval.harness import ArenaNotFoundError, Paths, evaluate_video, summarise, video_rears
from eval.spend import Call, Usage
from pharmavision.segmenter.synthetic import SyntheticSegmenter

from .dlc_csv import write_dlc
from .scenes import FLOOR, N_FRAMES, REAR, TEMPLATE, rears_in, walk_stop_walk
from .videos import make_video

SOURCE_FPS = 25


def walk(frame: float) -> tuple[float, float]:
    """One lap of a 60 px circle about the floor's centre over the trial, at any frame time."""
    angle = 2 * math.pi * frame / (N_FRAMES - 1)
    return (180 + 60 * math.cos(angle), 180 + 60 * math.sin(angle))


class FakeMeter:
    """What the API billed: nothing before the video, then one SAM 3.1 track."""

    def __init__(self) -> None:
        self.pending = [[], [Call(model="sam-3.1", usage=Usage(video_frames=60), usd=0.012)]]

    def take(self) -> list[Call]:
        return self.pending.pop(0) if self.pending else []


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(root=tmp_path / "eval-data")


@pytest.fixture
def video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "OFT_11.mp4", seconds=4, fps=SOURCE_FPS)


@pytest.fixture
def dlc_csv(tmp_path: Path) -> Path:
    # DLC ran at the source's 25 fps, on the same walk: frame k is at our frame k * 15 / 25.
    points = [(*walk(k * 15 / SOURCE_FPS), 0.99) for k in range(4 * SOURCE_FPS)]
    return write_dlc(tmp_path / "3_01_A_1DeepCut.csv", points)


def test_a_video_is_tracked_compared_with_dlc_and_its_spend_logged(
    paths: Paths, video: Path, dlc_csv: Path
) -> None:
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=walk)

    result = evaluate_video("OFT_11", video, dlc_csv, segmenter, FakeMeter(), paths, TEMPLATE)

    tracking = result.tracking
    assert tracking.frames == 60
    assert tracking.tracked_pct == 100.0
    assert tracking.frames_compared == 60
    assert tracking.distance_cm is not None
    # The walk moves about 1 cm per frame; DLC's nearest source frame is within 0.2 frames.
    assert tracking.distance_cm.max < 0.5
    assert result.floor_prompt == TEMPLATE.floor_prompt
    assert result.spend["sam-3.1"].usage.video_frames == 60
    assert result.spent_usd == pytest.approx(0.012)
    assert json.loads(paths.result("OFT_11").read_text())["video_id"] == "OFT_11"
    (logged,) = [json.loads(line) for line in paths.spend_log.read_text().splitlines()]
    assert logged["video_id"] == "OFT_11"
    assert logged["usd"] == pytest.approx(0.012)
    assert logged["by_model"]["sam-3.1"]["usage"]["video_frames"] == 60


def test_the_next_floor_prompt_is_tried_when_sam_finds_no_square_floor(
    paths: Paths, video: Path, dlc_csv: Path
) -> None:
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=walk, floor_prompt="white floor")

    result = evaluate_video("OFT_11", video, dlc_csv, segmenter, FakeMeter(), paths, TEMPLATE)

    assert result.floor_prompt == "white floor"


def test_a_video_with_no_floor_found_still_logs_its_spend(
    paths: Paths, video: Path, dlc_csv: Path
) -> None:
    with pytest.raises(ArenaNotFoundError):
        evaluate_video("OFT_12", video, dlc_csv, SyntheticSegmenter(), FakeMeter(), paths, TEMPLATE)

    (logged,) = [json.loads(line) for line in paths.spend_log.read_text().splitlines()]
    assert logged["video_id"] == "OFT_12"
    assert logged["finished"] is False
    assert not paths.result("OFT_12").exists()


def test_the_summary_lists_eval_videos_with_no_result_and_sums_every_attempt(
    paths: Paths, video: Path, dlc_csv: Path
) -> None:
    with pytest.raises(ArenaNotFoundError):
        evaluate_video("OFT_12", video, dlc_csv, SyntheticSegmenter(), FakeMeter(), paths, TEMPLATE)
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=walk)
    evaluate_video("OFT_11", video, dlc_csv, segmenter, FakeMeter(), paths, TEMPLATE)

    summary = summarise(["OFT_11", "OFT_12"], paths)

    assert [r.video_id for r in summary.videos] == ["OFT_11"]
    assert summary.missing == ["OFT_12"]
    assert summary.spent_usd == pytest.approx(0.024)


def test_a_finished_video_is_not_run_again(paths: Paths, video: Path, dlc_csv: Path) -> None:
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=walk)
    first = evaluate_video("OFT_11", video, dlc_csv, segmenter, FakeMeter(), paths, TEMPLATE)

    again = evaluate_video(
        "OFT_11", video, dlc_csv, SyntheticSegmenter(), FakeMeter(), paths, TEMPLATE
    )

    assert again.run_id == first.run_id
    assert again.tracking == first.tracking
    assert len(list(paths.runs.iterdir())) == 1


def test_a_tracked_eval_run_gives_its_rear_candidates_in_seconds_beside_the_raters_rears(
    paths: Paths, video: Path, dlc_csv: Path
) -> None:
    # Still for frames 15-44, rearing in frames 25-39: one candidate, 25/15 s to 40/15 s.
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=walk_stop_walk, body=rears_in(REAR))
    evaluate_video("OFT_11", video, dlc_csv, segmenter, FakeMeter(), paths, TEMPLATE)
    raters = {"Jin": [(1.7, 2.6)], "Oliver": []}

    rears = video_rears("OFT_11", paths, raters)

    assert rears is not None
    [candidate] = rears.candidates
    assert (candidate.start_s, candidate.end_s) == (pytest.approx(25 / 15), pytest.approx(40 / 15))
    assert 0 <= candidate.score <= 1
    assert rears.raters == raters


def test_an_eval_video_never_tracked_has_no_rears_to_compare(paths: Paths) -> None:
    assert video_rears("OFT_12", paths, {"Jin": []}) is None
