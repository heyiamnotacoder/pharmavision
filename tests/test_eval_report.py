"""The eval report: one command, from what the harness stored (tracked runs, tracking results,
study reports) and the label file, writes eval/REPORT.md. It calls no model. Videos, labels and
studies are synthetic, never Sturman data."""

import json
from pathlib import Path

import pytest

from eval.harness import Paths, evaluate_video
from eval.rearing import RearingSettings
from eval.report import build_report, render, write_report
from eval.split import Split
from pharmavision.segmenter.synthetic import SyntheticSegmenter

from .dlc_csv import write_dlc
from .scenes import FLOOR, REAR, TEMPLATE, rears_in, walk_stop_walk
from .test_eval_audit_rate import reported
from .test_eval_harness import FakeMeter
from .test_eval_sturman import labels
from .test_report import HONEST
from .videos import make_video

# Each tracked video holds one candidate, 1.667 s to 2.667 s.
LABEL_ROWS = (
    '"OFT_11_Jin.csv";"Jin";1.7;2.6;"Supported";"OFT_11";"a.csv"',
    '"OFT_11_OS.csv";"Oliver";1.8;2.5;"Unsupported";"OFT_11";"a.csv"',
    '"OFT_11_FA.csv";"Furkan";0;0;"Start/End";"OFT_11";"a.csv"',
    '"OFT_14_Jin.csv";"Jin";1.7;2.6;"Supported";"OFT_14";"b.csv"',
    '"OFT_14_OS.csv";"Oliver";1.7;2.6;"Supported";"OFT_14";"b.csv"',
    '"OFT_14_OS.csv";"Oliver";3.5;3.9;"Supported";"OFT_14";"b.csv"',
    '"OFT_12_Jin.csv";"Jin";1.0;2.0;"Supported";"OFT_12";"c.csv"',
)
SPLIT = Split(seed=1, train_ids=(), eval_ids=("OFT_11", "OFT_12", "OFT_14"))
EVERY_SCORE = RearingSettings(count_threshold=0.0, tolerance_s=0.5)


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    """OFT_11 and OFT_14 tracked, compared with DLC and their spend logged; OFT_12 never."""
    paths = Paths(root=tmp_path / "eval-data")
    video = make_video(tmp_path / "trial.mp4", seconds=4, fps=25)
    dlc = write_dlc(tmp_path / "dlc.csv", [(180.0, 180.0, 0.99)] * 100)
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=walk_stop_walk, body=rears_in(REAR))
    for video_id in ("OFT_11", "OFT_14"):
        evaluate_video(video_id, video, dlc, segmenter, FakeMeter(), paths, TEMPLATE)
    return paths


@pytest.fixture
def studies(tmp_path: Path, runs_dir: Path) -> Path:
    path = tmp_path / "studies"
    path.mkdir()
    (tmp_path / "honest").mkdir()
    reported(tmp_path, path, runs_dir, "honest", ("vehicle", "diazepam"), HONEST)
    return path


def test_one_command_writes_the_validation_table_from_stored_results_only(
    tmp_path: Path, paths: Paths, studies: Path, runs_dir: Path
) -> None:
    spend_before = paths.spend_log.read_text()

    report = build_report(
        SPLIT, paths, labels(tmp_path, *LABEL_ROWS), studies, runs_dir, EVERY_SCORE
    )
    out = tmp_path / "REPORT.md"
    write_report(report, out, paths)

    markdown = out.read_text()
    assert markdown == render(report)
    assert paths.spend_log.read_text() == spend_before  # no model was called
    rearing = report.rearing
    assert rearing is not None
    assert rearing.raters == ["Furkan", "Jin", "Oliver"]
    assert [v.video_id for v in rearing.per_video] == ["OFT_11", "OFT_14"]
    assert report.rearing_missing == ["OFT_12"]
    # One window per video, on Jin's one rear per video.
    assert rearing.scored["Jin"].events.f1 == pytest.approx(1.0)
    assert "| Rear events, F1 (±0.5 s): candidates scoring ≥ 0.00 | 0.00 | 1.00 | 0.80 |" in (
        markdown
    )
    assert "| Number-audit pass rate | 1 of 1 study reports (100%) | 100% | yes |" in markdown
    assert "OFT_12" in markdown
    stored = json.loads(paths.report_json.read_text())
    assert stored["audit"]["studies"][0]["study_id"] == "honest"


def test_the_report_is_the_same_each_time_it_is_made(
    tmp_path: Path, paths: Paths, studies: Path, runs_dir: Path
) -> None:
    label_file = labels(tmp_path, *LABEL_ROWS)

    first = render(build_report(SPLIT, paths, label_file, studies, runs_dir, EVERY_SCORE))
    again = render(build_report(SPLIT, paths, label_file, studies, runs_dir, EVERY_SCORE))

    assert first == again
