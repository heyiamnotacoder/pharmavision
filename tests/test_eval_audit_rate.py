"""The number-audit pass rate (eval): every stored study report audited again, deterministically,
against the tables its tools return now. Reports are written by a scripted judge through the
real study-level pipeline."""

import shutil
from pathlib import Path

import pytest

from eval.audit_rate import audit_studies
from pharmavision.pipeline import StudyFolder, study_report, study_stats

from .judges import ScriptedJudge
from .studies import write_study
from .test_report import DIAZEPAM, HONEST, INVENTED, RESULTS, VEHICLE

INVENTING = HONEST.replace(RESULTS, f"{RESULTS} {INVENTED}")


@pytest.fixture
def studies(tmp_path: Path) -> Path:
    path = tmp_path / "studies"
    path.mkdir()
    return path


def reported(
    tmp_path: Path, studies: Path, runs_dir: Path, name: str, groups: tuple[str, str], *drafts: str
) -> StudyFolder:
    """A study named `name` in `studies`, its stats run and its report written from `drafts`."""
    made = write_study(tmp_path / name, runs_dir, {groups[0]: VEHICLE, groups[1]: DIAZEPAM})
    study = StudyFolder(made.path.rename(studies / name))
    study_stats(study, runs_dir)
    study_report(study, runs_dir, ScriptedJudge(drafts=list(drafts)))
    return study


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    for name in ("honest", "retried", "inventing"):
        (tmp_path / name).mkdir()
    return tmp_path


def test_each_report_is_audited_again_and_the_pass_rate_counts_the_ones_that_pass(
    tmp: Path, studies: Path, runs_dir: Path
) -> None:
    reported(tmp, studies, runs_dir, "honest", ("vehicle", "diazepam"), HONEST)
    reported(tmp, studies, runs_dir, "retried", ("saline", "drug"), INVENTING, HONEST)
    reported(tmp, studies, runs_dir, "inventing", ("control", "dosed"), INVENTING, INVENTING)

    rate = audit_studies(studies, runs_dir)

    by_id = {s.study_id: s for s in rate.studies}
    assert sorted(by_id) == ["honest", "inventing", "retried"]
    assert (by_id["honest"].passed, by_id["honest"].first_draft_passed) == (True, True)
    assert (by_id["retried"].passed, by_id["retried"].first_draft_passed) == (True, False)
    assert by_id["retried"].drafts == 2
    assert by_id["inventing"].passed is False
    assert by_id["inventing"].unmatched == ["18%"]
    assert (rate.passed, rate.total) == (2, 3)
    assert rate.rate == pytest.approx(2 / 3)
    assert not rate.met


def test_a_report_whose_runs_are_gone_cannot_be_checked_and_does_not_count_as_passing(
    tmp: Path, studies: Path, runs_dir: Path
) -> None:
    reported(tmp, studies, runs_dir, "honest", ("vehicle", "diazepam"), HONEST)
    shutil.rmtree(runs_dir / "vehicle-0")

    rate = audit_studies(studies, runs_dir)

    [study] = rate.studies
    assert study.passed is None
    assert study.reason is not None
    assert "vehicle-0" in study.reason
    assert (rate.passed, rate.total, rate.met) == (0, 1, False)


def test_a_report_that_cannot_be_read_or_has_no_design_counts_as_not_passing(
    tmp: Path, studies: Path, runs_dir: Path
) -> None:
    honest = reported(tmp, studies, runs_dir, "honest", ("vehicle", "diazepam"), HONEST)
    (studies / "orphan").mkdir()
    shutil.copy(honest.path / "report.json", studies / "orphan" / "report.json")
    (honest.path / "report.json").write_text('{"drafts": []}')

    rate = audit_studies(studies, runs_dir)

    assert [(s.study_id, s.passed) for s in rate.studies] == [("honest", None), ("orphan", None)]
    assert all(s.reason for s in rate.studies)
    assert (rate.passed, rate.total, rate.met) == (0, 2, False)


def test_with_no_study_reports_the_pass_rate_abstains(studies: Path, runs_dir: Path) -> None:
    (studies / "no-report").mkdir()
    (studies / "no-report" / "study.yaml").write_text("name: x\n")

    rate = audit_studies(studies, runs_dir)

    assert rate.studies == []
    assert rate.rate is None
    assert not rate.met
    assert audit_studies(studies / "missing", runs_dir).rate is None
