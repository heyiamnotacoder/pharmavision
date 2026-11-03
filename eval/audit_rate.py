"""The number-audit pass rate (FR-9, NFR-1; the eval's one pass/fail bar, 100%). Every stored
study report is audited again: the shipped draft against the tables its tools return now, so a
report left behind by changed stats or edited runs fails here even though it passed when it was
written. Deterministic: it reads `report.json` and calls no model."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from pharmavision.artifacts import Report, StudyFolder, StudyNotReadyError
from pharmavision.audit import audit_numbers, known_values
from pharmavision.stages.report import tool_outputs


class StudyAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    study_id: str
    drafts: int | None
    """None when the report could not be read."""
    first_draft_passed: bool | None
    """As audited when it was written: False when the agent needed the re-prompt."""
    passed: bool | None
    """The shipped draft, audited again now; None when it could not be (see `reason`)."""
    unmatched: list[str]
    """The numbers no table holds, as written."""
    reason: str | None = None


class AuditRate(BaseModel):
    model_config = ConfigDict(frozen=True)

    studies: list[StudyAudit]

    @property
    def total(self) -> int:
        return len(self.studies)

    @property
    def passed(self) -> int:
        """Reports that pass now; one that could not be checked does not."""
        return sum(s.passed is True for s in self.studies)

    @property
    def rate(self) -> float | None:
        """None with no report to audit: the rate abstains."""
        return self.passed / self.total if self.total else None

    @property
    def met(self) -> bool:
        return self.total > 0 and self.passed == self.total


def audit_studies(studies_dir: Path, runs_dir: Path) -> AuditRate:
    """Every study folder in `studies_dir` holding a report, its runs read from `runs_dir`. A
    report that can't be read, or whose tables can't (no design, a run gone), is counted and
    does not pass: it is never left out."""
    folders = sorted(p for p in studies_dir.iterdir() if p.is_dir()) if studies_dir.is_dir() else []
    studies = [StudyFolder(p) for p in folders]
    return AuditRate(studies=[_audit(s, runs_dir) for s in studies if s.has("report")])


def _audit(study: StudyFolder, runs_dir: Path) -> StudyAudit:
    try:
        report = study.report()
    except ValidationError as error:
        return _unchecked(study, None, f"its report.json could not be read: {error}")
    try:
        tables = tool_outputs(study, runs_dir)
    except (FileNotFoundError, StudyNotReadyError, ValidationError) as error:
        return _unchecked(study, report, f"its tables could not be read: {error}")
    shipped = report.drafts[-1]
    audit = audit_numbers(shipped.markdown, known_values(tables), report.settings.audit)
    return StudyAudit(
        study_id=study.study_id,
        drafts=len(report.drafts),
        first_draft_passed=report.drafts[0].audit.passed,
        passed=audit.passed,
        unmatched=[n.text for n in audit.unmatched],
    )


def _unchecked(study: StudyFolder, report: Report | None, reason: str) -> StudyAudit:
    return StudyAudit(
        study_id=study.study_id,
        drafts=None if report is None else len(report.drafts),
        first_draft_passed=None if report is None else report.drafts[0].audit.passed,
        passed=None,
        unmatched=[],
        reason=reason,
    )
