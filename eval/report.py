"""The eval report, `eval/REPORT.md` (FR-11): the validation table the README and the demo
quote, made by one command from what the harness stored. It calls no model, so it costs
nothing and can be made again whenever the code changes:

- rearing: the tracked eval runs, measured afresh, against each rater (`eval.rearing`);
- the number-audit pass rate: every stored study report, audited again (`eval.audit_rate`);
- tracking against DeepLabCut and the spend: `tracking/<ID>.json` and `spend.jsonl`.

Anything missing abstains and is listed, never filled in."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pharmavision.artifacts import MeasureSettings

from .audit_rate import AuditRate, audit_studies
from .harness import Paths, TrackingSummary, summarise, video_rears
from .rearing import Metric, RearingAgreement, RearingSettings, VideoRears, rearing_agreement
from .split import Split
from .sturman import rater_rears

REPORT_MD = Path(__file__).parent / "REPORT.md"


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    split_seed: int
    eval_ids: list[str]
    measure_settings_version: int
    rearing: RearingAgreement | None
    """None when no eval video has a tracked run and labels."""
    rearing_missing: list[str]
    """Eval videos with no tracked run or no labels: left out of the rearing agreement."""
    audit: AuditRate
    tracking: TrackingSummary


def build_report(
    split: Split,
    paths: Paths,
    labels: Path,
    studies_dir: Path,
    runs_dir: Path,
    rearing_settings: RearingSettings | None = None,
) -> EvalReport:
    """Everything the report shows, from stored results and the label file."""
    measure = MeasureSettings()
    by_video = rater_rears(labels)
    videos: list[VideoRears] = []
    missing: list[str] = []
    for video_id in split.eval_ids:
        found = (
            video_rears(video_id, paths, by_video[video_id], measure)
            if (video_id in by_video)
            else None
        )
        if found is None:
            missing.append(video_id)
        else:
            videos.append(found)
    settings = rearing_settings or RearingSettings()
    return EvalReport(
        split_seed=split.seed,
        eval_ids=list(split.eval_ids),
        measure_settings_version=measure.version,
        rearing=rearing_agreement(videos, settings) if videos else None,
        rearing_missing=missing,
        audit=audit_studies(studies_dir, runs_dir),
        tracking=summarise(split.eval_ids, paths),
    )


def write_report(report: EvalReport, out: Path, paths: Paths) -> None:
    """The Markdown to `out`, and every number behind it to `paths.report_json`."""
    out.write_text(render(report))
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text(report.model_dump_json(indent=2) + "\n")


def render(report: EvalReport) -> str:
    parts = [
        "# PharmaVision eval report",
        f"Made by `python -m eval report` from stored results: no model is called, so it costs "
        f"nothing to make again. Sturman et al. 2020 open-field videos (Zenodo 3608658, "
        f"CC BY 4.0), the {len(report.eval_ids)} eval videos of the fixed split "
        f"(`eval/split.json`, seed {report.split_seed}); human labels from DLCAnalyzer "
        f"(GPL-3.0, not in this repo). Measure settings v{report.measure_settings_version}.",
        "## Validation table",
        *_validation(report),
        "## Rearing per video",
        *_per_video(report),
        "## Rear events against each rater",
        *_events(report),
        "## Number audit",
        *_audit(report.audit),
        "## Tracking against DeepLabCut",
        *_tracking(report.tracking),
        "## Not measured here",
        "- **Supported vs unsupported rears:** the label set is binary (rearing / none), since "
        "Muse Spark could not tell them apart (ARCHITECTURE, open decision 3).\n"
        "- **Muse Spark's rear labels against the raters and the score** on the eval split, "
        "which ADR 0003 (point 3) asks the harness to keep measuring: it needs the behaviours "
        "stage run on the eval videos (new Muse Spark spend), so it is not in this report yet. "
        "Its train-split agreement is in ARCHITECTURE, open decision 3.",
    ]
    return "\n\n".join(parts) + "\n"


def _threshold(rearing: RearingAgreement) -> str:
    return f"{rearing.settings.count_threshold:.2f}"


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _yes_no(value: bool | None) -> str:
    return "—" if value is None else "yes" if value else "no"


def _range(bounds: tuple[float, float] | None) -> str:
    return "—" if bounds is None else f"{bounds[0]:.2f} to {bounds[1]:.2f}"


def _met(values: Sequence[float | None], bounds: tuple[float, float] | None) -> str:
    """Within the human-human range: every rater's agreement at least the lowest between two
    raters. Abstains when a value is undefined."""
    known = [v for v in values if v is not None]
    if bounds is None or len(known) != len(values):
        return "—"
    return "yes" if min(known) >= bounds[0] else "no"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _validation(report: EvalReport) -> list[str]:
    rearing = report.rearing
    audit = report.audit
    parts: list[str] = []
    if rearing is None:
        parts.append("Rearing agreement abstains: no eval video has a tracked run and labels.")
    else:
        threshold = _threshold(rearing)
        tolerance = f"±{rearing.settings.tolerance_s:g} s"
        metrics: tuple[tuple[str, Metric], ...] = (
            ("Rear count, ICC(2,1)", "count_icc"),
            (f"Rear events, F1 ({tolerance})", "f1"),
        )
        rows: list[list[str]] = []
        for name, against, bar in (
            (f"candidates scoring ≥ {threshold}", rearing.scored, True),
            ("every candidate", rearing.every_candidate, False),
        ):
            for label, metric in metrics:
                values = [against[r].value(metric) for r in rearing.raters]
                bounds = rearing.human_range(metric)
                rows.append(
                    [
                        f"{label}: {name}",
                        *(_num(v) for v in values),
                        _range(bounds),
                        "within the raters' range" if bar else "report only",
                        _met(values, bounds) if bar else "—",
                    ]
                )
        parts.append(
            _table(
                [
                    "Measure",
                    *(f"vs {r}" for r in rearing.raters),
                    "Raters against each other",
                    "Bar",
                    "Met",
                ],
                rows,
            )
        )
        parts.append(
            f"Over {len(rearing.per_video)} eval videos. The pipeline makes no rear count (ADR "
            f"0003): the eval counts the candidates scoring at least {threshold} (a threshold "
            f"set on the train split's counts, in the eval only) to measure how far such a "
            f"count would be from the raters, and every candidate beside it. A window and a "
            f"rear match when they overlap or the gap between them is at most "
            f"{rearing.settings.tolerance_s:g} s (not their onsets: a long window can match a rear "
            f"anywhere in it), each in one pair at most, pairing as many as possible."
        )
        if report.rearing_missing:
            parts.append(
                f"Left out, with no tracked run or no labels: {', '.join(report.rearing_missing)}."
            )
    rate = "—" if audit.rate is None else f"{100 * audit.rate:.0f}%"
    parts.append(
        _table(
            ["Report", "Pass rate", "Bar", "Met"],
            [
                [
                    "Number-audit pass rate",
                    f"{audit.passed} of {audit.total} study reports ({rate})",
                    "100%",
                    "—" if audit.rate is None else "yes" if audit.met else "no",
                ]
            ],
        )
    )
    return parts


def _per_video(report: EvalReport) -> list[str]:
    rearing = report.rearing
    if rearing is None:
        return ["No eval video to show."]
    threshold = _threshold(rearing)
    rows = [
        [v.video_id, str(v.scored), str(v.candidates), *(str(v.raters[r]) for r in rearing.raters)]
        for v in rearing.per_video
    ]
    return [
        _table(
            ["Video", f"Scoring ≥ {threshold}", "Candidates", *rearing.raters],
            rows,
        )
    ]


def _events(report: EvalReport) -> list[str]:
    rearing = report.rearing
    if rearing is None:
        return ["No eval video to show."]
    threshold = _threshold(rearing)
    rows: list[list[str]] = []
    labelled = {
        **{f"scoring ≥ {threshold} vs {r}": a for r, a in rearing.scored.items()},
        **{f"every candidate vs {r}": a for r, a in rearing.every_candidate.items()},
        **rearing.rater_pairs,
    }
    for who, agreement in labelled.items():
        e = agreement.events
        rows.append(
            [
                who,
                str(e.matched),
                str(e.windows),
                str(e.rears),
                _num(e.precision),
                _num(e.recall),
                _num(e.f1),
            ]
        )
    return [
        _table(
            ["Windows vs rears", "Matched", "Windows", "Rears", "Precision", "Recall", "F1"], rows
        ),
        "Between two raters, the first rater's rears are the windows.",
    ]


def _audit(audit: AuditRate) -> list[str]:
    if not audit.studies:
        return ["No study report found: the pass rate abstains."]
    rows = [
        [
            s.study_id,
            "—" if s.drafts is None else str(s.drafts),
            _yes_no(s.first_draft_passed),
            _yes_no(s.passed),
            ", ".join(s.unmatched) or s.reason or "",
        ]
        for s in audit.studies
    ]
    return [
        _table(["Study", "Drafts", "First draft passed", "Passes now", "Unmatched numbers"], rows),
        "Each report's shipped draft is audited again against the tables its tools return now.",
    ]


def _tracking(tracking: TrackingSummary) -> list[str]:
    parts: list[str] = []
    if tracking.videos:
        rows: list[list[str]] = []
        for v in tracking.videos:
            t = v.tracking
            d = t.distance_cm
            rows.append(
                [
                    v.video_id,
                    _num(t.tracked_pct, 1),
                    str(t.frames_compared),
                    _num(d and d.median),
                    _num(d and d.p95),
                    _num(t.over_disagreement_pct, 1),
                    "—" if t.best_lag is None else str(t.best_lag.lag_frames),
                ]
            )
        parts.append(
            _table(
                [
                    "Video",
                    "Tracked %",
                    "Frames compared",
                    "Median cm",
                    "p95 cm",
                    "≥ 5 cm %",
                    "Best lag (frames)",
                ],
                rows,
            )
        )
        parts.append(
            "DeepLabCut is a reference tracker, not ground truth: no pass bar. Distance is from "
            "our mask centroid to DLC's body centre."
        )
    else:
        parts.append("No tracking result yet: run `python -m eval track`.")
    if tracking.missing:
        parts.append(f"No tracking result: {', '.join(tracking.missing)}.")
    parts.append(f"Model API spend logged for these videos: ${tracking.spent_usd:.2f}.")
    return parts
