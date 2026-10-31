"""Report (judges, then the number audit), at study level: the agent reads the study through
read-only tools and writes the ARRIVE 2.0 Essential 10 methods and results; every number in it
must match a value the tools returned, or the agent is re-prompted with the failing numbers.
When its last draft fails too, the report ships with those sentences struck out and flagged."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from .. import arrive
from ..artifacts import (
    Report,
    ReportDraft,
    ReportSettings,
    RunFolder,
    StudyFolder,
    StudyNotReadyError,
)
from ..audit import audit_numbers, known_values
from ..audit.known import Json
from ..judge import AuditFailure, Judge, RejectedDraft, ReportBrief, ReportTool

NO_REAR_COUNT = (
    "The pipeline makes no rear count: neither the rear candidates' scores nor the model's "
    "labels agreed with human raters per video (ADR 0003). Each candidate window is evidence "
    "only."
)

_EVENT_INTERNALS = {"low_confidence", "source"}
"""Fields every low-confidence event shares: the summary lists only those events, all judged by
the model."""

DESCRIPTIONS: dict[str, str] = {
    "get_protocol": "The study design (groups, treatments, doses, trials by blinding code), "
    "the test, the arena, the video, the models and the thresholds every measure, QC rule and "
    "statistical test used.",
    "get_metrics_table": "The per-trial metrics table: each trial's blinding code, group, "
    "metric values and every QC exclusion in it. A null value could not be computed.",
    "get_stats": "Per metric: n analysed, excluded and missing per group, mean, SD and median, "
    "the normality and variance checks, the test the rule chose and why, its statistic, "
    "degrees of freedom and p, the effect size with its interval, and any post-hoc pairs.",
    "get_qc": "Per trial: every QC flag with its frames, severity and reason, every model "
    "review of a flag with its verdict and reason, and every exclusion.",
    "get_events_summary": "Per trial: the number of rear candidate windows and every "
    "low-confidence behaviour event, and why the pipeline makes no rear count.",
}


def run(study: StudyFolder, runs_dir: Path, judge: Judge, settings: ReportSettings) -> Report:
    tables = tool_outputs(study, runs_dir)
    known = known_values(tables)
    tools = tuple(ReportTool(name, DESCRIPTIONS[name], _reader(t)) for name, t in tables.items())
    drafts: list[ReportDraft] = []
    rejected: list[RejectedDraft] = []
    for _ in range(settings.max_drafts):
        text = judge.write_report(ReportBrief(arrive.SECTIONS, tools, tuple(rejected)))
        markdown = arrive.assemble(text)
        audit = audit_numbers(markdown, known, settings.audit)
        drafts.append(ReportDraft(text=text, markdown=markdown, audit=audit))
        if audit.passed:
            break
        failures = tuple(AuditFailure(n.text, n.sentence) for n in audit.unmatched)
        rejected.append(RejectedDraft(text, failures))
    last = drafts[-1]
    report = Report(
        settings=settings,
        model_id=judge.model_id,
        drafts=drafts,
        markdown=arrive.strike(last.markdown, last.audit.unmatched),
    )
    study.write_report(report)
    return report


def tool_outputs(study: StudyFolder, runs_dir: Path) -> dict[str, Json]:
    """What each tool returns, by tool name: views of the stored artifacts that select and
    rename fields, never compute one. A count is a stored list's length, which the audit
    knows anyway."""
    design = study.design()
    try:
        trials = study.trials()
        stats = study.stats()
    except FileNotFoundError as missing:
        raise StudyNotReadyError(f"study {design.name!r}: run the study stats first") from missing
    runs = {t.blinding_code: RunFolder(runs_dir / t.run_id) for t in design.trials}
    manifest = next(iter(runs.values())).manifest()
    template = manifest.template
    measure, qc = manifest.measure, manifest.qc
    protocol: Json = {
        "study": design.name,
        "groups": [g.model_dump(mode="json") for g in design.groups],
        "reference_group": design.groups[0].name,
        "trials": [{"blinding_code": t.blinding_code, "group": t.group} for t in design.trials],
        "test": {
            "type": template.test_type,
            "animal": template.animal_prompt,
            "arena_width_cm": template.arena_width_cm,
            "arena_height_cm": template.arena_height_cm,
            "centre_area_fraction": template.centre_area_fraction,
            "duration_s": template.protocol_duration_s,
        },
        "video": {"camera": "top view", "analysed_fps": manifest.video.fps},
        "models": manifest.models,
        "software": f"PharmaVision {manifest.code_version}",
        "measure": {
            "max_interpolated_gap_s": measure.max_gap_s,
            "thigmotaxis_wall_cm": measure.thigmotaxis_wall_cm,
            "immobility_change_max": measure.immobility_change_max,
            "immobility_min_s": measure.immobility_min_s,
        },
        "qc": {
            "max_tracking_loss_fraction": qc.max_tracking_loss_fraction,
            "out_of_arena_margin_cm": qc.out_of_arena_margin_cm,
            "second_mask_min_area_cm2": qc.second_mask_min_area_cm2,
            "max_speed_cm_s": qc.max_speed_cm_s,
            "model_review": qc.review is not None,
        },
    }
    qc_by_trial: list[Json] = []
    events_by_trial: list[Json] = []
    for code, run in runs.items():
        qc_by_trial.append({"blinding_code": code, **run.qc().model_dump(mode="json")})
        low = [e for e in run.events().events if e.low_confidence]
        events_by_trial.append(
            {
                "blinding_code": code,
                "rear_candidate_windows": len(run.metrics().rear_candidates),
                "low_confidence_events": [
                    e.model_dump(mode="json", exclude=_EVENT_INTERNALS) for e in low
                ],
            }
        )
    return {
        "get_protocol": protocol,
        "get_metrics_table": _without_run_ids(trials.model_dump(mode="json")),
        "get_stats": stats.model_dump(mode="json"),
        "get_qc": {"trials": qc_by_trial},
        "get_events_summary": {"rear_count": NO_REAR_COUNT, "trials": events_by_trial},
    }


def _without_run_ids(table: Mapping[str, Json]) -> Json:
    """The trials as the scorer saw them: by blinding code, not by run folder."""
    rows = table["trials"]
    assert isinstance(rows, list)
    return {
        "trials": [
            {k: v for k, v in row.items() if k != "run_id"}
            for row in rows
            if isinstance(row, Mapping)
        ]
    }


def _reader(table: Json) -> Callable[[], str]:
    text = json.dumps(table, indent=1)
    return lambda: text
