"""Report (judges, then the number audit), at study level: the agent reads the study through
read-only tools and writes the ARRIVE 2.0 Essential 10 methods and results; every number in it
must match a value the tools returned, or the agent is re-prompted with the failing numbers.
When its last draft fails too, the report ships with those sentences struck out and flagged."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import arrive
from ..artifacts import (
    Exclusion,
    Manifest,
    QcFlag,
    Report,
    ReportDraft,
    ReportSettings,
    RunFolder,
    StudyDesign,
    StudyFolder,
    StudyNotReadyError,
    StudyStats,
    TrialTable,
)
from ..audit import audit_numbers, known_values, rule_reason_values
from ..audit.known import Json
from ..judge import AuditFailure, Judge, RejectedDraft, ReportBrief, ReportTool

NO_REAR_COUNT = (
    "The pipeline makes no rear count: neither the rear candidates' scores nor the model's "
    "labels agreed with human raters per video (ADR 0003). Each candidate window is evidence "
    "only."
)


def run(study: StudyFolder, runs_dir: Path, judge: Judge, settings: ReportSettings) -> Report:
    tables = tool_outputs(study, runs_dir)
    known = known_values(tables)
    tools = tuple(
        ReportTool(tool.name, tool.description, _reader(tables[tool.name])) for tool in TOOLS
    )
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


@dataclass(frozen=True)
class _Study:
    """Everything the tools read, loaded once and checked to describe one protocol."""

    design: StudyDesign
    trials: TrialTable
    stats: StudyStats
    runs: dict[str, RunFolder]
    """By blinding code, in the design's order."""
    manifest: Manifest
    """The first trial's: every trial's agrees with it on what the protocol tool reports."""


@dataclass(frozen=True)
class _Tool:
    name: str
    description: str
    output: Callable[[_Study], Json]
    """A view of the stored artifacts: it selects and renames fields and computes nothing. A
    count is a stored list's length, which the audit knows anyway."""


def tool_outputs(study: StudyFolder, runs_dir: Path) -> dict[str, Json]:
    """What each tool returns, by tool name, in `TOOLS` order."""
    loaded = _load(study, runs_dir)
    return {tool.name: tool.output(loaded) for tool in TOOLS}


def _load(study: StudyFolder, runs_dir: Path) -> _Study:
    design = study.design()
    try:
        trials = study.trials()
        stats = study.stats()
    except FileNotFoundError as missing:
        raise StudyNotReadyError(f"study {design.name!r}: run the study stats first") from missing
    runs = {t.blinding_code: RunFolder(runs_dir / t.run_id) for t in design.trials}
    manifests = {code: run.manifest() for code, run in runs.items()}
    first = next(iter(manifests.values()))
    differing = [code for code, m in manifests.items() if _protocol(m) != _protocol(first)]
    if differing:
        raise StudyNotReadyError(
            f"trials recorded or judged differently (template, video, behaviour settings or "
            f"models): {', '.join(differing)} against {design.trials[0].blinding_code}"
        )
    return _Study(design, trials, stats, runs, first)


def _protocol(manifest: Manifest) -> tuple[object, ...]:
    """What the protocol tool reports from one manifest; the stats stage already checked that
    the measure and QC settings agree. Not the floor prompt: it only found the arena, which the
    scientist then confirmed, and the eval tries others when the first finds no floor."""
    template = manifest.template.model_copy(update={"floor_prompt": ""})
    return (template, manifest.video.fps, manifest.behaviours, manifest.models)


def _protocol_output(study: _Study) -> Json:
    manifest = study.manifest
    template, measure, qc = manifest.template, manifest.measure, manifest.qc
    return {
        "study": study.design.name,
        "groups": [g.model_dump(mode="json") for g in study.design.groups],
        "reference_group": study.design.groups[0].name,
        "trials": [
            {"blinding_code": t.blinding_code, "group": t.group} for t in study.design.trials
        ],
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


def _metrics_table_output(study: _Study) -> Json:
    """The trials as the scorer saw them: by blinding code, not by run folder."""
    return {
        "trials": [
            {
                "blinding_code": row.blinding_code,
                "group": row.group,
                "values": row.model_dump(mode="json")["values"],
                "exclusions": [_exclusion(e) for e in row.exclusions],
            }
            for row in study.trials.trials
        ]
    }


def _stats_output(study: _Study) -> Json:
    """The results, and only the settings the methods cite: no version, no bootstrap seed."""
    settings = study.stats.settings
    return {
        "rule": settings.model_dump(
            mode="json", include={"normality_alpha", "variance_alpha", "ci_level", "min_group_n"}
        ),
        "metrics": [m.model_dump(mode="json") for m in study.stats.metrics],
    }


def _qc_output(study: _Study) -> Json:
    trials: list[Json] = []
    for code, run in study.runs.items():
        qc = run.qc()
        trials.append(
            {
                "blinding_code": code,
                "flags": [_flag(f) for f in qc.flags],
                "reviews": [
                    r.model_dump(mode="json", exclude={"flag"}) | {"flag": _flag(r.flag)}
                    for r in qc.reviews
                ],
                "exclusions": [_exclusion(e) for e in qc.exclusions],
            }
        )
    return {"trials": trials}


def _events_summary_output(study: _Study) -> Json:
    """No model confidence: it is the model's own number, not a measured one (NFR-1). An event
    is low-confidence when the model's confidence was under `min_confidence`."""
    trials: list[Json] = []
    for code, run in study.runs.items():
        low = [e for e in run.events().events if e.low_confidence]
        trials.append(
            {
                "blinding_code": code,
                "rear_candidate_windows": len(run.metrics().rear_candidates),
                "low_confidence_events": [
                    e.model_dump(mode="json", include={"first_frame", "last_frame", "label"})
                    | {"model_rationale": e.rationale}
                    for e in low
                ],
            }
        )
    return {
        "rear_count": NO_REAR_COUNT,
        "min_confidence": study.manifest.behaviours.min_confidence,
        "trials": trials,
    }


def _flag(flag: QcFlag) -> Json:
    """A flag, its rule-written reason's numbers as fields: QC flags are always a rule's."""
    return flag.model_dump(mode="json") | {"reason_values": rule_reason_values(flag.reason)}


def _exclusion(exclusion: Exclusion) -> Json:
    """An exclusion; a rule's with its reason's numbers as fields, a model's without."""
    dumped = exclusion.model_dump(mode="json")
    if exclusion.source == "rule":
        dumped["reason_values"] = rule_reason_values(exclusion.reason)
    return dumped


def _reader(table: Json) -> Callable[[], str]:
    text = json.dumps(table, indent=1)
    return lambda: text


TOOLS: tuple[_Tool, ...] = (
    _Tool(
        "get_protocol",
        "The study design (groups, treatments, doses, trials by blinding code), the test, the "
        "arena, the video, the models and the thresholds the measures and QC rules used.",
        _protocol_output,
    ),
    _Tool(
        "get_metrics_table",
        "The per-trial metrics table: each trial's blinding code, group, metric values and "
        "every QC exclusion in it. A null value could not be computed.",
        _metrics_table_output,
    ),
    _Tool(
        "get_stats",
        "The test-selection rule's thresholds and, per metric: n analysed, excluded and missing "
        "per group, mean, SD and median, the normality and variance checks, the test the rule "
        "chose and why, its statistic, degrees of freedom and p, the effect size with its "
        "interval, and any post-hoc pairs.",
        _stats_output,
    ),
    _Tool(
        "get_qc",
        "Per trial: every QC flag with its frames, severity and reason, every model review of a "
        "flag with its verdict and reason, and every exclusion. `reason_values` holds the "
        "numbers in a rule's reason.",
        _qc_output,
    ),
    _Tool(
        "get_events_summary",
        "Per trial: the number of rear candidate windows and every low-confidence behaviour "
        "event with the model's rationale, and why the pipeline makes no rear count.",
        _events_summary_output,
    ),
)
