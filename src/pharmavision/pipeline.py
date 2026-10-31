"""The pipeline entry: the one test seam. A run goes ingest → arena, stops at the arena gate,
and continues with track → measure → behaviours → QC once a human confirms the zones. At study
level, the stats run over a study's design and its finished runs, then the report over both."""

import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path

from .artifacts import (
    Arena,
    ArenaNotConfirmedError,
    ArenaSettings,
    BehaviourSettings,
    IngestSettings,
    Manifest,
    MeasureSettings,
    Metrics,
    QcSettings,
    Report,
    ReportSettings,
    RunFolder,
    StatsSettings,
    StudyFolder,
    StudyNotReadyError,
    StudyStats,
)
from .judge import Judge
from .segmenter import Segmenter
from .stages import arena, behaviours, ingest, measure, qc, report, stats, track
from .stages.arena import ArenaNotFoundError
from .stages.ingest import VideoTooShortError
from .templates import TestTemplate

__all__ = [
    "ArenaNotConfirmedError",
    "ArenaNotFoundError",
    "RunFolder",
    "Stage",
    "StudyFolder",
    "StudyNotReadyError",
    "VideoTooShortError",
    "confirm_arena",
    "finish_run",
    "new_run_id",
    "run_stage",
    "start_run",
    "study_report",
    "study_stats",
]


class Stage(StrEnum):
    """The stages after the arena gate, in pipeline order."""

    TRACK = "track"
    MEASURE = "measure"
    BEHAVIOURS = "behaviours"
    QC = "qc"


START_STAGES = ("ingest", "arena")
"""The stages `start_run` runs, in order, before the arena gate."""

StageDone = Callable[[str], None]
"""Told each stage's name as it finishes, for progress reporting."""


def new_run_id(now: datetime | None = None) -> str:
    """A fresh run id: the UTC time it was made, then a random suffix."""
    return f"{now or datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def start_run(
    video: Path,
    template: TestTemplate,
    segmenter: Segmenter,
    runs_dir: Path,
    ingest_settings: IngestSettings | None = None,
    arena_settings: ArenaSettings | None = None,
    measure_settings: MeasureSettings | None = None,
    qc_settings: QcSettings | None = None,
    behaviour_settings: BehaviourSettings | None = None,
    run_id: str | None = None,
    on_stage_done: StageDone = lambda _: None,
) -> RunFolder:
    """Ingest the video into a new run folder and detect the arena, which waits unconfirmed.
    All or nothing: if the video is too short or no floor is found, no run folder is left.
    `run_id` names the folder (a fresh `new_run_id()` by default); `on_stage_done` hears
    "ingest" and then "arena"."""
    ingest_settings = ingest_settings or IngestSettings()
    source = ingest.check_source(video, template)
    created_at = datetime.now(UTC)
    run = RunFolder(runs_dir / (run_id or new_run_id(created_at)))
    run.path.mkdir()
    try:
        normalised = ingest.normalise(video, run.video_path, ingest_settings)
        run.write_manifest(
            Manifest(
                run_id=run.run_id,
                created_at=created_at,
                code_version=version("pharmavision"),
                template=template,
                ingest=ingest_settings,
                arena=arena_settings or ArenaSettings(),
                measure=measure_settings or MeasureSettings(),
                behaviours=behaviour_settings or BehaviourSettings(),
                qc=qc_settings or QcSettings(),
                source=source,
                video=normalised,
            )
        )
        on_stage_done("ingest")
        run.write_arena(arena.detect(run, segmenter))
        _record_model(run, "arena", segmenter.model_id)
        on_stage_done("arena")
    except BaseException:
        shutil.rmtree(run.path)
        raise
    return run


def confirm_arena(run: RunFolder) -> Arena:
    """The human gate: the scientist has checked the zones over a frame of the video."""
    confirmed = run.arena().model_copy(update={"confirmed": True})
    run.write_arena(confirmed)
    return confirmed


def finish_run(
    run: RunFolder,
    segmenter: Segmenter,
    judge: Judge,
    on_stage_done: StageDone = lambda _: None,
) -> Metrics:
    """Every stage after the arena gate, in order; `on_stage_done` hears each one's name."""
    for stage in Stage:
        run_stage(run, stage, segmenter, judge)
        on_stage_done(stage)
    return run.metrics()


def run_stage(run: RunFolder, stage: Stage, segmenter: Segmenter, judge: Judge) -> None:
    """Run one stage after the arena gate. Raises ArenaNotConfirmedError, writing nothing, while
    the arena is unconfirmed."""
    match stage:
        case Stage.TRACK:
            track.run(run, segmenter)
            _record_model(run, "track", segmenter.model_id)
        case Stage.MEASURE:
            measure.run(run)
        case Stage.BEHAVIOURS:
            behaviours.run(run, judge)
            _record_model(run, "behaviours", judge.model_id)
        case Stage.QC:
            qc.run(run, segmenter, judge)
            if run.qc().reviews:
                _record_model(run, "qc_hands", segmenter.model_id)
                _record_model(run, "qc", judge.model_id)


def study_stats(
    study: StudyFolder, runs_dir: Path, settings: StatsSettings | None = None
) -> StudyStats:
    """The study-level stats: every trial in `study.yaml` read from its run in `runs_dir`, the
    per-trial metrics table and the stats written to the study folder, and any report on
    earlier stats removed. Raises StudyNotReadyError, writing nothing, while any of its runs
    has not finished QC."""
    return stats.run(study, runs_dir, settings or StatsSettings())


def study_report(
    study: StudyFolder, runs_dir: Path, judge: Judge, settings: ReportSettings | None = None
) -> Report:
    """The study's methods and results, written by the judge from the stored tables and gated
    by the number audit; `report.json` and `report.md` written to the study folder. Raises
    StudyNotReadyError, writing nothing, until the study stats have run."""
    return report.run(study, runs_dir, judge, settings or ReportSettings())


def _record_model(run: RunFolder, stage: str, model_id: str) -> None:
    manifest = run.manifest()
    models = {**manifest.models, stage: model_id}
    run.write_manifest(manifest.model_copy(update={"models": models}))
