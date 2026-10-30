"""The pipeline entry: the one test seam. A run goes ingest → arena, stops at the arena gate,
and continues with track → measure → behaviours → QC once a human confirms the zones."""

import shutil
import uuid
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
    RunFolder,
)
from .judge import Judge
from .segmenter import Segmenter
from .stages import arena, behaviours, ingest, measure, qc, track
from .stages.arena import ArenaNotFoundError
from .stages.ingest import VideoTooShortError
from .templates import TestTemplate

__all__ = [
    "ArenaNotConfirmedError",
    "ArenaNotFoundError",
    "RunFolder",
    "Stage",
    "VideoTooShortError",
    "confirm_arena",
    "finish_run",
    "run_stage",
    "start_run",
]


class Stage(StrEnum):
    """The stages after the arena gate, in pipeline order."""

    TRACK = "track"
    MEASURE = "measure"
    BEHAVIOURS = "behaviours"
    QC = "qc"


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
) -> RunFolder:
    """Ingest the video into a new run folder and detect the arena, which waits unconfirmed.
    All or nothing: if the video is too short or no floor is found, no run folder is left."""
    ingest_settings = ingest_settings or IngestSettings()
    source = ingest.check_source(video, template)
    created_at = datetime.now(UTC)
    run = RunFolder(runs_dir / f"{created_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}")
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
        run.write_arena(arena.detect(run, segmenter))
        _record_model(run, "arena", segmenter.model_id)
    except BaseException:
        shutil.rmtree(run.path)
        raise
    return run


def confirm_arena(run: RunFolder) -> Arena:
    """The human gate: the scientist has checked the zones over a frame of the video."""
    confirmed = run.arena().model_copy(update={"confirmed": True})
    run.write_arena(confirmed)
    return confirmed


def finish_run(run: RunFolder, segmenter: Segmenter, judge: Judge) -> Metrics:
    """Every stage after the arena gate, in order."""
    for stage in Stage:
        run_stage(run, stage, segmenter, judge)
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
            qc.run(run)


def _record_model(run: RunFolder, stage: str, model_id: str) -> None:
    manifest = run.manifest()
    models = {**manifest.models, stage: model_id}
    run.write_manifest(manifest.model_copy(update={"models": models}))
