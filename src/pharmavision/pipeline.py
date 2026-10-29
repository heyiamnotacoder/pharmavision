"""The pipeline entry: the one test seam. A run goes ingest → arena, stops at the arena gate,
and continues with track → measure once a human confirms the zones."""

import shutil
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path

from .artifacts import (
    Arena,
    ArenaNotConfirmedError,
    IngestSettings,
    Manifest,
    Metrics,
    RunFolder,
)
from .segmenter import Segmenter
from .stages import arena, ingest, measure, track
from .stages.ingest import VideoTooShortError
from .templates import TestTemplate

__all__ = [
    "ArenaNotConfirmedError",
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


def start_run(
    video: Path,
    template: TestTemplate,
    segmenter: Segmenter,
    runs_dir: Path,
    settings: IngestSettings | None = None,
) -> RunFolder:
    """Ingest the video into a new run folder and detect the arena, which waits unconfirmed.
    Raises before creating the folder if the video is too short for the protocol."""
    settings = settings or IngestSettings()
    source = ingest.check_source(video, template)
    created_at = datetime.now(UTC)
    run = RunFolder(runs_dir / f"{created_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}")
    run.path.mkdir()
    try:
        normalised = ingest.normalise(video, run.video_path, settings)
        run.write_manifest(
            Manifest(
                run_id=run.run_id,
                created_at=created_at,
                code_version=version("pharmavision"),
                template=template,
                ingest=settings,
                source=source,
                video=normalised,
            )
        )
    except BaseException:
        shutil.rmtree(run.path)
        raise
    run.write_arena(arena.detect(run, segmenter))
    _record_model(run, "arena", segmenter)
    return run


def confirm_arena(run: RunFolder) -> Arena:
    """The human gate: the scientist has checked the zones over a frame of the video."""
    confirmed = run.arena().model_copy(update={"confirmed": True})
    run.write_arena(confirmed)
    return confirmed


def finish_run(run: RunFolder, segmenter: Segmenter) -> Metrics:
    """Every stage after the arena gate, in order."""
    for stage in Stage:
        run_stage(run, stage, segmenter)
    return run.metrics()


def run_stage(run: RunFolder, stage: Stage, segmenter: Segmenter) -> None:
    """Run one stage after the arena gate. Raises ArenaNotConfirmedError, writing nothing, while
    the arena is unconfirmed."""
    match stage:
        case Stage.TRACK:
            track.run(run, segmenter)
            _record_model(run, "track", segmenter)
        case Stage.MEASURE:
            measure.run(run)


def _record_model(run: RunFolder, stage: str, segmenter: Segmenter) -> None:
    manifest = run.manifest()
    models = {**manifest.models, stage: segmenter.model_id}
    run.write_manifest(manifest.model_copy(update={"models": models}))
