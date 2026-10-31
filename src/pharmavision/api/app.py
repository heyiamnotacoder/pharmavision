"""The pipeline over HTTP. A run is created from an uploaded video and a template name, and its
stages run as background jobs in this process, their status in SQLite: ingest and arena up to
the arena gate, then, once the zones are confirmed, every later stage. Precomputed run folders
are served read-only beside the live runs, so the demo never re-runs anything."""

import asyncio
import logging
import re
import shutil
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field

from ..artifacts import Artifact, RunFolder, StudyFolder
from ..judge import Judge
from ..pipeline import (
    START_STAGES,
    Stage,
    confirm_arena,
    finish_run,
    new_run_id,
    start_run,
    study_stats,
)
from ..segmenter import Segmenter
from ..templates import TestTemplate
from .jobs import RESTING, JobKind, JobStatus, JobStore

log = logging.getLogger(__name__)

RUN_STAGES: tuple[str, ...] = (*START_STAGES, *Stage)
"""Every run-level stage, in pipeline order."""

STUDY_STAGES: tuple[str, ...] = ("stats",)
"""The stages a study's report job runs. The report agent (T8b) joins them as "report"."""

RunArtifact = Literal["manifest", "arena", "track", "metrics", "events", "qc"]
_RUN_FILES: dict[str, str] = {
    "manifest": "manifest.json",
    "arena": "arena.json",
    "track": "track.parquet",
    "metrics": "metrics.json",
    "events": "events.parquet",
    "qc": "qc.json",
    "video": "video.mp4",
}
"""What each artifact is stored as in a run folder."""
_STAGE_ARTIFACTS: tuple[str, ...] = ("manifest", "arena", "track", "metrics", "events", "qc")
"""The artifact each of `RUN_STAGES` writes last, in the same order."""

StudyArtifact = Literal["design", "trials", "stats"]
_STUDY_FILES: dict[str, str] = {
    "design": "study.yaml",
    "trials": "trials.json",
    "stats": "stats.json",
}

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
"""A run or study id: a folder name that can't climb out of its parent."""


class ApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    runs_dir: Path = Path("runs")
    """Live runs, one folder each, beside `runs/model_cache` as the command line keeps them."""
    studies_dir: Path = Path("studies")
    """One folder per study, holding its `study.yaml`."""
    database: Path = Path("runs/jobs.sqlite3")
    precomputed_runs_dir: Path | None = None
    """Finished run folders served read-only, for the demo."""
    max_upload_bytes: int = Field(default=1 << 30, gt=0)
    """The SAM 3.1 Files API takes up to 1 GiB per video."""
    stream_poll_s: float = Field(default=0.5, gt=0)
    """How often an open progress stream looks for new stages."""


class RunView(BaseModel):
    """A run's status and the artifacts it has so far, each with the URL that serves it."""

    run_id: str
    status: JobStatus
    stage: str | None
    """The stage running now; after a failure, the stage that failed."""
    error: str | None
    read_only: bool
    """A precomputed run: served as it is, never re-run or confirmed."""
    stages_done: list[str]
    artifacts: dict[str, str]


class StudyView(BaseModel):
    """A study's report job and the artifacts it has so far. `status` None: no report has been
    asked for yet."""

    study_id: str
    status: JobStatus | None
    stage: str | None
    error: str | None
    stages_done: list[str]
    artifacts: dict[str, str]


def create_app(
    settings: ApiSettings,
    segmenter: Segmenter,
    judge: Judge,
    templates: Mapping[str, TestTemplate],
) -> FastAPI:
    """The API over the given model backends. Jobs left queued or running by an earlier
    process are marked failed: nothing is running them now."""
    store = JobStore(settings.database)
    store.fail_interrupted("interrupted: the API restarted while this job ran")
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    uploads = settings.runs_dir / ".uploads"
    app = FastAPI(title="PharmaVision")

    def find(run_id: str) -> tuple[RunFolder, bool]:
        """The run's folder, and whether it is precomputed (read-only). 404 for no such run."""
        if _ID.fullmatch(run_id):
            if store.get("run", run_id) is not None:
                return RunFolder(settings.runs_dir / run_id), False
            if settings.precomputed_runs_dir is not None:
                run = RunFolder(settings.precomputed_runs_dir / run_id)
                if (run.path / "manifest.json").is_file():
                    return run, True
        raise HTTPException(404, f"no run {run_id}")

    def progress(run_id: str, after: int = 0) -> tuple[RunView, list[tuple[int, str]]]:
        """The run's view, and each stage it finished after event `after`, with its event id.
        A precomputed run's stages are the artifacts in its folder, numbered in order."""
        run, read_only = find(run_id)
        if read_only:
            done = _stages_in(run)
            complete = len(done) == len(RUN_STAGES)
            status: JobStatus = "done" if complete else "failed"
            stage = None
            error = None if complete else f"the precomputed run stops after {done[-1]}"
            events = list(enumerate(done, start=1))
        else:
            job = store.get("run", run_id)
            assert job is not None  # find() saw it, and jobs are never deleted
            status, stage, error = job.status, job.stage, job.error
            events = [(s.seq, s.stage) for s in store.stages("run", run_id)]
        view = RunView(
            run_id=run_id,
            status=status,
            stage=stage,
            error=error,
            read_only=read_only,
            stages_done=[stage for _, stage in events],
            artifacts=_artifact_urls(run),
        )
        return view, [(seq, stage) for seq, stage in events if seq > after]

    def view(run_id: str) -> RunView:
        return progress(run_id)[0]

    @app.get("/runs")
    def list_runs() -> list[RunView]:
        """Every live run, and every precomputed one."""
        ids = {job.id for job in store.jobs("run")}
        if settings.precomputed_runs_dir is not None and settings.precomputed_runs_dir.is_dir():
            ids |= {
                p.name for p in settings.precomputed_runs_dir.iterdir() if _ID.fullmatch(p.name)
            }
        views: list[RunView] = []
        for run_id in sorted(ids):
            try:
                views.append(view(run_id))
            except HTTPException:  # a precomputed folder without a manifest
                continue
        return views

    @app.post("/runs", status_code=202)
    async def create_run(
        background: BackgroundTasks,
        video: Annotated[UploadFile, File()],
        template: Annotated[str, Form()],
    ) -> RunView:
        """Upload a video and choose its test template. Ingest and arena detection run in the
        background, up to the arena gate."""
        if template not in templates:
            raise HTTPException(422, f"no template {template!r}; choose from {sorted(templates)}")
        run_id = new_run_id()
        upload = uploads / run_id / _filename(video.filename)
        upload.parent.mkdir(parents=True)
        size = 0
        with upload.open("wb") as out:
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    out.close()
                    shutil.rmtree(upload.parent)
                    raise HTTPException(413, f"videos up to {settings.max_upload_bytes} bytes")
                out.write(chunk)
        store.start("run", run_id)

        def work(done: Callable[[str], None]) -> JobStatus:
            try:
                start_run(
                    upload,
                    templates[template],
                    segmenter,
                    settings.runs_dir,
                    run_id=run_id,
                    on_stage_done=done,
                )
            finally:
                shutil.rmtree(upload.parent, ignore_errors=True)
            return "awaiting zone confirmation"

        background.add_task(_run_job, store, "run", run_id, START_STAGES, work)
        return view(run_id)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> RunView:
        return view(run_id)

    @app.post("/runs/{run_id}/arena/confirm", status_code=202)
    def confirm(run_id: str, background: BackgroundTasks) -> RunView:
        """The human gate: the scientist has checked the zones. Every later stage then runs in
        the background. Refused (409) unless the run is waiting at the gate."""
        run, read_only = find(run_id)
        if read_only:
            raise HTTPException(409, f"run {run_id} is precomputed: served as it is")
        if not store.claim("run", run_id, expected="awaiting zone confirmation"):
            raise HTTPException(409, f"run {run_id} is {view(run_id).status}, not at the gate")

        def work(done: Callable[[str], None]) -> JobStatus:
            confirm_arena(run)
            finish_run(run, segmenter, judge, on_stage_done=done)
            return "done"

        background.add_task(_run_job, store, "run", run_id, list(Stage), work)
        return view(run_id)

    @app.get("/runs/{run_id}/stream", response_class=EventSourceResponse)
    async def stream(
        run_id: str,
        _: Annotated[RunView, Depends(view)],  # a 404 before the stream starts
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """Server-sent events: one `stage` event per stage the run has finished, then a
        `status` event once it rests (at the gate, done or failed), and the stream ends. Each
        event's id lets a reconnecting client resume after the last one it saw."""
        after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
        while True:
            # The status is read before the stages, so a resting job's stages are all there.
            current, stages = progress(run_id, after)
            for seq, stage in stages:
                yield ServerSentEvent(id=str(seq), event="stage", data={"stage": stage})
                after = seq
            if current.status in RESTING:
                yield ServerSentEvent(
                    event="status",
                    data=current.model_dump(include={"status", "stage", "error"}),
                )
                return
            await asyncio.sleep(settings.stream_poll_s)

    @app.get("/runs/{run_id}/artifacts/{name}", response_model=None)
    def get_artifact(run_id: str, name: RunArtifact) -> Response:
        """One artifact as JSON, validated against its type as it is read."""
        run, _ = find(run_id)
        if not (run.path / _RUN_FILES[name]).is_file():
            raise HTTPException(404, f"run {run_id} has no {name} yet")
        artifact: Artifact = getattr(run, name)()
        return Response(artifact.model_dump_json(), media_type="application/json")

    @app.get("/runs/{run_id}/video", response_model=None)
    def get_video(run_id: str) -> FileResponse:
        """The normalised video the stages saw, so overlays line up frame for frame."""
        run, _ = find(run_id)
        if not run.video_path.is_file():
            raise HTTPException(404, f"run {run_id} has no video yet")
        return FileResponse(run.video_path, media_type="video/mp4")

    def study_folder(study_id: str) -> StudyFolder:
        study = StudyFolder(settings.studies_dir / study_id)
        if not _ID.fullmatch(study_id) or not (study.path / "study.yaml").is_file():
            raise HTTPException(404, f"no study {study_id}")
        return study

    def study_view(study_id: str) -> StudyView:
        study = study_folder(study_id)
        job = store.get("study", study_id)
        return StudyView(
            study_id=study_id,
            status=job and job.status,
            stage=job and job.stage,
            error=job and job.error,
            stages_done=[s.stage for s in store.stages("study", study_id)],
            artifacts={
                name: f"/studies/{study_id}/artifacts/{name}"
                for name, file in _STUDY_FILES.items()
                if (study.path / file).is_file()
            },
        )

    @app.post("/studies/{study_id}/report", status_code=202)
    def generate_report(study_id: str, background: BackgroundTasks) -> StudyView:
        """Generate the study's report in the background, from its `study.yaml` and its runs'
        artifacts. Refused (409) while a report for it is queued or running."""
        study = study_folder(study_id)
        if not store.start("study", study_id):
            raise HTTPException(409, f"a report for study {study_id} is already running")

        def work(done: Callable[[str], None]) -> JobStatus:
            study_stats(study, settings.runs_dir)
            done("stats")
            return "done"

        background.add_task(_run_job, store, "study", study_id, STUDY_STAGES, work)
        return study_view(study_id)

    @app.get("/studies/{study_id}")
    def get_study(study_id: str) -> StudyView:
        return study_view(study_id)

    @app.get("/studies/{study_id}/artifacts/{name}", response_model=None)
    def get_study_artifact(study_id: str, name: StudyArtifact) -> Response:
        study = study_folder(study_id)
        if not (study.path / _STUDY_FILES[name]).is_file():
            raise HTTPException(404, f"study {study_id} has no {name} yet")
        artifact: Artifact = getattr(study, name)()
        return Response(artifact.model_dump_json(), media_type="application/json")

    return app


def _run_job(
    store: JobStore,
    kind: JobKind,
    id: str,
    stages: Sequence[str],
    work: Callable[[Callable[[str], None]], JobStatus],
) -> None:
    """Run `work` as the job's body: it hears each stage as it finishes and returns the status
    the job rests in. Any error fails the job, at the stage that raised it, with its message."""
    remaining = list(stages)
    store.update(kind, id, "running", remaining[0])

    def done(stage: str) -> None:
        store.stage_done(kind, id, stage)
        remaining.remove(stage)
        store.update(kind, id, "running", remaining[0] if remaining else None)

    try:
        status = work(done)
    except Exception as e:
        log.exception("%s %s failed", kind, id)
        store.update(kind, id, "failed", remaining[0] if remaining else None, str(e) or repr(e))
    else:
        store.update(kind, id, status)


def _stages_in(run: RunFolder) -> list[str]:
    """The stages whose artifacts are in the folder, up to the first one missing."""
    done: list[str] = []
    for stage, name in zip(RUN_STAGES, _STAGE_ARTIFACTS, strict=True):
        if not (run.path / _RUN_FILES[name]).is_file():
            break
        done.append(stage)
    return done


def _artifact_urls(run: RunFolder) -> dict[str, str]:
    base = f"/runs/{run.run_id}"
    return {
        name: f"{base}/video" if name == "video" else f"{base}/artifacts/{name}"
        for name, file in _RUN_FILES.items()
        if (run.path / file).is_file()
    }


def _filename(name: str | None) -> str:
    """The upload's own name, without any folders, which the manifest records."""
    base = Path(name or "").name
    return base if base not in ("", ".", "..") else "video"
