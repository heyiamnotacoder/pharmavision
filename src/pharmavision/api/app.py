"""The pipeline over HTTP. A run is created from an uploaded video and a template name, and its
stages run as background jobs in this process, their status in SQLite: ingest and arena up to
the arena gate, then, once the zones are confirmed, every later stage. Precomputed run folders
are served read-only beside the live runs, so the demo never re-runs anything."""

import asyncio
import logging
import re
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field

from ..artifacts import RunArtifactName, RunFolder, StudyArtifactName, StudyFolder
from ..judge import Judge
from ..pipeline import (
    START_STAGES,
    Stage,
    StageDone,
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

STAGE_ARTIFACTS: dict[str, RunArtifactName] = {
    "ingest": "manifest",
    "arena": "arena",
    Stage.TRACK: "track",
    Stage.MEASURE: "metrics",
    Stage.BEHAVIOURS: "events",
    Stage.QC: "qc",
}
"""Every run-level stage in pipeline order, and the artifact it writes last."""

STUDY_STAGES: tuple[str, ...] = ("stats",)
"""The stages a study's report job runs. The report agent (T8b) joins them as "report"."""

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
"""A run or study id: a folder name that can't climb out of its parent."""

_FORM_OVERHEAD_BYTES = 64 * 1024
"""Room for the multipart boundaries and the template field beside the video."""


class ApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runs_dir: Path = Path("runs")
    """Live runs, one folder each, beside `runs/model_cache` as the command line keeps them.
    Uploads wait in `runs/.uploads/` until ingest has read them."""
    studies_dir: Path = Path("studies")
    """One folder per study, holding its `study.yaml`."""
    database: Path = Path("runs/jobs.sqlite3")
    """The background jobs' status, which outlives the process; the artifacts stay in the
    run and study folders."""
    precomputed_runs_dir: Path | None = None
    """Run folders served read-only, for the demo."""
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
    process are marked failed: nothing is running them now. Run it in one process only."""
    store = JobStore(settings.database)
    store.fail_interrupted("interrupted: the API restarted while this job ran")
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    uploads = settings.runs_dir / ".uploads"
    too_large = f"videos up to {settings.max_upload_bytes} bytes"
    app = FastAPI(title="PharmaVision")

    @app.middleware("http")
    async def refuse_large_uploads(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Before the body is read: the form parser spools all of it to disk first. A body
        sent without a length is caught as it is copied instead."""
        length = request.headers.get("content-length", "")
        limit = settings.max_upload_bytes + _FORM_OVERHEAD_BYTES
        if request.method == "POST" and length.isdigit() and int(length) > limit:
            return JSONResponse({"detail": too_large}, status_code=413)
        return await call_next(request)

    def find(run_id: str) -> tuple[RunFolder, bool]:
        """The run's folder, and whether it is precomputed (read-only). 404 for no such run."""
        if _ID.fullmatch(run_id):
            if store.get("run", run_id) is not None:
                return RunFolder(settings.runs_dir / run_id), False
            if settings.precomputed_runs_dir is not None:
                run = RunFolder(settings.precomputed_runs_dir / run_id)
                if run.has("manifest"):
                    return run, True
        raise HTTPException(404, f"no run {run_id}")

    def progress(run_id: str, after: int = 0) -> tuple[RunView, list[tuple[int, str]]]:
        """The run's view, and each stage it finished after event `after`, with its event id.
        The stages are the whole run's, from ingest on, not only the latest job's. A
        precomputed run's are the artifacts in its folder, numbered in order."""
        run, read_only = find(run_id)
        if read_only:
            done = _stages_in(run)
            status, stage, error = _precomputed_status(run, done)
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
            artifacts=_artifact_urls(f"/runs/{run_id}", run),
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
    def create_run(
        background: BackgroundTasks,
        video: Annotated[UploadFile, File()],
        template: Annotated[str, Form()],
    ) -> RunView:
        """Upload a video and choose its test template. Ingest and arena detection run in the
        background, up to the arena gate. A failed run is not retried: processing the video
        again is a new run."""
        if template not in templates:
            raise HTTPException(422, f"no template {template!r}; choose from {sorted(templates)}")
        run_id = new_run_id()
        upload = uploads / run_id / _filename(video.filename)
        upload.parent.mkdir(parents=True)
        try:
            _copy(video, upload, settings.max_upload_bytes)
        except BaseException:
            shutil.rmtree(upload.parent, ignore_errors=True)
            raise
        store.start("run", run_id)

        def work(done: StageDone) -> JobStatus:
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

        def work(done: StageDone) -> JobStatus:
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
            current, stages = await asyncio.to_thread(progress, run_id, after)
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
    def get_artifact(run_id: str, name: RunArtifactName) -> Response:
        """One artifact as JSON, validated against its type as it is read."""
        run, _ = find(run_id)
        if not run.has(name):
            raise HTTPException(404, f"run {run_id} has no {name} yet")
        return _json(run.read(name).model_dump_json())

    @app.get("/runs/{run_id}/video", response_model=None)
    def get_video(run_id: str) -> FileResponse:
        """The normalised video the stages saw, so overlays line up frame for frame."""
        run, _ = find(run_id)
        if not run.has("video"):
            raise HTTPException(404, f"run {run_id} has no video yet")
        return FileResponse(run.video_path, media_type="video/mp4")

    def study_folder(study_id: str) -> StudyFolder:
        study = StudyFolder(settings.studies_dir / study_id)
        if not _ID.fullmatch(study_id) or not study.has("design"):
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
            artifacts=_artifact_urls(f"/studies/{study_id}", study),
        )

    @app.post("/studies/{study_id}/report", status_code=202)
    def generate_report(study_id: str, background: BackgroundTasks) -> StudyView:
        """Generate the study's report in the background, from its `study.yaml` and its runs
        in the runs folder (not the precomputed one). Refused (409) while a report for it is
        queued or running."""
        study = study_folder(study_id)
        if not store.start("study", study_id):
            raise HTTPException(409, f"a report for study {study_id} is already running")

        def work(done: StageDone) -> JobStatus:
            study_stats(study, settings.runs_dir)
            done("stats")
            return "done"

        background.add_task(_run_job, store, "study", study_id, STUDY_STAGES, work)
        return study_view(study_id)

    @app.get("/studies/{study_id}")
    def get_study(study_id: str) -> StudyView:
        return study_view(study_id)

    @app.get("/studies/{study_id}/artifacts/{name}", response_model=None)
    def get_study_artifact(study_id: str, name: StudyArtifactName) -> Response:
        """One artifact as JSON, validated against its type as it is read."""
        study = study_folder(study_id)
        if not study.has(name):
            raise HTTPException(404, f"study {study_id} has no {name} yet")
        return _json(study.read(name).model_dump_json())

    return app


def _run_job(
    store: JobStore,
    kind: JobKind,
    id: str,
    stages: Sequence[str],
    work: Callable[[StageDone], JobStatus],
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
    for stage, artifact in STAGE_ARTIFACTS.items():
        if not run.has(artifact):
            break
        done.append(stage)
    return done


def _precomputed_status(
    run: RunFolder, done: list[str]
) -> tuple[JobStatus, str | None, str | None]:
    """Status, stage and error, read from the artifacts of a run nothing is running."""
    if len(done) == len(STAGE_ARTIFACTS):
        return "done", None, None
    if done[-1] == "arena" and not run.arena().confirmed:
        return "awaiting zone confirmation", None, None
    return "failed", None, f"the precomputed run stops after {done[-1]}"


def _artifact_urls(base: str, folder: RunFolder | StudyFolder) -> dict[str, str]:
    return {
        name: f"{base}/video" if name == "video" else f"{base}/artifacts/{name}"
        for name in folder.FILES
        if folder.has(name)
    }


def _copy(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    """Copy the upload to `dest`, refusing (413) one larger than `max_bytes`."""
    size = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"videos up to {max_bytes} bytes")
            out.write(chunk)


def _filename(name: str | None) -> str:
    """The upload's own name, without any folders, which the manifest records."""
    base = Path(name or "").name
    return base if base not in ("", ".", "..") else "video"


def _json(text: str) -> Response:
    return Response(text, media_type="application/json")
