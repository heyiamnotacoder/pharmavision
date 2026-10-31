"""Background jobs' status in SQLite: one row per job (a run or a study's report) and one row
per stage it finished, which the progress stream reads. A connection per call, so the API's
request threads and its background jobs never share one."""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

JobKind = Literal["run", "study"]
JobStatus = Literal["queued", "running", "awaiting zone confirmation", "done", "failed"]
RESTING: frozenset[JobStatus] = frozenset({"awaiting zone confirmation", "done", "failed"})
"""Statuses no background job will move on from by itself: a human has to act, or it ended."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);
CREATE TABLE IF NOT EXISTS stages_done (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    stage TEXT NOT NULL,
    at TEXT NOT NULL
);
"""


class Job(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: JobKind
    id: str
    status: JobStatus
    stage: str | None
    """The stage running now; on a failure, the stage that failed."""
    error: str | None
    updated_at: datetime


class StageDone(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    """Increasing across every job, so a stream can resume after the last one it sent."""
    stage: str
    at: datetime


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    def start(self, kind: JobKind, id: str) -> bool:
        """Queue a job, or queue it again if it is resting, forgetting its earlier stages. False,
        changing nothing, while it is queued or running."""
        now = _now()
        with self._connect() as db:
            resting = ", ".join("?" * len(RESTING))
            changed = db.execute(
                "INSERT INTO jobs VALUES (?, ?, 'queued', NULL, NULL, ?) "
                "ON CONFLICT (kind, id) DO UPDATE SET status = 'queued', stage = NULL, "
                f"error = NULL, updated_at = excluded.updated_at WHERE status IN ({resting})",
                (kind, id, now, *sorted(RESTING)),
            ).rowcount
            if changed:
                db.execute("DELETE FROM stages_done WHERE kind = ? AND id = ?", (kind, id))
        return bool(changed)

    def claim(
        self, kind: JobKind, id: str, expected: JobStatus, status: JobStatus = "queued"
    ) -> bool:
        """Move a job from `expected` to `status` in one step, so two requests can't both move
        it. False, changing nothing, when it is not in `expected`."""
        with self._connect() as db:
            return bool(
                db.execute(
                    "UPDATE jobs SET status = ?, error = NULL, updated_at = ? "
                    "WHERE kind = ? AND id = ? AND status = ?",
                    (status, _now(), kind, id, expected),
                ).rowcount
            )

    def update(
        self,
        kind: JobKind,
        id: str,
        status: JobStatus,
        stage: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status = ?, stage = ?, error = ?, updated_at = ? "
                "WHERE kind = ? AND id = ?",
                (status, stage, error, _now(), kind, id),
            )

    def stage_done(self, kind: JobKind, id: str, stage: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO stages_done (kind, id, stage, at) VALUES (?, ?, ?, ?)",
                (kind, id, stage, _now()),
            )

    def get(self, kind: JobKind, id: str) -> Job | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT kind, id, status, stage, error, updated_at FROM jobs "
                "WHERE kind = ? AND id = ?",
                (kind, id),
            ).fetchone()
        return _job(row) if row else None

    def jobs(self, kind: JobKind) -> list[Job]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT kind, id, status, stage, error, updated_at FROM jobs "
                "WHERE kind = ? ORDER BY id",
                (kind,),
            ).fetchall()
        return [_job(row) for row in rows]

    def stages(self, kind: JobKind, id: str, after: int = 0) -> list[StageDone]:
        """The stages the job finished, in order, from after `seq` `after`."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT seq, stage, at FROM stages_done "
                "WHERE kind = ? AND id = ? AND seq > ? ORDER BY seq",
                (kind, id, after),
            ).fetchall()
        return [StageDone(seq=seq, stage=stage, at=at) for seq, stage, at in rows]

    def fail_interrupted(self, reason: str) -> None:
        """Fail every queued or running job: the jobs run in the API's process, so after a
        restart nothing is running them."""
        active = [s for s in get_args(JobStatus) if s not in RESTING]
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? "
                f"WHERE status IN ({', '.join('?' * len(active))})",
                (reason, _now(), *active),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            yield db


def _job(row: Sequence[str | None]) -> Job:
    kind, id, status, stage, error, updated_at = row
    return Job.model_validate(
        {
            "kind": kind,
            "id": id,
            "status": status,
            "stage": stage,
            "error": error,
            "updated_at": updated_at,
        }
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
