"""The job store: the status of background jobs, shared by the API's request threads and the
jobs themselves, so its moves must be single steps."""

from pathlib import Path

from pharmavision.api.jobs import JobStore


def test_a_job_is_not_started_again_while_it_runs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    assert store.start("study", "s")
    store.update("study", "s", "running", "stats")

    assert not store.start("study", "s")
    job = store.get("study", "s")
    assert job is not None
    assert (job.status, job.stage) == ("running", "stats")


def test_a_resting_job_starts_again_without_its_old_stages(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.start("study", "s")
    store.stage_done("study", "s", "stats")
    store.update("study", "s", "failed", "report", "the audit failed")

    assert store.start("study", "s")
    job = store.get("study", "s")
    assert job is not None
    assert (job.status, job.stage, job.error) == ("queued", None, None)
    assert store.stages("study", "s") == []


def test_only_one_claim_moves_a_job_on(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.start("run", "r")
    store.update("run", "r", "awaiting zone confirmation")

    claims = [store.claim("run", "r", expected="awaiting zone confirmation") for _ in range(2)]

    assert claims == [True, False]


def test_stages_resume_after_the_last_one_seen(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.start("run", "r")
    store.start("run", "other")
    for stage in ("ingest", "arena"):
        store.stage_done("run", "r", stage)
        store.stage_done("run", "other", stage)

    first, second = store.stages("run", "r")

    assert [s.stage for s in store.stages("run", "r", after=first.seq)] == ["arena"]
    assert store.stages("run", "r", after=second.seq) == []
