"""Tests at the API seam: HTTP in, run and study folders out. The synthetic Segmenter and the
scripted Judge stand in for the models, as at the pipeline seam."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pharmavision.api import ApiSettings, create_app
from pharmavision.api.jobs import JobStore
from pharmavision.judge import Clip, ClipJudgement, FlaggedWindow, Judge, WindowVerdict
from pharmavision.pipeline import start_run
from pharmavision.segmenter import FrameMasks, Mask, Segmenter
from pharmavision.segmenter.synthetic import SyntheticSegmenter

from .judges import ScriptedJudge
from .scenes import FLOOR, TEMPLATE, circle_walk, finished_run
from .studies import write_study
from .videos import make_video

SCENE = SyntheticSegmenter(floor=FLOOR, trajectory=circle_walk(radius_px=90))


def api(
    tmp_path: Path,
    segmenter: Segmenter = SCENE,
    judge: Judge | None = None,
    precomputed: Path | None = None,
    max_upload_bytes: int = 1 << 30,
) -> TestClient:
    settings = ApiSettings(
        runs_dir=tmp_path / "runs",
        studies_dir=tmp_path / "studies",
        database=tmp_path / "jobs.sqlite3",
        precomputed_runs_dir=precomputed,
        max_upload_bytes=max_upload_bytes,
    )
    app = create_app(settings, segmenter, judge or ScriptedJudge(), {"open_field": TEMPLATE})
    return TestClient(app)


def create_run(client: TestClient, video: Path, template: str = "open_field") -> str:
    with video.open("rb") as f:
        response = client.post(
            "/runs", files={"video": (video.name, f, "video/mp4")}, data={"template": template}
        )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


@pytest.fixture
def video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "trial.mp4", seconds=4)


def test_a_new_run_stops_at_the_arena_gate(tmp_path: Path, video: Path) -> None:
    client = api(tmp_path)

    run_id = create_run(client, video)

    run = client.get(f"/runs/{run_id}").json()
    assert run["status"] == "awaiting zone confirmation"
    assert run["error"] is None
    assert run["read_only"] is False
    assert run["stages_done"] == ["ingest", "arena"]
    assert set(run["artifacts"]) == {"manifest", "arena", "video"}
    arena = client.get(run["artifacts"]["arena"]).json()
    assert arena["confirmed"] is False
    assert client.get(run["artifacts"]["manifest"]).json()["source"]["filename"] == "trial.mp4"


def stream(
    client: TestClient, run_id: str, last_event_id: str | None = None
) -> list[dict[str, str]]:
    """Every event of the run's progress stream, which ends once the run rests."""
    headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
    with client.stream("GET", f"/runs/{run_id}/stream", headers=headers) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        return list(parse_events(response.iter_lines()))


def parse_events(lines: Iterator[str]) -> Iterator[dict[str, str]]:
    event: dict[str, str] = {}
    for line in lines:
        if not line:
            if event:
                yield event
            event = {}
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            event[field] = value.removeprefix(" ")
    if event:
        yield event


def stages_and_status(events: list[dict[str, str]]) -> tuple[list[str], str]:
    *stages, last = events
    assert all(e["event"] == "stage" for e in stages)
    assert last["event"] == "status"
    return [json.loads(e["data"])["stage"] for e in stages], json.loads(last["data"])["status"]


def test_the_progress_stream_emits_one_event_per_stage(tmp_path: Path, video: Path) -> None:
    client = api(tmp_path)
    run_id = create_run(client, video)

    assert stages_and_status(stream(client, run_id)) == (
        ["ingest", "arena"],
        "awaiting zone confirmation",
    )

    assert client.post(f"/runs/{run_id}/arena/confirm").status_code == 202

    assert stages_and_status(stream(client, run_id)) == (
        ["ingest", "arena", "track", "measure", "behaviours", "qc"],
        "done",
    )
    run = client.get(f"/runs/{run_id}").json()
    assert set(run["artifacts"]) == {
        "manifest", "arena", "track", "metrics", "events", "qc", "video"
    }  # fmt: skip
    assert client.get(run["artifacts"]["arena"]).json()["confirmed"] is True
    assert client.get(run["artifacts"]["metrics"]).json()["distance_travelled_cm"] > 0


def test_a_reconnected_stream_resumes_after_the_last_event_it_saw(
    tmp_path: Path, video: Path
) -> None:
    client = api(tmp_path)
    run_id = create_run(client, video)
    client.post(f"/runs/{run_id}/arena/confirm")
    arena = stream(client, run_id)[1]

    resumed = stream(client, run_id, last_event_id=arena["id"])

    assert stages_and_status(resumed) == (["track", "measure", "behaviours", "qc"], "done")


def test_the_arena_is_confirmed_once_and_only_at_the_gate(tmp_path: Path, video: Path) -> None:
    client = api(tmp_path)
    run_id = create_run(client, video)
    client.post(f"/runs/{run_id}/arena/confirm")

    again = client.post(f"/runs/{run_id}/arena/confirm")

    assert again.status_code == 409
    assert client.get(f"/runs/{run_id}").json()["status"] == "done"


class Untouchable:
    """A segmenter and judge in one that fails any call: nothing may be re-run."""

    model_id = "untouchable"

    def segment_image(self, image: object, prompt: str) -> list[Mask]:
        raise AssertionError("segment_image called")

    def track_video(self, video: Path, prompt: str, max_objects: int) -> Iterator[FrameMasks]:
        raise AssertionError("track_video called")

    def classify_clip(self, clip: Clip) -> ClipJudgement:
        raise AssertionError("classify_clip called")

    def review_window(self, window: FlaggedWindow) -> WindowVerdict:
        raise AssertionError("review_window called")


def snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_a_precomputed_run_is_served_without_re_running_anything(tmp_path: Path) -> None:
    precomputed = tmp_path / "precomputed"
    precomputed.mkdir()
    run = finished_run(tmp_path, precomputed, circle_walk(radius_px=90))
    before = snapshot(run.path)
    client = api(tmp_path, Untouchable(), Untouchable(), precomputed)

    view = client.get(f"/runs/{run.run_id}").json()
    assert view["status"] == "done"
    assert view["read_only"] is True
    assert view["stages_done"] == ["ingest", "arena", "track", "measure", "behaviours", "qc"]
    metrics = client.get(view["artifacts"]["metrics"]).json()
    assert metrics == json.loads(run.metrics().model_dump_json())
    assert client.get(view["artifacts"]["video"]).content == run.video_path.read_bytes()
    assert stages_and_status(stream(client, run.run_id)) == (view["stages_done"], "done")
    assert client.post(f"/runs/{run.run_id}/arena/confirm").status_code == 409
    assert [r["run_id"] for r in client.get("/runs").json()] == [run.run_id]
    assert snapshot(run.path) == before


def test_a_video_shorter_than_the_protocol_fails_its_run_with_the_reason(
    tmp_path: Path,
) -> None:
    client = api(tmp_path)
    run_id = create_run(client, make_video(tmp_path / "short.mp4", seconds=2))

    run = client.get(f"/runs/{run_id}").json()
    assert run["status"] == "failed"
    assert run["stage"] == "ingest"
    assert run["error"] == "short.mp4 lasts 2.0 s; the protocol needs 3 s"
    assert run["artifacts"] == {}
    assert stages_and_status(stream(client, run_id)) == ([], "failed")
    assert client.post(f"/runs/{run_id}/arena/confirm").status_code == 409


def test_a_floor_never_found_fails_the_run_at_the_arena(tmp_path: Path, video: Path) -> None:
    client = api(tmp_path, segmenter=SyntheticSegmenter(floor=[]))
    run_id = create_run(client, video)

    run = client.get(f"/runs/{run_id}").json()
    assert (run["status"], run["stage"], run["stages_done"]) == ("failed", "arena", ["ingest"])


def test_an_unknown_template_is_refused_before_any_run_exists(tmp_path: Path, video: Path) -> None:
    client = api(tmp_path)

    with video.open("rb") as f:
        response = client.post(
            "/runs", files={"video": ("trial.mp4", f)}, data={"template": "plus_maze"}
        )

    assert response.status_code == 422
    assert client.get("/runs").json() == []


@pytest.mark.parametrize("run_id", ["no-such-run", "..", ".uploads", "%2E%2E%2Fjobs.sqlite3"])
def test_an_unknown_run_is_not_found(run_id: str, tmp_path: Path) -> None:
    client = api(tmp_path, precomputed=tmp_path)

    for path in ("", "/stream", "/video", "/artifacts/manifest"):
        assert client.get(f"/runs/{run_id}{path}").status_code == 404


def test_a_precomputed_run_stopped_at_the_gate_reads_as_waiting_there(
    tmp_path: Path, video: Path
) -> None:
    precomputed = tmp_path / "precomputed"
    precomputed.mkdir()
    run = start_run(video, TEMPLATE, SCENE, precomputed)
    client = api(tmp_path, Untouchable(), Untouchable(), precomputed)

    view = client.get(f"/runs/{run.run_id}").json()

    assert (view["status"], view["read_only"]) == ("awaiting zone confirmation", True)
    assert view["stages_done"] == ["ingest", "arena"]
    assert client.post(f"/runs/{run.run_id}/arena/confirm").status_code == 409


@pytest.mark.parametrize(
    "limit",
    [
        pytest.param(50_000, id="by its declared length, before the body is read"),
        pytest.param(150_000, id="while it is copied"),  # within the multipart overhead
    ],
)
def test_a_video_over_the_upload_limit_is_refused_before_any_run_exists(
    limit: int, tmp_path: Path
) -> None:
    client = api(tmp_path, max_upload_bytes=limit)

    response = client.post(
        "/runs", files={"video": ("big.mp4", bytes(200_000))}, data={"template": "open_field"}
    )

    assert response.status_code == 413
    assert client.get("/runs").json() == []
    assert [p for p in (tmp_path / "runs").rglob("*") if p.is_file()] == []


def test_a_restart_fails_the_jobs_it_interrupted(tmp_path: Path) -> None:
    # The jobs run in the API's process: one left running by the last process has stopped.
    JobStore(tmp_path / "jobs.sqlite3").start("run", "20261031T120000Z-abcdef")

    client = api(tmp_path)

    run = client.get("/runs/20261031T120000Z-abcdef").json()
    assert run["status"] == "failed"
    assert run["error"] == "interrupted: the API restarted while this job ran"


def fixture_study(tmp_path: Path) -> Path:
    """A two-group study under the API's studies folder, its runs finished, in its runs folder."""
    (tmp_path / "studies").mkdir()
    (tmp_path / "runs").mkdir()
    distances = {"vehicle": [100.0, 110.0, 120.0], "drug": [150.0, 160.0, 175.0]}
    return write_study(tmp_path / "studies", tmp_path / "runs", distances).path


def test_a_study_report_job_writes_the_study_stats(tmp_path: Path) -> None:
    study = fixture_study(tmp_path)
    client = api(tmp_path)

    response = client.post(f"/studies/{study.name}/report")

    assert response.status_code == 202
    view = client.get(f"/studies/{study.name}").json()
    assert (view["status"], view["error"], view["stages_done"]) == ("done", None, ["stats"])
    assert set(view["artifacts"]) == {"design", "trials", "stats"}
    stats = client.get(view["artifacts"]["stats"]).json()
    distance = next(m for m in stats["metrics"] if m["metric"] == "distance_travelled_cm")
    assert [g["n"] for g in distance["groups"]] == [3, 3]
    assert distance["comparison"]["test"] == "student_t"
    trials = client.get(view["artifacts"]["trials"]).json()["trials"]
    assert [t["blinding_code"] for t in trials] == ["B01", "B02", "B03", "B04", "B05", "B06"]


def test_a_study_with_an_unfinished_run_fails_its_report_with_the_reason(
    tmp_path: Path,
) -> None:
    study = fixture_study(tmp_path)
    (tmp_path / "runs" / "drug-1" / "qc.json").unlink()
    client = api(tmp_path)

    client.post(f"/studies/{study.name}/report")

    view = client.get(f"/studies/{study.name}").json()
    assert (view["status"], view["stage"]) == ("failed", "stats")
    assert "drug-1" in view["error"]
    assert set(view["artifacts"]) == {"design"}


def test_a_study_without_a_design_is_not_found(tmp_path: Path) -> None:
    client = api(tmp_path)

    assert client.post("/studies/no-such-study/report").status_code == 404
    assert client.get("/studies/no-such-study").status_code == 404
