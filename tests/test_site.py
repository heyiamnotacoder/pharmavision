"""The deployed site: the built frontend and the API from one process, as the demo URL serves
them. The frontend reaches the API at `/api` (its default `VITE_API_BASE`)."""

from pathlib import Path

from fastapi.testclient import TestClient

from pharmavision.api import with_frontend

from .judges import ScriptedJudge
from .scenes import circle_walk, finished_run
from .studies import write_study
from .test_api import Untouchable, make_app
from .videos import make_video


def built_frontend(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>PharmaVision</title>")
    (dist / "assets" / "index.js").write_text("console.log('pv')")
    return dist


def test_the_frontend_is_served_at_the_root_and_the_api_under_api(tmp_path: Path) -> None:
    precomputed = tmp_path / "precomputed"
    precomputed.mkdir()
    run = finished_run(tmp_path, precomputed, circle_walk(radius_px=90))
    api = make_app(tmp_path, judge=ScriptedJudge(), precomputed=precomputed)
    client = TestClient(with_frontend(api, built_frontend(tmp_path)))

    assert "<title>PharmaVision</title>" in client.get("/").text
    assert client.get("/assets/index.js").text == "console.log('pv')"
    [view] = client.get("/api/runs").json()
    assert view["run_id"] == run.run_id
    # The API's artifact URLs are relative to its base, as the frontend's `apiUrl` reads them.
    video = client.get(f"/api{view['artifacts']['video']}")
    assert video.content == run.video_path.read_bytes()
    assert client.get("/api/runs/no-such-run").status_code == 404


def test_a_read_only_site_refuses_everything_that_would_start_a_job(tmp_path: Path) -> None:
    """The public demo: no visitor can spend the Model API budget or re-run the demo study."""
    (tmp_path / "studies").mkdir()
    (tmp_path / "runs").mkdir()
    distances = {"vehicle": [1.0, 2.0, 3.0], "drug": [4.0, 5.0, 6.0]}
    write_study(tmp_path / "studies", tmp_path / "runs", distances)
    api = make_app(tmp_path, Untouchable(), Untouchable(), read_only=True)
    client = TestClient(with_frontend(api, built_frontend(tmp_path)))
    [study] = client.get("/api/studies").json()
    runs_before = client.get("/api/runs").json()

    with make_video(tmp_path / "upload.mp4", seconds=4).open("rb") as video:
        upload = client.post("/api/runs", files={"video": video}, data={"template": "open_field"})
    assert upload.status_code == 403
    assert "read-only" in upload.json()["detail"]
    assert client.post("/api/runs/vehicle-0/arena/confirm").status_code == 403
    assert client.post(f"/api/studies/{study['study_id']}/report").status_code == 403
    assert client.get(f"/api/studies/{study['study_id']}").json()["status"] is None
    assert client.get("/api/runs").json() == runs_before
