"""The deployed site: the built frontend and the API from one process, as the demo URL serves
them. The frontend reaches the API at `/api` (its default `VITE_API_BASE`)."""

from pathlib import Path

from fastapi.testclient import TestClient

from pharmavision.api import ApiSettings, create_app, with_frontend

from .judges import ScriptedJudge
from .scenes import TEMPLATE, circle_walk, finished_run
from .test_api import SCENE


def site(tmp_path: Path, dist: Path, precomputed: Path) -> TestClient:
    settings = ApiSettings(
        runs_dir=tmp_path / "runs",
        studies_dir=tmp_path / "studies",
        database=tmp_path / "jobs.sqlite3",
        precomputed_runs_dir=precomputed,
    )
    api = create_app(settings, SCENE, ScriptedJudge(), {"open_field": TEMPLATE})
    return TestClient(with_frontend(api, dist))


def test_the_frontend_is_served_at_the_root_and_the_api_under_api(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>PharmaVision</title>")
    (dist / "assets" / "index.js").write_text("console.log('pv')")
    precomputed = tmp_path / "precomputed"
    precomputed.mkdir()
    run = finished_run(tmp_path, precomputed, circle_walk(radius_px=90))
    client = site(tmp_path, dist, precomputed)

    assert "<title>PharmaVision</title>" in client.get("/").text
    assert client.get("/assets/index.js").text == "console.log('pv')"
    [view] = client.get("/api/runs").json()
    assert view["run_id"] == run.run_id
    # The API's artifact URLs are relative to its base, as the frontend's `apiUrl` reads them.
    video = client.get(f"/api{view['artifacts']['video']}")
    assert video.content == run.video_path.read_bytes()
    assert client.get("/api/runs/no-such-run").status_code == 404
