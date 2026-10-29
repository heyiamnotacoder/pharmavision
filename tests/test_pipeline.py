"""Tests at the one seam: the pipeline entry. They assert on artifacts and refusals only."""

import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from pharmavision.pipeline import (
    ArenaNotConfirmedError,
    Stage,
    VideoTooShortError,
    confirm_arena,
    finish_run,
    run_stage,
    start_run,
)
from pharmavision.segmenter.synthetic import SyntheticSegmenter
from pharmavision.templates import OPEN_FIELD

from .conftest import make_video

TEMPLATE = OPEN_FIELD.model_copy(update={"protocol_duration_s": 3.0})

# A 300 px square floor in a 360 px frame: 45 cm / 300 px.
FLOOR = [(30.0, 30.0), (330.0, 30.0), (330.0, 330.0), (30.0, 330.0)]
PX_PER_CM = 300 / 45
N_FRAMES = 60  # 4 s at 15 fps


def circle_walk(radius_px: float, laps: int = 1) -> Callable[[int], tuple[float, float]]:
    """Centre of the floor, constant speed, ending where it started on the last frame."""

    def position(frame: int) -> tuple[float, float]:
        angle = 2 * math.pi * laps * frame / (N_FRAMES - 1)
        return (180 + radius_px * math.cos(angle), 180 + radius_px * math.sin(angle))

    return position


def ffprobe_streams(video: Path) -> list[dict[str, str]]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)["streams"]


def test_rejects_a_video_shorter_than_the_protocol(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "short.mp4", seconds=2)

    with pytest.raises(VideoTooShortError):
        start_run(video, TEMPLATE, SyntheticSegmenter(), runs_dir)

    assert list(runs_dir.iterdir()) == []


def test_ingest_normalises_the_video_and_records_the_source(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4, fps=25)

    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=FLOOR), runs_dir)

    streams = ffprobe_streams(run.video_path)
    assert [s["codec_type"] for s in streams] == ["video"]
    assert streams[0]["codec_name"] == "h264"
    assert streams[0]["avg_frame_rate"] == "15/1"
    manifest = run.manifest()
    assert manifest.source.sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert manifest.source.duration_s == pytest.approx(4.0, abs=0.1)
    assert (manifest.source.width, manifest.source.height) == (360, 360)
    assert manifest.video.fps == 15
    assert manifest.video.n_frames == 60
    assert manifest.template == TEMPLATE


def test_arena_derives_the_centre_zone_and_scale_from_the_floor(
    tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)

    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=FLOOR), runs_dir)

    arena = run.arena()
    assert not arena.confirmed
    floor, centre = Polygon(arena.floor_px), Polygon(arena.centre_px)
    assert floor.symmetric_difference(Polygon(FLOOR)).area < 0.02 * Polygon(FLOOR).area
    assert arena.px_per_cm == pytest.approx(PX_PER_CM, rel=0.01)
    assert centre.area == pytest.approx(0.5 * floor.area, rel=1e-6)
    assert centre.centroid.distance(floor.centroid) < 1e-6
    assert floor.contains(centre)


@pytest.mark.parametrize("stage", [Stage.TRACK, Stage.MEASURE])
def test_stages_after_arena_refuse_an_unconfirmed_arena(
    stage: Stage, tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=lambda _: (180.0, 180.0))
    run = start_run(video, TEMPLATE, segmenter, runs_dir)
    before = sorted(p.name for p in run.path.iterdir())

    with pytest.raises(ArenaNotConfirmedError):
        run_stage(run, stage, segmenter)

    assert sorted(p.name for p in run.path.iterdir()) == before


def test_a_circle_walk_gives_its_known_distance(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=circle_walk(radius_px=90, laps=2))
    run = start_run(video, TEMPLATE, segmenter, runs_dir)

    confirm_arena(run)
    finish_run(run, segmenter)

    # 90 px is 13.5 cm on a 45 cm floor 300 px wide; two laps are 2 x 2 pi r.
    assert run.metrics().distance_travelled_cm == pytest.approx(2 * 2 * math.pi * 13.5, rel=0.01)
    frames = run.track().frames
    assert [f.frame_index for f in frames] == list(range(N_FRAMES))
    assert all(f.detected for f in frames)
    assert run.manifest().models == {"arena": "synthetic", "track": "synthetic"}


def test_track_is_keyed_by_frame_index_whatever_the_delivery_order(
    tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    walk, dropped = circle_walk(radius_px=90), {10, 11, 12, 40}
    segmenter = SyntheticSegmenter(
        floor=FLOOR,
        trajectory=lambda i: None if i in dropped else walk(i),
        shuffle_seed=7,
    )
    run = start_run(video, TEMPLATE, segmenter, runs_dir)

    confirm_arena(run)
    finish_run(run, segmenter)

    frames = run.track().frames
    assert [f.frame_index for f in frames] == list(range(N_FRAMES))
    assert {f.frame_index for f in frames if not f.detected} == dropped
    for frame in frames:
        if frame.position is not None:
            assert math.dist(frame.position, walk(frame.frame_index)) < 0.5


def test_an_animal_never_found_gives_gaps_and_no_distance(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(floor=FLOOR)  # empty stream
    run = start_run(video, TEMPLATE, segmenter, runs_dir)

    confirm_arena(run)
    finish_run(run, segmenter)

    assert not any(f.detected for f in run.track().frames)
    assert run.metrics().distance_travelled_cm is None
