"""The SAM 3.1 backend through the pipeline entry, replaying recorded and synthetic Responses
streams, so CI never calls the Meta Model API."""

import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import Polygon
from shapely.geometry import box as box_polygon

from pharmavision.artifacts import Track
from pharmavision.pipeline import confirm_arena, finish_run, start_run
from pharmavision.segmenter import Segmenter
from pharmavision.segmenter.config import make_segmenter
from pharmavision.segmenter.meta_api import MetaApiSegmenter, MetaApiSettings
from pharmavision.templates import OPEN_FIELD

from .sam_replay import (
    Event,
    Instance,
    ReplayTransport,
    disc,
    frame_line,
    polygon,
    read_fixture,
    recorded_track_lines,
    responses_events,
)
from .videos import make_video

TEMPLATE = OPEN_FIELD.model_copy(update={"protocol_duration_s": 3.0})
N_FRAMES = 60  # 4 s at 15 fps


def segmenter(cache_dir: Path, transport: ReplayTransport, **settings: Any) -> MetaApiSegmenter:
    return MetaApiSegmenter(MetaApiSettings(cache_dir=cache_dir, **settings), transport)


def test_the_recorded_oft_16_responses_give_its_arena_and_track(
    tmp_path: Path, runs_dir: Path
) -> None:
    # The Sturman frame size, so the recorded boxes and masks land where SAM 3.1 put them.
    video = make_video(tmp_path / "OFT_16.mp4", seconds=4, size=928, height=576)
    lines = recorded_track_lines(below_frame=N_FRAMES)  # the recording starts at frame 15
    transport = ReplayTransport([read_fixture("OFT_16_arena"), responses_events(lines)])
    sam = segmenter(tmp_path / "cache", transport)

    run = start_run(video, TEMPLATE, sam, runs_dir)
    confirm_arena(run)
    finish_run(run, sam)

    # The recorded floor mask is a trapezoid filling most of its box, x 285-671 by y 159-471.
    floor, box = Polygon(run.arena().floor_px), box_polygon(285, 159, 671, 471)
    assert box.buffer(1).contains(floor)
    assert floor.area > 0.85 * box.area
    boxes = {
        int(m["frame"]): tuple(int(m[k]) for k in ("x1", "y1", "x2", "y2"))
        for m in (_BOX.match(line) for line in lines)
        if m
    }
    frames = run.track().frames
    assert {f.frame_index for f in frames if f.detected} == set(range(15, N_FRAMES))
    for frame in frames:
        if frame.position is not None:
            left, top, right, bottom = boxes[frame.frame_index]
            assert left <= frame.position[0] <= right
            assert top <= frame.position[1] <= bottom
    assert run.manifest().models == {"arena": "sam-3.1", "track": "sam-3.1"}
    assert transport.uploaded_frame_counts == [N_FRAMES]


_BOX = re.compile(
    r"<(?P<frame>\d+)f>\d+<\|box;x1=(?P<x1>\d+);y1=(?P<y1>\d+);x2=(?P<x2>\d+);y2=(?P<y2>\d+)"
)


# A 300 px square floor in a 360 px frame, as in test_pipeline.
SHAPE = (360, 360)
FLOOR = [(30.0, 30.0), (330.0, 30.0), (330.0, 330.0), (30.0, 330.0)]


def walk(frame: int) -> tuple[float, float]:
    """A diagonal walk, 2 px per frame, on integer pixels so a disc's centroid is its centre."""
    return (100.0 + 2 * frame, 90.0 + 2 * frame)


def arena_stream() -> list[Event]:
    return responses_events([frame_line(0, [Instance("0", polygon(SHAPE, FLOOR), 0.66)])])


def track_stream(frames: Iterable[int]) -> list[Event]:
    """The animal at `walk(frame)` in each listed frame, in the order listed."""
    return responses_events(
        [frame_line(frame, [Instance("0", disc(SHAPE, walk(frame), 8), 0.71)]) for frame in frames]
    )


def tracked(tmp_path: Path, runs_dir: Path, sam: Segmenter) -> Track:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    run = start_run(video, TEMPLATE, sam, runs_dir)
    confirm_arena(run)
    finish_run(run, sam)
    return run.track()


def test_track_follows_the_frame_markers_when_frames_skip_and_arrive_out_of_order(
    tmp_path: Path, runs_dir: Path
) -> None:
    skipped = {0, 10, 11, 12, 40}
    order = [i for i in range(N_FRAMES) if i not in skipped]
    random.Random(7).shuffle(order)
    transport = ReplayTransport([arena_stream(), track_stream(order)])

    track = tracked(tmp_path, runs_dir, segmenter(tmp_path / "cache", transport))

    assert [f.frame_index for f in track.frames] == list(range(N_FRAMES))
    assert {f.frame_index for f in track.frames if not f.detected} == skipped
    for frame in track.frames:
        if frame.position is not None:
            assert frame.position == pytest.approx(walk(frame.frame_index), abs=0.01)


def test_an_empty_stream_means_animal_not_found_and_gives_gaps(
    tmp_path: Path, runs_dir: Path
) -> None:
    transport = ReplayTransport([arena_stream(), read_fixture("OFT_16_arena_no_match")])

    track = tracked(tmp_path, runs_dir, segmenter(tmp_path / "cache", transport))

    assert len(track.frames) == N_FRAMES
    assert not any(f.detected for f in track.frames)


def test_a_rerun_is_served_from_the_cache_without_calling_the_api(
    tmp_path: Path, runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ReplayTransport([arena_stream(), track_stream(range(N_FRAMES))])
    track = tracked(tmp_path, runs_dir, segmenter(tmp_path / "cache", first))

    # The backend as config chooses it, with no API key: any API call would fail.
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # and no .env to find
    rerun = tracked(tmp_path, runs_dir, make_segmenter(MetaApiSettings(cache_dir=Path("cache"))))

    assert rerun.frames == track.frames
    # The key holds the model id: another model is a cache miss.
    other = ReplayTransport([arena_stream(), track_stream(range(N_FRAMES))])
    tracked(tmp_path, runs_dir, segmenter(tmp_path / "cache", other, model="sam-3.2"))
    assert len(other.requests) == 2
