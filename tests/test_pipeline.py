"""Tests at the one seam: the pipeline entry. They assert on artifacts and refusals only."""

import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from shapely.geometry import Polygon

from pharmavision.artifacts import ArenaSettings, Gap, MeasureSettings
from pharmavision.pipeline import (
    ArenaNotConfirmedError,
    InvalidFloorError,
    Stage,
    VideoTooShortError,
    confirm_arena,
    finish_run,
    run_stage,
    start_run,
)
from pharmavision.segmenter import Mask
from pharmavision.segmenter.synthetic import SyntheticSegmenter

from .judges import ScriptedJudge
from .scenes import (
    FLOOR,
    N_FRAMES,
    PX_PER_CM,
    REAR,
    TEMPLATE,
    WALKING_BODY,
    circle_walk,
    finished_run,
    rears_in,
    walk_stop_walk,
)
from .videos import make_video


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


def test_accepts_a_video_short_of_the_protocol_by_container_rounding(
    tmp_path: Path, runs_dir: Path
) -> None:
    # A 10-min recording can probe as 599.96 s; the template tolerates up to 0.5 s.
    video = make_video(tmp_path / "rounded.mp4", seconds=2.6)

    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=FLOOR), runs_dir)

    assert run.manifest().source.duration_s == pytest.approx(2.6, abs=0.1)


def test_a_floor_never_found_waits_at_the_gate_with_an_empty_arena(
    tmp_path: Path, runs_dir: Path
) -> None:
    # SAM 3.1 drops a floor it scores under about 0.5: the scientist draws it instead (#16).
    video = make_video(tmp_path / "trial.mp4", seconds=4)

    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=[]), runs_dir)

    arena = run.arena()
    assert (arena.floor_px, arena.centre_px, arena.confirmed) == ([], [], False)
    assert run.manifest().models["arena"] == "synthetic"
    with pytest.raises(ArenaNotConfirmedError):
        run_stage(run, Stage.TRACK, SyntheticSegmenter(), ScriptedJudge())


def test_an_empty_arena_cannot_be_confirmed_without_a_floor(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=[]), runs_dir)

    with pytest.raises(InvalidFloorError, match="draw the floor"):
        confirm_arena(run)

    assert not run.arena().confirmed


def test_an_edited_floor_moves_the_centre_zone_and_the_scale(
    tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=FLOOR), runs_dir)
    # The scientist pulls two corners in: a 270 px square, 6 cm/px of 45 cm.
    edited = [(30.0, 30.0), (300.0, 30.0), (300.0, 300.0), (30.0, 300.0)]

    confirm_arena(run, edited)

    arena = run.arena()
    assert arena.confirmed
    assert arena.floor_px == edited
    assert arena.px_per_cm == pytest.approx(6.0)
    # The centre: half the floor's area (270² / 2 = 36,450 px²), about the floor's centre.
    centre = Polygon(arena.centre_px)
    assert centre.area == pytest.approx(36_450)
    assert (centre.centroid.x, centre.centroid.y) == pytest.approx((165.0, 165.0))


def test_a_drawn_floor_confirms_an_empty_arena(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=[]), runs_dir)

    confirm_arena(run, FLOOR)

    assert run.arena().px_per_cm == pytest.approx(PX_PER_CM)
    run_stage(run, Stage.TRACK, SyntheticSegmenter(), ScriptedJudge())  # the gate is open


@pytest.mark.parametrize(
    ("floor", "reason"),
    [
        pytest.param([(30.0, 30.0), (330.0, 30.0)], "at least 3 corners", id="two corners"),
        pytest.param([*FLOOR[:3], (30.0, 361.0)], "outside the 360 x 360", id="off the frame"),
        pytest.param([FLOOR[0], FLOOR[2], FLOOR[1], FLOOR[3]], "cross", id="a bow tie"),
        pytest.param([(30.0, 30.0), (180.0, 30.0), (330.0, 30.0)], "no area", id="a line"),
    ],
)
def test_a_floor_the_zones_cannot_come_from_is_refused(
    floor: list[tuple[float, float]], reason: str, tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=FLOOR), runs_dir)
    before = run.arena()

    with pytest.raises(InvalidFloorError, match=reason):
        confirm_arena(run, floor)

    assert run.arena() == before


def test_the_editor_gets_the_background_frame_the_floor_was_looked_for_on(
    tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "dim.mp4", seconds=4, floor_box=(100, 100, 160, 160))

    run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=[]), runs_dir)

    background = cv2.imread(str(run.background_path))
    assert background is not None
    assert background.shape == (360, 360, 3)
    # Plain, not equalised: the scientist sees the floor as the camera did.
    assert background[180, 180].mean() - background[180, 50].mean() == pytest.approx(8, abs=3)


class ImageSpy(SyntheticSegmenter):
    def __init__(self) -> None:
        super().__init__(floor=FLOOR)
        self.images: list[NDArray[np.uint8]] = []

    def segment_image(self, image: NDArray[np.uint8], prompt: str) -> list[Mask]:
        self.images.append(image)
        return super().segment_image(image, prompt)


def test_the_floor_prompt_sees_the_background_equalised(tmp_path: Path, runs_dir: Path) -> None:
    # SAM 3.1 drops a floor it scores under about 0.5. On two Sturman cameras the plain
    # background frame scored just under, and the equalised one did not (Oct 30).
    video = make_video(tmp_path / "dim.mp4", seconds=4, floor_box=(100, 100, 160, 160))
    plain, equalised = ImageSpy(), ImageSpy()

    off = ArenaSettings(contrast_clip_limit=None)
    start_run(video, TEMPLATE, plain, runs_dir, arena_settings=off)
    run = start_run(video, TEMPLATE, equalised, runs_dir)

    def edge_step(image: NDArray[np.uint8]) -> float:
        # Across the floor's left edge (x = 100) on its middle row, inside one CLAHE tile.
        grey = image.mean(axis=2)
        return float(grey[180, 102:108].mean() - grey[180, 92:98].mean())

    assert edge_step(equalised.images[0]) > edge_step(plain.images[0]) > 0
    assert run.manifest().arena.contrast_clip_limit == ArenaSettings().contrast_clip_limit


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


@pytest.mark.parametrize("stage", list(Stage))
def test_stages_after_arena_refuse_an_unconfirmed_arena(
    stage: Stage, tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=lambda _: (180.0, 180.0))
    run = start_run(video, TEMPLATE, segmenter, runs_dir)
    before = sorted(p.name for p in run.path.iterdir())

    with pytest.raises(ArenaNotConfirmedError):
        run_stage(run, stage, segmenter, ScriptedJudge())

    assert sorted(p.name for p in run.path.iterdir()) == before


def test_a_circle_walk_gives_its_known_distance(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=circle_walk(radius_px=90, laps=2))
    run = start_run(video, TEMPLATE, segmenter, runs_dir)

    confirm_arena(run)
    finish_run(run, segmenter, ScriptedJudge())

    # 90 px is 13.5 cm on a 45 cm floor 300 px wide; two laps are 2 x 2 pi r.
    assert run.metrics().distance_travelled_cm == pytest.approx(2 * 2 * math.pi * 13.5, rel=0.01)
    frames = run.track().frames
    assert [f.frame_index for f in frames] == list(range(N_FRAMES))
    assert all(f.detected for f in frames)
    assert run.manifest().models == {
        "arena": "synthetic",
        "track": "synthetic",
        "behaviours": "scripted",
    }


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
    finish_run(run, segmenter, ScriptedJudge())

    frames = run.track().frames
    assert [f.frame_index for f in frames] == list(range(N_FRAMES))
    assert {f.frame_index for f in frames if not f.detected} == dropped
    for frame in frames:
        if frame.position is not None:
            assert math.dist(frame.position, walk(frame.frame_index)) < 0.5


def test_an_animal_never_found_leaves_every_metric_unknown(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(floor=FLOOR)  # empty stream
    run = start_run(video, TEMPLATE, segmenter, runs_dir)

    confirm_arena(run)
    finish_run(run, segmenter, ScriptedJudge())

    assert not any(f.detected for f in run.track().frames)
    metrics = run.metrics()
    assert metrics.gaps == [Gap(first_frame=0, last_frame=N_FRAMES - 1, interpolated=False)]
    assert metrics.rear_candidates == []
    assert all(
        value is None for value in metrics.model_dump(exclude={"gaps", "rear_candidates"}).values()
    )


def test_the_manifest_records_the_versioned_measure_settings(
    tmp_path: Path, runs_dir: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    custom = MeasureSettings(max_gap_s=1.0, zone_hysteresis_cm=2.0)

    default_run = start_run(video, TEMPLATE, SyntheticSegmenter(floor=FLOOR), runs_dir)
    custom_run = start_run(
        video, TEMPLATE, SyntheticSegmenter(floor=FLOOR), runs_dir, measure_settings=custom
    )

    assert default_run.manifest().measure == MeasureSettings()
    assert default_run.manifest().measure.version == 4
    assert custom_run.manifest().measure == custom


def test_a_constant_speed_walk_gives_its_known_velocity(tmp_path: Path, runs_dir: Path) -> None:
    # 3 px per frame along the floor's midline, at 15 fps.
    run = finished_run(tmp_path, runs_dir, lambda i: (90.0 + 3 * i, 180.0))

    assert run.metrics().mean_velocity_cm_s == pytest.approx(3 * 15 / PX_PER_CM, rel=0.01)


def test_short_gaps_are_interpolated_and_long_gaps_stay_flagged(
    tmp_path: Path, runs_dir: Path
) -> None:
    # 3 frames (0.2 s) and 15 frames (1 s) dropped; the maximum interpolated gap is 0.5 s.
    dropped = {10, 11, 12, *range(30, 45)}
    run = finished_run(
        tmp_path, runs_dir, lambda i: None if i in dropped else (90.0 + 3 * i, 180.0)
    )

    metrics = run.metrics()
    assert metrics.gaps == [
        Gap(first_frame=10, last_frame=12, interpolated=True),
        Gap(first_frame=30, last_frame=44, interpolated=False),
    ]
    # Steps 0-29 (bridged across the short gap) and 45-59 count; the long gap is not guessed.
    assert metrics.distance_travelled_cm == pytest.approx((29 + 14) * 3 / PX_PER_CM, rel=0.01)
    assert metrics.mean_velocity_cm_s == pytest.approx(3 * 15 / PX_PER_CM, rel=0.01)


def test_a_one_frame_jump_in_the_track_adds_no_distance(tmp_path: Path, runs_dir: Path) -> None:
    # A still animal, mis-detected 50 px away on one frame: the median filter removes the spike.
    run = finished_run(tmp_path, runs_dir, lambda i: (230.0, 180.0) if i == 30 else (180.0, 180.0))

    assert run.metrics().distance_travelled_cm == pytest.approx(0.0, abs=1e-6)


def test_a_scripted_dwell_gives_the_centre_time_and_entries(tmp_path: Path, runs_dir: Path) -> None:
    # The centre zone spans x and y from about 74 to 286 px. Out 10 frames, in 20, out 10, in 20.
    def dwell(i: int) -> tuple[float, float]:
        in_centre = 10 <= i < 30 or 40 <= i < 60
        return (180.0, 180.0) if in_centre else (50.0, 180.0)

    run = finished_run(tmp_path, runs_dir, dwell)

    metrics = run.metrics()
    assert metrics.centre_time_s == pytest.approx(40 / 15)
    assert metrics.centre_entries == 2


@pytest.mark.parametrize(
    ("jitter_cm", "swing_frames"),
    [
        pytest.param(0.5, 10, id="slow swings inside the hysteresis band"),
        pytest.param(3.0, 1, id="swings across the band, shorter than the minimum dwell"),
    ],
)
def test_jitter_on_the_centre_zone_edge_adds_no_entries(
    jitter_cm: float, swing_frames: int, tmp_path: Path, runs_dir: Path
) -> None:
    # Out, then 20 frames swinging across the centre zone's left edge, then one real visit.
    edge_x = 180 - 150 * math.sqrt(0.5)

    def jitter(i: int) -> tuple[float, float]:
        if 10 <= i < 30:
            side = 1 if (i - 10) // swing_frames % 2 == 0 else -1
            return (edge_x + side * jitter_cm * PX_PER_CM, 180.0)
        return (180.0, 180.0) if 30 <= i < 50 else (50.0, 180.0)

    run = finished_run(tmp_path, runs_dir, jitter)

    assert run.metrics().centre_entries == 1


def test_a_visit_broken_by_brief_mis_detections_counts_once(tmp_path: Path, runs_dir: Path) -> None:
    # In the centre for frames 15-51, but detected in the periphery for 3 frames 4 times. The
    # 5-frame stretches between them are each shorter than the minimum dwell; the visit is not.
    def visit(i: int) -> tuple[float, float]:
        glitch = i in {*range(20, 23), *range(28, 31), *range(36, 39), *range(44, 47)}
        return (180.0, 180.0) if 15 <= i < 52 and not glitch else (50.0, 180.0)

    run = finished_run(tmp_path, runs_dir, visit)

    metrics = run.metrics()
    assert metrics.centre_entries == 1
    assert metrics.centre_time_s == pytest.approx(37 / 15)


def test_a_long_gap_does_not_join_two_glimpses_into_a_visit(tmp_path: Path, runs_dir: Path) -> None:
    # Two 5-frame glimpses of the centre, each shorter than the minimum dwell, on either side of
    # a 1 s gap that is too long to interpolate.
    def glimpses(i: int) -> tuple[float, float] | None:
        if 25 <= i < 40:
            return None
        return (180.0, 180.0) if 20 <= i < 45 else (50.0, 180.0)

    run = finished_run(tmp_path, runs_dir, glimpses)

    metrics = run.metrics()
    assert metrics.centre_entries == 0
    assert metrics.centre_time_s == 0


def test_wall_hugging_gives_the_known_thigmotaxis(tmp_path: Path, runs_dir: Path) -> None:
    # 45 frames walking along the left wall 2 cm from it, then 15 frames in the middle.
    def hug(i: int) -> tuple[float, float]:
        return (30 + 2 * PX_PER_CM, 100 + 160 * i / 44) if i < 45 else (180.0, 180.0)

    run = finished_run(tmp_path, runs_dir, hug)

    assert run.metrics().thigmotaxis == pytest.approx(45 / 60)


def test_a_static_mask_gives_full_immobility(tmp_path: Path, runs_dir: Path) -> None:
    run = finished_run(tmp_path, runs_dir, lambda _: (180.0, 180.0))

    assert run.metrics().immobility_s == pytest.approx(N_FRAMES / 15)


def test_only_pauses_of_the_minimum_duration_count_as_immobility(
    tmp_path: Path, runs_dir: Path
) -> None:
    # Walking 3 px per frame, with a pause of 10 identical frames (10-19, 0.67 s) and one of
    # exactly 15 (30-44, 1 s) in mid-video. Immobility bouts must last at least 1 s.
    def progress(i: int) -> int:
        if i < 10:
            return i
        if i < 20:
            return 10
        if i < 30:
            return i - 9
        return 21 if i < 45 else i - 23

    run = finished_run(tmp_path, runs_dir, lambda i: (90.0 + 3 * progress(i), 180.0))

    assert run.metrics().immobility_s == pytest.approx(15 / 15)


def test_the_same_video_and_settings_give_identical_metrics(tmp_path: Path, runs_dir: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    walk, dropped = circle_walk(radius_px=120), {5, 20, 21, *range(40, 50)}
    segmenter = SyntheticSegmenter(
        floor=FLOOR, trajectory=lambda i: None if i in dropped else walk(i), shuffle_seed=3
    )
    runs = [start_run(video, TEMPLATE, segmenter, runs_dir) for _ in range(2)]

    for run in runs:
        confirm_arena(run)
        finish_run(run, segmenter, ScriptedJudge())

    first, second = (run.metrics() for run in runs)
    assert first.distance_travelled_cm is not None
    assert first == second
    for artifact in ("metrics.json", "qc.json"):
        assert (runs[0].path / artifact).read_bytes() == (runs[1].path / artifact).read_bytes()

    assert (runs[0].path / "metrics.json").read_bytes() == (
        runs[1].path / "metrics.json"
    ).read_bytes()


def test_a_still_shrink_and_elongate_episode_gives_one_rear_candidate(
    tmp_path: Path, runs_dir: Path
) -> None:
    run = finished_run(tmp_path, runs_dir, walk_stop_walk, body=rears_in(REAR))

    [candidate] = run.metrics().rear_candidates
    assert (candidate.first_frame, candidate.last_frame) == (25, 39)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(lambda _: WALKING_BODY, id="walking normally"),
        pytest.param(rears_in(REAR), id="shrinking and elongating at walking speed"),
    ],
)
def test_walking_gives_no_rear_candidate(
    body: Callable[[int], tuple[float, float]], tmp_path: Path, runs_dir: Path
) -> None:
    # 21.6 cm/s around a circle.
    run = finished_run(tmp_path, runs_dir, circle_walk(radius_px=90), body=body)

    assert run.metrics().rear_candidates == []


def test_standing_still_with_the_usual_body_gives_no_rear_candidate(
    tmp_path: Path, runs_dir: Path
) -> None:
    # Still for 2 s, under every speed limit: only the mask can rule it out.
    run = finished_run(tmp_path, runs_dir, walk_stop_walk, body=lambda _: WALKING_BODY)

    assert run.metrics().rear_candidates == []


def test_the_rear_thresholds_come_from_the_runs_measure_settings(
    tmp_path: Path, runs_dir: Path
) -> None:
    # The episode keeps 46% of the walking area: not a drop of 60%.
    strict = MeasureSettings(rear_area_drop=0.6)

    run = finished_run(
        tmp_path, runs_dir, walk_stop_walk, body=rears_in(REAR), measure_settings=strict
    )

    assert run.manifest().measure.rear_area_drop == 0.6
    assert run.metrics().rear_candidates == []
