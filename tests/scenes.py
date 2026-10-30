"""The synthetic scene the seam tests share: a 300 px square floor and one 4 s trial."""

import math
from collections.abc import Callable
from pathlib import Path

from pharmavision.artifacts import MeasureSettings, QcSettings
from pharmavision.pipeline import RunFolder, confirm_arena, finish_run, start_run
from pharmavision.segmenter.synthetic import SyntheticSegmenter
from pharmavision.templates import OPEN_FIELD

from .videos import make_video

TEMPLATE = OPEN_FIELD.model_copy(update={"protocol_duration_s": 3.0})

# A 300 px square floor in a 360 px frame: 45 cm / 300 px.
FLOOR = [(30.0, 30.0), (330.0, 30.0), (330.0, 330.0), (30.0, 330.0)]
PX_PER_CM = 300 / 45
N_FRAMES = 60  # 4 s at 15 fps

Trajectory = Callable[[int], tuple[float, float] | None]


def circle_walk(radius_px: float, laps: int = 1) -> Callable[[int], tuple[float, float]]:
    """Centre of the floor, constant speed, ending where it started on the last frame."""

    def position(frame: int) -> tuple[float, float]:
        angle = 2 * math.pi * laps * frame / (N_FRAMES - 1)
        return (180 + radius_px * math.cos(angle), 180 + radius_px * math.sin(angle))

    return position


def finished_run(
    tmp_path: Path,
    runs_dir: Path,
    trajectory: Trajectory,
    intruder: Trajectory = lambda _: None,
    qc_settings: QcSettings | None = None,
    body: Callable[[int], tuple[float, float]] = lambda _: (8.0, 8.0),
    measure_settings: MeasureSettings | None = None,
) -> RunFolder:
    """A run through every stage, on a 4 s video, with the animal at `trajectory(frame)`, its
    body an ellipse with semi-axes `body(frame)` (along x, along y), and a second, less
    confident object at `intruder(frame)`."""
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(floor=FLOOR, trajectory=trajectory, body=body, intruder=intruder)
    run = start_run(
        video,
        TEMPLATE,
        segmenter,
        runs_dir,
        measure_settings=measure_settings,
        qc_settings=qc_settings,
    )
    confirm_arena(run)
    finish_run(run, segmenter)
    return run
