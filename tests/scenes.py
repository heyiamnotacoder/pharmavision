"""The synthetic scene the seam tests share: a 300 px square floor and one 4 s trial."""

import math
from collections.abc import Callable
from pathlib import Path

from pharmavision.artifacts import BehaviourSettings, MeasureSettings, QcSettings
from pharmavision.judge import Judge
from pharmavision.pipeline import RunFolder, confirm_arena, finish_run, start_run
from pharmavision.segmenter.synthetic import SyntheticSegmenter
from pharmavision.templates import OPEN_FIELD

from .judges import ScriptedJudge
from .videos import make_video

TEMPLATE = OPEN_FIELD.model_copy(update={"protocol_duration_s": 3.0})

# A 300 px square floor in a 360 px frame: 45 cm / 300 px.
FLOOR = [(30.0, 30.0), (330.0, 30.0), (330.0, 330.0), (30.0, 330.0)]
PX_PER_CM = 300 / 45
N_FRAMES = 60  # 4 s at 15 fps

Trajectory = Callable[[int], tuple[float, float] | None]

# A walking body, 10 x 7 px semi-axes (elongation 1.43), and a rear seen from above: the mask
# shrinks to 8 x 4 px (46% of the area) and gets longer and thinner (elongation 2.0).
WALKING_BODY = (10.0, 7.0)
REAR = range(25, 40)  # 1 s


def rears_in(episode: range) -> Callable[[int], tuple[float, float]]:
    """The walking body, with the rear's body in the `episode` frames."""
    return lambda i: (8.0, 4.0) if i in episode else WALKING_BODY


def walk_stop_walk(i: int) -> tuple[float, float]:
    """6 px per frame (13.5 cm/s) along the midline, standing still for frames 15-44."""
    return (60.0 + 6 * (i if i < 15 else 15 if i < 45 else i - 30), 180.0)


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
    judge: Judge | None = None,
    behaviour_settings: BehaviourSettings | None = None,
    segmenter: SyntheticSegmenter | None = None,
) -> RunFolder:
    """A run through every stage, on a 4 s video, with the animal at `trajectory(frame)`, its
    body an ellipse with semi-axes `body(frame)` (along x, along y), and a second, less
    confident object at `intruder(frame)`. `judge` judges the rear candidates' clips and the
    flagged windows. A `segmenter` given replaces the scene the other arguments describe."""
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = segmenter or SyntheticSegmenter(
        floor=FLOOR, trajectory=trajectory, body=body, intruder=intruder
    )
    run = start_run(
        video,
        TEMPLATE,
        segmenter,
        runs_dir,
        measure_settings=measure_settings,
        qc_settings=qc_settings,
        behaviour_settings=behaviour_settings,
    )
    confirm_arena(run)
    finish_run(run, segmenter, judge or ScriptedJudge())
    return run
