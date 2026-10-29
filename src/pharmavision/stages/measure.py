"""Measure (measures): track and arena to metrics, through the pure functions in `measure/`."""

from ..artifacts import Metrics, RunFolder
from ..measure.distance import distance_travelled_cm


def run(run: RunFolder) -> None:
    arena = run.confirmed_arena()
    trajectory = [f.position for f in run.track().frames]
    run.write_metrics(
        Metrics(distance_travelled_cm=distance_travelled_cm(trajectory, arena.px_per_cm))
    )
