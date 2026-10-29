"""Measure (measures): track and arena to metrics, through the pure functions in `measure/`."""

from ..artifacts import RunFolder
from ..measure.open_field import open_field_metrics


def run(run: RunFolder) -> None:
    arena = run.confirmed_arena()
    manifest = run.manifest()
    run.write_metrics(open_field_metrics(run.track(), arena, manifest.video, manifest.measure))
