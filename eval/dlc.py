"""DeepLabCut output as DLCAnalyzer ships it: a CSV with three header rows (scorer, body
parts, coords), then one row per frame of the source video: the frame index, then x, y and
likelihood for each body part, in source-video pixels. A reference tracker, not ground truth."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class DlcFileError(ValueError):
    """The file is not DeepLabCut output in the layout expected."""


@dataclass(frozen=True)
class DlcTrack:
    """One body part per source frame: element i is frame i."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    likelihood: NDArray[np.float64]


BODYPART = "bodycentre"
"""The part compared with our mask centroid: the Sturman network's body centre."""


def read_dlc(path: Path) -> DlcTrack:
    """The body centre's path."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        _scorer, parts, coords = next(reader), next(reader), next(reader)
    columns = {
        coord: i
        for i, (part, coord) in enumerate(zip(parts, coords, strict=True))
        if part == BODYPART
    }
    if set(columns) != {"x", "y", "likelihood"}:
        tracked = sorted(set(parts[1:]))
        raise DlcFileError(f"{path.name} has no {BODYPART!r} x/y/likelihood; it has {tracked}")
    table = np.loadtxt(
        path,
        delimiter=",",
        skiprows=3,
        usecols=(0, columns["x"], columns["y"], columns["likelihood"]),
        ndmin=2,
    )
    frames = table[:, 0]
    if not np.array_equal(frames, np.arange(len(frames))):
        raise DlcFileError(f"{path.name}: rows are not frames 0..{len(frames) - 1} in order")
    return DlcTrack(x=table[:, 1], y=table[:, 2], likelihood=table[:, 3])
