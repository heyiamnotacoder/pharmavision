"""DeepLabCut output CSVs in DLCAnalyzer's layout, written from synthetic positions."""

from collections.abc import Sequence
from pathlib import Path

PARTS = ("tl", "bodycentre", "tailbase")
"""A few of the 18 parts the Sturman network tracks: an arena corner, the body centre, the tail."""


def write_dlc(path: Path, bodycentre: Sequence[tuple[float, float, float]]) -> Path:
    """One row per frame: each part's x, y and likelihood. The body centre is at `bodycentre`
    (x, y, likelihood); the other parts sit elsewhere, so a parser reading the wrong columns
    gets wrong numbers."""
    scorer = "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    lines = [
        ",".join(["scorer", *[scorer] * 3 * len(PARTS)]),
        ",".join(["bodyparts", *[p for p in PARTS for _ in range(3)]]),
        ",".join(["coords", *["x", "y", "likelihood"] * len(PARTS)]),
    ]
    for i, (x, y, likelihood) in enumerate(bodycentre):
        values = {
            "tl": (1.0, 2.0, 0.99),
            "bodycentre": (x, y, likelihood),
            "tailbase": (7.0, 8.0, 0.5),
        }
        lines.append(",".join([str(i), *[str(v) for p in PARTS for v in values[p]]]))
    path.write_text("\n".join(lines) + "\n")
    return path
