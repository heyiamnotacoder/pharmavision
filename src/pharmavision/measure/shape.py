"""Shape features of binary masks, and how a mask changes from one frame to the next."""

import itertools
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as coco_mask


class MaskShape(NamedTuple):
    x_px: float
    y_px: float
    area_px: int


def mask_shape(mask: NDArray[np.bool_]) -> MaskShape | None:
    """Centroid (pixel centres) and area of the mask, or None for an empty mask."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return MaskShape(x_px=float(cols.mean()), y_px=float(rows.mean()), area_px=int(rows.size))


def mask_changes(rles: Sequence[str | None], height: int, width: int) -> list[float | None]:
    """The change from each frame's mask to the next one's, None where either is missing."""
    return [
        _mask_change(a, b, height, width) if a is not None and b is not None else None
        for a, b in itertools.pairwise(rles)
    ]


def mask_elongations(rles: Sequence[str | None], height: int, width: int) -> list[float | None]:
    """Each frame's mask elongation, None where the mask is missing or has no width."""
    return [_elongation(rle, height, width) if rle is not None else None for rle in rles]


def _elongation(rle: str, height: int, width: int) -> float | None:
    """Major over minor axis of the ellipse with the mask's second moments: 1 for a disc, a / b
    for an ellipse with semi-axes a and b. None for a mask that is a line or a point."""
    with warnings.catch_warnings():
        # pycocotools' Cython decoder predates NumPy 2's `copy` keyword; the result is right.
        warnings.filterwarnings("ignore", "__array__ implementation", DeprecationWarning)
        decoded: NDArray[np.uint8] = coco_mask.decode(
            {"size": [height, width], "counts": rle.encode("ascii")}
        )
    rows, cols = np.nonzero(decoded)
    if rows.size < 2:
        return None
    (var_x, cov), (_, var_y) = np.cov(np.stack([cols, rows]).astype(np.float64)).tolist()
    spread = float(np.hypot((var_x - var_y) / 2, cov))
    major, minor = (var_x + var_y) / 2 + spread, (var_x + var_y) / 2 - spread
    # Relative, as collinear pixels leave rounding error, not zero, in the minor axis.
    return float(np.sqrt(major / minor)) if minor > 1e-9 * major else None


def _mask_change(before_rle: str, after_rle: str, height: int, width: int) -> float:
    """Fraction of two masks' pixels that changed, 1 - IoU (symmetric difference over union),
    computed on the COCO RLE counts without decoding. 0 for an identical mask, 1 for masks that
    don't overlap."""
    size = [height, width]
    iou = coco_mask.iou(
        [{"size": size, "counts": before_rle.encode("ascii")}],
        [{"size": size, "counts": after_rle.encode("ascii")}],
        [0],
    )
    return 1.0 - float(iou[0][0])
