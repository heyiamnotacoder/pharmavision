"""Shape features of one binary mask."""

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


def mask_change(before_rle: str, after_rle: str, height: int, width: int) -> float:
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
