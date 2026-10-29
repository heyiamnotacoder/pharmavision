"""Shape features of one binary mask."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


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
