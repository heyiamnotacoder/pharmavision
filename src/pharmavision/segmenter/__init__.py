"""The single boundary to the "sees" layer: pixels in, masks out. Geometry only."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Mask:
    """One object's binary mask over the whole frame, row-major (height, width)."""

    pixels: NDArray[np.bool_]
    confidence: float | None = None


@dataclass(frozen=True)
class FrameMasks:
    """Every mask for one frame. Backends may skip frames or deliver them out of order, so
    consumers key on `frame_index`, never on arrival position."""

    frame_index: int
    masks: tuple[Mask, ...]


class Segmenter(Protocol):
    @property
    def model_id(self) -> str:
        """Recorded in the manifest so a run says which model saw it."""
        ...

    def segment_image(self, image: NDArray[np.uint8], prompt: str) -> list[Mask]:
        """Masks for every instance of `prompt` in one image. No match gives an empty list."""
        ...

    def track_video(self, video: Path, prompt: str, max_objects: int) -> Iterator[FrameMasks]:
        """Masks per frame for `prompt` across the video. No match gives an empty stream."""
        ...
