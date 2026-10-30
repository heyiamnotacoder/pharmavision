"""A stand-in for SAM 3.1 that draws the masks of a scripted scene, so every metric has a
closed-form answer through the real pipeline."""

import random
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ..measure.geometry import Point
from . import FrameMasks, Mask

_SHIFT = 4


class SyntheticSegmenter:
    """Ignores pixels and prompts: images always show `floor`, videos always show one animal
    at `trajectory(frame_index)`, an axis-aligned ellipse with semi-axes `body(frame_index)`
    (along x, along y), and a second, less confident and larger object (an intruding hand, a
    disc) at `intruder(frame_index)`. Like SAM 3.1, it skips frames where neither is there,
    keeps the `max_objects` most confident, and with `shuffle_seed` delivers frames out of
    order."""

    model_id = "synthetic"

    def __init__(
        self,
        floor: Sequence[Point] = (),
        trajectory: Callable[[int], Point | None] = lambda _: None,
        body: Callable[[int], tuple[float, float]] = lambda _: (8.0, 8.0),
        intruder: Callable[[int], Point | None] = lambda _: None,
        intruder_radius_px: float = 30.0,
        shuffle_seed: int | None = None,
    ) -> None:
        self.floor = floor
        self.trajectory = trajectory
        self.body = body
        self.intruder = intruder
        self.intruder_radius_px = intruder_radius_px
        self.shuffle_seed = shuffle_seed

    def segment_image(self, image: NDArray[np.uint8], prompt: str) -> list[Mask]:
        if not self.floor:
            return []
        canvas = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(canvas, [np.round(np.array(self.floor)).astype(np.int32)], color=1)
        return [Mask(canvas.astype(bool), confidence=1.0)]

    def track_video(self, video: Path, prompt: str, max_objects: int) -> Iterator[FrameMasks]:
        capture = cv2.VideoCapture(str(video))
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = (
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        capture.release()
        order = list(range(n_frames))
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(order)
        for index in order:
            hand = (self.intruder_radius_px, self.intruder_radius_px)
            objects = [
                Mask(self._ellipse(shape, centre, semi_axes), confidence=confidence)
                for centre, semi_axes, confidence in (
                    (self.trajectory(index), self.body(index), 0.9),
                    (self.intruder(index), hand, 0.5),
                )
                if centre is not None
            ]
            if objects:
                yield FrameMasks(index, tuple(objects[:max_objects]))

    def _ellipse(
        self, shape: tuple[int, int], centre: Point, semi_axes: tuple[float, float]
    ) -> NDArray[np.bool_]:
        canvas = np.zeros(shape, dtype=np.uint8)
        # Fixed-point coordinates (4 fractional bits) so the centroid isn't snapped to the grid.
        scale = 1 << _SHIFT
        x, y = centre
        cv2.ellipse(
            canvas,
            (round(x * scale), round(y * scale)),
            (round(semi_axes[0] * scale), round(semi_axes[1] * scale)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
            shift=_SHIFT,
        )
        return canvas.astype(bool)
