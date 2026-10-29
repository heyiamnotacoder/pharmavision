"""Arena (sees): find the floor on a background frame, then derive the zones and scale in code."""

import cv2
import numpy as np
from numpy.typing import NDArray

from ..artifacts import Arena, RunFolder
from ..measure.geometry import Point, px_per_cm, shrink_to_area_fraction
from ..segmenter import Mask, Segmenter

BACKGROUND_SAMPLES = 25


class ArenaNotFoundError(RuntimeError):
    pass


def detect(run: RunFolder, segmenter: Segmenter) -> Arena:
    template = run.manifest().template
    masks = segmenter.segment_image(background_frame(run), template.floor_prompt)
    if not masks:
        raise ArenaNotFoundError(f"no {template.floor_prompt!r} found in run {run.run_id}")
    floor = mask_to_polygon(_best(masks))
    return Arena(
        floor_px=floor,
        centre_px=shrink_to_area_fraction(floor, template.centre_area_fraction),
        px_per_cm=px_per_cm(floor, template.arena_width_cm, template.arena_height_cm),
        confirmed=False,
    )


def background_frame(run: RunFolder) -> NDArray[np.uint8]:
    """Per-pixel median over evenly sampled frames, so the animal disappears from the floor."""
    n_frames = run.manifest().video.n_frames
    capture = cv2.VideoCapture(str(run.video_path))
    frames: list[NDArray[np.uint8]] = []
    try:
        for index in np.linspace(0, n_frames - 1, min(BACKGROUND_SAMPLES, n_frames), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(np.asarray(frame, dtype=np.uint8))
    finally:
        capture.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def mask_to_polygon(mask: NDArray[np.bool_]) -> list[Point]:
    """Outline of the largest connected region, simplified to within 1% of its perimeter."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(outline, 0.01 * cv2.arcLength(outline, closed=True), closed=True)
    return [(float(x), float(y)) for x, y in simplified.reshape(-1, 2)]


def _best(masks: list[Mask]) -> NDArray[np.bool_]:
    return max(masks, key=lambda m: (m.confidence or 0.0, int(m.pixels.sum()))).pixels
