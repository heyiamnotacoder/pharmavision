"""Arena (sees): find the floor on a background frame, then derive the zones and scale in code.
The floor is SAM 3.1's or, where it found none or the scientist moved its corners, the one
drawn in the zone editor; the centre zone and the scale always follow from it."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from ..artifacts import Arena, RunFolder
from ..measure.geometry import (
    InvalidFloorError,
    Point,
    check_floor,
    px_per_cm,
    shrink_to_area_fraction,
)
from ..segmenter import Segmenter, best_mask
from ..templates import TestTemplate

EMPTY = Arena(floor_px=[], centre_px=[], px_per_cm=0.0, confirmed=False)
"""What the stage leaves when no floor was found: the scientist draws one at the gate."""


def run(run: RunFolder, segmenter: Segmenter) -> Arena:
    """Write the background frame, and the arena found on it (EMPTY if no usable floor was)."""
    manifest = run.manifest()
    template, settings, video = manifest.template, manifest.arena, manifest.video
    background = background_frame(run, settings.background_samples)
    if not cv2.imwrite(str(run.background_path), background):
        raise OSError(f"could not write {run.background_path}")
    image = equalise(background, settings.contrast_clip_limit, settings.contrast_tiles)
    mask = best_mask(segmenter.segment_image(image, template.floor_prompt))
    floor = mask_to_polygon(mask.pixels, settings.outline_tolerance) if mask else []
    try:
        check_floor(floor, video.width, video.height)
    except InvalidFloorError:
        return EMPTY
    return zones(floor, template)


def zones(floor: Sequence[Point], template: TestTemplate) -> Arena:
    """The unconfirmed arena on `floor`: the centre zone and the scale come from it, so an
    edited floor moves both."""
    return Arena(
        floor_px=list(floor),
        centre_px=shrink_to_area_fraction(floor, template.centre_area_fraction),
        px_per_cm=px_per_cm(floor, template.arena_width_cm, template.arena_height_cm),
        confirmed=False,
    )


def background_frame(run: RunFolder, samples: int) -> NDArray[np.uint8]:
    """Per-pixel median over evenly sampled frames, so the animal disappears from the floor."""
    n_frames = run.manifest().video.n_frames
    capture = cv2.VideoCapture(str(run.video_path))
    frames: list[NDArray[np.uint8]] = []
    try:
        for index in np.linspace(0, n_frames - 1, min(samples, n_frames), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(np.asarray(frame, dtype=np.uint8))
    finally:
        capture.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def equalise(image: NDArray[np.uint8], clip_limit: float | None, tiles: int) -> NDArray[np.uint8]:
    """Equalise the lightness channel tile by tile (CLAHE), so colours keep their hue. The
    image unchanged when `clip_limit` is None."""
    if clip_limit is None:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), dtype=np.uint8)


def mask_to_polygon(mask: NDArray[np.bool_], tolerance: float) -> list[Point]:
    """Outline of the largest connected region, simplified to within `tolerance` of its
    perimeter. Empty for an empty mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    outline = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(
        outline, tolerance * cv2.arcLength(outline, closed=True), closed=True
    )
    return [(float(x), float(y)) for x, y in simplified.reshape(-1, 2)]
