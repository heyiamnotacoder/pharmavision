"""Arena (sees): find the floor on a background frame, then derive the zones and scale in code."""

import cv2
import numpy as np
from numpy.typing import NDArray

from ..artifacts import Arena, RunFolder
from ..measure.geometry import Point, polygon_area, px_per_cm, shrink_to_area_fraction
from ..segmenter import Segmenter, best_mask


class ArenaNotFoundError(RuntimeError):
    pass


def detect(run: RunFolder, segmenter: Segmenter) -> Arena:
    manifest = run.manifest()
    template, settings = manifest.template, manifest.arena
    background = background_frame(run, settings.background_samples)
    image = equalise(background, settings.contrast_clip_limit)
    mask = best_mask(segmenter.segment_image(image, template.floor_prompt))
    floor = mask_to_polygon(mask.pixels, settings.outline_tolerance) if mask else []
    if len(floor) < 3 or polygon_area(floor) == 0:
        raise ArenaNotFoundError(f"no {template.floor_prompt!r} found in run {run.run_id}")
    return Arena(
        floor_px=floor,
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


def equalise(image: NDArray[np.uint8], clip_limit: float | None) -> NDArray[np.uint8]:
    """Local contrast stretch (CLAHE) on the lightness channel only, so colours keep their hue.
    The image unchanged when `clip_limit` is None."""
    if clip_limit is None:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lab[..., 0])
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
