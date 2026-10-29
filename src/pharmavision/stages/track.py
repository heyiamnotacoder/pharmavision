"""Track (sees): the animal's mask in every frame, keyed by frame index."""

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as coco_mask

from ..artifacts import RunFolder, Track, TrackFrame
from ..measure.shape import mask_shape
from ..segmenter import FrameMasks, Segmenter, best_mask


def run(run: RunFolder, segmenter: Segmenter) -> None:
    run.confirmed_arena()
    manifest = run.manifest()
    template, n_frames = manifest.template, manifest.video.n_frames
    by_frame_index: dict[int, TrackFrame] = {}
    stream = segmenter.track_video(run.video_path, template.animal_prompt, template.max_animals)
    for frame in stream:
        if not 0 <= frame.frame_index < n_frames:
            raise ValueError(
                f"{segmenter.model_id} returned frame {frame.frame_index} of {n_frames}"
            )
        animal = _animal_in(frame)
        if animal is not None:
            by_frame_index[frame.frame_index] = animal
    frames = [by_frame_index.get(i) or TrackFrame.gap(i) for i in range(n_frames)]
    run.write_track(Track(frames=frames))


def _animal_in(frame: FrameMasks) -> TrackFrame | None:
    """The single animal: the best mask in the frame, or None (a gap) when there is none."""
    best = best_mask(frame.masks)
    shape = mask_shape(best.pixels) if best else None
    if best is None or shape is None:
        return None
    return TrackFrame(
        frame_index=frame.frame_index,
        detected=True,
        x_px=shape.x_px,
        y_px=shape.y_px,
        area_px=shape.area_px,
        confidence=best.confidence,
        mask_rle=_rle(best.pixels),
    )


def _rle(pixels: NDArray[np.bool_]) -> str:
    """COCO RLE counts; the size is the video's frame size in the manifest."""
    counts: str | bytes = coco_mask.encode(np.asfortranarray(pixels.astype(np.uint8)))["counts"]
    # The stubs say str; pycocotools returns bytes.
    return counts.decode("ascii") if isinstance(counts, bytes) else counts
