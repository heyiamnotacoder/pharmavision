"""Track (sees): the animal's mask in every frame, keyed by frame index."""

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as coco_mask

from ..artifacts import RunFolder, Track, TrackFrame
from ..measure.shape import mask_shape
from ..segmenter import FrameMasks, Mask, Segmenter


def run(run: RunFolder, segmenter: Segmenter) -> None:
    run.confirmed_arena()
    manifest = run.manifest()
    template, n_frames = manifest.template, manifest.video.n_frames
    found: dict[int, TrackFrame] = {}
    stream = segmenter.track_video(run.video_path, template.animal_prompt, template.max_animals)
    for frame in stream:
        if not 0 <= frame.frame_index < n_frames:
            raise ValueError(
                f"{segmenter.model_id} returned frame {frame.frame_index} of {n_frames}"
            )
        record = _record(frame)
        if record is not None:
            found[frame.frame_index] = record
    frames = [found.get(i) or TrackFrame.gap(i) for i in range(n_frames)]
    run.write_track(Track(frames=frames))


def _record(frame: FrameMasks) -> TrackFrame | None:
    """The single animal: the highest-confidence non-empty mask, or None (a gap)."""
    candidates = [(m, s) for m in frame.masks if (s := mask_shape(m.pixels)) is not None]
    if not candidates:
        return None
    best, shape = max(candidates, key=lambda c: _confidence(c[0]))
    return TrackFrame(
        frame_index=frame.frame_index,
        detected=True,
        x_px=shape.x_px,
        y_px=shape.y_px,
        area_px=shape.area_px,
        confidence=best.confidence,
        mask_rle=_rle(best.pixels),
    )


def _confidence(mask: Mask) -> float:
    return mask.confidence if mask.confidence is not None else 0.0


def _rle(pixels: NDArray[np.bool_]) -> str:
    """COCO RLE counts; the size is the video's frame size in the manifest."""
    counts: str | bytes = coco_mask.encode(np.asfortranarray(pixels.astype(np.uint8)))["counts"]
    # The stubs say str; pycocotools returns bytes.
    return counts.decode("ascii") if isinstance(counts, bytes) else counts
