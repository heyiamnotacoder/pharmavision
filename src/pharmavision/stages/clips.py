"""Short clips of a run's video for the judge: a span of frames, cropped to a box and scaled, with
the tracked mask outlined, encoded bit-exact so the same frames always give the same bytes. The
video is read once, front to back, for all of a stage's clips; the model never sees all of it."""

import subprocess
from collections.abc import Collection, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ..artifacts import NormalisedVideo, RunFolder, TrackFrame
from ..judge import Clip
from ..measure.shape import decode_mask
from ..model_cache import input_hash

OUTLINE_BGR = (0, 255, 0)

Box = tuple[int, int, int, int]
"""A crop: left and top edge, width and height, in pixels of the normalised video."""


@dataclass(frozen=True)
class ClipView:
    """Which frames a clip shows, the part of each frame it crops to, and its (width, height)."""

    span: range
    box: Box
    size: tuple[int, int]


def span_about(
    first: int, last: int, video: NormalisedVideo, pad_s: float, min_s: float, max_s: float
) -> range:
    """The frames of a clip around frames `first` to `last`: padded by `pad_s` each side,
    widened to `min_s` or cut to `max_s` about their middle, then shifted to lie inside the
    video."""
    fps, n_frames = video.fps, video.n_frames
    pad = round(pad_s * fps)
    shortest, longest = round(min_s * fps), round(max_s * fps)
    start, stop = first - pad, last + 1 + pad
    length = min(max(stop - start, shortest), longest, n_frames)
    middle = (first + last + 1) / 2
    start = min(max(round(middle - length / 2), 0), n_frames - length)
    return range(start, start + length)


def cut_clips(run: RunFolder, views: Sequence[ClipView], dest: Path) -> Generator[tuple[int, Clip]]:
    """Each view's clip, as `dest/<k>.mp4` for the k-th view, in the order their spans end."""
    video = run.manifest().video
    frames = run.track().frames
    buffers: dict[int, list[NDArray[np.uint8]]] = {k: [] for k in range(len(views))}
    capture = cv2.VideoCapture(str(run.video_path))
    try:
        last_needed = max((view.span.stop for view in views), default=0)
        for index in range(last_needed):
            image = _read(capture, run.video_path, index)
            for k, view in enumerate(views):
                if index not in view.span:
                    continue
                buffers[k].append(_clip_frame(image, frames[index], video, view))
                if index == view.span.stop - 1:
                    yield k, _clip(buffers.pop(k), video.fps, dest / f"{k}.mp4")
    finally:
        capture.release()


def read_frames(video: Path, indices: Collection[int]) -> dict[int, NDArray[np.uint8]]:
    """The frames at `indices`, read front to back: seeking an H.264 stream may land on another
    frame, and the frame's bytes are a model's cache key."""
    wanted = set(indices)
    found: dict[int, NDArray[np.uint8]] = {}
    capture = cv2.VideoCapture(str(video))
    try:
        for index in range(max(wanted, default=-1) + 1):
            image = _read(capture, video, index)
            if index in wanted:
                found[index] = image
    finally:
        capture.release()
    return found


def mask_of(frame: TrackFrame, video: NormalisedVideo) -> NDArray[np.uint8] | None:
    """The frame's tracked mask over the whole frame, or None in a gap."""
    if frame.mask_rle is None:
        return None
    return decode_mask(frame.mask_rle, video.height, video.width)


def _read(capture: cv2.VideoCapture, video: Path, index: int) -> NDArray[np.uint8]:
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"{video.name}: could not read frame {index}")
    return np.asarray(image, dtype=np.uint8)


def _clip_frame(
    image: NDArray[np.uint8], frame: TrackFrame, video: NormalisedVideo, view: ClipView
) -> NDArray[np.uint8]:
    """The crop scaled to the clip size, with the frame's mask outlined where there is one."""
    x, y, width, height = view.box
    crop = cv2.resize(image[y : y + height, x : x + width], view.size, interpolation=cv2.INTER_AREA)
    mask = mask_of(frame, video)
    if mask is not None:
        local = cv2.resize(
            mask[y : y + height, x : x + width], view.size, interpolation=cv2.INTER_NEAREST
        )
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, contours, -1, OUTLINE_BGR, thickness=2)
    return np.asarray(crop, dtype=np.uint8)


def _clip(frames: Sequence[NDArray[np.uint8]], fps: float, dest: Path) -> Clip:
    """The frames as an H.264 MP4, keyed by the hash of the frames themselves and the frame
    rate, which every machine computes alike from the same run folder."""
    height, width = frames[0].shape[:2]
    frames_sha256 = input_hash(f"{width}x{height} at {fps} fps", *(f.tobytes() for f in frames))
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18",
            "-threads", "1", "-flags:v", "+bitexact", "-fflags", "+bitexact",
            "-map_metadata", "-1", "-movflags", "+faststart", str(dest),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True, capture_output=True,
    )  # fmt: skip
    return Clip(dest, frames_sha256)
