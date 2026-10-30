"""Behaviours (judges): each rear candidate's clip, judged by the model, as one event.

The clip is 2-4 s of the normalised video around the candidate window, cropped to a square
around the animal and with its tracked mask outlined, so the model sees the animal rather than
the arena. The video is read once for all clips; the model never sees the full video."""

import subprocess
import tempfile
from collections.abc import Generator, Sequence
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ..artifacts import (
    BehaviourEvent,
    BehaviourSettings,
    Events,
    NormalisedVideo,
    RearCandidate,
    RunFolder,
    TrackFrame,
)
from ..judge import Clip, Judge
from ..measure.shape import decode_mask
from ..model_cache import input_hash

OUTLINE_BGR = (0, 255, 0)

Box = tuple[int, int, int]
"""A square crop: left and top edge, and side, in pixels of the normalised video."""


def run(run: RunFolder, judge: Judge) -> None:
    run.confirmed_arena()  # the gate, before reading anything else
    manifest = run.manifest()
    settings = manifest.behaviours
    candidates = run.metrics().rear_candidates
    spans = [clip_span(c, manifest.video, settings) for c in candidates]
    events: dict[int, BehaviourEvent] = {}
    with tempfile.TemporaryDirectory() as tmp, closing(cut_clips(run, spans, Path(tmp))) as clips:
        for k, clip in clips:
            judgement = judge.classify_clip(clip)
            events[k] = BehaviourEvent(
                first_frame=candidates[k].first_frame,
                last_frame=candidates[k].last_frame,
                label=judgement.label,
                confidence=judgement.confidence,
                low_confidence=judgement.confidence < settings.min_confidence,
                rationale=judgement.rationale,
                source="model",
            )
    run.write_events(Events(events=[events[k] for k in range(len(spans))]))


def cut_clips(run: RunFolder, spans: Sequence[range], dest: Path) -> Generator[tuple[int, Clip]]:
    """Each span's clip, cropped and outlined, as `dest/<k>.mp4` for the k-th span. The video is
    read once, front to back, so clips come in the order their spans end."""
    arena = run.confirmed_arena()
    manifest = run.manifest()
    settings, video = manifest.behaviours, manifest.video
    frames = run.track().frames
    margin_px = settings.crop_margin_cm * arena.px_per_cm
    boxes = [_crop_box(frames[span.start : span.stop], video, margin_px) for span in spans]
    buffers: dict[int, list[NDArray[np.uint8]]] = {k: [] for k in range(len(spans))}
    capture = cv2.VideoCapture(str(run.video_path))
    try:
        last_needed = max((span.stop for span in spans), default=0)
        for index in range(last_needed):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"{run.video_path.name}: could not read frame {index}")
            for k, span in enumerate(spans):
                if index not in span:
                    continue
                crop = _clip_frame(
                    np.asarray(image, dtype=np.uint8), frames[index], video, boxes[k], settings
                )
                buffers[k].append(crop)
                if index == span.stop - 1:
                    yield k, _clip(buffers.pop(k), video.fps, dest / f"{k}.mp4")
    finally:
        capture.release()


def clip_span(
    candidate: RearCandidate, video: NormalisedVideo, settings: BehaviourSettings
) -> range:
    """The frames of the clip shown for `candidate`: the window padded by `clip_pad_s` each
    side, widened to `clip_min_s` or cut to `clip_max_s` about the window's middle, then
    shifted to lie inside the video."""
    fps, n_frames = video.fps, video.n_frames
    pad = round(settings.clip_pad_s * fps)
    shortest, longest = round(settings.clip_min_s * fps), round(settings.clip_max_s * fps)
    start, stop = candidate.first_frame - pad, candidate.last_frame + 1 + pad
    length = min(max(stop - start, shortest), longest, n_frames)
    middle = (candidate.first_frame + candidate.last_frame + 1) / 2
    start = min(max(round(middle - length / 2), 0), n_frames - length)
    return range(start, start + length)


def _crop_box(frames: Sequence[TrackFrame], video: NormalisedVideo, margin_px: float) -> Box:
    """The square around every mask in the clip's frames, plus the margin, inside the frame.
    Without any mask, the largest square in the middle of the frame."""
    masks = [mask for frame in frames if (mask := _mask(frame, video)) is not None]
    if masks:
        rows, cols = np.nonzero(np.logical_or.reduce(masks))
        left, top = float(cols.min()), float(rows.min())
        right, bottom = float(cols.max() + 1), float(rows.max() + 1)
    else:
        left, top, right, bottom = 0.0, 0.0, float(video.width), float(video.height)
    side = min(round(max(right - left, bottom - top) + 2 * margin_px), video.width, video.height)
    x = min(max(round((left + right - side) / 2), 0), video.width - side)
    y = min(max(round((top + bottom - side) / 2), 0), video.height - side)
    return (x, y, side)


def _clip_frame(
    image: NDArray[np.uint8],
    frame: TrackFrame,
    video: NormalisedVideo,
    box: Box,
    settings: BehaviourSettings,
) -> NDArray[np.uint8]:
    """The crop scaled to the clip size, with the frame's mask outlined where there is one."""
    x, y, side = box
    size = (settings.clip_side_px, settings.clip_side_px)
    crop = cv2.resize(image[y : y + side, x : x + side], size, interpolation=cv2.INTER_AREA)
    mask = _mask(frame, video)
    if mask is not None:
        local = cv2.resize(mask[y : y + side, x : x + side], size, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, contours, -1, OUTLINE_BGR, thickness=2)
    return np.asarray(crop, dtype=np.uint8)


def _mask(frame: TrackFrame, video: NormalisedVideo) -> NDArray[np.uint8] | None:
    """The frame's tracked mask over the whole frame, or None in a gap."""
    if frame.mask_rle is None:
        return None
    return decode_mask(frame.mask_rle, video.height, video.width)


def _clip(frames: Sequence[NDArray[np.uint8]], fps: float, dest: Path) -> Clip:
    """The frames as an H.264 MP4, keyed by the hash of the frames themselves and the frame
    rate, which every machine computes alike from the same run folder."""
    side = frames[0].shape[0]
    frames_sha256 = input_hash(f"{side}x{side} at {fps} fps", *(f.tobytes() for f in frames))
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{side}x{side}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18",
            "-threads", "1", "-flags:v", "+bitexact", "-fflags", "+bitexact",
            "-map_metadata", "-1", "-movflags", "+faststart", str(dest),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True, capture_output=True,
    )  # fmt: skip
    return Clip(dest, frames_sha256)
