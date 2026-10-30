"""Behaviours (judges): each rear candidate's clip, judged by the model, as one event.

The clip is 2-4 s of the normalised video around the candidate window, cropped to a square
around the animal and with its tracked mask outlined, so the model sees the animal rather than
the arena. The video is read once for all clips; the model never sees the full video."""

import tempfile
from collections.abc import Generator, Sequence
from contextlib import closing
from pathlib import Path

import numpy as np

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
from .clips import Box, ClipView, mask_of, span_about
from .clips import cut_clips as cut_views


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
    """Each span's clip, cropped to a square around the animal and outlined, as `dest/<k>.mp4`
    for the k-th span, in the order the spans end."""
    arena = run.confirmed_arena()
    manifest = run.manifest()
    settings, video = manifest.behaviours, manifest.video
    frames = run.track().frames
    margin_px = settings.crop_margin_cm * arena.px_per_cm
    size = (settings.clip_side_px, settings.clip_side_px)
    views = [
        ClipView(span, _crop_box(frames[span.start : span.stop], video, margin_px), size)
        for span in spans
    ]
    return cut_views(run, views, dest)


def clip_span(
    candidate: RearCandidate, video: NormalisedVideo, settings: BehaviourSettings
) -> range:
    """The frames of the clip shown for `candidate`: the window padded by `clip_pad_s` each
    side, widened to `clip_min_s` or cut to `clip_max_s` about the window's middle, then
    shifted to lie inside the video."""
    return span_about(
        candidate.first_frame,
        candidate.last_frame,
        video,
        settings.clip_pad_s,
        settings.clip_min_s,
        settings.clip_max_s,
    )


def _crop_box(frames: Sequence[TrackFrame], video: NormalisedVideo, margin_px: float) -> Box:
    """The square around every mask in the clip's frames, plus the margin, inside the frame.
    Without any mask, the largest square in the middle of the frame."""
    masks = [mask for frame in frames if (mask := mask_of(frame, video)) is not None]
    if masks:
        rows, cols = np.nonzero(np.logical_or.reduce(masks))
        left, top = float(cols.min()), float(rows.min())
        right, bottom = float(cols.max() + 1), float(rows.max() + 1)
    else:
        left, top, right, bottom = 0.0, 0.0, float(video.width), float(video.height)
    side = min(round(max(right - left, bottom - top) + 2 * margin_px), video.width, video.height)
    x = min(max(round((left + right - side) / 2), 0), video.width - side)
    y = min(max(round((top + bottom - side) / 2), 0), video.height - side)
    return (x, y, side, side)
