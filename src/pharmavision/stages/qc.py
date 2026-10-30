"""QC (measures, then judges): the deterministic flags and the exclusions they lead to, through
`measure/qc.py`; then the model review of each warning. SAM 3.1 looks for hands on a few of the
warning's frames, and the judge sees a short whole-frame clip around it and gives a verdict,
which applies automatically. Only flagged windows go to the models: with the review off, or
nothing flagged, or the trial already excluded by a rule, no model is called."""

import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..artifacts import (
    Arena,
    Exclusion,
    NormalisedVideo,
    Qc,
    QcFlag,
    QcReview,
    QcReviewSettings,
    RunFolder,
)
from ..judge import FlaggedWindow, HandSighting, Judge
from ..measure.qc import trial_qc
from ..segmenter import Segmenter
from .clips import ClipView, cut_clips, read_frames, span_about


def run(run: RunFolder, segmenter: Segmenter, judge: Judge) -> None:
    arena = run.confirmed_arena()
    manifest = run.manifest()
    qc = trial_qc(run.track(), arena, run.metrics().gaps, manifest.video, manifest.qc)
    settings = manifest.qc.review
    if settings is not None and not qc.trial_excluded:
        qc = _reviewed(run, qc, arena, segmenter, judge, settings)
    run.write_qc(qc)


def _reviewed(
    run: RunFolder,
    qc: Qc,
    arena: Arena,
    segmenter: Segmenter,
    judge: Judge,
    settings: QcReviewSettings,
) -> Qc:
    """`qc` with every warning reviewed, and the exclusions the verdicts lead to added."""
    video = run.manifest().video
    warnings = [flag for flag in qc.flags if flag.severity == "warning"]
    if not warnings:
        return qc
    sampled = [_sampled_frames(flag, settings.hand_frames) for flag in warnings]
    stills = read_frames(run.video_path, {i for frames in sampled for i in frames})
    views = [_whole_frame_view(flag, video, settings) for flag in warnings]
    reviews: dict[int, QcReview] = {}
    with tempfile.TemporaryDirectory() as tmp, closing(cut_clips(run, views, Path(tmp))) as clips:
        for k, clip in clips:
            flag = warnings[k]
            hands = [
                sighting
                for i in sampled[k]
                for prompt in settings.hand_prompts
                if (sighting := _hand(segmenter, stills[i], i, prompt, arena.px_per_cm))
            ]
            window = FlaggedWindow(
                clip, flag.kind, flag.first_frame, flag.last_frame, flag.reason, tuple(hands)
            )
            verdict = judge.review_window(window)
            reviews[k] = QcReview(
                flag=flag, hands=hands, verdict=verdict.verdict, reason=verdict.reason
            )
    ordered = [reviews[k] for k in range(len(warnings))]
    excluded = [_exclusion(r, video) for r in ordered if r.verdict != "keep"]
    return qc.model_copy(update={"exclusions": qc.exclusions + excluded, "reviews": ordered})


def _sampled_frames(flag: QcFlag, n: int) -> list[int]:
    """`n` frames evenly spaced from the flag's first to its last (its middle for one)."""
    if n == 1:
        return [(flag.first_frame + flag.last_frame) // 2]
    spaced = np.linspace(flag.first_frame, flag.last_frame, n).round().astype(int)
    return sorted({int(i) for i in spaced})


def _hand(
    segmenter: Segmenter, image: NDArray[np.uint8], frame_index: int, prompt: str, px_per_cm: float
) -> HandSighting | None:
    """The prompt's largest mask on the frame, or None when it finds nothing."""
    areas = [int(mask.pixels.sum()) for mask in segmenter.segment_image(image, prompt)]
    largest = max((a for a in areas if a), default=0)
    if not largest:
        return None
    return HandSighting(prompt=prompt, frame_index=frame_index, area_cm2=largest / px_per_cm**2)


def _whole_frame_view(flag: QcFlag, video: NormalisedVideo, settings: QcReviewSettings) -> ClipView:
    """The clip around the flag: the whole frame, as a hand comes in from the edge, scaled down
    (never up) to fit `clip_max_side_px`, each side even for H.264."""
    span = span_about(
        flag.first_frame,
        flag.last_frame,
        video,
        settings.clip_pad_s,
        settings.clip_min_s,
        settings.clip_max_s,
    )
    scale = min(1.0, settings.clip_max_side_px / max(video.width, video.height))
    size = (2 * round(video.width * scale / 2), 2 * round(video.height * scale / 2))
    return ClipView(span, (0, 0, video.width, video.height), size)


def _exclusion(review: QcReview, video: NormalisedVideo) -> Exclusion:
    if review.verdict == "exclude_trial":
        first, last, scope = 0, video.n_frames - 1, "trial"
    else:
        first, last, scope = review.flag.first_frame, review.flag.last_frame, "window"
    return Exclusion(
        scope=scope, first_frame=first, last_frame=last, reason=review.reason, source="model"
    )
