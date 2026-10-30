"""The model review of QC flags through the pipeline seam: each flagged window is shown to SAM 3.1
("hand", "glove") and to the judge, and the verdicts change the exclusions, each with its reason.
With the review off, the deterministic QC stands alone."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pharmavision.artifacts import Exclusion, QcSettings
from pharmavision.judge import Judge, WindowVerdict
from pharmavision.judge.muse_spark import (
    JudgeResponseError,
    MuseSparkJudge,
    MuseSparkSettings,
    RecordedJudge,
)
from pharmavision.measure.geometry import Point
from pharmavision.pipeline import RunFolder
from pharmavision.segmenter import Mask
from pharmavision.segmenter.synthetic import SyntheticSegmenter

from .judges import ScriptedJudge
from .muse_replay import decode_video, frame_size_and_count, verdict, videos_sent
from .sam_replay import ReplayTransport
from .scenes import FLOOR, N_FRAMES, PX_PER_CM, circle_walk, finished_run

# A 10 x 10 cm square in the frame's top-left corner, outside the floor.
HAND = [(0.0, 0.0), (10 * PX_PER_CM, 0.0), (10 * PX_PER_CM, 10 * PX_PER_CM), (0.0, 10 * PX_PER_CM)]


class PromptSpy(SyntheticSegmenter):
    """Records every image prompt after the arena's."""

    def __init__(self, seen: dict[str, list[Point]] | None = None) -> None:
        super().__init__(
            floor=FLOOR,
            trajectory=lambda _: (180.0, 180.0),
            intruder=lambda i: (280.0, 280.0) if 40 <= i < 50 else None,
            seen=seen,
        )
        self.prompts: list[str] = []

    def segment_image(self, image: NDArray[np.uint8], prompt: str) -> list[Mask]:
        if prompt != self.floor_prompt:
            self.prompts.append(prompt)
        return super().segment_image(image, prompt)


def intruded_run(
    tmp_path: Path,
    runs_dir: Path,
    judge: Judge,
    qc_settings: QcSettings | None = None,
    segmenter: SyntheticSegmenter | None = None,
) -> RunFolder:
    """A still animal with a hand-sized disc beside it on frames 40-49: one `second_mask` flag."""
    return finished_run(
        tmp_path,
        runs_dir,
        lambda _: (180.0, 180.0),
        intruder=lambda i: (280.0, 280.0) if 40 <= i < 50 else None,
        judge=judge,
        qc_settings=qc_settings,
        segmenter=segmenter,
    )


def test_a_verdict_to_exclude_the_window_excludes_the_flagged_frames(
    tmp_path: Path, runs_dir: Path
) -> None:
    verdict = WindowVerdict(verdict="exclude_window", reason="A hand reaches in over the mouse.")

    run = intruded_run(tmp_path, runs_dir, ScriptedJudge(verdict=verdict))

    qc = run.qc()
    assert [(f.kind, f.first_frame, f.last_frame) for f in qc.flags] == [("second_mask", 40, 49)]
    assert qc.exclusions == [
        Exclusion(
            scope="window",
            first_frame=40,
            last_frame=49,
            reason="A hand reaches in over the mouse.",
            source="model",
        )
    ]
    assert not qc.trial_excluded
    [review] = qc.reviews
    assert review.flag == qc.flags[0]
    assert (review.verdict, review.reason) == ("exclude_window", verdict.reason)


def test_a_verdict_to_exclude_the_trial_excludes_it_with_the_models_reason(
    tmp_path: Path, runs_dir: Path
) -> None:
    verdict = WindowVerdict(verdict="exclude_trial", reason="The mouse is lifted out.")

    run = intruded_run(tmp_path, runs_dir, ScriptedJudge(verdict=verdict))

    qc = run.qc()
    assert qc.exclusions == [
        Exclusion(
            scope="trial",
            first_frame=0,
            last_frame=N_FRAMES - 1,
            reason="The mouse is lifted out.",
            source="model",
        )
    ]
    assert qc.trial_excluded
    assert run.manifest().models["qc"] == "scripted"
    assert run.manifest().models["qc_hands"] == "synthetic"


def test_a_verdict_to_keep_the_window_is_listed_and_excludes_nothing(
    tmp_path: Path, runs_dir: Path
) -> None:
    verdict = WindowVerdict(verdict="keep", reason="A reflection; the mouse is tracked.")

    run = intruded_run(tmp_path, runs_dir, ScriptedJudge(verdict=verdict))

    qc = run.qc()
    assert qc.exclusions == []
    assert [(r.verdict, r.reason) for r in qc.reviews] == [("keep", verdict.reason)]


def test_with_the_review_off_the_deterministic_qc_stands_alone(
    tmp_path: Path, runs_dir: Path
) -> None:
    excluding = WindowVerdict(verdict="exclude_trial", reason="Never asked.")
    judge, spy = ScriptedJudge(verdict=excluding), PromptSpy()
    off = QcSettings(review=None)
    deterministic = intruded_run(tmp_path, runs_dir, ScriptedJudge(), QcSettings(review=None))

    run = intruded_run(tmp_path, runs_dir, judge, off, spy)

    assert run.manifest().qc.review is None
    assert run.qc() == deterministic.qc()
    assert [(f.kind, f.first_frame, f.last_frame) for f in run.qc().flags] == [
        ("second_mask", 40, 49)
    ]
    assert run.qc().exclusions == []
    assert run.qc().reviews == []
    assert judge.windows == []
    assert spy.prompts == []
    assert "qc" not in run.manifest().models


def test_a_clean_trial_shows_the_models_nothing(tmp_path: Path, runs_dir: Path) -> None:
    judge = ScriptedJudge(verdict=WindowVerdict(verdict="exclude_trial", reason="Never asked."))

    run = finished_run(tmp_path, runs_dir, circle_walk(radius_px=90, laps=2), judge=judge)

    assert run.qc().flags == []
    assert judge.windows == []
    assert "qc" not in run.manifest().models


def test_a_trial_a_rule_already_excluded_shows_the_models_nothing(
    tmp_path: Path, runs_dir: Path
) -> None:
    # 17% of frames lost: the rule excludes the trial, so no verdict could change it.
    judge = ScriptedJudge(verdict=WindowVerdict(verdict="keep", reason="Never asked."))

    run = finished_run(
        tmp_path, runs_dir, lambda i: None if 20 <= i < 30 else (180.0, 180.0), judge=judge
    )

    assert run.qc().trial_excluded
    assert [e.source for e in run.qc().exclusions] == ["rule"]
    assert judge.windows == []


def test_only_the_flagged_window_goes_to_the_models(tmp_path: Path, runs_dir: Path) -> None:
    judge, spy = ScriptedJudge(), PromptSpy(seen={"hand": HAND})

    run = intruded_run(tmp_path, runs_dir, judge, segmenter=spy)

    # Three frames of the flagged window (40-49), each asked for a hand and for a glove.
    assert spy.prompts == ["hand", "glove"] * 3
    [shown] = judge.windows
    window = shown.window
    assert (window.kind, window.first_frame, window.last_frame) == ("second_mask", 40, 49)
    assert window.reason == run.qc().flags[0].reason
    assert [(h.prompt, h.frame_index) for h in window.hands] == [
        ("hand", 40),
        ("hand", 44),
        ("hand", 49),
    ]
    assert all(h.area_cm2 == pytest.approx(100, rel=0.05) for h in window.hands)
    assert run.qc().reviews[0].hands == list(window.hands)
    # A short clip of the whole frame around the window, never the full video.
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(shown.clip)
    width, height, n_frames = frame_size_and_count(clip)
    assert (width, height) == (360, 360)
    assert n_frames < N_FRAMES
    assert 2 * 15 <= n_frames <= 6 * 15


def test_recorded_muse_spark_verdicts_change_the_exclusions(tmp_path: Path, runs_dir: Path) -> None:
    # A first run records Muse Spark's verdict in the response cache, as the laptop would; the
    # recorded Judge then replays it by the same key, and never calls the API.
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")
    transport = ReplayTransport([verdict("exclude_window", "A gloved hand enters the arena.")])
    hands = PromptSpy(seen={"glove": HAND})
    intruded_run(tmp_path, runs_dir, MuseSparkJudge(settings, transport), segmenter=hands)

    replay = intruded_run(
        tmp_path, runs_dir, RecordedJudge(settings), segmenter=PromptSpy(seen={"glove": HAND})
    )

    assert replay.qc().exclusions == [
        Exclusion(
            scope="window",
            first_frame=40,
            last_frame=49,
            reason="A gloved hand enters the arena.",
            source="model",
        )
    ]
    assert replay.manifest().models["qc"] == "muse-spark-1.3"
    # The model was told what the rule and the hand prompts found, and shown the window's clip.
    [request] = transport.requests
    [prompt] = [
        part["text"]
        for message in request["input"]
        for part in message["content"]
        if part["type"] == "input_text"
    ]
    assert replay.qc().flags[0].reason in prompt
    assert "glove" in prompt
    assert "frame 40" in prompt
    [seen, *_] = replay.qc().reviews[0].hands
    assert f"largest mask {seen.area_cm2:.0f} cm²" in prompt
    [part] = videos_sent(request)
    width, height, n_frames = frame_size_and_count(decode_video(part, tmp_path / "sent.mp4"))
    assert (width, height) == (360, 360)
    assert n_frames < N_FRAMES


def test_a_malformed_verdict_fails_the_stage_and_is_not_cached(
    tmp_path: Path, runs_dir: Path
) -> None:
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")
    judge = MuseSparkJudge(settings, ReplayTransport([verdict("discard", "Not a verdict.")]))

    with pytest.raises(JudgeResponseError):
        intruded_run(tmp_path, runs_dir, judge)

    assert not any(path.is_file() for path in settings.cache_dir.rglob("*"))
