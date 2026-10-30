"""The behaviours stage through the pipeline seam: rear candidates in, judged events out."""

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from pharmavision.artifacts import BehaviourEvent, BehaviourSettings
from pharmavision.judge import ClipJudgement
from pharmavision.judge.muse_spark import (
    FewShotExample,
    FewShotLeakError,
    JudgeResponseError,
    MuseSparkJudge,
    MuseSparkSettings,
    RecordedJudge,
    RecordingMissingError,
)
from pharmavision.pipeline import RunFolder

from .judges import ScriptedJudge
from .muse_replay import answer, decode_video, frame_size_and_count, videos_sent
from .sam_replay import ReplayTransport
from .scenes import N_FRAMES, REAR, WALKING_BODY, finished_run, rears_in, walk_stop_walk


def test_each_rear_candidate_becomes_one_judged_event(tmp_path: Path, runs_dir: Path) -> None:
    judgement = ClipJudgement(label="rearing", confidence=0.9, rationale="Stands up, then down.")
    judge = ScriptedJudge(judgement)

    run = finished_run(tmp_path, runs_dir, walk_stop_walk, body=rears_in(REAR), judge=judge)

    assert len(judge.clips) == 1
    assert run.events().events == [
        BehaviourEvent(
            first_frame=25,
            last_frame=39,
            label="rearing",
            confidence=0.9,
            low_confidence=False,
            rationale="Stands up, then down.",
            source="model",
        )
    ]


def test_a_run_without_rear_candidates_shows_the_model_nothing(
    tmp_path: Path, runs_dir: Path
) -> None:
    judge = ScriptedJudge()

    run = finished_run(tmp_path, runs_dir, walk_stop_walk, judge=judge)

    assert judge.clips == []
    assert run.events().events == []


def test_the_recorded_judge_replays_stored_responses_as_model_events(
    tmp_path: Path, runs_dir: Path
) -> None:
    # A first run records Muse Spark's answer in the response cache, as the laptop would; the
    # recorded Judge then replays it by the same key, and never calls the API.
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")
    live = MuseSparkJudge(settings, ReplayTransport([answer("rearing", 0.8, "Up on hind legs.")]))
    finished_run(tmp_path, runs_dir, walk_stop_walk, body=rears_in(REAR), judge=live)

    replay = finished_run(
        tmp_path, runs_dir, walk_stop_walk, body=rears_in(REAR), judge=RecordedJudge(settings)
    )

    assert replay.events().events == [
        BehaviourEvent(
            first_frame=25,
            last_frame=39,
            label="rearing",
            confidence=0.8,
            low_confidence=False,
            rationale="Up on hind legs.",
            source="model",
        )
    ]
    assert replay.manifest().models["behaviours"] == "muse-spark-1.3"


def test_the_recorded_judge_refuses_a_clip_it_has_no_response_for(
    tmp_path: Path, runs_dir: Path
) -> None:
    judge = RecordedJudge(MuseSparkSettings(cache_dir=tmp_path / "cache"))

    with pytest.raises(RecordingMissingError):
        finished_run(tmp_path, runs_dir, walk_stop_walk, body=rears_in(REAR), judge=judge)


def rear_run(
    tmp_path: Path,
    runs_dir: Path,
    judge: MuseSparkJudge,
    behaviour_settings: BehaviourSettings | None = None,
) -> RunFolder:
    """A run with one rear candidate (frames 25-39), judged by `judge`."""
    return finished_run(
        tmp_path,
        runs_dir,
        walk_stop_walk,
        body=rears_in(REAR),
        judge=judge,
        behaviour_settings=behaviour_settings,
    )


def test_a_re_run_of_the_same_clip_makes_no_api_call(tmp_path: Path, runs_dir: Path) -> None:
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")
    rear_run(tmp_path, runs_dir, MuseSparkJudge(settings, ReplayTransport([answer("rearing")])))
    again = ReplayTransport([])  # any API call fails the test

    run = rear_run(tmp_path, runs_dir, MuseSparkJudge(settings, again))

    assert again.requests == []
    assert [e.label for e in run.events().events] == ["rearing"]


@pytest.mark.parametrize(
    ("settings", "behaviour_settings"),
    [
        pytest.param({"tier": "contributor"}, None, id="another model id"),
        pytest.param({}, BehaviourSettings(clip_side_px=128), id="another clip"),
    ],
)
def test_responses_are_cached_by_model_id_and_clip(
    settings: dict[str, Any],
    behaviour_settings: BehaviourSettings | None,
    tmp_path: Path,
    runs_dir: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    first = MuseSparkSettings(cache_dir=cache_dir)
    rear_run(tmp_path, runs_dir, MuseSparkJudge(first, ReplayTransport([answer("rearing")])))
    changed = ReplayTransport([answer("none")])

    judge = MuseSparkJudge(MuseSparkSettings(cache_dir=cache_dir, **settings), changed)
    run = rear_run(tmp_path, runs_dir, judge, behaviour_settings)

    assert len(changed.requests) == 1
    assert [e.label for e in run.events().events] == ["none"]


@pytest.mark.parametrize(
    ("tier", "model"),
    [("standard", "muse-spark-1.3"), ("contributor", "muse-spark-1.3-contributor")],
)
def test_the_model_tier_is_a_config_value(
    tier: Literal["standard", "contributor"], model: str, tmp_path: Path, runs_dir: Path
) -> None:
    transport = ReplayTransport([answer("rearing")])
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache", tier=tier)

    run = rear_run(tmp_path, runs_dir, MuseSparkJudge(settings, transport))

    assert [request["model"] for request in transport.requests] == [model]
    assert run.manifest().models["behaviours"] == model


@pytest.mark.parametrize(("confidence", "low"), [(0.59, True), (0.6, False), (0.95, False)])
def test_a_judgement_under_the_minimum_confidence_is_a_low_confidence_event(
    confidence: float, low: bool, tmp_path: Path, runs_dir: Path
) -> None:
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")
    judge = MuseSparkJudge(settings, ReplayTransport([answer("rearing", confidence)]))

    run = rear_run(tmp_path, runs_dir, judge)

    assert run.manifest().behaviours.min_confidence == 0.6
    [event] = run.events().events
    assert (event.confidence, event.low_confidence) == (confidence, low)


def test_only_short_clips_cropped_to_the_animal_reach_the_model(
    tmp_path: Path, runs_dir: Path
) -> None:
    # Two rear candidates in a 4 s (60-frame, 360 x 360 px) video.
    transport = ReplayTransport([answer("rearing"), answer("none")])
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")

    run = finished_run(
        tmp_path,
        runs_dir,
        lambda _: (180.0, 180.0),
        body=lambda i: (8.0, 4.0) if i in range(8, 16) or i in range(44, 52) else WALKING_BODY,
        judge=MuseSparkJudge(settings, transport),
    )

    assert len(run.metrics().rear_candidates) == 2
    assert len(transport.requests) == 2
    whole_video = run.video_path.read_bytes()
    for n, request in enumerate(transport.requests):
        [part] = videos_sent(request)
        assert "file_id" not in part
        assert part["fps"] == 15
        clip = decode_video(part, tmp_path / f"clip-{n}.mp4")
        assert clip.read_bytes() != whole_video
        width, height, n_frames = frame_size_and_count(clip)
        assert (width, height) == (256, 256)
        assert 2 * 15 <= n_frames <= 4 * 15
        assert n_frames < N_FRAMES


def test_a_malformed_judgement_fails_the_stage_and_is_not_cached(
    tmp_path: Path, runs_dir: Path
) -> None:
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache")
    judge = MuseSparkJudge(settings, ReplayTransport([answer("grooming")]))

    with pytest.raises(JudgeResponseError):
        rear_run(tmp_path, runs_dir, judge)

    assert not any(path.is_file() for path in settings.cache_dir.rglob("*"))


def few_shot_example(tmp_path: Path, video_id: str) -> FewShotExample:
    clip = tmp_path / f"{video_id}.mp4"
    clip.write_bytes(b"an example clip")  # sent as is; the request is replayed, never decoded
    return FewShotExample(video_id=video_id, clip=clip, label="rearing", rationale="Rater rear.")


def test_few_shot_examples_from_outside_the_train_split_are_refused(tmp_path: Path) -> None:
    examples = (few_shot_example(tmp_path, "OFT_16"), few_shot_example(tmp_path, "OFT_11"))
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache", few_shot=examples)

    with pytest.raises(FewShotLeakError, match="OFT_11"):
        MuseSparkJudge(settings, ReplayTransport([]))


def test_few_shot_examples_go_ahead_of_the_clip_with_their_labels(
    tmp_path: Path, runs_dir: Path
) -> None:
    example = few_shot_example(tmp_path, "OFT_16")  # train split
    settings = MuseSparkSettings(cache_dir=tmp_path / "cache", few_shot=(example,))
    transport = ReplayTransport([answer("rearing")])

    rear_run(tmp_path, runs_dir, MuseSparkJudge(settings, transport))

    [request] = transport.requests
    roles = [message["role"] for message in request["input"]]
    assert roles == ["user", "assistant", "user"]
    shown, judged = (
        decode_video(p, tmp_path / f"{n}.mp4") for n, p in enumerate(videos_sent(request))
    )
    assert shown.read_bytes() == b"an example clip"
    assert judged.read_bytes() != b"an example clip"
    [reply] = request["input"][1]["content"]
    assert json.loads(reply["text"]) == {
        "label": "rearing",
        "confidence": 1.0,
        "rationale": "Rater rear.",
    }
