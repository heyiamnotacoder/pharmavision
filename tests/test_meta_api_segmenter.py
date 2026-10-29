"""The SAM 3.1 backend at the Segmenter interface, for what the track artifact can't show:
object identity across the overlapping chunks of a video over the frame limit."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from pharmavision.segmenter import Mask
from pharmavision.segmenter.meta_api import MetaApiSegmenter, MetaApiSettings, SamResponseError

from .sam_replay import (
    Event,
    Instance,
    ReplayTransport,
    disc,
    frame_line,
    output_text,
    responses_events,
)
from .videos import make_video

SHAPE = (360, 360)
DECOY_AT = (300.0, 60.0)


def walk(frame: int) -> tuple[float, float]:
    return (100.0 + 2 * frame, 90.0 + 2 * frame)


def animal(frame: int, object_id: str = "0") -> Instance:
    return Instance(object_id, disc(SHAPE, walk(frame), 8), 0.71)


def chunk_stream(start: int, end: int, animal_id: str, decoy_id: str) -> list[Event]:
    """SAM 3.1's answer for frames start..end-1 cut out as their own video: markers count from
    0, and object ids are whatever this response chose."""
    decoy = Instance(decoy_id, disc(SHAPE, DECOY_AT, 14), 0.35)
    return responses_events(
        [
            frame_line(frame - start, [animal(frame, animal_id), decoy])
            for frame in range(start, end)
        ]
    )


def output_lines(events: list[Event]) -> list[str]:
    return output_text(events).splitlines(keepends=True)


def centroid(mask: Mask) -> tuple[float, float]:
    ys, xs = np.nonzero(mask.pixels)
    return (float(xs.mean()), float(ys.mean()))


def nearest(masks: Sequence[Mask], point: tuple[float, float]) -> Mask:
    return min(masks, key=lambda m: np.hypot(*np.subtract(centroid(m), point)))


def test_chunks_over_the_frame_limit_are_relinked_by_mask_iou_in_the_overlap(
    tmp_path: Path,
) -> None:
    video = make_video(tmp_path / "long.mp4", seconds=4)  # 100 frames at 25 fps
    # 40 frames per request with a 10-frame overlap: frames 0-39, 30-69 and 60-99. Each response
    # numbers its objects its own way; the animal is "0" first, then "3", then "1".
    transport = ReplayTransport(
        [
            chunk_stream(0, 40, animal_id="0", decoy_id="1"),
            chunk_stream(30, 70, animal_id="3", decoy_id="0"),
            chunk_stream(60, 100, animal_id="1", decoy_id="2"),
        ]
    )
    settings = MetaApiSettings(
        cache_dir=tmp_path / "cache", max_frames_per_request=40, chunk_overlap_frames=10
    )
    sam = MetaApiSegmenter(settings, transport)

    frames = list(sam.track_video(video, "mouse", max_objects=2))

    assert transport.uploaded_frame_counts == [40, 40, 40]
    assert [f.frame_index for f in frames] == list(range(100))
    animal_ids, decoy_ids = set[str | None](), set[str | None]()
    for frame in frames:
        assert len(frame.masks) == 2
        animal = nearest(frame.masks, walk(frame.frame_index))
        decoy = nearest(frame.masks, DECOY_AT)
        assert centroid(animal) == pytest.approx(walk(frame.frame_index), abs=0.01)
        animal_ids.add(animal.object_id)
        decoy_ids.add(decoy.object_id)
    assert len(animal_ids) == 1 and len(decoy_ids) == 1
    assert animal_ids != decoy_ids


def test_a_single_animal_keeps_only_the_most_confident_instance(tmp_path: Path) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=2)  # 50 frames at 25 fps
    transport = ReplayTransport([chunk_stream(0, 50, animal_id="0", decoy_id="1")])
    sam = MetaApiSegmenter(MetaApiSettings(cache_dir=tmp_path / "cache"), transport)

    frames = list(sam.track_video(video, "mouse", max_objects=1))

    assert [len(f.masks) for f in frames] == [1] * 50
    for frame in frames:
        assert frame.masks[0].confidence == pytest.approx(0.71)
        assert centroid(frame.masks[0]) == pytest.approx(walk(frame.frame_index), abs=0.01)


def truncated(events: list[Event]) -> list[Event]:
    return [e for e in events if e["type"] != "response.completed"]


def past_the_end(events: list[Event]) -> list[Event]:
    """The stream plus one frame beyond the video's last."""
    return responses_events([*output_lines(events), frame_line(50, [animal(50)])])


@pytest.mark.parametrize("spoil", [truncated, past_the_end])
def test_a_bad_response_raises_and_is_not_cached(
    spoil: Callable[[list[Event]], list[Event]], tmp_path: Path
) -> None:
    video = make_video(tmp_path / "trial.mp4", seconds=2)  # 50 frames at 25 fps
    good = chunk_stream(0, 50, animal_id="0", decoy_id="1")
    transport = ReplayTransport([spoil(good), good])
    sam = MetaApiSegmenter(MetaApiSettings(cache_dir=tmp_path / "cache"), transport)

    with pytest.raises(SamResponseError):
        list(sam.track_video(video, "mouse", max_objects=1))
    frames = list(sam.track_video(video, "mouse", max_objects=1))

    assert len(transport.requests) == 2
    assert [f.frame_index for f in frames] == list(range(50))
