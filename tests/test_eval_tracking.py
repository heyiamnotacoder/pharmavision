"""Our track against DeepLabCut's body centre: % of frames tracked and the centroid distance in
cm. DLC is a reference, not ground truth, so the comparison reports, it does not grade."""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from eval.dlc import DlcTrack
from eval.tracking import (
    TrackingComparison,
    TrackingSettings,
    VideoMapping,
    compare_tracking,
    source_frame,
)
from pharmavision.artifacts import IngestSettings, Track, TrackFrame
from pharmavision.stages.ingest import normalise

PX_PER_CM = 1.5


def track(positions: Sequence[tuple[float, float] | None]) -> Track:
    return Track(
        frames=[
            TrackFrame.gap(i)
            if p is None
            else TrackFrame(
                frame_index=i, detected=True, x_px=p[0], y_px=p[1], area_px=50, mask_rle="r"
            )
            for i, p in enumerate(positions)
        ]
    )


def dlc(
    points: Sequence[tuple[float, float]], likelihood: Sequence[float] | None = None
) -> DlcTrack:
    return DlcTrack(
        x=np.array([p[0] for p in points], dtype=np.float64),
        y=np.array([p[1] for p in points], dtype=np.float64),
        likelihood=np.array(likelihood or [1.0] * len(points), dtype=np.float64),
    )


def compare(
    ours: Track,
    reference: DlcTrack,
    fps: float = 15.0,
    dlc_fps: float = 15.0,
    scale: tuple[float, float] = (1.0, 1.0),
    settings: TrackingSettings | None = None,
) -> TrackingComparison:
    mapping = VideoMapping(
        fps=fps,
        source_fps=dlc_fps,
        source_frames=len(reference.x),
        scale_x=scale[0],
        scale_y=scale[1],
        px_per_cm=PX_PER_CM,
    )
    return compare_tracking("OFT_11", ours, reference, mapping, settings or TrackingSettings())


def test_a_constant_offset_of_three_pixels_is_two_cm_on_every_frame() -> None:
    path = [(100.0, 100.0), (110.0, 100.0), (120.0, 105.0), (130.0, 110.0)]

    result = compare(track(path), dlc([(x + 3.0, y) for x, y in path]))

    assert result.frames == result.frames_tracked == result.frames_compared == 4
    assert result.tracked_pct == 100.0
    assert result.distance_cm is not None
    assert result.distance_cm.mean == pytest.approx(2.0)
    assert result.distance_cm.median == pytest.approx(2.0)
    assert result.distance_cm.p95 == pytest.approx(2.0)
    assert result.distance_cm.max == pytest.approx(2.0)
    assert result.over_disagreement_pct == 0.0
    assert result.best_lag is not None
    assert result.best_lag.lag_frames == 0


def test_frames_are_matched_by_time_when_dlc_ran_at_the_source_rate() -> None:
    # 25 fps DLC against our 15 fps: frames 0-3 kept source frames 0, 2, 4 and 5 (see the
    # ffmpeg test below). DLC's x is its own frame number, so only the right match gives 0 cm.
    reference = dlc([(float(k), 0.0) for k in range(8)])
    ours = track([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (5.0, 0.0)])

    result = compare(ours, reference, fps=15.0, dlc_fps=25.0)

    assert result.distance_cm is not None
    assert result.distance_cm.max == 0.0


def test_a_dlc_track_starting_late_is_found_by_the_alignment_check() -> None:
    # DLC's rows start 3 source frames late: its frame k shows what the source had at k - 3.
    kept = source_frame(np.arange(30), 15.0, 25.0)
    ours = track([(float(k), 0.0) for k in kept])
    reference = dlc([(float(k - 3), 0.0) for k in range(60)])

    result = compare(ours, reference, fps=15.0, dlc_fps=25.0)

    assert result.distance_cm is not None
    assert result.distance_cm.median == pytest.approx(3 / PX_PER_CM)
    assert result.best_lag is not None
    assert result.best_lag.lag_frames == 3
    assert result.best_lag.median_cm == 0.0
    assert (result.dlc_frames, result.source_frames) == (60, 60)


def test_dlc_pixels_are_scaled_to_the_normalised_video() -> None:
    # The source was twice the normalised size along x and 4/3 along y.
    path = [(100.0, 90.0), (120.0, 60.0)]

    result = compare(track(path), dlc([(2 * x, 4 / 3 * y) for x, y in path]), scale=(0.5, 0.75))

    assert result.distance_cm is not None
    assert result.distance_cm.max == pytest.approx(0.0)


def test_gaps_count_against_tracking_and_unsure_dlc_frames_are_left_out() -> None:
    path = [(100.0, 100.0), None, (120.0, 100.0), (130.0, 100.0)]
    # DLC is unsure of frame 2, where it puts the animal far off.
    reference = dlc(
        [(103.0, 100.0), (110.0, 100.0), (500.0, 500.0), (133.0, 100.0)],
        likelihood=[1.0, 1.0, 0.3, 1.0],
    )

    result = compare(track(path), reference)

    assert result.frames_tracked == 3
    assert result.tracked_pct == 75.0
    assert result.dlc_confident_pct == 75.0
    assert result.frames_compared == 2
    assert result.distance_cm is not None
    assert result.distance_cm.max == pytest.approx(2.0)


def test_frames_past_the_end_of_the_dlc_output_are_not_compared() -> None:
    result = compare(track([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]), dlc([(0.0, 0.0)]))

    assert result.frames_compared == 1
    assert result.dlc_confident_pct == pytest.approx(100 / 3)


def test_with_no_frame_to_compare_the_distance_abstains() -> None:
    result = compare(track([None, None]), dlc([(0.0, 0.0), (1.0, 1.0)]))

    assert result.tracked_pct == 0.0
    assert result.frames_compared == 0
    assert result.distance_cm is None
    assert result.over_disagreement_pct is None
    assert result.largest == []
    assert result.best_lag is None


def test_the_largest_disagreements_are_listed_for_checking_by_eye() -> None:
    # Offsets of 0, 15, 3 and 7.5 px: 0, 10, 2 and 5 cm.
    path = [(100.0, 100.0)] * 4
    reference = dlc([(100.0, 100.0), (115.0, 100.0), (103.0, 100.0), (107.5, 100.0)])
    settings = TrackingSettings(disagreement_cm=4.0, largest_n=2)

    result = compare(track(path), reference, settings=settings)

    assert [(d.frame_index, d.distance_cm) for d in result.largest] == [
        (1, pytest.approx(10.0)),
        (3, pytest.approx(5.0)),
    ]
    assert result.largest[0].time_s == pytest.approx(1 / 15)
    assert result.over_disagreement_pct == 50.0


def frame_numbers(video: Path) -> NDArray[np.int64]:
    """Each frame's number, read back from the luma that `numbered_video` gave it."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "extractplanes=y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    luma = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 64 * 64)
    return np.rint((np.median(luma, axis=1) - 16) / 1.5).astype(np.int64)


def test_each_frame_meets_the_source_frame_ingest_kept_for_it(tmp_path: Path) -> None:
    # A 25 fps source whose frame k is a flat grey of luma 16 + 1.5 k, through the real ingest.
    source = tmp_path / "numbered.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "nullsrc=s=64x64:r=25:d=4.8,geq=lum='16+1.5*N':cb=128:cr=128",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    normalised = normalise(source, tmp_path / "video.mp4", IngestSettings())
    kept = frame_numbers(tmp_path / "video.mp4")

    assert frame_numbers(source).tolist() == list(range(120))
    assert len(kept) == normalised.n_frames == 72
    assert source_frame(np.arange(72), normalised.fps, 25.0).tolist() == kept.tolist()
