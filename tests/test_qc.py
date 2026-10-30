"""QC through the pipeline seam: each deterministic flag from a scripted scene, with its frame
range and reason, and the exclusions they lead to. Nothing is dropped silently."""

from pathlib import Path

from pharmavision.artifacts import Qc, QcSettings
from pharmavision.pipeline import confirm_arena, finish_run, start_run
from pharmavision.segmenter.synthetic import SyntheticSegmenter

from .scenes import FLOOR, N_FRAMES, PX_PER_CM, TEMPLATE, circle_walk, finished_run
from .videos import make_video


def flagged(qc: Qc) -> list[tuple[str, int, int, str]]:
    """Each flag as (kind, first frame, last frame, severity), checking it gives a reason."""
    assert all(flag.reason for flag in qc.flags)
    return [(f.kind, f.first_frame, f.last_frame, f.severity) for f in qc.flags]


def test_a_trial_that_loses_the_animal_too_often_is_excluded(
    tmp_path: Path, runs_dir: Path
) -> None:
    # 10 of 60 frames (17%) lost in one stretch, too long to interpolate; the limit is 5%.
    run = finished_run(tmp_path, runs_dir, lambda i: None if 20 <= i < 30 else (180.0, 180.0))

    qc = run.qc()
    assert flagged(qc) == [
        ("tracking_loss", 20, 29, "warning"),
        ("tracking_loss", 0, N_FRAMES - 1, "critical"),
    ]
    [exclusion] = qc.exclusions
    assert (exclusion.scope, exclusion.first_frame, exclusion.last_frame) == (
        "trial",
        0,
        N_FRAMES - 1,
    )
    assert exclusion.source == "rule"
    assert "17%" in exclusion.reason
    assert qc.trial_excluded


def test_a_brief_loss_is_flagged_but_keeps_the_trial(tmp_path: Path, runs_dir: Path) -> None:
    # Missing on the first 2 frames (3%): nothing to interpolate from, but under the 5% limit.
    run = finished_run(tmp_path, runs_dir, lambda i: None if i < 2 else (180.0, 180.0))

    qc = run.qc()
    assert flagged(qc) == [("tracking_loss", 0, 1, "warning")]
    assert qc.exclusions == []
    assert not qc.trial_excluded


def test_a_clean_walk_raises_no_flags(tmp_path: Path, runs_dir: Path) -> None:
    # Two laps of 13.5 cm radius in 4 s: about 42 cm/s, inside the floor all the way.
    run = finished_run(tmp_path, runs_dir, circle_walk(radius_px=90, laps=2))

    assert run.qc() == Qc(flags=[], exclusions=[])


def test_frames_outside_the_arena_are_flagged(tmp_path: Path, runs_dir: Path) -> None:
    # Frames 20-24 at 3 cm left of the floor's left edge (x = 30 px); the margin is 2 cm.
    def leave(i: int) -> tuple[float, float]:
        return (30 - 3 * PX_PER_CM, 180.0) if 20 <= i < 25 else (40.0, 180.0)

    run = finished_run(tmp_path, runs_dir, leave)

    qc = run.qc()
    assert flagged(qc) == [("out_of_arena", 20, 24, "warning")]
    assert "3.0 cm" in qc.flags[0].reason
    assert qc.exclusions == []


def test_a_centroid_jump_faster_than_a_mouse_runs_is_flagged(
    tmp_path: Path, runs_dir: Path
) -> None:
    # A still animal mis-detected 80 px (12 cm) away on frame 30: 180 cm/s there and back.
    run = finished_run(tmp_path, runs_dir, lambda i: (260.0, 180.0) if i == 30 else (180.0, 180.0))

    qc = run.qc()
    assert flagged(qc) == [("speed_jump", 29, 31, "warning")]
    assert "180 cm/s" in qc.flags[0].reason
    assert qc.exclusions == []


def test_a_second_large_mask_beside_the_animal_is_flagged(tmp_path: Path, runs_dir: Path) -> None:
    # A hand-sized disc (30 px radius, about 64 cm²) on frames 40-49, away from a still animal.
    run = finished_run(
        tmp_path,
        runs_dir,
        lambda _: (180.0, 180.0),
        intruder=lambda i: (280.0, 280.0) if 40 <= i < 50 else None,
    )

    qc = run.qc()
    assert flagged(qc) == [("second_mask", 40, 49, "warning")]
    assert qc.exclusions == []
    # The animal is still the one tracked: the intruder is flagged, not followed.
    assert {f.position for f in run.track().frames} == {(180.0, 180.0)}


def test_the_manifest_records_the_versioned_qc_thresholds_and_they_decide(
    tmp_path: Path, runs_dir: Path
) -> None:
    # The same 17% loss as above, under a 20% limit: flagged, not excluded.
    video = make_video(tmp_path / "trial.mp4", seconds=4)
    segmenter = SyntheticSegmenter(
        floor=FLOOR, trajectory=lambda i: None if 20 <= i < 30 else (180.0, 180.0)
    )
    lenient = QcSettings(max_tracking_loss_fraction=0.2)
    run = start_run(video, TEMPLATE, segmenter, runs_dir, qc_settings=lenient)
    confirm_arena(run)
    finish_run(run, segmenter)

    assert run.manifest().qc == lenient
    assert run.manifest().qc.version == 1
    assert flagged(run.qc()) == [("tracking_loss", 20, 29, "warning")]
    assert run.qc().exclusions == []
