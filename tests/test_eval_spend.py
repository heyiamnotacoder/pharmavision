"""Spend: what each live model call cost, from the usage the API reports, and what a run of
the eval videos would cost before it starts."""

import pytest

from eval.spend import (
    MeteredTransport,
    UnknownPriceError,
    Usage,
    VideoPlan,
    cost_usd,
    format_projection,
    project,
)

from .sam_replay import ReplayTransport, read_fixture


def test_a_recorded_sam_track_is_billed_per_video_frame() -> None:
    # T2's OFT_16 recording: SAM 3.1 billed 9,142 frames, at $0.20 per 1,000.
    metered = MeteredTransport(ReplayTransport([read_fixture("OFT_16_track")]))

    events = list(metered.stream({"model": "sam-3.1", "input": []}))

    assert events[-1]["type"] == "response.completed"
    (call,) = metered.take()
    assert call.model == "sam-3.1"
    assert call.usage.video_frames == 9142
    assert call.usd == pytest.approx(1.8284)
    assert metered.take() == []


def test_a_short_video_request_is_billed_the_fifty_frame_minimum() -> None:
    assert cost_usd("sam-3.1", Usage(video_frames=10)) == pytest.approx(50 * 0.20 / 1000)


def test_an_image_request_is_billed_per_image() -> None:
    assert cost_usd("sam-3.1", Usage(images=2)) == pytest.approx(2 * 2.50 / 1000)


def test_muse_spark_tokens_are_billed_at_the_tier_price_with_cached_input_cheaper() -> None:
    # Contributor tier: $0.10 per M input, $0.002 per M cached input, $0.20 per M output.
    usage = Usage(input_tokens=3_000_000, cached_tokens=1_000_000, output_tokens=500_000)

    assert cost_usd("muse-spark-1.3-contributor", usage) == pytest.approx(0.2 + 0.002 + 0.1)


def test_a_model_with_no_known_price_is_refused_not_billed_as_free() -> None:
    with pytest.raises(UnknownPriceError, match=r"muse-wonder-1\.0"):
        cost_usd("muse-wonder-1.0", Usage(input_tokens=1))


def test_usage_is_read_from_the_responses_usage_block() -> None:
    usage = Usage.from_response(
        {
            "input_tokens": 2300,
            "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": None},
            "output_tokens": 1500,
            "output_tokens_details": {"reasoning_tokens": 1200},
            "total_tokens": 3800,
            "video_frames_processed": None,
        }
    )

    assert usage == Usage(input_tokens=2300, cached_tokens=100, output_tokens=1500)


def test_the_projection_prices_each_video_still_to_run_and_frees_those_done() -> None:
    # 609 s at 15 fps is 9,135 frames in one request ($1.827), plus one floor image ($0.0025).
    plans = [
        VideoPlan(video_id="OFT_11", duration_s=609.0),
        VideoPlan(video_id="OFT_12", duration_s=609.0, spent_usd=1.83),
    ]

    projection = project(plans, fps=15)

    assert [row.video_id for row in projection.rows] == ["OFT_11", "OFT_12"]
    assert projection.rows[0].projected_usd == pytest.approx(1.8295)
    assert projection.rows[1].projected_usd == 0.0
    assert projection.to_run == ["OFT_11"]
    assert projection.total_usd == pytest.approx(1.8295)
    assert projection.spent_usd == pytest.approx(1.83)


def test_a_video_over_the_frame_limit_is_projected_as_overlapping_chunks() -> None:
    # 1,100 s at 15 fps is 16,500 frames: chunks of 15,000 and 1,650 (150 overlap).
    projection = project([VideoPlan(video_id="OFT_11", duration_s=1100.0)], fps=15)

    assert projection.total_usd == pytest.approx(16_650 * 0.20 / 1000 + 2.50 / 1000)


def test_the_projection_is_printed_per_video_with_its_total_before_a_run() -> None:
    plans = [
        VideoPlan(video_id="OFT_11", duration_s=609.0),
        VideoPlan(video_id="OFT_12", duration_s=609.0, spent_usd=1.83),
    ]

    text = format_projection(project(plans, fps=15))

    assert text.splitlines() == [
        "video     status   projected $   spent $",
        "OFT_11    to run        1.8295         -",
        "OFT_12    done          0.0000    1.8300",
        "1 of 2 videos to run: projected $1.83 at most (cached responses cost nothing); "
        "$1.83 spent so far",
    ]
