"""The eval harness, tracking part (T12a). Laptop only: `track` calls SAM 3.1 with the
MODEL_API_KEY in .env.

Usage:
  uv run python -m eval fetch          # labels, DLC output and the 14 eval videos (~1.8 GB)
  uv run python -m eval plan           # the projected spend of `track`, per video
  uv run python -m eval track [--yes]  # print the projection, ask, then track and compare

Every command first checks that no label file can be committed (`eval.guard`) and, once the
labels are fetched, that `eval/split.json` is the seeded split of the labelled videos. Results:
`eval/data/tracking/<ID>.json` per video and `summary.json`; spend: `eval/data/spend.jsonl`.
Model responses are cached in `runs/model_cache`, shared with the app and the spike, so a video
already tracked there costs nothing.
"""

import argparse
import sys
from pathlib import Path

import cv2

from pharmavision.artifacts import IngestSettings
from pharmavision.model_api import OpenAiTransport
from pharmavision.segmenter.meta_api import MetaApiSegmenter, MetaApiSettings

from .guard import check_labels_untracked
from .harness import (
    ArenaNotFoundError,
    Paths,
    TrackingSummary,
    evaluate_video,
    spent_by_video,
    summarise,
)
from .spend import MeteredTransport, VideoPlan, format_projection, project
from .split import Split, check_split, load_split
from .sturman import LABELS, dlc_files, fetch_dlc, fetch_labels, fetch_video, video_path

REPO = Path(__file__).resolve().parents[1]
NOMINAL_DURATION_S = 609.5
"""A Sturman open-field video not downloaded yet is projected at this length (they run about
609 s: OFT_16 is 609.47 s)."""


def checked_split() -> tuple[Split, dict[str, str]]:
    """The committed split, checked against the label file, and each video's DLC file."""
    if not LABELS.exists():
        raise SystemExit(f"{LABELS} is missing: run `python -m eval fetch` first")
    dlc = dlc_files(LABELS)
    split = load_split()
    check_split(split, dlc)
    return split, dlc


def duration_s(video: Path) -> float:
    if not video.exists():
        return NOMINAL_DURATION_S
    capture = cv2.VideoCapture(str(video))
    try:
        return capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()


def plan(split: Split, paths: Paths) -> str:
    spent = spent_by_video(paths)
    plans = [
        VideoPlan(
            video_id=video_id,
            duration_s=duration_s(video_path(video_id)),
            spent_usd=spent.get(video_id, 0.0) if paths.result(video_id).exists() else None,
        )
        for video_id in split.eval_ids
    ]
    return format_projection(project(plans, fps=IngestSettings().fps))


def fetch() -> None:
    fetch_labels()
    split, dlc = checked_split()
    for video_id in split.eval_ids:
        fetch_dlc(dlc[video_id])
        fetch_video(video_id)


def track(yes: bool, paths: Paths) -> TrackingSummary:
    split, dlc = checked_split()
    print(plan(split, paths))
    if not yes and input("Track the eval videos? [y/N] ").strip().lower() != "y":
        raise SystemExit("nothing run")
    settings = MetaApiSettings()
    meter = MeteredTransport(OpenAiTransport(settings.base_url))
    segmenter = MetaApiSegmenter(settings, transport=meter)
    for video_id in split.eval_ids:
        video, dlc_csv = fetch_video(video_id), fetch_dlc(dlc[video_id])
        try:
            result = evaluate_video(video_id, video, dlc_csv, segmenter, meter, paths)
        except ArenaNotFoundError as error:
            print(f"{video_id}: skipped, {error}", flush=True)
            continue
        tracking = result.tracking
        median = "-" if tracking.distance_cm is None else f"{tracking.distance_cm.median:.2f} cm"
        print(
            f"{video_id}: {tracking.tracked_pct:.1f}% tracked, median {median} from DLC, "
            f"${result.spent_usd:.4f} ({result.floor_prompt!r})",
            flush=True,
        )
        unreported = sum(s.unreported_calls for s in result.spend.values())
        if unreported:
            print(f"{video_id}: {unreported} calls reported no usage; check the Meta console")
    summary = summarise(split.eval_ids, paths)
    (paths.results / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    return summary


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eval", description="The eval harness, tracking part (T12a)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="fetch the labels, DLC output and eval videos")
    commands.add_parser("plan", help="print the projected spend of `track`")
    track_parser = commands.add_parser("track", help="track the eval videos and compare with DLC")
    track_parser.add_argument("--yes", action="store_true", help="run without asking")
    args = parser.parse_args(argv)

    check_labels_untracked(REPO)
    paths = Paths()
    match args.command:
        case "fetch":
            fetch()
        case "plan":
            print(plan(checked_split()[0], paths))
        case "track":
            summary = track(args.yes, paths)
            print(f"{len(summary.videos)} videos compared, ${summary.spent_usd:.2f} spent")
            if summary.missing:
                raise SystemExit(f"no tracking result for {summary.missing}")
        case _:
            parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
