"""Run one video through the pipeline from the command line (laptop: needs MODEL_API_KEY in .env).

    uv run python -m pharmavision start VIDEO      # ingest + arena; stops at the arena gate
    uv run python -m pharmavision finish RUN_DIR   # confirm the arena, then track + measure

Check `arena.json` (floor and centre polygons, px) between the two steps: `finish` is the
confirmation. Model responses are cached in runs/model_cache, so a re-run costs nothing."""

import argparse
from pathlib import Path

from .pipeline import RunFolder, confirm_arena, finish_run, start_run
from .segmenter.config import make_segmenter
from .segmenter.meta_api import MetaApiSettings
from .templates import OPEN_FIELD


def main() -> None:
    parser = argparse.ArgumentParser(prog="pharmavision")
    steps = parser.add_subparsers(dest="step", required=True)
    start = steps.add_parser("start", help="ingest a video and detect the arena")
    start.add_argument("video", type=Path)
    start.add_argument("--runs-dir", type=Path, default=Path("runs"))
    finish = steps.add_parser("finish", help="confirm the arena, then track and measure")
    finish.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    segmenter = make_segmenter(MetaApiSettings())
    if args.step == "start":
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        run = start_run(args.video, OPEN_FIELD, segmenter, args.runs_dir)
        print(f"{run.path}\n{run.arena().model_dump_json(indent=2)}")
    else:
        run = RunFolder(args.run_dir)
        confirm_arena(run)
        metrics = finish_run(run, segmenter)
        frames = run.track().frames
        tracked = sum(f.detected for f in frames)
        print(f"tracked {tracked}/{len(frames)} frames")
        print(metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
