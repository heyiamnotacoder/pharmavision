"""Run one video through the pipeline from the command line (laptop: needs MODEL_API_KEY in .env).

    uv run python -m pharmavision start VIDEO      # ingest + arena; stops at the arena gate
    uv run python -m pharmavision finish RUN_DIR   # confirm the arena, then the other stages
    uv run python -m pharmavision finish RUN_DIR --tier contributor   # public Sturman data only
    uv run python -m pharmavision serve [--precomputed DIR]   # the HTTP API on localhost:8000

Check `arena.json` (floor and centre polygons, px) between the two steps: `finish` is the
confirmation. Model responses are cached in runs/model_cache, so a re-run costs nothing."""

import argparse
from pathlib import Path

from .judge.muse_spark import MuseSparkJudge, MuseSparkSettings
from .pipeline import RunFolder, confirm_arena, finish_run, start_run
from .segmenter.config import make_segmenter
from .segmenter.meta_api import MetaApiSettings
from .templates import OPEN_FIELD, TEMPLATES


def main() -> None:
    parser = argparse.ArgumentParser(prog="pharmavision")
    steps = parser.add_subparsers(dest="step", required=True)
    start = steps.add_parser("start", help="ingest a video and detect the arena")
    start.add_argument("video", type=Path)
    start.add_argument("--runs-dir", type=Path, default=Path("runs"))
    finish = steps.add_parser("finish", help="confirm the arena, then every later stage")
    finish.add_argument("run_dir", type=Path)
    serve = steps.add_parser("serve", help="the HTTP API: runs, progress, study reports")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--precomputed", type=Path, default=None, help="finished run folders, served read-only"
    )
    for step in (finish, serve):
        step.add_argument(
            "--tier",
            choices=["standard", "contributor"],
            default="standard",
            help="Muse Spark tier: Meta may train on Contributor-tier clips, so public data only",
        )
    args = parser.parse_args()

    segmenter = make_segmenter(MetaApiSettings())
    if args.step == "serve":
        import uvicorn

        from .api import ApiSettings, create_app

        judge = MuseSparkJudge(MuseSparkSettings(tier=args.tier))
        settings = ApiSettings(precomputed_runs_dir=args.precomputed)
        # One process: the background jobs run in it, so a second worker would not see them.
        uvicorn.run(
            create_app(settings, segmenter, judge, TEMPLATES), host=args.host, port=args.port
        )
    elif args.step == "start":
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        run = start_run(args.video, OPEN_FIELD, segmenter, args.runs_dir)
        print(f"{run.path}\n{run.arena().model_dump_json(indent=2)}")
    else:
        run = RunFolder(args.run_dir)
        confirm_arena(run)
        judge = MuseSparkJudge(MuseSparkSettings(tier=args.tier))
        metrics = finish_run(run, segmenter, judge)
        frames = run.track().frames
        tracked = sum(f.detected for f in frames)
        print(f"tracked {tracked}/{len(frames)} frames")
        print(metrics.model_dump_json(indent=2))
        print(run.events().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
