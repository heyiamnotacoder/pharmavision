"""How many human-labelled rears on the train split fall inside a rear-candidate window (laptop
only). Answers ARCHITECTURE open decision 2.

Usage: uv run python -m spike.rear_recall [setting=value ...]

Each train video goes through the pipeline once: ingest, arena (auto-confirmed: "square floor"
was checked on Sturman frames in T2), and track with SAM 3.1 on the Meta Model API, about $1.80
per video and cached in runs/model_cache. Its run folder is reused after that. Candidates are
recomputed from the stored track with the default measure settings, overridden by any
`setting=value` arguments (e.g. rear_area_drop=0.15), so trying thresholds costs nothing.
Needs MODEL_API_KEY in .env and `python -m spike.data` first.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pharmavision.artifacts import MeasureSettings
from pharmavision.measure.open_field import open_field_metrics
from pharmavision.pipeline import RunFolder, Stage, confirm_arena, run_stage, start_run
from pharmavision.segmenter import Segmenter
from pharmavision.segmenter.config import make_segmenter
from pharmavision.segmenter.meta_api import MetaApiSettings
from pharmavision.templates import OPEN_FIELD
from spike.data import LABELS, SPLIT, VIDEOS
from spike.rears import Rear, Recall, consensus_rears, load_rears, rear_recall

OUT = Path("eval/data/spike/rear_recall")


def tracked_run(video_id: str, segmenter: Segmenter) -> RunFolder:
    """The video's run folder, tracked; made on the first call and reused after."""
    pointer = OUT / f"{video_id}.run"
    if pointer.exists():
        return RunFolder(Path(pointer.read_text().strip()))
    runs = OUT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run = start_run(VIDEOS / f"{video_id}.mp4", OPEN_FIELD, segmenter, runs)
    confirm_arena(run)
    run_stage(run, Stage.TRACK, segmenter)
    pointer.write_text(f"{run.path}\n")
    return run


def candidate_windows(run: RunFolder, settings: MeasureSettings) -> list[tuple[float, float]]:
    """Rear-candidate windows in seconds: frame i spans [i / fps, (i + 1) / fps)."""
    video = run.manifest().video
    metrics = open_field_metrics(run.track(), run.confirmed_arena(), video, settings)
    return [
        (c.first_frame / video.fps, (c.last_frame + 1) / video.fps) for c in metrics.rear_candidates
    ]


def scores(rears: list[Rear], windows: list[tuple[float, float]]) -> dict[str, Recall]:
    """Recall against each rater, against all raters pooled, and against the consensus rears
    (at least two raters agree)."""
    by_rater = {
        rater: rear_recall([r for r in rears if r.rater == rater], windows)
        for rater in sorted({r.rater for r in rears})
    }
    return by_rater | {
        "all_raters": rear_recall(rears, windows),
        "consensus": rear_recall(consensus_rears(rears), windows),
    }


def total(recalls: list[Recall]) -> dict[str, Any]:
    """The per-video counts summed over videos, with the recall of the sums."""
    summed = Recall(*(sum(getattr(r, f) for r in recalls) for f in Recall.__dataclass_fields__))
    return asdict(summed) | {"recall": summed.recall}


def main(overrides: list[str]) -> None:
    settings = MeasureSettings.model_validate(dict(arg.split("=", 1) for arg in overrides))
    segmenter = make_segmenter(MetaApiSettings())
    train: list[str] = json.loads(SPLIT.read_text())["train"]
    labels = load_rears(LABELS)

    per_video: dict[str, dict[str, Any]] = {}
    for video_id in train:
        run = tracked_run(video_id, segmenter)
        windows = candidate_windows(run, settings)
        video_scores = scores([r for r in labels if r.video_id == video_id], windows)
        per_video[video_id] = {
            "run": str(run.path),
            "candidate_s": sum(end - start for start, end in windows),
            "duration_s": run.manifest().video.duration_s,
            "scores": video_scores,
        }
        consensus = video_scores["consensus"]
        print(
            f"{video_id}: {consensus.found}/{consensus.rears} consensus rears found, "
            f"{len(windows)} windows, {consensus.windows_on_a_rear} on a consensus rear"
        )

    keys = sorted({k for v in per_video.values() for k in v["scores"]})
    summary = {
        "settings": settings.model_dump(),
        "videos": train,
        "totals": {
            k: total([v["scores"][k] for v in per_video.values() if k in v["scores"]]) for k in keys
        },
        "candidate_time_fraction": sum(v["candidate_s"] for v in per_video.values())
        / sum(v["duration_s"] for v in per_video.values()),
        "per_video": {
            video_id: v | {"scores": {k: asdict(s) for k, s in v["scores"].items()}}
            for video_id, v in per_video.items()
        },
    }
    name = "-".join(overrides) or "defaults"
    (OUT / f"summary-v{settings.version}-{name}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("totals", "candidate_time_fraction")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
