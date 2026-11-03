"""One eval video through the pipeline (ingest, arena auto-confirmed, track) and compared with
DeepLabCut, with what its model calls cost logged whether or not it finished. Its run folder is
kept and reused, so comparing again costs nothing."""

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from pharmavision.artifacts import RunFolder
from pharmavision.judge import Clip, ClipJudgement, FlaggedWindow, ReportBrief, WindowVerdict
from pharmavision.pipeline import ArenaNotFoundError, Stage, confirm_arena, run_stage, start_run
from pharmavision.segmenter import Segmenter
from pharmavision.templates import OPEN_FIELD, TestTemplate

from .dlc import read_dlc
from .spend import Call, ModelSpend, by_model
from .sturman import DATA
from .tracking import TrackingComparison, TrackingSettings, compare_tracking

FLOOR_PROMPTS = ("square floor", "white floor", "open field arena", "floor", "box")
"""Tried in order until SAM 3.1 finds a floor: on the Sturman videos OFT_12 and OFT_52 took
"white floor" and OFT_39 "floor" (spike, Oct 31). The template's own prompt goes first."""


@dataclass(frozen=True)
class Paths:
    """Where the harness keeps what it makes, all under the git-ignored eval data folder."""

    root: Path = DATA

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def results(self) -> Path:
        return self.root / "tracking"

    @property
    def spend_log(self) -> Path:
        """One JSON line per video attempt: the live calls it made, by model, and their cost."""
        return self.root / "spend.jsonl"


class Meter(Protocol):
    def take(self) -> list[Call]:
        """The billed calls since the last take."""
        ...


class VideoResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    run_id: str
    floor_prompt: str
    tracking: TrackingComparison
    spend: dict[str, ModelSpend]
    """This attempt's live calls; a reused run or a cached response costs nothing."""
    spent_usd: float


class SpendEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    at: datetime
    finished: bool
    by_model: dict[str, ModelSpend]
    usd: float


class TrackingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    videos: list[VideoResult]
    missing: list[str]
    """Eval videos with no tracking result: their row in the report abstains."""
    spent_usd: float
    """Every attempt logged for these videos, finished or not."""


def evaluate_video(
    video_id: str,
    video: Path,
    dlc_csv: Path,
    segmenter: Segmenter,
    meter: Meter,
    paths: Paths,
    template: TestTemplate = OPEN_FIELD,
    settings: TrackingSettings | None = None,
) -> VideoResult:
    """Track the video (or reuse its tracked run), compare it with its DLC output, and write
    the result to `paths.results`. The spend of every attempt goes to `paths.spend_log`, also
    when the arena is never found."""
    meter.take()  # calls before this video are not its spend
    finished = False
    try:
        run = _tracked_run(video_id, video, segmenter, paths.runs, template)
        manifest = run.manifest()
        source, normalised = manifest.source, manifest.video
        tracking = compare_tracking(
            video_id,
            run.track(),
            read_dlc(dlc_csv),
            fps=normalised.fps,
            dlc_fps=source.fps,
            scale=(normalised.width / source.width, normalised.height / source.height),
            px_per_cm=run.confirmed_arena().px_per_cm,
            settings=settings or TrackingSettings(),
        )
        finished = True
    finally:
        spend = by_model(meter.take())
        _log_spend(paths.spend_log, video_id, finished, spend)
    result = VideoResult(
        video_id=video_id,
        run_id=run.run_id,
        floor_prompt=manifest.template.floor_prompt,
        tracking=tracking,
        spend=spend,
        spent_usd=sum(s.usd for s in spend.values()),
    )
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / f"{video_id}.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def summarise(video_ids: Sequence[str], paths: Paths) -> TrackingSummary:
    """The stored result of each video, the ones with none, and what they all cost."""
    videos: list[VideoResult] = []
    missing: list[str] = []
    for video_id in video_ids:
        path = paths.results / f"{video_id}.json"
        if path.exists():
            videos.append(VideoResult.model_validate_json(path.read_text()))
        else:
            missing.append(video_id)
    spent = spent_by_video(paths)
    return TrackingSummary(
        videos=videos, missing=missing, spent_usd=sum(spent.get(v, 0.0) for v in video_ids)
    )


def spent_by_video(paths: Paths) -> dict[str, float]:
    """Every logged attempt's cost, summed per video."""
    if not paths.spend_log.exists():
        return {}
    spent: dict[str, float] = {}
    for line in paths.spend_log.read_text().splitlines():
        entry = SpendEntry.model_validate_json(line)
        spent[entry.video_id] = spent.get(entry.video_id, 0.0) + entry.usd
    return spent


def _tracked_run(
    video_id: str, video: Path, segmenter: Segmenter, runs_dir: Path, template: TestTemplate
) -> RunFolder:
    """The video's run folder (named by its ID), tracked: reused when an earlier attempt got
    that far, made afresh otherwise. The arena is confirmed without a human: the floor prompts
    were checked by eye on Sturman frames in the spike."""
    run = RunFolder(runs_dir / video_id)
    if run.has("track"):
        return run
    if run.path.exists():
        shutil.rmtree(run.path)  # an attempt that stopped before its track; responses are cached
    runs_dir.mkdir(parents=True, exist_ok=True)
    prompts = dict.fromkeys((template.floor_prompt, *FLOOR_PROMPTS))
    for prompt in prompts:
        attempt = template.model_copy(update={"floor_prompt": prompt})
        try:
            run = start_run(video, attempt, segmenter, runs_dir, run_id=video_id)
            break
        except ArenaNotFoundError:
            continue
    else:
        raise ArenaNotFoundError(f"{video_id}: no floor found with any of {list(prompts)}")
    confirm_arena(run)
    run_stage(run, Stage.TRACK, segmenter, _TrackOnly())
    return run


def _log_spend(log: Path, video_id: str, finished: bool, spend: dict[str, ModelSpend]) -> None:
    entry = SpendEntry(
        video_id=video_id,
        at=datetime.now(UTC),
        finished=finished,
        by_model=spend,
        usd=sum(s.usd for s in spend.values()),
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(entry.model_dump(mode="json")) + "\n")


class _TrackOnly:
    """The judge the track stage is handed and never asks: the tracking eval calls no Muse
    Spark, so any call here is a bug."""

    model_id = "none"

    def classify_clip(self, clip: Clip) -> ClipJudgement:
        raise AssertionError("the tracking eval judges no clips")

    def review_window(self, window: FlaggedWindow) -> WindowVerdict:
        raise AssertionError("the tracking eval reviews no windows")

    def write_report(self, brief: ReportBrief) -> str:
        raise AssertionError("the tracking eval writes no report")
