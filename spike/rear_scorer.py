"""Fit the measured rear scorer on the train split and check it on the eval split (laptop only).
Settles ADR 0003: the score is good evidence per candidate, but no per-video rear count from it
agrees with the raters.

Usage:
  uv run python -m spike.rear_scorer track  # download and track the 14 eval videos (~$25, once)
  uv run python -m spike.rear_scorer fit    # fit on the 6 train videos, leave-one-video-out
  uv run python -m spike.rear_scorer eval   # the fitted defaults on the 14 eval videos

Each candidate the measure stage finds (settings v4) is labelled from the raters: a rear when it
overlaps a consensus rear (at least two raters), not a rear when no rater marked any rear within
`MARGIN_S`, and left out otherwise. The scorer is a logistic regression on the candidate's
features (`RearFeatures`), fitted with numpy on standardised features and written back in the
features' own units, ready to paste into `MeasureSettings.rear_scorer`.

Counts, as FR-6 asks: per video, the candidates scoring at least a count threshold (set on the
train split's counts, kept in fit.json only) against each rater's number of rears, beside the
raters against each other. Needs `python -m spike.rear_recall` first (the train runs).
"""

import json
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pharmavision.artifacts import MeasureSettings, RearCandidate, RearFeatures, RearScorer
from pharmavision.judge.muse_spark import MuseSparkJudge, MuseSparkSettings
from pharmavision.measure.open_field import open_field_metrics
from pharmavision.measure.rears import rear_score
from pharmavision.pipeline import RunFolder, Stage, confirm_arena, run_stage, start_run
from pharmavision.segmenter.config import make_segmenter
from pharmavision.segmenter.meta_api import MetaApiSettings
from pharmavision.stages.arena import ArenaNotFoundError
from pharmavision.templates import OPEN_FIELD
from spike.data import LABELS, SPLIT, VIDEOS, download
from spike.rear_recall import OUT as RUNS
from spike.rears import Agreement, Rear, agreement, consensus_rears, load_rears

OUT = Path("eval/data/spike/rear_scorer")
MARGIN_S = 1.0  # a negative candidate is this far from any rater's rear
L2 = 1e-3  # ridge penalty on the standardised weights, for a stable fit
FEATURES = list(RearFeatures.model_fields)
FLOOR_PROMPTS = ["square floor", "white floor", "open field arena", "floor", "box"]


@dataclass(frozen=True)
class Video:
    video_id: str
    candidates: list[RearCandidate]
    labels: list[int | None]  # 1 rear, 0 not a rear, None left out
    rater_counts: dict[str, int]


def candidates(video_id: str, settings: MeasureSettings) -> tuple[list[RearCandidate], float]:
    run = RunFolder(Path((RUNS / f"{video_id}.run").read_text().strip()))
    video = run.manifest().video
    metrics = open_field_metrics(run.track(), run.confirmed_arena(), video, settings)
    return metrics.rear_candidates, video.fps


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def label(window: tuple[float, float], rears: list[Rear], agreed: list[Rear]) -> int | None:
    if any(overlaps(window, (r.start, r.end)) for r in agreed):
        return 1
    start, end = window
    if all(not overlaps((r.start, r.end), (start - MARGIN_S, end + MARGIN_S)) for r in rears):
        return 0
    return None


def load(video_ids: list[str], settings: MeasureSettings) -> list[Video]:
    every = load_rears(LABELS)
    videos: list[Video] = []
    for video_id in video_ids:
        found, fps = candidates(video_id, settings)
        rears = [r for r in every if r.video_id == video_id]
        agreed = consensus_rears(rears)
        windows = [(c.first_frame / fps, (c.last_frame + 1) / fps) for c in found]
        raters = sorted({r.rater for r in every})
        videos.append(
            Video(
                video_id,
                found,
                [label(w, rears, agreed) for w in windows],
                {rater: sum(r.rater == rater for r in rears) for rater in raters},
            )
        )
        print(f"{video_id}: {len(found)} candidates", flush=True)
    return videos


def matrix(videos: list[Video]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows = [
        ([getattr(c.features, f) for f in FEATURES], y)
        for v in videos
        for c, y in zip(v.candidates, v.labels, strict=True)
        if y is not None
    ]
    return np.array([x for x, _ in rows]), np.array([y for _, y in rows], dtype=np.float64)


def fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> RearScorer:
    """Ridge logistic regression by gradient descent on standardised features, returned in the
    features' own units."""
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-12
    z = np.c_[np.ones(len(x)), (x - mu) / sd]
    w = np.zeros(z.shape[1])
    for _ in range(20_000):
        p = 1 / (1 + np.exp(-z @ w))
        grad = z.T @ (p - y) / len(y) + L2 * np.r_[0.0, w[1:]]
        w -= 0.5 * grad
    weights = w[1:] / sd
    intercept = float(w[0] - (w[1:] * mu / sd).sum())
    return RearScorer(
        intercept=intercept,
        weights=RearFeatures(**{f: float(v) for f, v in zip(FEATURES, weights, strict=True)}),
    )


def rescored(videos: list[Video], scorer: RearScorer) -> list[Video]:
    return [
        Video(
            v.video_id,
            [c.model_copy(update={"score": rear_score(c.features, scorer)}) for c in v.candidates],
            v.labels,
            v.rater_counts,
        )
        for v in videos
    ]


def auc(scores: list[float], truth: list[int]) -> float:
    pos = [s for s, t in zip(scores, truth, strict=True) if t]
    neg = [s for s, t in zip(scores, truth, strict=True) if not t]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def candidate_scores(videos: list[Video], thresholds: list[float]) -> dict[str, Any]:
    """Candidate labels against the raters: each video's scores at its own threshold."""
    rows = [
        (c.score, c.score >= t, y)
        for v, t in zip(videos, thresholds, strict=True)
        for c, y in zip(v.candidates, v.labels, strict=True)
        if y is not None
    ]
    scores, called, truth = [s for s, _, _ in rows], [c for _, c, _ in rows], [y for *_, y in rows]
    hits = sum(c and t for c, t in zip(called, truth, strict=True))
    score: Agreement = agreement([str(t) for t in truth], [str(int(c)) for c in called])
    return {
        "candidates": len(rows),
        "rears": sum(truth),
        "auc": auc(scores, truth),
        "rear_vs_none": asdict(score),
        "precision": hits / max(sum(called), 1),
        "recall": hits / max(sum(truth), 1),
    }


def count_agreement(videos: list[Video], thresholds: list[float]) -> dict[str, Any]:
    """Per-video rear counts: the scorer (each video at its own threshold) against each rater,
    and each pair of raters. Pearson r, and the mean absolute difference as a share of the
    pair's mean count."""

    def compare(a: list[float], b: list[float]) -> dict[str, float]:
        x, y = np.array(a), np.array(b)
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
        return {"pearson_r": r, "mean_abs_diff_pct": float(np.mean(np.abs(x - y) / ((x + y) / 2)))}

    model = [
        float(sum(c.score >= t for c in v.candidates))
        for v, t in zip(videos, thresholds, strict=True)
    ]
    raters = sorted(videos[0].rater_counts)
    per_rater = {rater: compare(model, [v.rater_counts[rater] for v in videos]) for rater in raters}
    pairs = {
        f"{a}~{b}": compare(
            [v.rater_counts[a] for v in videos], [v.rater_counts[b] for v in videos]
        )
        for a, b in combinations(raters, 2)
    }

    def mean(group: dict[str, dict[str, float]], key: str) -> float:
        return float(np.mean([g[key] for g in group.values()]))

    return {
        "per_video": {
            v.video_id: {"scorer": m, **v.rater_counts} for v, m in zip(videos, model, strict=True)
        },
        "scorer_vs_rater": per_rater,
        "rater_vs_rater": pairs,
        "mean": {
            "scorer_vs_rater": {k: mean(per_rater, k) for k in ("pearson_r", "mean_abs_diff_pct")},
            "rater_vs_rater": {k: mean(pairs, k) for k in ("pearson_r", "mean_abs_diff_pct")},
        },
    }


def calibrated_threshold(videos: list[Video], scorer: RearScorer) -> float:
    """The threshold whose total count over `videos` comes closest to the raters' mean total:
    counts are what FR-6 judges, so the threshold is set on them, on the train split only."""
    target = np.mean([sum(v.rater_counts[r] for v in videos) for r in videos[0].rater_counts])
    scores = [c.score for v in rescored(videos, scorer) for c in v.candidates]
    grid = np.linspace(0.05, 0.95, 91)
    return float(min(grid, key=lambda t: abs(sum(s >= t for s in scores) - target)))


def calibrated(videos: list[Video]) -> tuple[RearScorer, float]:
    """Weights fitted on the videos' labelled candidates, and a count threshold set on their
    counts. The threshold lives here only: the pipeline makes no count (ADR 0003)."""
    scorer = fit(*matrix(videos))
    return scorer, calibrated_threshold(videos, scorer)


def fit_train() -> None:
    settings = MeasureSettings()
    train: list[str] = json.loads(SPLIT.read_text())["train"]
    videos = load(train, settings)
    # Leave one video out: fit the weights and the threshold on five, score the sixth.
    held_out: list[Video] = []
    thresholds: list[float] = []
    for k, v in enumerate(videos):
        scorer, threshold = calibrated(videos[:k] + videos[k + 1 :])
        held_out += rescored([v], scorer)
        thresholds.append(threshold)
    final, count_threshold = calibrated(videos)
    summary = {
        "settings_version": settings.version,
        "scorer": final.model_dump(),
        "count_threshold": count_threshold,
        "lovo_thresholds": thresholds,
        "lovo_candidates": candidate_scores(held_out, thresholds),
        "lovo_counts": count_agreement(held_out, thresholds),
    }
    (OUT / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "lovo_counts"}, indent=2))
    print(json.dumps(summary["lovo_counts"]["mean"], indent=2))


def eval_split() -> None:
    settings = MeasureSettings()
    held_out: list[str] = json.loads(SPLIT.read_text())["eval"]
    # The count threshold set on the train split by `fit`, as the settings carry none.
    threshold: float = json.loads((OUT / "fit.json").read_text())["count_threshold"]
    videos = load([v for v in held_out if (RUNS / f"{v}.run").exists()], settings)
    summary = {
        "settings_version": settings.version,
        "scorer": settings.rear_scorer.model_dump(),
        "videos": [v.video_id for v in videos],
        "count_threshold": threshold,
        "candidates": candidate_scores(videos, [threshold] * len(videos)),
        "counts": count_agreement(videos, [threshold] * len(videos)),
    }
    (OUT / "eval.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=2))
    print(json.dumps(summary["counts"]["mean"], indent=2))


def track() -> None:
    """Track each eval video once, as `spike.rear_recall` tracks the train videos. Where SAM 3.1
    finds no "square floor", the other floor prompts are tried in order (OFT_12 and OFT_52 took
    "white floor", OFT_39 "floor"; px/cm stayed within 7.26-7.45 on all 20 videos)."""
    segmenter = make_segmenter(MetaApiSettings())
    for video_id in json.loads(SPLIT.read_text())["eval"]:
        pointer = RUNS / f"{video_id}.run"
        if pointer.exists():
            continue
        download(video_id)
        for prompt in FLOOR_PROMPTS:
            template = OPEN_FIELD.model_copy(update={"floor_prompt": prompt})
            try:
                run = start_run(VIDEOS / f"{video_id}.mp4", template, segmenter, RUNS / "runs")
                break
            except ArenaNotFoundError:
                print(f"{video_id}: no {prompt!r}", flush=True)
        else:
            print(f"{video_id}: skipped, no arena found", flush=True)
            continue
        confirm_arena(run)
        # Track only: the judge is never called.
        judge = MuseSparkJudge(MuseSparkSettings(tier="contributor"))
        run_stage(run, Stage.TRACK, segmenter, judge)
        pointer.write_text(f"{run.path}\n")
        frames = run.track().frames
        print(f"{video_id}: {prompt!r}, {sum(f.detected for f in frames)}/{len(frames)} tracked")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"track": track, "fit": fit_train, "eval": eval_split}[sys.argv[1]]()
