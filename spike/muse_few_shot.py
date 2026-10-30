"""Few-shot examples for the Muse Spark judge, and what they change (laptop only). Settles the
rest of open decision 3: are cropped, outlined, few-shot clips good enough to trust?

Usage:
  uv run python -m spike.muse_few_shot pool    # cut + zero-shot-label candidate clips (train split)
  uv run python -m spike.muse_few_shot build   # cut the examples in few_shot/examples.json
  uv run python -m spike.muse_few_shot score   # T2 clips and the pool hold-out, with and without
  uv run python -m spike.muse_few_shot again   # the hold-out asked a second time: label noise
  uv run python -m spike.muse_few_shot smoke [few]   # the 60 s smoke run (OFT_16 540-600 s)

Every clip is cut the way the behaviours stage cuts it (`cut_clips`), from the train videos' run
folders that `spike.rear_recall` made. The pool is rear-candidate clips that either show a rear
all three raters marked or lie a second away from any rater's rear (hard negatives: postures
the candidates flag and no rater calls a rear). It keeps away from the 20 T2 clips
and from OFT_16 540-600 s (the live smoke run). Contributor tier, as this is Sturman data; cost
is also given at the standard rate. Needs MODEL_API_KEY in .env, `python -m spike.data` and
`python -m spike.rear_recall` first.
"""

import json
import random
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pharmavision.artifacts import MeasureSettings, RunFolder
from pharmavision.judge import Clip
from pharmavision.judge.muse_spark import (
    FEW_SHOT_DIR,
    FewShotExample,
    MuseSparkJudge,
    MuseSparkSettings,
    few_shot_examples,
)
from pharmavision.measure.open_field import open_field_metrics
from pharmavision.model_cache import Event, input_hash
from pharmavision.pipeline import Stage, confirm_arena, run_stage, start_run
from pharmavision.segmenter.config import make_segmenter
from pharmavision.segmenter.meta_api import MetaApiSettings
from pharmavision.stages import behaviours
from pharmavision.stages.behaviours import clip_span, cut_clips
from pharmavision.templates import OPEN_FIELD
from spike.data import LABELS, SPLIT, VIDEOS
from spike.muse_clips import N_PER_KIND, duration_s
from spike.rears import Agreement, Rear, agreement, consensus_rears, load_rears, pick_clips

OUT = Path("eval/data/spike/muse_few_shot")
RUNS = Path("eval/data/spike/rear_recall")
# Per M tokens, dev.meta.ai/docs/pricing-rate-limits, Oct 30, 2026
USD_PER_M = {"standard": (1.25, 4.25), "contributor": (0.10, 0.20)}
KEEP_OUT_S = 2.0  # pool clips stay this far from a T2 clip
SMOKE = ("OFT_16", 540.0, 600.0)
NEGATIVE_MARGIN_S = 1.0  # a hard negative is this far from any rater's rear
MIN_REAR_S = 0.6  # a clear rear lasts this long, by consensus
ALL_RATERS = 3  # a clear rear is marked by every rater
MIN_INSIDE = 0.8  # and lies at least this share inside the clip
PER_VIDEO = {"supported": 3, "unsupported": 3, "none": 8}
SEED = 20261030
WORKERS = 8  # parallel requests, well under the 150 RPM limit

Kind = Literal["supported", "unsupported", "none"]


@dataclass(frozen=True)
class PoolClip:
    video_id: str
    first_frame: int
    stop_frame: int
    start_s: float
    end_s: float
    truth: Kind


def run_folder(video_id: str) -> RunFolder:
    return RunFolder(Path((RUNS / f"{video_id}.run").read_text().strip()))


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def kept_out(
    video_id: str, window: tuple[float, float], t2: Sequence[tuple[str, float, float]]
) -> bool:
    near = [(s - KEEP_OUT_S, e + KEEP_OUT_S) for v, s, e in (*t2, SMOKE) if v == video_id]
    return any(overlaps(window, w) for w in near)


def classify(window: tuple[float, float], rears: list[Rear]) -> Kind | None:
    """Clear rear (all three raters, mostly inside the clip), hard negative, or neither."""
    start, end = window
    if all(
        not overlaps((r.start, r.end), (start - NEGATIVE_MARGIN_S, end + NEGATIVE_MARGIN_S))
        for r in rears
    ):
        return "none"
    for c in consensus_rears(rears):
        raters = {
            r.rater
            for r in rears
            if r.kind == c.kind and overlaps((r.start, r.end), (c.start, c.end))
        }
        inside = min(c.end, end) - max(c.start, start)
        long_enough = c.end - c.start >= MIN_REAR_S
        if len(raters) == ALL_RATERS and long_enough and inside >= MIN_INSIDE * (c.end - c.start):
            return c.kind
    return None


def t2_windows() -> list[tuple[str, float, float, Kind]]:
    train: list[str] = json.loads(SPLIT.read_text())["train"]
    video_s = min(duration_s(VIDEOS / f"{v}.mp4") for v in train)
    return [
        (c.video_id, c.start, c.end, c.truth)
        for c in pick_clips(load_rears(LABELS), train, video_s, N_PER_KIND)
    ]


def judge(few_shot: Sequence[FewShotExample] = ()) -> MuseSparkJudge:
    return MuseSparkJudge(MuseSparkSettings(tier="contributor", few_shot=tuple(few_shot)))


def usage(events: list[Event]) -> dict[str, Any]:
    done = next(e for e in events if e.get("type") == "response.completed")
    return dict(done["response"]["usage"])  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]


def cost_usd(u: dict[str, Any], tier: str) -> float:
    per_in, per_out = USD_PER_M[tier]
    return (u["input_tokens"] * per_in + u["output_tokens"] * per_out) / 1e6


def judge_spans(
    j: MuseSparkJudge, video_id: str, spans: list[range], tag: str
) -> list[dict[str, Any]]:
    """Each span cut from the video's run folder and judged; one row per span, in order."""
    dest = OUT / "clips" / tag / video_id
    dest.mkdir(parents=True, exist_ok=True)
    clips = dict(cut_clips(run_folder(video_id), spans, dest))

    def one(k: int) -> dict[str, Any]:
        events = j.response(clips[k])
        judgement = j.classify_clip(clips[k])  # from the cache just filled
        return judgement.model_dump() | {"usage": usage(events), "clip": str(clips[k].video)}

    with ThreadPoolExecutor(WORKERS) as executor:
        return list(executor.map(one, range(len(spans))))


def pool() -> None:
    labels = load_rears(LABELS)
    t2 = [(v, s, e) for v, s, e, _ in t2_windows()]
    rng = random.Random(SEED)
    chosen: list[PoolClip] = []
    for video_id in json.loads(SPLIT.read_text())["train"]:
        run = run_folder(video_id)
        video, settings = run.manifest().video, run.manifest().behaviours
        metrics = open_field_metrics(run.track(), run.confirmed_arena(), video, MeasureSettings())
        rears = [r for r in labels if r.video_id == video_id]
        by_kind: dict[Kind, list[PoolClip]] = {"supported": [], "unsupported": [], "none": []}
        for c in metrics.rear_candidates:
            span = clip_span(c, video, settings)
            window = (span.start / video.fps, span.stop / video.fps)
            kind = classify(window, rears)
            if kind is None or kept_out(video_id, window, t2):
                continue
            by_kind[kind].append(PoolClip(video_id, span.start, span.stop, *window, kind))
        for kind, clips in by_kind.items():
            chosen += rng.sample(clips, min(PER_VIDEO[kind], len(clips)))
        print(video_id, {k: len(v) for k, v in by_kind.items()})
    rows = score_pool(chosen, judge(), "zero-shot")
    (OUT / "pool.json").write_text(json.dumps(rows, indent=2) + "\n")
    summarise("pool, zero-shot", rows)


def score_pool(clips: list[PoolClip], j: MuseSparkJudge, tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_id in sorted({c.video_id for c in clips}):
        mine = [c for c in clips if c.video_id == video_id]
        spans = [range(c.first_frame, c.stop_frame) for c in mine]
        rows += [
            asdict(c) | r for c, r in zip(mine, judge_spans(j, video_id, spans, tag), strict=True)
        ]
    return rows


def build() -> None:
    """Cut each example in examples.json from its run folder into the package."""
    entries = json.loads((FEW_SHOT_DIR / "examples.json").read_text())
    for e in entries:
        run = run_folder(e["video_id"])
        fps = run.manifest().video.fps
        span = range(round(e["start_s"] * fps), round(e["end_s"] * fps))
        tmp = OUT / "build"
        tmp.mkdir(parents=True, exist_ok=True)
        ((_, clip),) = cut_clips(run, [span], tmp)
        dest = FEW_SHOT_DIR / e["clip"]
        clip.video.replace(dest)
        print(dest, dest.stat().st_size)


def summarise(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = ["rear" if r["truth"] != "none" else "none" for r in rows]
    predicted = ["rear" if r["label"] == "rearing" else "none" for r in rows]
    called = [t for t, p in zip(truth, predicted, strict=True) if p == "rear"]
    score: Agreement = agreement(truth, predicted)
    out = {
        "clips": len(rows),
        "rears": truth.count("rear"),
        "rear_vs_none": asdict(score),
        "recall": sum(t == p == "rear" for t, p in zip(truth, predicted, strict=True))
        / max(truth.count("rear"), 1),
        "precision": called.count("rear") / len(called) if called else None,
        "mean_input_tokens": sum(r["usage"]["input_tokens"] for r in rows) / len(rows),
        "mean_output_tokens": sum(r["usage"]["output_tokens"] for r in rows) / len(rows),
        "cost_usd": {t: sum(cost_usd(r["usage"], t) for r in rows) for t in USD_PER_M},
    }
    print(name, json.dumps(out, indent=1))
    return out


def score() -> None:
    examples = few_shot_examples()
    entries = json.loads((FEW_SHOT_DIR / "examples.json").read_text())
    shown = {(e["video_id"], e["start_s"]) for e in entries}
    t2: list[PoolClip] = []
    for v, s, e, k in t2_windows():
        fps = run_folder(v).manifest().video.fps
        t2.append(PoolClip(v, round(s * fps), round(e * fps), s, e, k))
    pool_rows = json.loads((OUT / "pool.json").read_text())
    held_out = [
        PoolClip(**{f: r[f] for f in PoolClip.__dataclass_fields__})
        for r in pool_rows
        if (r["video_id"], round(r["start_s"], 3)) not in shown
    ]
    results: dict[str, Any] = {}
    for name, j in (("zero-shot", judge()), ("few-shot", judge(examples))):
        for set_name, clips in (("t2", t2), ("pool_held_out", held_out)):
            rows = score_pool(clips, j, f"{set_name}-{name}")
            results[f"{set_name}/{name}"] = summarise(f"{set_name}, {name}", rows) | {"rows": rows}
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


def again() -> None:
    """Ask the hold-out clips once more, from a separate cache: how often the label changes."""
    summary = json.loads((OUT / "summary.json").read_text())
    results: dict[str, Any] = {}
    for name, few in (("zero-shot", ()), ("few-shot", few_shot_examples())):
        settings = MuseSparkSettings(
            tier="contributor", few_shot=few, cache_dir=OUT / "again_cache"
        )
        j = MuseSparkJudge(settings)
        rows = summary[f"pool_held_out/{name}"]["rows"]

        def one(row: dict[str, Any], j: MuseSparkJudge = j) -> str:
            # Keyed by the clip's path: this cache only ever holds these clips.
            return j.classify_clip(Clip(Path(row["clip"]), input_hash(row["clip"]))).label

        with ThreadPoolExecutor(WORKERS) as executor:
            labels = list(executor.map(one, rows))
        second = [r | {"label": label} for r, label in zip(rows, labels, strict=True)]
        results[name] = summarise(f"pool_held_out, {name}, asked again", second) | {
            "same_label_as_first": sum(
                r["label"] == new for r, new in zip(rows, labels, strict=True)
            )
            / len(rows)
        }
        print(name, "same label as first ask:", results[name]["same_label_as_first"])
    (OUT / "again.json").write_text(json.dumps(results, indent=2) + "\n")


def smoke_source() -> RunFolder:
    """The smoke run: OFT_16 540-600 s cut to its own 60 s video, ingested with the protocol
    length set to 60 s (ingest refuses a short clip otherwise), tracked and measured. Made once;
    SAM 3.1 responses are cached."""
    pointer = OUT / "smoke.run"
    if pointer.exists():
        return RunFolder(Path(pointer.read_text().strip()))
    video_id, start, end = SMOKE
    clip = OUT / f"{video_id}_{start:.0f}-{end:.0f}.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-to", f"{end}",
         "-i", str(VIDEOS / f"{video_id}.mp4"), "-c:v", "libx264", "-an", str(clip)],
        check=True,
    )  # fmt: skip
    template = OPEN_FIELD.model_copy(update={"protocol_duration_s": end - start})
    segmenter = make_segmenter(MetaApiSettings())
    runs = OUT / "smoke-runs"
    runs.mkdir(parents=True, exist_ok=True)
    run = start_run(clip, template, segmenter, runs)
    confirm_arena(run)
    for stage in (Stage.TRACK, Stage.MEASURE):
        run_stage(run, stage, segmenter, judge())  # the judge is never called
    pointer.write_text(f"{run.path}\n")
    return run


def smoke(few: bool) -> None:
    """Re-judge a copy of the smoke run on the standard tier, like its zero-shot baseline, and
    score its events against the raters: a `rearing` event is on a rear when its window
    overlaps any rater's rear."""
    run = RunFolder(OUT / f"smoke-{'few' if few else 'zero'}-shot")
    shutil.rmtree(run.path, ignore_errors=True)
    shutil.copytree(smoke_source().path, run.path)
    settings = MuseSparkSettings(few_shot=few_shot_examples() if few else ())
    behaviours.run(run, MuseSparkJudge(settings))
    video_id, offset, end = SMOKE
    labels = [r for r in load_rears(LABELS) if r.video_id == video_id]
    rears = [(r.start, r.end) for r in labels if overlaps((r.start, r.end), (offset, end))]
    agreed = [
        (c.start, c.end)
        for c in consensus_rears(labels)
        if overlaps((c.start, c.end), (offset, end))
    ]
    fps = run.manifest().video.fps
    windows = [
        ((e.first_frame / fps + offset, (e.last_frame + 1) / fps + offset), e.label)
        for e in run.events().events
    ]
    called = [w for w, label in windows if label == "rearing"]
    on_a_rear = [w for w in called if any(overlaps(w, r) for r in rears)]
    found = [c for c in agreed if any(overlaps(w, c) for w in called)]
    print(
        f"{len(windows)} events, {len(called)} rearing: {len(on_a_rear)}/{len(called)} on a rear; "
        f"consensus rears found {len(found)}/{len(agreed)}"
    )


if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        smoke(few=sys.argv[2:] == ["few"])
    else:
        {"pool": pool, "build": build, "score": score, "again": again}[sys.argv[1]]()
