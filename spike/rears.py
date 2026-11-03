"""Pure helpers for the T2 spike: which labelled rears become clips, and how Muse Spark scores.

Ground truth comes from the Sturman et al. 2020 human labels (DLCAnalyzer, GPL-3.0). The label
file stays in the git-ignored `eval/data/`; this module only reads it.
"""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from eval.split import SPLIT_SEED

RearKind = Literal["supported", "unsupported"]
Truth = Literal["supported", "unsupported", "none"]
_KINDS: dict[str, RearKind] = {"Supported": "supported", "Unsupported": "unsupported"}


@dataclass(frozen=True)
class Rear:
    """One rear as one rater labelled it, in seconds from the start of the video."""

    video_id: str
    rater: str
    start: float
    end: float
    kind: RearKind


def load_rears(path: Path) -> list[Rear]:
    """Read the DLCAnalyzer open-field label file and keep supported and unsupported rears."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return [
        Rear(
            video_id=row["ID"],
            rater=row["Experimenter"],
            start=float(row["from"]),
            end=float(row["to"]),
            kind=_KINDS[row["type"]],
        )
        for row in rows
        if row["type"] in _KINDS
    ]


def _iou(a: Rear, b: Rear) -> float:
    overlap = min(a.end, b.end) - max(a.start, b.start)
    union = max(a.end, b.end) - min(a.start, b.start)
    return max(overlap, 0.0) / union if union > 0 else 0.0


def consensus_rears(rears: list[Rear], min_iou: float = 0.5) -> list[Rear]:
    """Rears at least two raters agree on: same video, same kind, time IoU ≥ `min_iou`.

    Each consensus rear spans the time every agreeing rater shares, and is reported once
    however many raters saw it.
    """
    ordered = sorted(rears, key=lambda r: (r.video_id, r.start, r.rater))
    used: set[int] = set()
    out: list[Rear] = []
    for i, ref in enumerate(ordered):
        if i in used:
            continue
        matches = [
            j
            for j, other in enumerate(ordered)
            if j != i
            and j not in used
            and other.video_id == ref.video_id
            and other.rater != ref.rater
            and other.kind == ref.kind
            and _iou(ref, other) >= min_iou
        ]
        if not matches:
            continue
        used.update([i, *matches])
        agreed = [ref, *(ordered[j] for j in matches)]
        out.append(
            Rear(
                video_id=ref.video_id,
                rater="consensus",
                start=max(r.start for r in agreed),
                end=min(r.end for r in agreed),
                kind=ref.kind,
            )
        )
    return out


def clip_window(
    rear: Rear, video_s: float, pad_s: float = 0.5, min_s: float = 2.0, max_s: float = 4.0
) -> tuple[float, float]:
    """Clip (start, end) in seconds: the rear plus `pad_s` each side, held to 2-4 s, centred on the
    rear and shifted to stay inside the video. Rounded to milliseconds."""
    length = min(max(rear.end - rear.start + 2 * pad_s, min_s), max_s)
    start = (rear.start + rear.end) / 2 - length / 2
    start = min(max(start, 0.0), video_s - length)
    return round(start, 3), round(start + length, 3)


@dataclass(frozen=True)
class Clip:
    """A 2-4 s window of one video, with the label the human raters agree it shows."""

    video_id: str
    start: float
    end: float
    truth: Truth


def _no_rear_windows(
    rears: list[Rear], video_s: float, length_s: float = 3.0, margin_s: float = 1.0
) -> list[tuple[float, float]]:
    """Windows on a 1 s grid where no rater marked any rear within `margin_s`."""
    windows: list[tuple[float, float]] = []
    for t in range(int(video_s - length_s) + 1):
        start, end = float(t), t + length_s
        if all(end + margin_s <= r.start or r.end <= start - margin_s for r in rears):
            windows.append((start, end))
    return windows


def pick_clips(
    rears: list[Rear],
    train_ids: list[str],
    video_s: float,
    n_per_kind: dict[Truth, int],
    seed: int = SPLIT_SEED,
) -> list[Clip]:
    """Sample clips from train videos only: consensus rears of each kind, plus no-rear windows."""
    rng = random.Random(seed)
    train = [r for r in rears if r.video_id in train_ids]
    clips: list[Clip] = []
    agreed = consensus_rears(train)
    for kind in ("supported", "unsupported"):
        pool = [r for r in agreed if r.kind == kind]
        for r in rng.sample(pool, min(n_per_kind.get(kind, 0), len(pool))):
            start, end = clip_window(r, video_s)
            clips.append(Clip(r.video_id, start, end, kind))
    quiet = [
        (video_id, *window)
        for video_id in sorted(train_ids)
        for window in _no_rear_windows([r for r in train if r.video_id == video_id], video_s)
    ]
    for video_id, start, end in rng.sample(quiet, min(n_per_kind.get("none", 0), len(quiet))):
        clips.append(Clip(video_id, start, end, "none"))
    return clips


@dataclass(frozen=True)
class Agreement:
    n: int
    accuracy: float
    kappa: float | None  # Cohen's kappa; None when chance agreement is 1 and kappa is undefined


def agreement(truth: list[str], predicted: list[str]) -> Agreement:
    """Model labels against the human consensus: raw accuracy and Cohen's kappa."""
    n = len(truth)
    observed = sum(t == p for t, p in zip(truth, predicted, strict=True)) / n
    truth_counts, predicted_counts = Counter(truth), Counter(predicted)
    chance = sum(truth_counts[k] * predicted_counts[k] for k in truth_counts) / n**2
    kappa = None if chance == 1 else (observed - chance) / (1 - chance)
    return Agreement(n=n, accuracy=observed, kappa=kappa)


@dataclass(frozen=True)
class Recall:
    """How many labelled rears overlap a rear-candidate window, and how many windows overlap a
    labelled rear (the share of clips stage 5 would spend on a real rear)."""

    rears: int
    found: int
    windows: int
    windows_on_a_rear: int

    @property
    def recall(self) -> float | None:
        return self.found / self.rears if self.rears else None


def rear_recall(rears: list[Rear], windows: list[tuple[float, float]]) -> Recall:
    """Rears against candidate windows, both in seconds. A rear is found when some window
    overlaps it for a non-zero time; touching at an end is not enough. Overlap, not
    containment, because stage 5 judges a 2-4 s clip cut around each window, not the window."""

    def overlaps(r: Rear, window: tuple[float, float]) -> bool:
        return min(r.end, window[1]) > max(r.start, window[0])

    return Recall(
        rears=len(rears),
        found=sum(any(overlaps(r, w) for w in windows) for r in rears),
        windows=len(windows),
        windows_on_a_rear=sum(any(overlaps(r, w) for r in rears) for w in windows),
    )
