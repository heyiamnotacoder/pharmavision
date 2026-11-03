"""Rearing agreement (measures): the pipeline's rear candidates against each Sturman rater, as
FR-6 asks, beside the raters against each other.

- **Per-video count:** ICC(2,1) between a count from the candidates and each rater's count of
  rears, over the eval videos.
- **Events:** each candidate window matched one-to-one to a rater's rears, a pair matching when
  the two lie within `tolerance_s` of each other; precision, recall and F1 pooled over the
  videos.

The pipeline makes no rear count (ADR 0003). The eval counts anyway, to measure how far a count
would be from the raters: the candidates scoring at least `count_threshold` (set on the train
split's counts by `spike.rear_scorer fit`, Oct 31), and every candidate."""

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

Window = tuple[float, float]
"""A start and an end, in seconds from the start of the video."""
Metric = Literal["count_icc", "f1"]


class RearingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    count_threshold: float = Field(default=0.55, ge=0, le=1)
    """The eval's count: candidates scoring at least this. Set on the train split's counts by
    `spike.rear_scorer fit` (Oct 31, measure settings v4). Here only: the pipeline makes no
    count (ADR 0003)."""
    tolerance_s: float = Field(default=0.5, ge=0)
    """A window and a rear match when they overlap or lie at most this far apart."""


class Candidate(BaseModel):
    """A rear candidate window in seconds, with its rear score."""

    model_config = ConfigDict(frozen=True)

    start_s: float
    end_s: float
    score: float = Field(ge=0, le=1)


class VideoRears(BaseModel):
    """One eval video: the pipeline's rear candidates, and each rater's rears. A rater who
    marked no rear in the video has an empty list."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    candidates: list[Candidate]
    raters: dict[str, list[Window]]


class EventScore(BaseModel):
    """Windows against rears, matched one-to-one and pooled over the videos. Precision, recall
    and F1 are None when they are undefined (no windows, no rears)."""

    model_config = ConfigDict(frozen=True)

    matched: int
    windows: int
    rears: int
    precision: float | None
    recall: float | None
    f1: float | None

    @classmethod
    def of(cls, matched: int, windows: int, rears: int) -> "EventScore":
        precision = matched / windows if windows else None
        recall = matched / rears if rears else None
        f1 = 2 * matched / (windows + rears) if windows + rears else None
        return cls(
            matched=matched,
            windows=windows,
            rears=rears,
            precision=precision,
            recall=recall,
            f1=f1,
        )


class Agreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    count_icc: float | None
    """ICC(2,1) of the per-video counts; None when undefined."""
    events: EventScore

    def value(self, metric: Metric) -> float | None:
        return self.count_icc if metric == "count_icc" else self.events.f1


class VideoCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    scored: int
    """Candidates scoring at least the count threshold."""
    candidates: int
    raters: dict[str, int]


class RearingAgreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    settings: RearingSettings
    raters: list[str]
    per_video: list[VideoCounts]
    scored: dict[str, Agreement]
    """The candidates at or above the count threshold, against each rater."""
    every_candidate: dict[str, Agreement]
    """Every candidate, against each rater: the recall-first windows (ADR 0001) as they are."""
    rater_pairs: dict[str, Agreement]
    """Each pair of raters, `A~B`, against each other: the human-human range."""

    def human_range(self, metric: Metric) -> tuple[float, float] | None:
        """The lowest and highest agreement between two raters; None when none is defined."""
        values = [v for v in (a.value(metric) for a in self.rater_pairs.values()) if v is not None]
        return (min(values), max(values)) if values else None


def rearing_agreement(videos: Sequence[VideoRears], settings: RearingSettings) -> RearingAgreement:
    """The scored count and every candidate against each rater, and the raters against each
    other, over `videos`."""
    raters = sorted({r for v in videos for r in v.raters})

    def scored_windows(v: VideoRears) -> list[Window]:
        return [(c.start_s, c.end_s) for c in v.candidates if c.score >= settings.count_threshold]

    def every_window(v: VideoRears) -> list[Window]:
        return [(c.start_s, c.end_s) for c in v.candidates]

    def rears_of(rater: str) -> Callable[[VideoRears], list[Window]]:
        return lambda v: v.raters.get(rater, [])

    def agree(
        a: Callable[[VideoRears], list[Window]], b: Callable[[VideoRears], list[Window]]
    ) -> Agreement:
        pairs = [(a(v), b(v)) for v in videos]
        return Agreement(
            count_icc=icc_a1([[len(x), len(y)] for x, y in pairs]),
            events=EventScore.of(
                matched=sum(matched_pairs(x, y, settings.tolerance_s) for x, y in pairs),
                windows=sum(len(x) for x, _ in pairs),
                rears=sum(len(y) for _, y in pairs),
            ),
        )

    return RearingAgreement(
        settings=settings,
        raters=raters,
        per_video=[
            VideoCounts(
                video_id=v.video_id,
                scored=len(scored_windows(v)),
                candidates=len(v.candidates),
                raters={r: len(rears_of(r)(v)) for r in raters},
            )
            for v in videos
        ],
        scored={r: agree(scored_windows, rears_of(r)) for r in raters},
        every_candidate={r: agree(every_window, rears_of(r)) for r in raters},
        rater_pairs={
            f"{a}~{b}": agree(rears_of(a), rears_of(b)) for a, b in combinations(raters, 2)
        },
    )


def matched_pairs(windows: Sequence[Window], rears: Sequence[Window], tolerance_s: float) -> int:
    """The most window-rear pairs that can be made, each window and each rear in one pair at
    most, a pair being a window and a rear no more than `tolerance_s` apart."""
    if not windows or not rears:
        return 0
    reach = np.array(
        [[max(w[0], r[0]) - min(w[1], r[1]) <= tolerance_s for r in rears] for w in windows]
    )
    rows, cols = linear_sum_assignment(reach, maximize=True)
    return int(reach[rows, cols].sum())


def icc_a1(ratings: Sequence[Sequence[float]]) -> float | None:
    """ICC(2,1) of an n videos x k raters table: two-way random effects, absolute agreement, a
    single rater (Shrout & Fleiss 1979). None when it is undefined: fewer than two videos, or
    no spread to agree on."""
    y = np.asarray(ratings, dtype=np.float64)
    n, k = y.shape
    if n < 2 or k < 2:
        return None
    grand = float(y.mean())
    ms_rows = k * float(((y.mean(axis=1) - grand) ** 2).sum()) / (n - 1)
    ms_cols = n * float(((y.mean(axis=0) - grand) ** 2).sum()) / (k - 1)
    residual = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + grand
    ms_error = float((residual**2).sum()) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator <= 0:
        return None
    return (ms_rows - ms_error) / denominator
