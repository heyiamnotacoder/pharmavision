"""Post-hoc pairs after a test of 3 or more groups: every pair in the design's order, the later
group minus the earlier."""

import itertools
import math
import statistics
from collections import Counter
from collections.abc import Sequence

from scipy import stats

from ..artifacts import PairwiseComparison


def tukey_hsd(
    names: Sequence[str], samples: Sequence[Sequence[float]], level: float
) -> list[PairwiseComparison]:
    """Tukey's HSD: p and intervals from the studentized range on the pooled variance. The
    statistic is t, the difference over sqrt(MSE (1/n_a + 1/n_b))."""
    result = stats.tukey_hsd(*samples)
    ci = result.confidence_interval(level)
    within = sum((len(s) - 1) * statistics.variance(s) for s in samples)
    mse = within / (sum(len(s) for s in samples) - len(samples))
    pairs: list[PairwiseComparison] = []
    for i, j in itertools.combinations(range(len(samples)), 2):
        difference = float(result.statistic[j, i])
        se = math.sqrt(mse * (1 / len(samples[i]) + 1 / len(samples[j])))
        pairs.append(
            PairwiseComparison(
                group_a=names[i],
                group_b=names[j],
                difference=difference,
                statistic=difference / se,
                p_adjusted=float(result.pvalue[j, i]),
                ci_low=float(ci.low[j, i]),
                ci_high=float(ci.high[j, i]),
            )
        )
    return pairs


def games_howell(
    names: Sequence[str], samples: Sequence[Sequence[float]], level: float
) -> list[PairwiseComparison]:
    """Games-Howell: each pair's own variances and Welch degrees of freedom, with p and
    intervals from the studentized range. The statistic is t, the difference over
    sqrt(var_a / n_a + var_b / n_b)."""
    k = len(samples)
    pairs: list[PairwiseComparison] = []
    for i, j in itertools.combinations(range(k), 2):
        a, b = samples[i], samples[j]
        va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
        se = math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
        difference = statistics.fmean(b) - statistics.fmean(a)
        t = difference / se
        half = float(stats.studentized_range.ppf(level, k, df)) / math.sqrt(2) * se
        pairs.append(
            PairwiseComparison(
                group_a=names[i],
                group_b=names[j],
                difference=difference,
                statistic=t,
                p_adjusted=float(stats.studentized_range.sf(abs(t) * math.sqrt(2), k, df)),
                ci_low=difference - half,
                ci_high=difference + half,
            )
        )
    return pairs


def dunn(names: Sequence[str], samples: Sequence[Sequence[float]]) -> list[PairwiseComparison]:
    """Dunn's test: mean ranks over all the groups' values together, z with the tie
    correction, and p Holm-adjusted over the pairs. It gives no interval."""
    pooled = [v for s in samples for v in s]
    n = len(pooled)
    ranks = stats.rankdata(pooled)
    bounds = list(itertools.accumulate((len(s) for s in samples), initial=0))
    mean_ranks = [float(ranks[lo:hi].mean()) for lo, hi in itertools.pairwise(bounds)]
    ties = sum(t**3 - t for t in Counter(pooled).values())
    variance = n * (n + 1) / 12 - ties / (12 * (n - 1))
    raw: list[tuple[int, int, float, float]] = []
    for i, j in itertools.combinations(range(len(samples)), 2):
        difference = mean_ranks[j] - mean_ranks[i]
        z = difference / math.sqrt(variance * (1 / len(samples[i]) + 1 / len(samples[j])))
        raw.append((i, j, difference, z))
    adjusted = _holm([2 * float(stats.norm.sf(abs(z))) for *_, z in raw])
    return [
        PairwiseComparison(
            group_a=names[i],
            group_b=names[j],
            difference=difference,
            statistic=z,
            p_adjusted=p,
            ci_low=None,
            ci_high=None,
        )
        for (i, j, difference, z), p in zip(raw, adjusted, strict=True)
    ]


def _holm(p: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment: the k-th smallest of m p values times m - k + 1, kept from
    falling below the one before it and capped at 1."""
    m = len(p)
    adjusted = [0.0] * m
    running = 0.0
    for k, i in enumerate(sorted(range(m), key=lambda i: p[i])):
        running = max(running, min(1.0, (m - k) * p[i]))
        adjusted[i] = running
    return adjusted
