"""Three or more groups: the omnibus test, its effect size, and the post-hoc pairs."""

import statistics
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ..artifacts import Comparison, EffectSize, StatsSettings
from .intervals import bootstrap_ci, eta_squared_ci
from .posthoc import dunn, games_howell, tukey_hsd


def anova(
    names: Sequence[str], samples: Sequence[Sequence[float]], settings: StatsSettings
) -> Comparison:
    """One-way ANOVA, with eta² (between-group over total sum of squares) and Tukey's HSD."""
    result = stats.f_oneway(*samples)
    f = float(result.statistic)
    n = sum(len(s) for s in samples)
    df1, df2 = len(samples) - 1, n - len(samples)
    low, high = eta_squared_ci(f, df1, df2, n, settings.ci_level)
    return Comparison(
        test="anova",
        statistic=f,
        df=(df1, df2),
        p=float(result.pvalue),
        effect=EffectSize(
            name="eta_squared", estimate=f * df1 / (f * df1 + df2), ci_low=low, ci_high=high
        ),
        posthoc="tukey_hsd",
        pairwise=tukey_hsd(names, samples, settings.ci_level),
    )


def welch_anova(
    names: Sequence[str], samples: Sequence[Sequence[float]], settings: StatsSettings
) -> Comparison:
    """Welch's one-way ANOVA, which weights each group by n / variance, with Games-Howell. The
    effect is the same descriptive eta² as for ANOVA; its interval is bootstrapped within the
    groups, since an interval from the noncentral F would assume the equal variances that
    Welch's test does not."""
    k = len(samples)
    n = [len(s) for s in samples]
    means = [statistics.fmean(s) for s in samples]
    weights = [n_i / statistics.variance(s) for n_i, s in zip(n, samples, strict=True)]
    total = sum(weights)
    centre = sum(w * m for w, m in zip(weights, means, strict=True)) / total
    between = sum(w * (m - centre) ** 2 for w, m in zip(weights, means, strict=True)) / (k - 1)
    spread = sum((1 - w / total) ** 2 / (n_i - 1) for w, n_i in zip(weights, n, strict=True))
    f = between / (1 + 2 * (k - 2) / (k**2 - 1) * spread)
    df1, df2 = k - 1, (k**2 - 1) / (3 * spread)
    low, high = bootstrap_ci(samples, _eta_squared, settings)
    estimate = float(_eta_squared(*(np.asarray(s, dtype=np.float64) for s in samples), axis=-1))
    return Comparison(
        test="welch_anova",
        statistic=f,
        df=(df1, df2),
        p=float(stats.f.sf(f, df1, df2)),
        effect=EffectSize(name="eta_squared", estimate=estimate, ci_low=low, ci_high=high),
        posthoc="games_howell",
        pairwise=games_howell(names, samples, settings.ci_level),
    )


def _eta_squared(*samples: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """The between-group sum of squares over the total, along `axis` of every sample."""
    pooled = np.concatenate(samples, axis=axis)
    grand = pooled.mean(axis=axis)
    between = np.zeros_like(grand)
    for s in samples:
        between += s.shape[axis] * (s.mean(axis=axis) - grand) ** 2
    total = ((pooled - np.expand_dims(grand, axis)) ** 2).sum(axis=axis)
    return between / total


def kruskal_wallis(
    names: Sequence[str], samples: Sequence[Sequence[float]], settings: StatsSettings
) -> Comparison:
    """The Kruskal-Wallis H test (tie-corrected), with epsilon² = H / (n - 1) and Dunn's test.
    epsilon² has no closed-form interval, so it is bootstrapped within the groups."""
    result = stats.kruskal(*samples)
    h = float(result.statistic)
    n = sum(len(s) for s in samples)

    def epsilon_squared(*resampled: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
        # A resample of all-equal values has no H (0 / 0); it separates no group, so 0.
        h = np.asarray(stats.kruskal(*resampled, axis=axis).statistic, dtype=np.float64)
        return np.nan_to_num(h / (n - 1))

    low, high = bootstrap_ci(samples, epsilon_squared, settings)
    return Comparison(
        test="kruskal_wallis",
        statistic=h,
        df=(len(samples) - 1,),
        p=float(result.pvalue),
        effect=EffectSize(name="epsilon_squared", estimate=h / (n - 1), ci_low=low, ci_high=high),
        posthoc="dunn",
        pairwise=dunn(names, samples),
    )
