"""Critical values and the intervals no scipy function gives directly."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from ..artifacts import StatsSettings

Statistic = Callable[..., NDArray[np.float64]]
"""A statistic of several samples along their last axis, as `scipy.stats.bootstrap` calls it
vectorised: `statistic(*samples, axis=-1)`."""


def z_critical(level: float) -> float:
    """The two-sided critical value of the standard normal at `level`."""
    return float(stats.norm.ppf((1 + level) / 2))


def eta_squared_ci(f: float, df1: float, df2: float, n: int, level: float) -> tuple[float, float]:
    """The interval of eta² from an F statistic, through its noncentrality: the noncentralities
    at which `f` is the upper and lower (1 - level) / 2 quantile, each as nc / (nc + n). An end
    whose noncentrality would be negative is 0."""
    tail = (1 - level) / 2
    central = float(stats.f.cdf(f, df1, df2))

    def noncentrality(p: float) -> float:
        return float(special.ncfdtrinc(df1, df2, p, f)) if central > p else 0.0

    low, high = noncentrality(1 - tail), noncentrality(tail)
    return low / (low + n), high / (high + n)


def bootstrap_ci(
    samples: Sequence[Sequence[float]], statistic: Statistic, settings: StatsSettings
) -> tuple[float, float]:
    """The percentile bootstrap interval of `statistic`, resampling within each group, with the
    settings' fixed seed so the same values give the same interval."""
    result = stats.bootstrap(
        tuple(np.asarray(s, dtype=np.float64) for s in samples),
        statistic,
        n_resamples=settings.bootstrap_resamples,
        confidence_level=settings.ci_level,
        method="percentile",
        rng=np.random.default_rng(settings.bootstrap_seed),
    )
    ci = result.confidence_interval
    return float(ci.low), float(ci.high)
