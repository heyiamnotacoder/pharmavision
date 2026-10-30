"""Critical values and the intervals no scipy function gives directly."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from ..artifacts import StatsSettings

Statistic = Callable[..., NDArray[np.float64]]
"""A statistic of several samples along their last axis, as `scipy.stats.bootstrap` calls it
vectorised: `statistic(*samples, axis=-1)`."""


def z_critical(level: float) -> float:
    """The two-sided critical value of the standard normal at `level`."""
    return float(stats.norm.ppf((1 + level) / 2))


def eta_squared_ci(f: float, df1: float, df2: float, level: float) -> tuple[float, float]:
    """The interval of eta² from an F statistic, through its noncentrality: the noncentralities
    at which `f` is the upper and lower (1 - level) / 2 quantile, each put on the estimate's
    scale as nc / (nc + df2), as R's `effectsize` does (eta² = F df1 / (F df1 + df2), with
    F df1 for nc). On the population scale, nc / (nc + n), a large effect's estimate can fall
    outside its own interval. An end whose noncentrality would be negative is 0."""
    tail = (1 - level) / 2
    central = float(stats.f.cdf(f, df1, df2))

    def excess(nc: float, p: float) -> float:
        return float(stats.ncf.cdf(f, df1, df2, nc)) - p

    def noncentrality(p: float) -> float:
        # Root-found on the cdf: scipy.special.ncfdtrinc misses it by about 1e-6.
        if central <= p:
            return 0.0
        high = max(1.0, f * df1)
        while excess(high, p) > 0:
            high *= 2
        return float(optimize.brentq(excess, 0.0, high, args=(p,)))

    low, high = noncentrality(1 - tail), noncentrality(tail)
    return low / (low + df2), high / (high + df2)


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
