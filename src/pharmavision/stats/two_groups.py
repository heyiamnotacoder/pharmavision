"""Two groups: the later group `b` against the reference `a`."""

import math
import statistics
from collections.abc import Sequence

from scipy import stats

from ..artifacts import Comparison, EffectSize, StatsSettings
from .intervals import z_critical


def t_test(
    a: Sequence[float], b: Sequence[float], equal_var: bool, settings: StatsSettings
) -> Comparison:
    """Student's or Welch's t-test of `b` against `a`, with the difference in means and its
    interval from the same t distribution."""
    result = stats.ttest_ind(b, a, equal_var=equal_var)
    ci = result.confidence_interval(settings.ci_level)
    estimate = statistics.fmean(b) - statistics.fmean(a)
    return Comparison(
        test="student_t" if equal_var else "welch_t",
        statistic=float(result.statistic),
        df=(float(result.df),),
        p=float(result.pvalue),
        effect=EffectSize(
            name="mean_difference",
            estimate=estimate,
            ci_low=float(ci.low),
            ci_high=float(ci.high),
        ),
    )


def mann_whitney(a: Sequence[float], b: Sequence[float], settings: StatsSettings) -> Comparison:
    """The Mann-Whitney U test of `b` against `a` (exact p when scipy can), with the rank-biserial
    correlation and its interval through Fisher's z. Groups that do not overlap give r = ±1,
    where z is infinite and the interval shrinks to r itself, as R's `effectsize` reports it."""
    result = stats.mannwhitneyu(b, a, alternative="two-sided")
    u = float(result.statistic)
    n_a, n_b = len(a), len(b)
    r = 2 * u / (n_a * n_b) - 1
    half = z_critical(settings.ci_level) * math.sqrt((n_a + n_b + 1) / (3 * n_a * n_b))
    z = math.atanh(r) if abs(r) < 1 else math.copysign(math.inf, r)
    return Comparison(
        test="mann_whitney_u",
        statistic=u,
        df=(),
        p=float(result.pvalue),
        effect=EffectSize(
            name="rank_biserial",
            estimate=r,
            ci_low=math.tanh(z - half),
            ci_high=math.tanh(z + half),
        ),
    )
