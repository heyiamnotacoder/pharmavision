"""The test-selection rule: Shapiro-Wilk on every group and Levene across them decide the test,
and the reason records what they gave."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats

from ..artifacts import Comparison, NormalityCheck, StatsSettings, TestName, VarianceCheck
from .many_groups import anova, kruskal_wallis, welch_anova
from .two_groups import mann_whitney, t_test


@dataclass(frozen=True)
class RuleOutcome:
    normality: list[NormalityCheck]
    variance: VarianceCheck | None
    reason: str
    comparison: Comparison | None


def compare(
    names: Sequence[str], samples: Sequence[Sequence[float]], settings: StatsSettings
) -> RuleOutcome:
    """Compare the groups' `samples`, given in the design's order (the first is the reference)."""
    groups = list(zip(names, samples, strict=True))
    small = [f"{name} has {len(s)}" for name, s in groups if len(s) < settings.min_group_n]
    if small:
        reason = (
            f"no test: {', '.join(small)} trials analysed, and the rule needs at least "
            f"{settings.min_group_n} in every group"
        )
        return RuleOutcome([], None, reason, None)
    pooled = {v for sample in samples for v in sample}
    if len(pooled) == 1:
        reason = f"no test: every trial analysed has the same value, {pooled.pop():g}"
        return RuleOutcome([], None, reason, None)
    flat = [name for name, sample in groups if len(set(sample)) == 1]
    normality = [_shapiro(name, sample) for name, sample in groups if name not in flat]
    levene = stats.levene(*samples, center="median")
    # Every group without spread leaves Levene 0 / 0: no check, and the rule has gone
    # non-parametric already.
    variance = (
        None
        if math.isnan(levene.pvalue)
        else VarianceCheck(statistic=float(levene.statistic), p=float(levene.pvalue))
    )
    alpha = settings.normality_alpha
    not_normal = [f"{name} has no spread, so Shapiro-Wilk cannot check it" for name in flat] + [
        f"Shapiro-Wilk p = {c.p:.3g} <= {alpha:g} in {c.group}" for c in normality if c.p <= alpha
    ]
    two = len(samples) == 2
    if not_normal or variance is None:  # no variance check means no spread anywhere
        test: TestName = "mann_whitney_u" if two else "kruskal_wallis"
        reason = f"{'; '.join(not_normal)}: {_LABELS[test]}"
    else:
        equal = variance.p > settings.variance_alpha
        test = ("student_t" if two else "anova") if equal else ("welch_t" if two else "welch_anova")
        reason = (
            f"Shapiro-Wilk p > {alpha:g} in every group and Levene p = {variance.p:.3g} "
            f"{'>' if equal else '<='} {settings.variance_alpha:g}: {_LABELS[test]}"
        )
    return RuleOutcome(normality, variance, reason, _run(test, names, samples, settings))


_LABELS: dict[TestName, str] = {
    "student_t": "Student's t-test",
    "welch_t": "Welch's t-test",
    "mann_whitney_u": "Mann-Whitney U test",
    "anova": "one-way ANOVA with Tukey's HSD",
    "welch_anova": "Welch's ANOVA with Games-Howell",
    "kruskal_wallis": "Kruskal-Wallis test with Dunn's test (Holm)",
}


def _run(
    test: TestName,
    names: Sequence[str],
    samples: Sequence[Sequence[float]],
    settings: StatsSettings,
) -> Comparison:
    match test:
        case "student_t" | "welch_t":
            return t_test(samples[0], samples[1], test == "student_t", settings)
        case "mann_whitney_u":
            return mann_whitney(samples[0], samples[1], settings)
        case "anova":
            return anova(names, samples, settings)
        case "welch_anova":
            return welch_anova(names, samples, settings)
        case "kruskal_wallis":
            return kruskal_wallis(names, samples, settings)


def _shapiro(name: str, sample: Sequence[float]) -> NormalityCheck:
    result = stats.shapiro(sample)
    return NormalityCheck(group=name, w=float(result.statistic), p=float(result.pvalue))
