"""Study statistics through the study-level seam: the deterministic rule picks the test, and
every number matches a reference computed independently: by scipy or statsmodels called
directly on the fixture values, or a literal from pingouin 0.7.0 / scikit-posthocs 0.17.1
(which the project does not depend on), for the tests scipy lacks."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray
from pydantic import ValidationError
from scipy import stats as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from pharmavision.artifacts import MeasureSettings, MetricStats, StudyStats
from pharmavision.pipeline import StudyNotReadyError, study_stats

from .studies import TRIAL_EXCLUSION, distance_only, fixture_run, write_study

# Fixture distances (cm). Shapiro-Wilk p and Levene p (median-centred) as scipy gives them:
NORMAL_A = [10.1, 11.3, 9.8, 10.6, 12.0, 10.9]  # Shapiro 0.947
NORMAL_B = [12.4, 13.1, 11.8, 12.9, 14.0, 12.2]  # Shapiro 0.861; A vs B: Levene 0.948
NORMAL_C = [11.0, 11.9, 10.5, 12.3, 11.4, 11.7]  # Shapiro 0.976; A, B, C: Levene 0.868
TIGHT = [10.0, 10.2, 9.9, 10.1, 10.3, 9.8]  # Shapiro 0.961
WIDE = [9.0, 15.5, 12.0, 18.0, 6.5, 13.5]  # Shapiro 0.976; TIGHT vs WIDE: Levene 0.00765
NARROW = [11.2, 11.0, 11.4, 11.3, 11.1, 11.5]  # Shapiro 0.961; TIGHT, WIDE, NARROW: Levene 0.00112
OUTLIER = [10.0, 10.1, 10.2, 10.3, 10.4, 25.0]  # Shapiro 4.28e-05


def pairs(metric: MetricStats) -> list[tuple[str, str]]:
    assert metric.comparison is not None
    return [(c.group_a, c.group_b) for c in metric.comparison.pairwise]


def tukey_reference(
    groups: dict[str, list[float]],
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """statsmodels' Tukey HSD per pair (a, b): b minus a, adjusted p and the interval, both ways
    round. Its summary table is rounded to 4 decimals."""
    values = [v for vs in groups.values() for v in vs]
    labels = [g for g, vs in groups.items() for _ in vs]
    table = pairwise_tukeyhsd(values, labels, alpha=0.05).summary().data[1:]
    reference: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for a, b, difference, p, low, high, _ in table:
        reference[(a, b)] = (difference, p, low, high)
        reference[(b, a)] = (-difference, p, -high, -low)
    return reference


def eta_squared(*samples: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Between-group over total sum of squares, along `axis` of each sample (for st.bootstrap)."""
    pooled = np.concatenate(samples, axis=axis)
    grand = pooled.mean(axis=axis, keepdims=True)
    between = sum(s.shape[axis] * (s.mean(axis=axis) - grand.squeeze(axis)) ** 2 for s in samples)
    return between / ((pooled - grand) ** 2).sum(axis=axis)


def distance(stats: StudyStats) -> MetricStats:
    [metric] = [m for m in stats.metrics if m.metric == "distance_travelled_cm"]
    return metric


def test_two_normal_groups_with_equal_variances_get_students_t_test(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": NORMAL_A, "diazepam": NORMAL_B})

    stats = study_stats(study, runs_dir)

    metric = distance(stats)
    assert metric.comparison is not None
    reference = st.ttest_ind(NORMAL_B, NORMAL_A)
    assert metric.comparison.test == "student_t"
    assert "Shapiro-Wilk" in metric.reason and "Levene p = 0.948" in metric.reason
    assert metric.comparison.statistic == pytest.approx(reference.statistic)
    assert metric.comparison.df == pytest.approx((10,))
    assert metric.comparison.p == pytest.approx(reference.pvalue)
    effect = metric.comparison.effect
    ci = reference.confidence_interval(0.95)
    assert effect.name == "mean_difference"
    assert (effect.estimate, effect.ci_low, effect.ci_high) == pytest.approx(
        (sum(NORMAL_B) / 6 - sum(NORMAL_A) / 6, ci.low, ci.high)
    )
    assert [(g.group, g.n) for g in metric.groups] == [("vehicle", 6), ("diazepam", 6)]
    assert study.stats() == stats


def test_two_normal_groups_with_unequal_variances_get_welchs_t_test(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": TIGHT, "diazepam": WIDE})

    metric = distance(study_stats(study, runs_dir))

    assert metric.comparison is not None
    reference = st.ttest_ind(WIDE, TIGHT, equal_var=False)
    assert metric.comparison.test == "welch_t"
    assert "Levene p = 0.00765" in metric.reason
    assert metric.variance is not None
    assert metric.variance.p == pytest.approx(st.levene(TIGHT, WIDE).pvalue)
    assert (metric.comparison.statistic, metric.comparison.p) == pytest.approx(
        (reference.statistic, reference.pvalue)
    )
    assert metric.comparison.df == pytest.approx((reference.df,))
    ci = reference.confidence_interval(0.95)
    effect = metric.comparison.effect
    assert (effect.ci_low, effect.ci_high) == pytest.approx((ci.low, ci.high))


def test_a_group_that_is_not_normal_gets_the_mann_whitney_u_test(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": OUTLIER, "diazepam": NORMAL_B})

    metric = distance(study_stats(study, runs_dir))

    assert metric.comparison is not None
    assert metric.comparison.test == "mann_whitney_u"
    assert "Shapiro-Wilk p = 4.28e-05 <= 0.05 in vehicle" in metric.reason
    # pingouin.mwu(NORMAL_B, OUTLIER): U 30, exact p 0.0649350649, rank-biserial 0.6666666667.
    assert metric.comparison.statistic == 30
    assert metric.comparison.df == ()
    assert metric.comparison.p == pytest.approx(0.0649350649)
    effect = metric.comparison.effect
    assert effect.name == "rank_biserial"
    assert effect.estimate == pytest.approx(0.6666666667)
    # Its interval through Fisher's z, with standard error sqrt((n1 + n2 + 1) / (3 n1 n2)).
    half = st.norm.ppf(0.975) * math.sqrt(13 / (3 * 6 * 6))
    z = math.atanh(2 / 3)
    assert (effect.ci_low, effect.ci_high) == pytest.approx(
        (math.tanh(z - half), math.tanh(z + half))
    )


def test_three_normal_groups_with_equal_variances_get_anova_and_tukeys_hsd(
    tmp_path: Path, runs_dir: Path
) -> None:
    groups = {"vehicle": NORMAL_A, "low": NORMAL_B, "high": NORMAL_C}
    study = write_study(tmp_path, runs_dir, groups)

    metric = distance(study_stats(study, runs_dir))

    assert metric.comparison is not None
    assert metric.comparison.test == "anova"
    assert metric.comparison.posthoc == "tukey_hsd"
    assert "Levene p = 0.868 > 0.05" in metric.reason
    reference = anova_oneway(list(groups.values()), use_var="equal")
    assert (metric.comparison.statistic, metric.comparison.p) == pytest.approx(
        (reference.statistic, reference.pvalue)
    )
    assert metric.comparison.df == pytest.approx((2, 15))
    effect = metric.comparison.effect
    assert effect.name == "eta_squared"
    assert effect.estimate == pytest.approx(0.5843857952)  # pingouin.anova np2
    # The interval is of the noncentrality at which F is the 97.5th and 2.5th percentile, put
    # on the estimate's scale as nc / (nc + df2), as R's effectsize does.
    for eta_squared, q in ((effect.ci_low, 0.975), (effect.ci_high, 0.025)):
        noncentrality = 15 * eta_squared / (1 - eta_squared)
        assert st.ncf.cdf(reference.statistic, 2, 15, noncentrality) == pytest.approx(q, abs=1e-9)
    # Every pair in design order, later group minus earlier, against statsmodels' Tukey HSD.
    assert pairs(metric) == [("vehicle", "low"), ("vehicle", "high"), ("low", "high")]
    tukey = tukey_reference(groups)
    for pair in metric.comparison.pairwise:
        difference, p, low, high = tukey[(pair.group_a, pair.group_b)]
        assert (pair.difference, pair.p_adjusted, pair.ci_low, pair.ci_high) == pytest.approx(
            (difference, p, low, high), abs=1e-4
        )
    # pingouin.pairwise_tukey T, the difference over its standard error: low vs vehicle 4.5256.
    assert metric.comparison.pairwise[0].statistic == pytest.approx(4.5255111696)


def test_three_normal_groups_with_unequal_variances_get_welchs_anova_and_games_howell(
    tmp_path: Path, runs_dir: Path
) -> None:
    groups = {"vehicle": TIGHT, "low": WIDE, "high": NARROW}
    study = write_study(tmp_path, runs_dir, groups)

    metric = distance(study_stats(study, runs_dir))

    assert metric.comparison is not None
    assert metric.comparison.test == "welch_anova"
    assert metric.comparison.posthoc == "games_howell"
    assert "Levene p = 0.00112 <= 0.05" in metric.reason
    # pingouin.welch_anova: F 57.9017430598 on 2 and 8.8947235314 df, p 7.9449e-06, np2 0.1587.
    assert metric.comparison.statistic == pytest.approx(57.9017430598)
    assert metric.comparison.df == pytest.approx((2, 8.8947235314))
    assert metric.comparison.p == pytest.approx(7.9449e-06, rel=1e-4)
    effect = metric.comparison.effect
    assert effect.name == "eta_squared"
    assert effect.estimate == pytest.approx(0.1587379889)
    # Its interval by percentile bootstrap within each group: Welch's F assumes no equal
    # variances, and neither does the interval.
    samples = tuple(np.array(v) for v in groups.values())
    ci = st.bootstrap(
        samples, eta_squared, n_resamples=9999, method="percentile", rng=np.random.default_rng(0)
    ).confidence_interval
    assert (effect.ci_low, effect.ci_high) == pytest.approx((ci.low, ci.high))
    # pingouin.pairwise_gameshowell, as (difference, T, df, p) with the sign turned to later
    # group minus earlier; the interval is the difference -/+ q(0.95, 3, df) / sqrt(2) x se.
    assert pairs(metric) == [("vehicle", "low"), ("vehicle", "high"), ("low", "high")]
    reference = [
        (2.3666666667, 1.3749520558, 5.0197274949, 0.4194086886, 1.7212721394),
        (1.2000000000, 11.1098411973, 10.0, 0.0000016394, 0.1080123450),
        (-1.1666666667, -0.6777932670, 5.0197274949, 0.7858251427, 1.7212721394),
    ]
    for pair, (difference, t, df, p, se) in zip(metric.comparison.pairwise, reference, strict=True):
        half = st.studentized_range.ppf(0.95, 3, df) / math.sqrt(2) * se
        assert (pair.difference, pair.statistic) == pytest.approx((difference, t))
        assert pair.p_adjusted == pytest.approx(p, rel=1e-4)
        assert (pair.ci_low, pair.ci_high) == pytest.approx((difference - half, difference + half))


def test_three_groups_one_not_normal_get_kruskal_wallis_and_dunns_test(
    tmp_path: Path, runs_dir: Path
) -> None:
    groups = {"vehicle": OUTLIER, "low": NORMAL_B, "high": NORMAL_C}
    study = write_study(tmp_path, runs_dir, groups)

    metric = distance(study_stats(study, runs_dir))

    assert metric.comparison is not None
    assert metric.comparison.test == "kruskal_wallis"
    assert metric.comparison.posthoc == "dunn"
    assert "Shapiro-Wilk p = 4.28e-05 <= 0.05 in vehicle" in metric.reason
    # pingouin.kruskal: H 7.6842105263 on 2 df, p 0.0214483993.
    assert metric.comparison.statistic == pytest.approx(7.6842105263)
    assert metric.comparison.df == (2,)
    assert metric.comparison.p == pytest.approx(0.0214483993)
    # epsilon² = H / (n - 1), its interval by percentile bootstrap within each group.
    effect = metric.comparison.effect
    assert effect.name == "epsilon_squared"
    assert effect.estimate == pytest.approx(7.6842105263 / 17)

    def epsilon_squared(*samples: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
        return np.asarray(st.kruskal(*samples, axis=axis).statistic) / 17

    samples = tuple(np.array(v) for v in groups.values())
    ci = st.bootstrap(
        samples, epsilon_squared, n_resamples=9999, method="percentile",
        rng=np.random.default_rng(0),
    ).confidence_interval  # fmt: skip
    assert (effect.ci_low, effect.ci_high) == pytest.approx((ci.low, ci.high))
    # Dunn's test: the difference in mean ranks over all 18 trials, Holm-adjusted p as
    # scikit_posthocs.posthoc_dunn(p_adjust="holm") gives them, and no interval.
    assert pairs(metric) == [("vehicle", "low"), ("vehicle", "high"), ("low", "high")]
    ranks = st.rankdata([v for vs in groups.values() for v in vs]).reshape(3, 6).mean(axis=1)
    holm = [0.0174594509, 0.2561449666, 0.2095149797]
    differences = [ranks[1] - ranks[0], ranks[2] - ranks[0], ranks[2] - ranks[1]]
    for pair, difference, p in zip(metric.comparison.pairwise, differences, holm, strict=True):
        assert (pair.difference, pair.p_adjusted) == pytest.approx((difference, p))
        assert (pair.ci_low, pair.ci_high) == (None, None)


def test_excluded_trials_are_left_out_and_n_after_exclusions_is_reported(
    tmp_path: Path, runs_dir: Path
) -> None:
    # A seventh vehicle trial at 40 cm, excluded by QC, and an eighth whose distance is unknown.
    vehicle = [*NORMAL_A, 40.0, None]
    study = write_study(
        tmp_path, runs_dir, {"vehicle": vehicle, "diazepam": NORMAL_B},
        excluded=frozenset({"vehicle-6"}),
    )  # fmt: skip

    stats = study_stats(study, runs_dir)

    metric = distance(stats)
    assert metric.comparison is not None
    assert metric.comparison.p == pytest.approx(st.ttest_ind(NORMAL_B, NORMAL_A).pvalue)
    [vehicle_n, diazepam_n] = metric.groups
    assert (vehicle_n.n, vehicle_n.n_excluded, vehicle_n.n_missing) == (6, 1, 1)
    assert (diazepam_n.n, diazepam_n.n_excluded, diazepam_n.n_missing) == (6, 0, 0)
    assert vehicle_n.mean == pytest.approx(sum(NORMAL_A) / 6)
    # The excluded trial stays in the per-trial table, with its value and the reason.
    [row] = [r for r in study.trials().trials if r.run_id == "vehicle-6"]
    assert row.excluded
    assert row.values["distance_travelled_cm"] == 40.0
    assert row.exclusions == [TRIAL_EXCLUSION]
    assert [r.blinding_code for r in study.trials().trials] == [f"B{i:02d}" for i in range(1, 15)]


def test_the_rule_abstains_when_a_group_has_too_few_trials_left(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(
        tmp_path, runs_dir, {"vehicle": [10.0, 11.0, 12.0], "diazepam": NORMAL_B},
        excluded=frozenset({"vehicle-0"}),
    )  # fmt: skip

    stats = study_stats(study, runs_dir)

    metric = distance(stats)
    assert metric.comparison is None
    assert metric.reason == (
        "no test: vehicle has 2 trials analysed, and the rule needs at least 3 in every group"
    )
    assert [(g.n, g.n_excluded) for g in metric.groups] == [(2, 1), (6, 0)]
    # A metric no trial has abstains the same way, rather than reporting zero.
    [velocity] = [m for m in stats.metrics if m.metric == "mean_velocity_cm_s"]
    assert velocity.comparison is None
    assert [g.mean for g in velocity.groups] == [None, None]


def test_a_group_with_no_spread_cannot_be_checked_for_normality(
    tmp_path: Path, runs_dir: Path
) -> None:
    # No vehicle animal entered the centre zone, say: every value the same.
    study = write_study(tmp_path, runs_dir, {"vehicle": [0.0] * 6, "diazepam": NORMAL_B})

    metric = distance(study_stats(study, runs_dir))

    assert metric.comparison is not None
    assert metric.comparison.test == "mann_whitney_u"
    assert "vehicle has no spread, so Shapiro-Wilk cannot check it" in metric.reason
    assert [c.group for c in metric.normality] == ["diazepam"]


def test_the_rule_abstains_when_every_trial_has_the_same_value(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": [0.0] * 6, "diazepam": [0.0] * 6})

    metric = distance(study_stats(study, runs_dir))

    assert metric.comparison is None
    assert metric.reason == "no test: every trial analysed has the same value, 0"


def test_refuses_a_study_with_a_run_that_has_not_finished_qc(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": NORMAL_A, "diazepam": NORMAL_B})
    (runs_dir / "diazepam-2" / "qc.json").unlink()

    with pytest.raises(StudyNotReadyError, match="B09: run diazepam-2 has not finished QC"):
        study_stats(study, runs_dir)

    assert sorted(p.name for p in study.path.iterdir()) == ["study.yaml"]


def test_refuses_to_compare_trials_measured_with_different_settings(
    tmp_path: Path, runs_dir: Path
) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": NORMAL_A, "diazepam": NORMAL_B})
    # Re-measure one trial under a different zone hysteresis.
    (runs_dir / "diazepam-0").rename(tmp_path / "old")
    fixture_run(
        runs_dir, "diazepam-0", distance_only(12.4), measure=MeasureSettings(zone_hysteresis_cm=2)
    )

    with pytest.raises(StudyNotReadyError, match="different measure or QC settings: B07"):
        study_stats(study, runs_dir)

    assert sorted(p.name for p in study.path.iterdir()) == ["study.yaml"]


def test_refuses_a_design_whose_trial_names_no_group_of_it(tmp_path: Path, runs_dir: Path) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": NORMAL_A, "diazepam": NORMAL_B})
    design = yaml.safe_load((study.path / "study.yaml").read_text())
    design["trials"][0]["group"] = "vehicel"
    (study.path / "study.yaml").write_text(yaml.safe_dump(design))

    with pytest.raises(ValidationError, match="trials in unknown groups: vehicel"):
        study_stats(study, runs_dir)


@pytest.mark.parametrize(
    ("groups", "test"),
    [
        # Counts, as centre entries are: some resamples hold one value only, with no eta².
        ([[5.0, 6, 4, 5], [5, 10, 9, 5], [5, 6, 3, 5]], "welch_anova"),
        # An effect so large that an interval on another scale than the estimate's misses it.
        ([[1.0, 1.1, 0.9], [5, 5.1, 4.9], [9, 9.1, 8.9]], "anova"),
    ],
)
def test_the_effect_size_interval_is_finite_and_holds_its_estimate(
    tmp_path: Path, runs_dir: Path, groups: list[list[float]], test: str
) -> None:
    study = write_study(
        tmp_path, runs_dir, dict(zip(["vehicle", "low", "high"], groups, strict=True))
    )

    stats = study_stats(study, runs_dir)

    metric = distance(stats)
    assert metric.comparison is not None
    assert metric.comparison.test == test
    effect = metric.comparison.effect
    assert effect.ci_low <= effect.estimate <= effect.ci_high
    assert study.stats() == stats


def test_the_same_study_gives_the_same_stats(tmp_path: Path, runs_dir: Path) -> None:
    study = write_study(tmp_path, runs_dir, {"vehicle": TIGHT, "low": WIDE, "high": NARROW})

    study_stats(study, runs_dir)
    first = (study.path / "stats.json").read_bytes()
    study_stats(study, runs_dir)

    assert (study.path / "stats.json").read_bytes() == first
