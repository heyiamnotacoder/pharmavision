"""The per-trial metrics table to the stats: each metric's values per group, after exclusions,
through the test-selection rule."""

import statistics
from collections.abc import Sequence

from ..artifacts import (
    METRIC_UNITS,
    METRICS,
    GroupSummary,
    MetricName,
    MetricStats,
    StatsSettings,
    StudyDesign,
    StudyStats,
    TrialRow,
    TrialTable,
)
from .rule import compare


def table_stats(table: TrialTable, design: StudyDesign, settings: StatsSettings) -> StudyStats:
    """Every metric compared between the design's groups, in the design's group order."""
    return StudyStats(
        settings=settings,
        metrics=[_metric_stats(metric, table, design, settings) for metric in METRICS],
    )


def _metric_stats(
    metric: MetricName, table: TrialTable, design: StudyDesign, settings: StatsSettings
) -> MetricStats:
    names = [g.name for g in design.groups]
    rows = {name: [r for r in table.trials if r.group == name] for name in names}
    samples = {name: _analysed(metric, rows[name]) for name in names}
    outcome = compare(names, [samples[name] for name in names], settings)
    return MetricStats(
        metric=metric,
        unit=METRIC_UNITS[metric],
        groups=[_summary(name, metric, rows[name], samples[name]) for name in names],
        normality=outcome.normality,
        variance=outcome.variance,
        reason=outcome.reason,
        comparison=outcome.comparison,
    )


def _analysed(metric: MetricName, rows: Sequence[TrialRow]) -> list[float]:
    return [v for r in rows if (v := r.analysed_value(metric)) is not None]


def _summary(
    group: str, metric: MetricName, rows: Sequence[TrialRow], values: Sequence[float]
) -> GroupSummary:
    kept = [r for r in rows if not r.excluded]
    return GroupSummary(
        group=group,
        n=len(values),
        n_excluded=len(rows) - len(kept),
        n_missing=sum(1 for r in kept if r.values[metric] is None),
        mean=statistics.fmean(values) if values else None,
        sd=statistics.stdev(values) if len(values) >= 2 else None,
        median=statistics.median(values) if values else None,
    )
