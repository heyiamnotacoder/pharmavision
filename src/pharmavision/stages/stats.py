"""Stats (measures), at study level: the study design and its finished runs to the per-trial
metrics table, then to the stats, through the pure functions in `stats/`."""

from pathlib import Path

from ..artifacts import (
    METRICS,
    RunFolder,
    StatsSettings,
    StudyFolder,
    StudyNotReadyError,
    StudyStats,
    StudyTrial,
    TrialRow,
    TrialTable,
)
from ..stats.study import table_stats


def run(study: StudyFolder, runs_dir: Path, settings: StatsSettings) -> StudyStats:
    design = study.design()
    runs = [RunFolder(runs_dir / t.run_id) for t in design.trials]
    table = TrialTable(trials=[_row(run, t) for run, t in zip(runs, design.trials, strict=True)])
    _check_same_settings(runs, design.trials)
    result = table_stats(table, design, settings)
    study.write_trials(table)
    study.write_stats(result)
    return result


def _check_same_settings(runs: list[RunFolder], trials: list[StudyTrial]) -> None:
    """Metrics measured under different thresholds are not alike, so they are never compared."""
    settings = [(m.measure, m.qc) for m in (run.manifest() for run in runs)]
    differing = [t.blinding_code for t, s in zip(trials, settings, strict=True) if s != settings[0]]
    if differing:
        raise StudyNotReadyError(
            f"trials measured with different measure or QC settings: {', '.join(differing)} "
            f"against {trials[0].blinding_code}"
        )


def _row(run: RunFolder, trial: StudyTrial) -> TrialRow:
    try:
        metrics = run.metrics()
        qc = run.qc()
    except FileNotFoundError as missing:
        raise StudyNotReadyError(
            f"trial {trial.blinding_code}: run {trial.run_id} has not finished QC"
        ) from missing
    return TrialRow(
        run_id=trial.run_id,
        blinding_code=trial.blinding_code,
        group=trial.group,
        values={m: getattr(metrics, m) for m in METRICS},
        exclusions=qc.exclusions,
    )
