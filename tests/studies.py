"""Fixture studies for the study-level seam: run folders holding the artifacts the run-level
stages write (manifest, metrics, QC), with metric values chosen per trial, and a study design
over them. The run-level tests check how those artifacts are made; these choose their numbers."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

from pharmavision.artifacts import (
    ArenaSettings,
    Events,
    Exclusion,
    IngestSettings,
    Manifest,
    MeasureSettings,
    Metrics,
    NormalisedVideo,
    Qc,
    QcSettings,
    RunFolder,
    SourceVideo,
    StudyFolder,
)

from .scenes import N_FRAMES, TEMPLATE

TRIAL_EXCLUSION = Exclusion(
    scope="trial",
    first_frame=0,
    last_frame=N_FRAMES - 1,
    reason="the animal was not found on 17% of frames (limit 5%)",
    source="rule",
)


def fixture_run(
    runs_dir: Path,
    run_id: str,
    metrics: Metrics,
    exclusions: Sequence[Exclusion] = (),
    measure: MeasureSettings | None = None,
) -> RunFolder:
    """A run folder as the QC stage leaves it, holding `metrics`, no behaviour events and
    `exclusions`."""
    run = RunFolder(runs_dir / run_id)
    run.path.mkdir()
    run.write_manifest(
        Manifest(
            run_id=run_id,
            created_at=datetime(2026, 10, 30, tzinfo=UTC),
            code_version="0.1.0",
            template=TEMPLATE,
            ingest=IngestSettings(),
            arena=ArenaSettings(),
            measure=measure or MeasureSettings(),
            qc=QcSettings(),
            source=SourceVideo(
                filename=f"{run_id}.mp4", sha256="0" * 64, width=360, height=360, fps=15,
                duration_s=4.0,
            ),
            video=NormalisedVideo(width=360, height=360, fps=15, duration_s=4.0, n_frames=60),
        )
    )  # fmt: skip
    run.write_metrics(metrics)
    run.write_events(Events(events=[]))
    run.write_qc(Qc(flags=[], exclusions=list(exclusions)))
    return run


def distance_only(distance_cm: float | None) -> Metrics:
    """Metrics with only the distance travelled known."""
    return Metrics(
        distance_travelled_cm=distance_cm,
        mean_velocity_cm_s=None,
        centre_time_s=None,
        centre_entries=None,
        thigmotaxis=None,
        immobility_s=None,
        gaps=[],
        rear_candidates=[],
    )


def write_study(
    tmp_path: Path,
    runs_dir: Path,
    distances: Mapping[str, Sequence[float | None]],
    excluded: frozenset[str] = frozenset(),
) -> StudyFolder:
    """A study with one group per key of `distances`, in that order, and one trial per value:
    run `<group>-<i>`, blinded as `B01`, `B02`, ... The runs named in `excluded` carry a trial
    exclusion."""
    trials: list[dict[str, str]] = []
    for group, values in distances.items():
        for i, distance in enumerate(values):
            run_id = f"{group}-{i}"
            exclusions = [TRIAL_EXCLUSION] if run_id in excluded else []
            fixture_run(runs_dir, run_id, distance_only(distance), exclusions)
            code = f"B{len(trials) + 1:02d}"
            trials.append({"run_id": run_id, "blinding_code": code, "group": group})
    design = {
        "name": "Fixture study",
        "groups": [
            {"name": group, "treatment": "saline" if k == 0 else "diazepam", "dose": k,
             "dose_unit": "mg/kg"}
            for k, group in enumerate(distances)
        ],
        "trials": trials,
    }  # fmt: skip
    study = StudyFolder(tmp_path / "study")
    study.path.mkdir()
    (study.path / "study.yaml").write_text(yaml.safe_dump(design, sort_keys=False))
    return study
