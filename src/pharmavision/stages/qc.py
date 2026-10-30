"""QC (measures): deterministic flags and the exclusions they lead to, through `measure/qc.py`."""

from ..artifacts import RunFolder
from ..measure.qc import trial_qc


def run(run: RunFolder) -> None:
    arena = run.confirmed_arena()
    manifest = run.manifest()
    run.write_qc(trial_qc(run.track(), arena, run.metrics().gaps, manifest.video, manifest.qc))
