"""QC (measures): deterministic flags and the exclusions they lead to, through `measure/qc.py`."""

from ..artifacts import RunFolder
from ..measure.qc import qc_flags


def run(run: RunFolder) -> None:
    arena = run.confirmed_arena()
    manifest = run.manifest()
    run.write_qc(qc_flags(run.track(), arena, run.metrics().gaps, manifest.video, manifest.qc))
