"""The licence guard: the Sturman labels and DLCAnalyzer's DeepLabCut output are GPL-3.0 and
this repo is MIT, so they live in the git-ignored `eval/data/` and are never committed. Every
harness command runs this check first, and a test runs it in CI."""

import re
import subprocess
from pathlib import Path

DATA_DIR = "eval/data"
SAMPLE_PATHS = (
    f"{DATA_DIR}/labels/AllLabDataOFT_final.csv",
    f"{DATA_DIR}/dlc/3_01_A_190507121945DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
    f"{DATA_DIR}/videos/OFT_11.mp4",
)
"""Where the harness writes what it fetches; each must be ignored."""

LABEL_LIKE = re.compile(r"AllLabData|DeepCut|(^|/)OFT_\d+_(Jin|FA|OS)\w*\.(csv|json)$")
"""Names of the DLCAnalyzer files: the label table, DLC output, and the per-rater label files."""


class LabelsNotIgnoredError(RuntimeError):
    """A label or DLC file could be committed, or already is."""


def check_labels_untracked(repo: Path) -> None:
    """Raise unless git ignores the eval data folder and tracks nothing under it or named like
    a DLCAnalyzer file, staged files included."""
    for path in SAMPLE_PATHS:
        # --no-index: judge the ignore rules alone, even for a path forced into the index.
        ignored = _git(repo, "check-ignore", "-q", "--no-index", path, check=False)
        if ignored.returncode != 0:
            raise LabelsNotIgnoredError(f"{path} is not git-ignored: add {DATA_DIR}/ to .gitignore")
    tracked = _git(repo, "ls-files", "-z").stdout.split("\0")
    offending = [p for p in tracked if p.startswith(f"{DATA_DIR}/") or LABEL_LIKE.search(p)]
    if offending:
        raise LabelsNotIgnoredError(
            f"GPL-3.0 Sturman label or DLC files are tracked by git: {offending}; "
            f"`git rm --cached` them"
        )


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True
    )
