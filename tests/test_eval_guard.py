"""No Sturman label or DLC file can be committed: the ignore rule holds, and nothing like one is
tracked. The labels are GPL-3.0 and this repo is MIT."""

import shutil
import subprocess
from pathlib import Path

import pytest

from eval.guard import LabelsNotIgnoredError, check_labels_untracked

REPO = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fresh repo with this repo's .gitignore."""
    git(tmp_path, "init", "-q")
    shutil.copy(REPO / ".gitignore", tmp_path / ".gitignore")
    return tmp_path


def test_this_repo_keeps_the_eval_data_out() -> None:
    check_labels_untracked(REPO)


def test_a_repo_without_the_ignore_rule_is_refused(repo: Path) -> None:
    (repo / ".gitignore").write_text("runs/\n")

    with pytest.raises(LabelsNotIgnoredError, match="eval/data"):
        check_labels_untracked(repo)


def test_a_label_file_forced_past_the_ignore_rule_is_refused(repo: Path) -> None:
    labels = repo / "eval" / "data" / "labels" / "AllLabDataOFT_final.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text("x\n")
    git(repo, "add", "-f", str(labels))

    with pytest.raises(LabelsNotIgnoredError, match="AllLabDataOFT_final"):
        check_labels_untracked(repo)


def test_a_dlc_output_copied_elsewhere_is_refused(repo: Path) -> None:
    copied = repo / "tests" / "3_01_A_190507121945DeepCut_resnet50_Blockcourse1May9shuffle1.csv"
    copied.parent.mkdir()
    copied.write_text("x\n")
    git(repo, "add", str(copied))

    with pytest.raises(LabelsNotIgnoredError, match="DeepCut"):
        check_labels_untracked(repo)
