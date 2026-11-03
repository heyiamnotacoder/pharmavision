"""The committed eval split is the record: the harness checks it, never rewrites it."""

import json
from pathlib import Path

import pytest

from eval.split import SPLIT_PATH, Split, SplitMismatchError, check_split, load_split

# The 20 open-field video IDs of Sturman et al. 2020, as the label file's ID column names them.
STURMAN_OFT_IDS = [
    "OFT_11", "OFT_12", "OFT_14", "OFT_15", "OFT_16", "OFT_23", "OFT_24", "OFT_38", "OFT_39",
    "OFT_41", "OFT_43", "OFT_44", "OFT_49", "OFT_5", "OFT_50", "OFT_51", "OFT_52", "OFT_54",
    "OFT_58", "OFT_6",
]  # fmt: skip


def test_the_committed_split_holds_six_train_and_fourteen_eval_videos() -> None:
    split = load_split()

    assert SPLIT_PATH.parent.name == "eval"
    assert split.seed == 20261029
    assert split.train_ids == ("OFT_16", "OFT_38", "OFT_44", "OFT_50", "OFT_54", "OFT_58")
    assert len(split.eval_ids) == 14
    assert set(split.train_ids).isdisjoint(split.eval_ids)


def test_the_committed_split_is_the_seeded_draw_from_the_label_ids() -> None:
    # Label rows come in any order and repeat each ID: the draw sorts and dedupes first.
    check_split(load_split(), list(reversed(STURMAN_OFT_IDS)) * 3)


def test_a_label_file_naming_other_videos_is_refused() -> None:
    with pytest.raises(SplitMismatchError, match="OFT_99"):
        check_split(load_split(), [*STURMAN_OFT_IDS[1:], "OFT_99"])


def test_a_split_file_edited_by_hand_is_refused(tmp_path: Path) -> None:
    split = load_split()
    swapped = Split(
        seed=split.seed,
        train_ids=(split.eval_ids[0], *split.train_ids[1:]),
        eval_ids=(split.train_ids[0], *split.eval_ids[1:]),
    )

    with pytest.raises(SplitMismatchError, match="seed"):
        check_split(swapped, STURMAN_OFT_IDS)


def test_load_split_reads_the_file_format(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"seed": 1, "train": ["B", "A"], "eval": ["C"]}))

    assert load_split(path) == Split(seed=1, train_ids=("B", "A"), eval_ids=("C",))
