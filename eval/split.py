"""The fixed 6 train / 14 eval split of the 20 open-field videos. `split.json` is the record:
it was drawn once with `SPLIT_SEED` and committed, and the harness only checks it, never
rewrites it. Few-shot examples and fitted settings come from the train videos only."""

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SPLIT_PATH = Path(__file__).parent / "split.json"
SPLIT_SEED = 20261029  # the spike date
N_TRAIN = 6


class SplitMismatchError(RuntimeError):
    """The committed split does not match the labelled videos or its own seeded draw."""


@dataclass(frozen=True)
class Split:
    seed: int
    train_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]


def train_eval_split(
    video_ids: Iterable[str], n_train: int = N_TRAIN, seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """A seeded random draw of `n_train` videos; the rest are the eval videos. Both sorted."""
    ids = sorted(set(video_ids))
    train = sorted(random.Random(seed).sample(ids, n_train))
    return train, [i for i in ids if i not in train]


def load_split(path: Path = SPLIT_PATH) -> Split:
    data = json.loads(path.read_text())
    return Split(seed=data["seed"], train_ids=tuple(data["train"]), eval_ids=tuple(data["eval"]))


def check_split(split: Split, labelled_ids: Iterable[str]) -> None:
    """Raise unless the split covers exactly the labelled videos (IDs as the label file's `ID`
    column names them) and is the draw its seed gives from them."""
    ids = set(labelled_ids)
    recorded = set(split.train_ids) | set(split.eval_ids)
    if ids != recorded:
        raise SplitMismatchError(
            f"split.json lists {sorted(recorded - ids)} not in the labels, "
            f"and the labels hold {sorted(ids - recorded)} not in split.json"
        )
    train, held_out = train_eval_split(ids, len(split.train_ids), split.seed)
    if (tuple(train), tuple(held_out)) != (split.train_ids, split.eval_ids):
        raise SplitMismatchError(
            f"split.json is not the draw its seed {split.seed} gives: train {train}"
        )
