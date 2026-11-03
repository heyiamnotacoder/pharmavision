"""Check eval/split.json and fetch the train-split open-field videos (laptop only).

Usage: uv run python -m spike.data

Videos: Sturman et al. 2020, Zenodo record 3608658, CC BY 4.0. Labels: DLCAnalyzer (GPL-3.0),
read from the git-ignored eval/data/labels/; `python -m eval fetch` fetches them.
"""

from pathlib import Path

from eval.split import SPLIT_PATH as SPLIT
from eval.split import check_split, load_split
from eval.sturman import LABELS, VIDEOS, fetch_video
from spike.rears import load_rears

__all__ = ["LABELS", "SPLIT", "VIDEOS", "checked_train_ids", "download"]


def checked_train_ids() -> list[str]:
    """The committed split's train videos, once it is checked against the labels."""
    split = load_split(SPLIT)
    check_split(split, (r.video_id for r in load_rears(LABELS)))
    return list(split.train_ids)


def download(video_id: str) -> Path:
    return fetch_video(video_id)


if __name__ == "__main__":
    for video_id in checked_train_ids():
        path = download(video_id)
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")
