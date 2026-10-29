"""Write eval/split.json and fetch the train-split open-field videos (laptop only).

Usage: uv run python -m spike.data

Videos: Sturman et al. 2020, Zenodo record 3608658, CC BY 4.0. Labels: DLCAnalyzer (GPL-3.0),
read from the git-ignored eval/data/labels/; fetch them first with
  curl -sSL -o eval/data/labels/AllLabDataOFT_final.csv \\
    https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/master/data/OFT/Labels/AllLabDataOFT_final.csv
"""

import json
from pathlib import Path

import httpx

from spike.rears import SPLIT_SEED, load_rears, train_eval_split

LABELS = Path("eval/data/labels/AllLabDataOFT_final.csv")
VIDEOS = Path("eval/data/videos")
SPLIT = Path("eval/split.json")
ZENODO = "https://zenodo.org/records/3608658/files/{name}?download=1"


def write_split() -> list[str]:
    ids = sorted({r.video_id for r in load_rears(LABELS)})
    train, held_out = train_eval_split(ids)
    SPLIT.write_text(
        json.dumps({"seed": SPLIT_SEED, "train": train, "eval": held_out}, indent=2) + "\n"
    )
    return train


def download(video_id: str) -> Path:
    dest = VIDEOS / f"{video_id}.mp4"
    if dest.exists():
        return dest
    VIDEOS.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    url = ZENODO.format(name=dest.name)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    part.rename(dest)
    return dest


if __name__ == "__main__":
    for video_id in write_split():
        path = download(video_id)
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")
