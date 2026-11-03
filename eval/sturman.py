"""The Sturman et al. 2020 open-field data, fetched at eval time into the git-ignored
`eval/data/` (see `eval.guard`):

- videos: Zenodo record 3608658, CC BY 4.0, one `<ID>.mp4` per video (e.g. `OFT_11.mp4`);
- human labels and DeepLabCut output: GitHub `ETHZ-INS/DLCAnalyzer`, GPL-3.0, read at a pinned
  commit so every run compares against the same files.

Videos are keyed by their label-file `ID` throughout. The DLC output is named by its recording
(`3_01_A_…DeepCut….csv` holds OFT_11), so it is found through the label file's `DLCFile` column,
never by the number in either name."""

import csv
from collections.abc import Callable
from pathlib import Path

import httpx

DATA = Path(__file__).parent / "data"
LABELS = DATA / "labels" / "AllLabDataOFT_final.csv"
DLC_DIR = DATA / "dlc"
VIDEOS = DATA / "videos"

DLCANALYZER_COMMIT = "cfdc34a85b6ce961bbddd8905a1a62b714dbc11a"  # master, checked Nov 3, 2026
_DLCANALYZER = f"https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/{DLCANALYZER_COMMIT}"
LABELS_URL = f"{_DLCANALYZER}/data/OFT/Labels/AllLabDataOFT_final.csv"
DLC_URL = _DLCANALYZER + "/data/OFT/Output_DLC/{name}"
VIDEO_URL = "https://zenodo.org/records/3608658/files/{video_id}.mp4?download=1"


class LabelFileError(ValueError):
    """The label file does not map each video to one DLC output file."""


def dlc_files(labels: Path) -> dict[str, str]:
    """Each labelled video's ID and the name of its DeepLabCut output file."""
    with labels.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    found: dict[str, set[str]] = {}
    for row in rows:
        found.setdefault(row["ID"], set()).add(row["DLCFile"])
    out: dict[str, str] = {}
    for video_id, names in sorted(found.items()):
        if len(names) != 1:
            raise LabelFileError(f"{video_id} has {len(names)} DLC files: {sorted(names)}")
        (name,) = names
        if not name or Path(name).name != name or name in {".", ".."}:
            raise LabelFileError(f"{video_id}: {name!r} is not a plain file name")
        out[video_id] = name
    return out


def fetch_labels() -> Path:
    return _fetch(LABELS_URL, LABELS)


def fetch_dlc(name: str) -> Path:
    return _fetch(DLC_URL.format(name=name), DLC_DIR / name)


def fetch_video(video_id: str) -> Path:
    return _fetch(VIDEO_URL.format(video_id=video_id), VIDEOS / f"{video_id}.mp4")


def _fetch(url: str, dest: Path, log: Callable[[str], None] = print) -> Path:
    """Download once: a file already there is reused, and a partial download never is."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with part.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    part.rename(dest)
    log(f"fetched {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest
