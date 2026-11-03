"""Reading a body part's path out of DeepLabCut output. Files are synthetic, never Sturman data."""

from pathlib import Path

import pytest

from eval.dlc import DlcFileError, read_dlc

from .dlc_csv import write_dlc


def test_reads_the_body_centre_per_frame(tmp_path: Path) -> None:
    path = write_dlc(tmp_path / "dlc.csv", [(10.5, 20.0, 0.99), (11.0, 21.5, 0.2)])

    dlc = read_dlc(path)

    assert dlc.x.tolist() == [10.5, 11.0]
    assert dlc.y.tolist() == [20.0, 21.5]
    assert dlc.likelihood.tolist() == [0.99, 0.2]


def test_reads_another_body_part(tmp_path: Path) -> None:
    path = write_dlc(tmp_path / "dlc.csv", [(10.5, 20.0, 0.99)])

    assert read_dlc(path, "tailbase").x.tolist() == [7.0]


def test_a_part_the_network_did_not_track_is_refused(tmp_path: Path) -> None:
    path = write_dlc(tmp_path / "dlc.csv", [(10.5, 20.0, 0.99)])

    with pytest.raises(DlcFileError, match="nose"):
        read_dlc(path, "nose")


def test_rows_out_of_frame_order_are_refused(tmp_path: Path) -> None:
    path = write_dlc(tmp_path / "dlc.csv", [(1.0, 1.0, 1.0), (2.0, 2.0, 1.0)])
    lines = path.read_text().splitlines()
    path.write_text("\n".join([*lines[:3], lines[4], lines[3]]) + "\n")

    with pytest.raises(DlcFileError, match="frame"):
        read_dlc(path)
