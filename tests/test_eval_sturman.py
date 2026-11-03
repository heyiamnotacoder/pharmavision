"""The label file maps each video by its ID to its DeepLabCut output. Rows are synthetic, never
Sturman data."""

from pathlib import Path

import pytest

from eval.sturman import LabelFileError, dlc_files, rater_rears

HEADER = '"file";"Experimenter";"from";"to";"type";"ID";"DLCFile"\n'


def labels(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + "".join(f"{row}\n" for row in rows))
    return path


def test_each_video_id_maps_to_its_dlc_file_not_to_the_number_in_its_name(
    tmp_path: Path,
) -> None:
    path = labels(
        tmp_path,
        '"OFT_11_Jin.csv";"Jin";2.6;3.7;"Start/End";"OFT_11";"3_01_A_1DeepCut_x.csv"',
        '"OFT_11_OS.csv";"Oliver";4.8;6.5;"Supported";"OFT_11";"3_01_A_1DeepCut_x.csv"',
        '"OFT_5_FA.csv";"Furkan";1.0;2.0;"Unsupported";"OFT_5";"13_01_A_2DeepCut_x.csv"',
    )

    assert dlc_files(path) == {"OFT_11": "3_01_A_1DeepCut_x.csv", "OFT_5": "13_01_A_2DeepCut_x.csv"}


def test_a_video_with_two_dlc_files_is_refused(tmp_path: Path) -> None:
    path = labels(
        tmp_path,
        '"OFT_11_Jin.csv";"Jin";2.6;3.7;"Supported";"OFT_11";"a.csv"',
        '"OFT_11_OS.csv";"Oliver";4.8;6.5;"Supported";"OFT_11";"b.csv"',
    )

    with pytest.raises(LabelFileError, match="OFT_11"):
        dlc_files(path)


@pytest.mark.parametrize("name", ["../x.csv", "a/b.csv", ""])
def test_a_dlc_file_name_that_is_not_a_plain_file_name_is_refused(
    tmp_path: Path, name: str
) -> None:
    # The name becomes a download URL and a path under eval/data/dlc/.
    path = labels(tmp_path, f'"OFT_11_Jin.csv";"Jin";2.6;3.7;"Supported";"OFT_11";"{name}"')

    with pytest.raises(LabelFileError):
        dlc_files(path)


def test_each_raters_rears_are_read_by_video_with_none_where_they_marked_none(
    tmp_path: Path,
) -> None:
    path = labels(
        tmp_path,
        '"OFT_11_Jin.csv";"Jin";2.6;3.7;"Start/End";"OFT_11";"a.csv"',
        '"OFT_11_OS.csv";"Oliver";4.8;6.5;"Supported";"OFT_11";"a.csv"',
        '"OFT_11_OS.csv";"Oliver";1.0;1.5;"Unsupported";"OFT_11";"a.csv"',
        '"OFT_11_OS.csv";"Oliver";9.0;9.5;"Grooming";"OFT_11";"a.csv"',
        '"OFT_5_Jin.csv";"Jin";7.0;8.0;"Supported";"OFT_5";"b.csv"',
    )

    assert rater_rears(path) == {
        "OFT_11": {"Jin": [], "Oliver": [(1.0, 1.5), (4.8, 6.5)]},
        "OFT_5": {"Jin": [(7.0, 8.0)], "Oliver": []},
    }
