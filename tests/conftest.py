"""Shared fixtures for tests at the pipeline seam."""

from pathlib import Path

import pytest


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "runs"
    path.mkdir()
    return path
