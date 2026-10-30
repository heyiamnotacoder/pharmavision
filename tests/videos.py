"""Test videos, made with ffmpeg."""

import subprocess
from pathlib import Path


def make_video(
    path: Path,
    seconds: float,
    fps: int = 25,
    size: int = 360,
    height: int | None = None,
    floor_box: tuple[int, int, int, int] | None = None,
) -> Path:
    """A plain grey clip with an audio track, as a phone recording would have. Square unless
    `height` is given. `floor_box` (x, y, w, h) draws a floor barely lighter than the walls."""
    source = f"color=c=gray:s={size}x{height or size}:r={fps}:d={seconds}"
    if floor_box:
        x, y, w, h = floor_box
        source += f",drawbox=x={x}:y={y}:w={w}:h={h}:color=0x888888:t=fill"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", source,
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path
