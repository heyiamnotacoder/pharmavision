"""Test videos, made with ffmpeg."""

import subprocess
from pathlib import Path


def make_video(path: Path, seconds: float, fps: int = 25, size: int = 360) -> Path:
    """A plain grey clip with an audio track, as a phone recording would have."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s={size}x{size}:r={fps}:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path
