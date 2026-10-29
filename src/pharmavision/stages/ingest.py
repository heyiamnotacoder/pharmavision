"""Ingest (measures): normalise the video with ffmpeg and record what came in."""

import hashlib
import json
import subprocess
from pathlib import Path

from ..artifacts import IngestSettings, NormalisedVideo, SourceVideo, VideoInfo
from ..templates import TestTemplate


class VideoTooShortError(ValueError):
    pass


def check_source(video: Path, template: TestTemplate) -> SourceVideo:
    """Probe and hash the source; reject it before any run folder exists if it is too short."""
    info, _ = _probe(video)
    if info.duration_s < template.protocol_duration_s - template.duration_tolerance_s:
        raise VideoTooShortError(
            f"{video.name} lasts {info.duration_s:.1f} s; "
            f"the protocol needs {template.protocol_duration_s:.0f} s"
        )
    return SourceVideo(**info.model_dump(), filename=video.name, sha256=_sha256(video))


def normalise(source: Path, dest: Path, settings: IngestSettings) -> NormalisedVideo:
    """H.264 MP4 at a fixed fps, no audio, downscaled (never upscaled) to fit the max side."""
    side = settings.max_side_px
    scale = (
        f"scale='min(iw,{side})':'min(ih,{side})'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-map", "0:v:0", "-an", "-vf", f"fps={settings.fps},{scale}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
            "-movflags", "+faststart", str(dest),
        ],
        check=True, capture_output=True,
    )  # fmt: skip
    info, n_frames = _probe(dest, count_frames=True)
    return NormalisedVideo(**info.model_dump(), n_frames=n_frames)


def _probe(video: Path, count_frames: bool = False) -> tuple[VideoInfo, int]:
    """Video stream info, plus the frame count when asked (0 otherwise; counting reads the
    whole file)."""
    count = ["-count_packets"] if count_frames else []
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", *count,
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets:format=duration",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    info = VideoInfo(
        width=stream["width"],
        height=stream["height"],
        fps=float(num) / float(den),
        duration_s=float(data["format"]["duration"]),
    )
    return info, int(stream.get("nb_read_packets", 0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
