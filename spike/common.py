"""What the spike scripts share: the Meta Model API key, request shape and 15 fps encoding."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_URL = "https://api.meta.ai/v1"
FPS = 15  # ingest rate: a 10-min video is 9,000 frames, under SAM 3.1's 15,000 limit


def api_key() -> str:
    """MODEL_API_KEY from .env (laptop only, never committed)."""
    load_dotenv()
    return os.environ["MODEL_API_KEY"]


def message(*content: dict[str, Any]) -> list[dict[str, Any]]:
    """One user message for the Responses API. The request is untyped because `input_video`
    and SAM's `metadata` keys are Meta extensions the OpenAI SDK's types lack."""
    return [{"type": "message", "role": "user", "content": list(content)}]


def input_hash(*parts: bytes | str | dict[str, Any]) -> str:
    """Cache key for a model call: hash of everything that decides its answer."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, dict):
            part = json.dumps(part, sort_keys=True)
        h.update(part.encode() if isinstance(part, str) else part)
        h.update(b"\0")
    return h.hexdigest()[:16]


def encode(src: Path, dest: Path, start_s: float | None = None, end_s: float | None = None) -> Path:
    """Re-encode to FPS fps H.264 MP4 without audio (what ingest does), optionally one window."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    window: list[str] = []
    if start_s is not None and end_s is not None:
        window = ["-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *window, "-i", str(src), "-r", str(FPS),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(dest)],
        check=True,
    )  # fmt: skip
    return dest
