"""Push one Sturman open-field video through SAM 3.1 and measure cost and wall-clock (laptop only).

Usage: uv run python -m spike.sam_run OFT_16

Steps: normalise to 15 fps H.264 (as ingest will) → background frame → arena prompt on it →
upload → track "mouse". Every Responses event is recorded to eval/data/spike/sam/ as gzipped
JSONL, then replayed through meta-sam-parser, so the replay path T4 needs is proven here.
Needs MODEL_API_KEY in .env.
"""

import asyncio
import base64
import gzip
import itertools
import json
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import (
    SegmentationMaskRecord,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI

OUT = Path("eval/data/spike/sam")
FIXTURES = Path("tests/fixtures/sam")
FPS = 15
USD_PER_FRAME = 0.20 / 1000  # sam-3.1 video, dev.meta.ai pricing, Oct 30, 2026
USD_PER_IMAGE = 2.50 / 1000
FIXTURE_LIMIT_BYTES = 5_000_000
TRIM_FRAMES = 300  # 20 s at 15 fps when the full recording is too big to commit


def normalise(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-r", str(FPS), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-an", str(dest)],
        check=True,
    )  # fmt: skip


def background_frame(video: Path, dest: Path, samples: int = 60) -> None:
    """Per-pixel median over evenly spaced frames, so the animal disappears."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    for i in np.linspace(0, total - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    cv2.imwrite(str(dest), np.median(np.stack(frames), axis=0).astype(np.uint8))


async def record(client: AsyncOpenAI, dest: Path, **request: Any) -> dict[str, Any]:
    """Stream one request, write every event to `dest` (gz JSONL), return timings and usage.

    A finished recording is reused (timings in a sidecar), so a re-run never pays twice."""
    sidecar = dest.with_suffix(".timing.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    started = time.monotonic()
    first_delta: float | None = None
    usage: dict[str, Any] | None = None
    events = 0
    stream = await client.responses.create(stream=True, **request)
    with gzip.open(dest, "wt") as f:
        async for event in stream:
            events += 1
            if event.type == "response.output_text.delta" and first_delta is None:
                first_delta = time.monotonic() - started
            if event.type == "response.completed" and event.response.usage is not None:
                usage = event.response.usage.model_dump(mode="json")
            f.write(json.dumps(event.model_dump(mode="json")) + "\n")
    timing = {
        "events": events,
        "seconds_to_first_frame": first_delta,
        "seconds_total": time.monotonic() - started,
        "usage": usage,
    }
    sidecar.write_text(json.dumps(timing, indent=2) + "\n")
    return timing


async def replay(path: Path) -> AsyncIterator[Mapping[str, object]]:
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


async def summarise_track(path: Path) -> dict[str, Any]:
    result = await parse_responses_stream(replay(path), video_segmentation_format()).final_result()
    masks = [r for r in result.records if isinstance(r, SegmentationMaskRecord)]
    order = [r.frame.frame_index for r in masks if r.frame is not None]
    per_frame: dict[int, int] = {}
    for i in order:
        per_frame[i] = per_frame.get(i, 0) + 1
    conf = [r.confidence for r in masks if r.confidence is not None]
    return {
        "outcome": type(result.outcome).__name__,
        "diagnostics": len(result.diagnostics),
        "frames_with_mask": len(per_frame),
        "frames_with_more_than_one_mask": sum(n > 1 for n in per_frame.values()),
        "out_of_order_steps": sum(b < a for a, b in itertools.pairwise(order)),
        "object_ids": sorted({r.object_id for r in masks}),
        "mean_confidence": sum(conf) / len(conf) if conf else None,
        "first_mask_size_hw": [masks[0].mask.height, masks[0].mask.width] if masks else None,
    }


async def summarise_arena(path: Path) -> dict[str, Any]:
    result = await parse_responses_stream(replay(path), image_segmentation_format()).final_result()
    masks = [r for r in result.records if isinstance(r, SegmentationMaskRecord)]
    return {
        "masks": len(masks),
        "bounds": [[m.bounds.left, m.bounds.top, m.bounds.right, m.bounds.bottom] for m in masks],
        "confidence": [m.confidence for m in masks],
    }


def write_fixture(src: Path, dest: Path) -> str:
    """Commit the whole recording if it is small, else the first TRIM_FRAMES deltas plus the
    stream's opening and closing events."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size <= FIXTURE_LIMIT_BYTES:
        dest.write_bytes(src.read_bytes())
        return "full"
    with gzip.open(src, "rt") as f:
        lines = f.readlines()
    is_delta = ['"response.output_text.delta"' in line for line in lines]
    deltas = [i for i, delta in enumerate(is_delta) if delta]
    keep_until = deltas[TRIM_FRAMES - 1] if len(deltas) >= TRIM_FRAMES else len(lines) - 1
    closing = [line for i, line in enumerate(lines) if i > keep_until and not is_delta[i]]
    with gzip.open(dest, "wt") as f:
        f.writelines(lines[: keep_until + 1] + closing)
    return f"first {TRIM_FRAMES} frame deltas"


def message(*content: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "message", "role": "user", "content": list(content)}]


async def main(video_id: str) -> None:
    load_dotenv()
    client = AsyncOpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"{video_id}_{FPS}fps.mp4"
    normalise(Path("eval/data/videos") / f"{video_id}.mp4", video)
    frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    background = OUT / f"{video_id}_background.png"
    background_frame(video, background)

    png = base64.b64encode(background.read_bytes()).decode()
    arena_path = OUT / f"{video_id}_arena.jsonl.gz"
    arena = await record(
        client,
        arena_path,
        model="sam-3.1",
        input=message(
            {"type": "input_text", "text": "arena floor"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{png}"},
        ),
        metadata={"include_confidence": "true"},
    )

    track_path = OUT / f"{video_id}_track.jsonl.gz"
    upload_file = OUT / f"{video_id}_upload.json"
    if not upload_file.exists():
        upload_started = time.monotonic()
        with video.open("rb") as f:
            uploaded = await client.files.create(purpose="user_data", file=f)
        upload_file.write_text(
            json.dumps({"file_id": uploaded.id, "seconds": time.monotonic() - upload_started})
        )
    upload = json.loads(upload_file.read_text())
    track = await record(
        client,
        track_path,
        model="sam-3.1",
        input=message(
            {"type": "input_text", "text": "mouse"},
            {"type": "input_video", "file_id": upload["file_id"]},
        ),
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
    )

    billed_frames = (track["usage"] or {}).get("video_frames_processed")
    billed_images = (arena["usage"] or {}).get("images_processed")
    summary = {
        "video_id": video_id,
        "frames_sent": frames,
        "upload_seconds": upload["seconds"],
        "arena": arena | await summarise_arena(arena_path),
        "track": track | await summarise_track(track_path),
        "cost_usd": {
            "track": billed_frames * USD_PER_FRAME if billed_frames is not None else None,
            "arena": billed_images * USD_PER_IMAGE if billed_images is not None else None,
        },
        "fixtures": {
            "arena": write_fixture(arena_path, FIXTURES / f"{video_id}_arena.jsonl.gz"),
            "track": write_fixture(track_path, FIXTURES / f"{video_id}_track.jsonl.gz"),
        },
    }
    (OUT / f"{video_id}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "OFT_16"))
