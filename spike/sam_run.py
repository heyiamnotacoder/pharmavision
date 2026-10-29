"""Push one Sturman open-field video through SAM 3.1 and measure cost and wall-clock (laptop only).

Usage: uv run python -m spike.sam_run OFT_16

Steps: encode to 15 fps H.264 (as ingest will) → background frame → arena prompt on it →
upload → track "mouse". Every Responses event is recorded to eval/data/spike/sam/ as gzipped
JSONL keyed by input hash, then replayed through meta-sam-parser, so the replay path T4 needs is
proven here. The committed fixture is replayed too. Needs MODEL_API_KEY in .env.
"""

import asyncio
import base64
import copy
import gzip
import hashlib
import itertools
import json
import re
import sys
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from meta_sam_parser import (
    SegmentationMaskRecord,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI

from spike.common import BASE_URL, FPS, api_key, encode, input_hash, message
from spike.data import VIDEOS

OUT = Path("eval/data/spike/sam")
FIXTURES = Path("tests/fixtures/sam")
# dev.meta.ai/docs/pricing-rate-limits, Oct 30, 2026
USD_PER_FRAME = 0.20 / 1000
USD_PER_IMAGE = 2.50 / 1000
STAY_ON_API_USD = 3.30  # per 10-min video; ARCHITECTURE open decision 1
FIXTURE_LIMIT_BYTES = 5_000_000
TRIM_FRAMES = 300  # frames 0-299 (20 s at 15 fps) when the full recording is too big to commit
DELTA = "response.output_text.delta"
FRAME_MARKER = re.compile(r"<(\d+)f>")
# "arena floor", "floor", "box" and "open field arena" return no mask on Sturman OFT (Oct 30)
ARENA_PROMPT = "square floor"


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


async def record(client: AsyncOpenAI, name: str, **request: Any) -> tuple[Path, dict[str, Any]]:
    """Stream one request and write every event to a gz JSONL file keyed by the request's hash.
    Returns the file and its timings and usage. A finished recording is reused, so a re-run with
    the same input never pays twice."""
    dest = OUT / f"{name}-{input_hash(request)}.jsonl.gz"
    sidecar = dest.with_suffix(".timing.json")
    if sidecar.exists():
        return dest, json.loads(sidecar.read_text())
    started = time.monotonic()
    first_delta: float | None = None
    usage: dict[str, Any] | None = None
    events = 0
    stream = await client.responses.create(stream=True, **request)
    with gzip.open(dest, "wt") as f:
        async for event in stream:
            events += 1
            if event.type == DELTA and first_delta is None:
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
    return dest, timing


def read_events(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


async def replay(path: Path) -> AsyncIterator[Mapping[str, object]]:
    for event in read_events(path):
        yield event


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


def _replace_text(value: Any, full: str, kept: str) -> Any:
    """Swap the full output text for the kept text wherever a closing event repeats it."""
    if value == full:
        return kept
    if isinstance(value, dict):
        return {k: _replace_text(v, full, kept) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_text(v, full, kept) for v in value]
    return value


def trim_events(events: list[dict[str, Any]], max_frame: int) -> list[dict[str, Any]]:
    """Keep frames 0..max_frame-1, chosen by their `<Nf>` index, not by line position (frames can
    skip and arrive out of order). Emits one delta per kept line and rewrites the closing events'
    copies of the full text, so the parser still sees deltas and final text agree."""
    deltas = [e for e in events if e["type"] == DELTA]
    full = "".join(e["delta"] for e in deltas)
    kept_lines = [
        line
        for line in full.splitlines(keepends=True)
        if (m := FRAME_MARKER.match(line)) and int(m.group(1)) < max_frame
    ]
    out: list[dict[str, Any]] = []
    for event in events:
        if event["type"] != DELTA:
            out.append(_replace_text(event, full, "".join(kept_lines)))
        elif event is deltas[0]:
            out.extend(copy.deepcopy(event) | {"delta": line} for line in kept_lines)
    for n, event in enumerate(out):
        if "sequence_number" in event:
            event["sequence_number"] = n
    return out


async def write_fixture(src: Path, dest: Path, *, video: bool) -> dict[str, Any]:
    """Commit the whole recording if small, else frames 0..TRIM_FRAMES-1; then prove it replays."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size <= FIXTURE_LIMIT_BYTES:
        dest.write_bytes(src.read_bytes())
        kind = "full"
    else:
        with gzip.open(dest, "wt") as f:
            f.writelines(json.dumps(e) + "\n" for e in trim_events(read_events(src), TRIM_FRAMES))
        kind = f"frames 0-{TRIM_FRAMES - 1}"
    replayed = await (summarise_track(dest) if video else summarise_arena(dest))
    return {"path": str(dest), "kind": kind, "bytes": dest.stat().st_size, "replay": replayed}


async def upload(client: AsyncOpenAI, video: Path) -> dict[str, Any]:
    """Upload once per video content; the file id is reused on re-runs."""
    receipt = OUT / f"upload-{hashlib.sha256(video.read_bytes()).hexdigest()[:16]}.json"
    if not receipt.exists():
        started = time.monotonic()
        with video.open("rb") as f:
            uploaded = await client.files.create(purpose="user_data", file=f)
        receipt.write_text(
            json.dumps({"file_id": uploaded.id, "seconds": time.monotonic() - started})
        )
    return json.loads(receipt.read_text())


async def main(video_id: str) -> None:
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key())
    OUT.mkdir(parents=True, exist_ok=True)
    video = encode(VIDEOS / f"{video_id}.mp4", OUT / f"{video_id}_{FPS}fps.mp4")
    frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    background = OUT / f"{video_id}_background.png"
    background_frame(video, background)

    png = base64.b64encode(background.read_bytes()).decode()
    arena_path, arena = await record(
        client,
        f"{video_id}_arena",
        model="sam-3.1",
        input=message(
            {"type": "input_text", "text": ARENA_PROMPT},
            {"type": "input_image", "image_url": f"data:image/png;base64,{png}"},
        ),
        metadata={"include_confidence": "true"},
    )
    uploaded = await upload(client, video)
    track_path, track = await record(
        client,
        f"{video_id}_track",
        model="sam-3.1",
        input=message(
            {"type": "input_text", "text": "mouse"},
            {"type": "input_video", "file_id": uploaded["file_id"]},
        ),
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
    )

    billed_frames = (track["usage"] or {}).get("video_frames_processed")
    billed_images = (arena["usage"] or {}).get("images_processed")
    cost = None
    if billed_frames is not None and billed_images is not None:
        total = billed_frames * USD_PER_FRAME + billed_images * USD_PER_IMAGE
        per_10_min = total * 600 / (frames / FPS)
        cost = {
            "video_usd": total,
            "per_10_min_usd": per_10_min,
            "decision": "stay on API" if per_10_min <= STAY_ON_API_USD else "self-host or cut",
        }
    summary = {
        "video_id": video_id,
        "frames_sent": frames,
        "billed": {"video_frames": billed_frames, "images": billed_images},
        "cost": cost,
        "wall_clock_s": {
            "upload": uploaded["seconds"],
            "track_first_frame": track["seconds_to_first_frame"],
            "track_total": track["seconds_total"],
            "arena_total": arena["seconds_total"],
        },
        "arena": arena | await summarise_arena(arena_path),
        "track": track | await summarise_track(track_path),
        "fixtures": {
            "arena": await write_fixture(
                arena_path, FIXTURES / f"{video_id}_arena.jsonl.gz", video=False
            ),
            "track": await write_fixture(
                track_path, FIXTURES / f"{video_id}_track.jsonl.gz", video=True
            ),
        },
    }
    (OUT / f"{video_id}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "OFT_16"))
