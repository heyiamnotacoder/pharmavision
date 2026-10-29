"""SAM 3.1 on the Meta Model API: the primary Segmenter backend.

One request per image or video: the video is uploaded once through the Files API, then the
Responses API streams one `<Nf>` line per frame, which `meta-sam-parser` turns into box-local
masks. Records are keyed by that frame index, never by line position."""

import asyncio
import base64
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
from meta_sam_parser import (
    IncompleteOutcome,
    ResponseFormat,
    SegmentationMaskRecord,
    SegmentationResult,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..model_cache import ResponseCache, file_sha256, input_hash
from . import FrameMasks, Mask
from .chunks import handover_frame, plan_chunks, relink

PROMPT_VERSION = "sam-v1"
"""The request shape (prompt wording aside, which is hashed): bump it when the request changes."""

Request = Mapping[str, Any]
Event = Mapping[str, object]


class SamResponseError(RuntimeError):
    """SAM 3.1 answered with something other than a complete, well-formed stream."""


class MetaApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["meta_api"] = "meta_api"
    model: str = "sam-3.1"
    base_url: str = "https://api.meta.ai/v1"
    cache_dir: Path = Path("runs/model_cache")
    """Shared by every run, so re-running a video costs nothing."""
    max_frames_per_request: int = Field(default=15_000, ge=2)
    """SAM 3.1's limit; longer videos go in overlapping chunks."""
    chunk_overlap_frames: int = Field(default=150, ge=1)
    """10 s at 15 fps: the frames two chunks share, for relinking their objects."""
    relink_min_iou: float = Field(default=0.5, gt=0, le=1)
    """Mean mask IoU over the overlap at which two chunks' objects are the same object."""

    @model_validator(mode="after")
    def _overlap_fits_in_a_chunk(self) -> "MetaApiSettings":
        if self.chunk_overlap_frames >= self.max_frames_per_request:
            raise ValueError("chunk_overlap_frames must be below max_frames_per_request")
        return self


class SamTransport(Protocol):
    """The two Model API calls the backend makes. Tests replay recordings through this."""

    def upload(self, video: Path) -> str:
        """Upload a video through the Files API; returns its file id."""
        ...

    def stream(self, request: Request) -> Iterable[Event]:
        """Stream one Responses API request; yields every event as a JSON mapping."""
        ...


class OpenAiTransport:
    """The Model API is OpenAI-compatible. The client is made on first use, so a run served
    entirely from the cache needs no API key."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._client: Any = None

    def upload(self, video: Path) -> str:
        with video.open("rb") as f:
            return self._openai().files.create(purpose="user_data", file=f).id

    def stream(self, request: Request) -> Iterator[Event]:
        for event in self._openai().responses.create(stream=True, **request):
            yield event.model_dump(mode="json")

    def _openai(self) -> Any:
        if self._client is None:
            from dotenv import find_dotenv, load_dotenv
            from openai import OpenAI

            load_dotenv(find_dotenv(usecwd=True))
            self._client = OpenAI(base_url=self.base_url, api_key=os.environ["MODEL_API_KEY"])
        return self._client


class MetaApiSegmenter:
    def __init__(self, settings: MetaApiSettings, transport: SamTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport or OpenAiTransport(settings.base_url)
        self.cache = ResponseCache(settings.cache_dir)

    @property
    def model_id(self) -> str:
        return self.settings.model

    def segment_image(self, image: NDArray[np.uint8], prompt: str) -> list[Mask]:
        metadata = {"include_confidence": "true"}

        def request() -> Request:
            ok, png = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("could not encode the image as PNG")
            data_url = f"data:image/png;base64,{base64.b64encode(png.tobytes()).decode()}"
            return self._request(prompt, {"type": "input_image", "image_url": data_url}, metadata)

        digest = input_hash("image", str(image.shape), image.tobytes(), prompt, metadata)
        result = self._respond(digest, request, image_segmentation_format())
        shape = (image.shape[0], image.shape[1])
        return [_to_mask(record, shape) for record in _masks(result)]

    def track_video(self, video: Path, prompt: str, max_objects: int) -> Iterator[FrameMasks]:
        """One request per chunk, chunks relinked into one set of object ids; then the
        `max_objects` most confident instances in each frame."""
        settings = self.settings
        shape, n_frames = _video_shape(video)
        video_sha = file_sha256(video)
        tracks: dict[int, dict[str, SegmentationMaskRecord]] = {}
        previous: dict[int, dict[str, SegmentationMaskRecord]] = {}
        previous_chunk: range | None = None
        chunks = plan_chunks(
            n_frames, settings.max_frames_per_request, settings.chunk_overlap_frames
        )
        for number, chunk in enumerate(chunks):
            found = self._track_chunk(video, video_sha, n_frames, chunk, prompt)
            sam_ids = {sam_id for objects in found.values() for sam_id in objects}
            if previous_chunk is None:
                ids, first = {sam_id: sam_id for sam_id in sam_ids}, 0
            else:
                overlap = range(chunk.start, previous_chunk.stop)
                linked = relink(
                    _decoded(previous, overlap, shape),
                    _decoded(found, overlap, shape),
                    settings.relink_min_iou,
                )
                ids = {sam_id: linked.get(sam_id, f"{number}.{sam_id}") for sam_id in sam_ids}
                first = handover_frame(previous_chunk, chunk)
            current = {
                frame: {ids[sam_id]: record for sam_id, record in objects.items()}
                for frame, objects in found.items()
            }
            tracks = {frame: objects for frame, objects in tracks.items() if frame < first}
            tracks.update({frame: o for frame, o in current.items() if frame >= first})
            previous, previous_chunk = current, chunk
        for frame in sorted(tracks):
            ranked = sorted(tracks[frame].items(), key=lambda item: -(item[1].confidence or 0.0))
            kept = ranked[:max_objects]
            yield FrameMasks(frame, tuple(_to_mask(r, shape, object_id) for object_id, r in kept))

    def _track_chunk(
        self, video: Path, video_sha: str, n_frames: int, chunk: range, prompt: str
    ) -> dict[int, dict[str, SegmentationMaskRecord]]:
        """SAM 3.1 on frames `chunk` of the video: frame index -> SAM object id -> mask, with
        the response's chunk-local frame markers shifted back to the video's frame indices."""
        metadata = {"mask_encoding": "one_bit", "include_confidence": "true"}

        def request() -> Request:
            with tempfile.TemporaryDirectory() as tmp:
                whole = chunk == range(n_frames)
                clip = video if whole else _cut(video, chunk, Path(tmp) / "chunk.mp4")
                file_id = self.transport.upload(clip)
            return self._request(prompt, {"type": "input_video", "file_id": file_id}, metadata)

        span = f"frames {chunk.start}-{chunk.stop}"
        digest = input_hash("video", video_sha, span, prompt, metadata)
        result = self._respond(digest, request, video_segmentation_format())
        found: dict[int, dict[str, SegmentationMaskRecord]] = defaultdict(dict)
        for record in _masks(result):
            assert record.frame is not None  # video records always carry their frame
            if record.frame.frame_index >= len(chunk):
                raise SamResponseError(
                    f"SAM 3.1 returned frame {record.frame.frame_index} of {len(chunk)} in a chunk"
                )
            found[chunk.start + record.frame.frame_index][record.object_id] = record
        return found

    def _respond[R: SegmentationResult](
        self, digest: str, request: Callable[[], Request], format_: ResponseFormat[Any, R]
    ) -> R:
        """The parsed response, from the cache when this exact input was seen before. `request`
        is built only on a miss, so a cached re-run uploads nothing. Only a complete,
        well-formed response is cached."""
        model = self.settings.model
        cached = self.cache.get(model, PROMPT_VERSION, digest)
        if cached is not None:
            return _parse(cached, format_)
        events = list(self.transport.stream(request()))
        result = _parse(events, format_)
        self.cache.put(model, PROMPT_VERSION, digest, events)
        return result

    def _request(self, prompt: str, media: dict[str, str], metadata: dict[str, str]) -> Request:
        """A Responses request, untyped because `input_video` and SAM's `metadata` keys are
        Meta extensions the OpenAI SDK's types lack."""
        content = [{"type": "input_text", "text": prompt}, media]
        return {
            "model": self.settings.model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "metadata": metadata,
        }


def _parse[R: SegmentationResult](events: Iterable[Event], format_: ResponseFormat[Any, R]) -> R:
    async def source() -> AsyncIterator[Event]:
        for event in events:
            yield event

    async def parse() -> R:
        return await parse_responses_stream(source(), format_).final_result()

    result = asyncio.run(parse())
    if isinstance(result.outcome, IncompleteOutcome):
        raise SamResponseError(f"incomplete SAM 3.1 response: {result.outcome}")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise SamResponseError(f"malformed SAM 3.1 output line {errors[0].line}: {errors[0].code}")
    return result


def _masks(result: SegmentationResult) -> list[SegmentationMaskRecord]:
    """The latest revision of each object's mask in each frame."""
    latest: dict[object, SegmentationMaskRecord] = {}
    for record in result.records:
        if isinstance(record, SegmentationMaskRecord):
            seen = latest.get(record.identity)
            if seen is None or record.revision > seen.revision:
                latest[record.identity] = record
    return list(latest.values())


def _to_mask(
    record: SegmentationMaskRecord, shape: tuple[int, int], object_id: str | None = None
) -> Mask:
    """Paste the box-local mask into a full-frame canvas at its box."""
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    local = raster.reshape(record.mask.height, record.mask.width)
    b = record.bounds
    left, top, right, bottom = int(b.left), int(b.top), int(b.right), int(b.bottom)
    if local.shape != (bottom - top, right - left):
        local = cv2.resize(local, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    pixels = np.zeros(shape, dtype=bool)
    height, width = shape
    pixels[top : min(bottom, height), left : min(right, width)] = local[
        : min(bottom, height) - top, : min(right, width) - left
    ].astype(bool)
    return Mask(pixels, confidence=record.confidence, object_id=object_id or record.object_id)


def _decoded(
    frames: Mapping[int, Mapping[str, SegmentationMaskRecord]],
    only: range,
    shape: tuple[int, int],
) -> dict[int, dict[str, NDArray[np.bool_]]]:
    return {
        frame: {object_id: _to_mask(r, shape).pixels for object_id, r in objects.items()}
        for frame, objects in frames.items()
        if frame in only
    }


def _video_shape(video: Path) -> tuple[tuple[int, int], int]:
    """(height, width) and the frame count."""
    capture = cv2.VideoCapture(str(video))
    try:
        shape = (
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        return shape, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _cut(video: Path, frames: range, dest: Path) -> Path:
    """Frames `frames` of the video as their own H.264 MP4, frame-exact (re-encoded)."""
    trim = f"trim=start_frame={frames.start}:end_frame={frames.stop},setpts=PTS-STARTPTS"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:v:0", "-an",
            "-vf", trim, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart", str(dest),
        ],
        check=True, capture_output=True,
    )  # fmt: skip
    _, n_frames = _video_shape(dest)
    if n_frames != len(frames):
        raise RuntimeError(f"cut {n_frames} frames from {video.name}, wanted {len(frames)}")
    return dest
