"""SAM 3.1 Responses streams for tests: T2's real recordings, and synthetic streams built in the
same wire format, replayed through a transport that never calls the API."""

import copy
import gzip
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Test-only: the parser ships the one-bit encoder privately; synthetic streams need it.
from meta_sam_parser._mask_codec import _encode_raster  # pyright: ignore[reportPrivateUsage]
from numpy.typing import NDArray

FIXTURES = Path(__file__).parent / "fixtures" / "sam"
DELTA = "response.output_text.delta"
_CLOSING = {
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
}
_FRAME_MARKER = re.compile(r"<(\d+)f>")

Event = dict[str, Any]


def read_fixture(name: str) -> list[Event]:
    with gzip.open(FIXTURES / f"{name}.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def output_text(events: Sequence[Event]) -> str:
    return "".join(e["delta"] for e in events if e["type"] == DELTA)


def responses_events(lines: Sequence[str]) -> list[Event]:
    """A completed Responses stream carrying `lines` (one delta each), in the envelope of the
    real "no match" recording."""
    text = "".join(lines)
    out: list[Event] = []
    for event in read_fixture("OFT_16_arena_no_match"):
        if event["type"] in _CLOSING:
            event = _with_text(event, text)
        out.append(event)
        if event["type"] == "response.content_part.added":
            out.extend(
                {
                    "type": DELTA,
                    "content_index": 0,
                    "item_id": event["item_id"],
                    "output_index": 0,
                    "delta": line,
                    "logprobs": [],
                }
                for line in lines
            )
    for n, event in enumerate(out):
        event["sequence_number"] = n
    return out


def recorded_track_lines(below_frame: int) -> list[str]:
    """Lines of the real OFT_16 track recording for frames below `below_frame`, chosen by their
    `<Nf>` marker."""
    return [
        line
        for line in output_text(read_fixture("OFT_16_track")).splitlines(keepends=True)
        if (m := _FRAME_MARKER.match(line)) and int(m.group(1)) < below_frame
    ]


@dataclass(frozen=True)
class Instance:
    """One object in one frame: its id in the response, a full-frame mask and a confidence."""

    object_id: str
    pixels: NDArray[np.bool_]
    confidence: float


def frame_line(frame_index: int, instances: Sequence[Instance]) -> str:
    """One `<Nf>` line in SAM 3.1's grammar: a box plus a box-local one-bit mask per object."""
    records: list[str] = []
    for instance in instances:
        ys, xs = np.nonzero(instance.pixels)
        top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
        local = instance.pixels[top : bottom + 1, left : right + 1].astype(np.uint8)
        h, w = local.shape
        height, width = instance.pixels.shape
        payload = _encode_raster(local.ravel().tolist(), w, h)
        c = instance.confidence
        records.append(
            f"{instance.object_id}<|box;x1={left};y1={top};x2={right};y2={bottom};"
            f"w={width};h={height};c={c}|><|mask;x=0;y=0;c={c};data={h},{w},{payload}|>"
        )
    return f"<{frame_index}f>{','.join(records)}\n"


def disc(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> NDArray[np.bool_]:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.circle(canvas, (round(centre[0]), round(centre[1])), round(radius), 1, thickness=-1)
    return canvas.astype(bool)


def polygon(shape: tuple[int, int], points: Sequence[tuple[float, float]]) -> NDArray[np.bool_]:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(np.array(points)).astype(np.int32)], color=1)
    return canvas.astype(bool)


@dataclass
class ReplayTransport:
    """Serves recorded Responses streams in call order, and records every call. Running out of
    streams fails the test: it means the code called the API when it should not have."""

    streams: list[list[Event]]
    requests: list[dict[str, Any]] = field(default_factory=list)
    uploaded_frame_counts: list[int] = field(default_factory=list)

    def upload(self, video: Path) -> str:
        capture = cv2.VideoCapture(str(video))
        self.uploaded_frame_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        return f"file-{len(self.uploaded_frame_counts)}"

    def stream(self, request: Mapping[str, Any]) -> Iterator[Mapping[str, object]]:
        self.requests.append(dict(request))
        if not self.streams:
            raise AssertionError(f"unexpected API call: {request.get('model')}")
        yield from copy.deepcopy(self.streams.pop(0))


def _with_text(value: Any, text: str) -> Any:
    """Fill every empty `text` field, as the closing events repeat the full output text."""
    if isinstance(value, dict):
        return {
            k: text if k == "text" and v == "" else _with_text(v, text) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_with_text(v, text) for v in value]
    return value
