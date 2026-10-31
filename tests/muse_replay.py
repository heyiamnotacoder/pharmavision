"""Muse Spark Responses streams for tests: a structured-output answer in the envelope of a real
Model API recording, replayed through the same transport as the SAM 3.1 tests."""

import base64
import json
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .sam_replay import Event, responses_events


def answer(label: str, confidence: float = 0.9, rationale: str = "Rears up.") -> list[Event]:
    """A completed stream whose output text is the JSON answer, in one delta."""
    text = json.dumps({"label": label, "confidence": confidence, "rationale": rationale})
    return responses_events([text])


def verdict(verdict: str, reason: str) -> list[Event]:
    """A completed stream whose output text is a QC review's JSON verdict, in one delta."""
    return responses_events([json.dumps({"verdict": verdict, "reason": reason})])


def text(markdown: str) -> list[Event]:
    """A completed stream whose output text is `markdown`, in one delta: an agent's last turn."""
    return responses_events([markdown])


def with_commentary(stream: list[Event], commentary: str) -> list[Event]:
    """`stream` with a `phase: "commentary"` message ahead of its answer, as Muse Spark sends
    one on the live API (Oct 31): a remark to the user, not part of the answer."""
    item = {"id": "msg_commentary", "type": "message", "role": "assistant", "phase": "commentary"}
    added = {"type": "response.output_item.added", "output_index": 0, "item": item}
    delta = {
        "type": "response.output_text.delta",
        "item_id": "msg_commentary",
        "output_index": 0,
        "content_index": 0,
        "delta": commentary,
    }
    done = {"type": "response.output_item.done", "output_index": 0, "item": item}
    events = [stream[0], added, delta, done, *stream[1:]]
    for n, event in enumerate(events):
        event["sequence_number"] = n
    return events


def tool_calls(*names: str) -> list[Event]:
    """A completed stream that calls each tool in `names` with no arguments, as `function_call`
    output items (`call-0`, `call-1`, ...), and writes no text."""
    items = [
        {
            "id": f"fc_{n}",
            "type": "function_call",
            "call_id": f"call-{n}",
            "name": name,
            "arguments": "{}",
            "status": "completed",
        }
        for n, name in enumerate(names)
    ]
    envelope = responses_events([""])
    created, completed = envelope[0], envelope[-1]
    events: list[Event] = [created]
    for n, item in enumerate(items):
        events.append({"type": "response.output_item.added", "output_index": n, "item": item})
        events.append({"type": "response.output_item.done", "output_index": n, "item": item})
    events.append({**completed, "response": {**completed["response"], "output": items}})
    for n, event in enumerate(events):
        event["sequence_number"] = n
    return events


def function_outputs(request: Mapping[str, Any]) -> dict[str, str]:
    """Each `function_call_output` in a Responses request's input, by its call id."""
    return {
        item["call_id"]: item["output"]
        for item in request["input"]
        if item.get("type") == "function_call_output"
    }


def videos_sent(request: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every `input_video` part in a Responses request, in order."""
    for message in request["input"]:
        for part in message["content"]:
            if part["type"] == "input_video":
                yield part


def decode_video(part: Mapping[str, Any], dest: Path) -> Path:
    """Write an `input_video` part's base64 data URL to `dest`."""
    header, data = part["video_url"].split(",", 1)
    assert header == "data:video/mp4;base64"
    dest.write_bytes(base64.b64decode(data))
    return dest


def frame_size_and_count(video: Path) -> tuple[int, int, int]:
    """Width, height and exact frame count (packets counted)."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=width,height,nb_read_packets", "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_read_packets"])
