"""Muse Spark on the Meta Model API: the primary Judge backend.

One Responses request per clip, streamed, with structured output. The clip goes inline as a
base64 MP4 sampled at `input_fps`; few-shot examples go ahead of it as earlier turns (a user
turn with the example clip, then an assistant turn replaying its label). Every response is
cached by model id, prompt version and the hash of the clip's frames (and of the prompt and
examples),
so a re-run makes no API call, and the recorded Judge replays the same cache with no API at all."""

import base64
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..model_api import OpenAiTransport, Request, ResponsesTransport
from ..model_cache import Event, ResponseCache, input_hash
from . import Clip, ClipJudgement, RearLabel

MAX_VIDEOS = 5
"""Videos the Model API takes in one request (a 400 above it, checked live Oct 30)."""

PROMPT_VERSION = "clip-v1"
"""The request shape and wording: bump it when either changes."""

PROMPT = """\
This clip is 2-4 seconds of a top-view camera over a mouse in an open-field arena, cropped to a
square around the mouse. The green outline is the mouse's tracked outline. Decide whether the
mouse rears at any point in the clip.

- rearing: the mouse stands up on its hind legs, against an arena wall or not.
- none: no rear (walking, turning, sitting still, grooming, sniffing).

From above, a rear shows as the body suddenly looking shorter and rounder while the mouse stays
in place, then lengthening again as it comes down. Answer with the single best label, your
confidence from 0 to 1 that it is right, and one sentence of rationale."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["rearing", "none"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "rationale"],
    "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "name": "rear_label", "schema": SCHEMA, "strict": True}


class JudgeResponseError(RuntimeError):
    """Muse Spark answered with something other than a complete, well-formed judgement."""


class RecordingMissingError(LookupError):
    """The recorded Judge has no stored response for this clip."""


class FewShotLeakError(ValueError):
    """A few-shot example comes from a video outside the train split: the model would have seen
    an evaluation video."""


class FewShotExample(BaseModel):
    """One labelled clip shown to the model ahead of the clip it judges, cut the way the
    behaviours stage cuts clips."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: str
    """The Sturman video it was cut from, e.g. "OFT_16": it must be in the train split."""
    clip: Path
    label: RearLabel
    rationale: str


FEW_SHOT_DIR = Path(__file__).parent / "few_shot"
"""The committed few-shot set: `examples.json` (clip references and labels) and the clips, cut
from the train split's run folders by `python -m spike.muse_few_shot build`."""


def few_shot_examples(directory: Path = FEW_SHOT_DIR) -> tuple[FewShotExample, ...]:
    """The few-shot set in `directory`, in the order the model sees it."""
    entries = json.loads((directory / "examples.json").read_text())
    return tuple(
        FewShotExample(
            video_id=e["video_id"],
            clip=directory / e["clip"],
            label=e["label"],
            rationale=e["rationale"],
        )
        for e in entries
    )


class MuseSparkSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str = "muse-spark-1.3"
    tier: Literal["standard", "contributor"] = "standard"
    """The Contributor tier is far cheaper, but Meta may train on its prompts and outputs: for
    the public Sturman data only, never a lab's own videos (ARCHITECTURE "Cost control")."""
    base_url: str = "https://api.meta.ai/v1"
    cache_dir: Path = Path("runs/model_cache")
    """Shared by every run, so re-running a video costs nothing."""
    input_fps: int | None = Field(default=15, gt=0)
    """`input_video` sampling. At 15 fps (every frame) Muse Spark agreed with the raters more
    than at its default rate, for about 7 times the tokens (T2). None leaves it to the API."""
    few_shot: tuple[FewShotExample, ...] = Field(default=(), max_length=MAX_VIDEOS - 1)
    """Labelled clips shown ahead of each clip; with the clip judged, at most `MAX_VIDEOS`."""
    split_file: Path = Path("eval/split.json")
    """The fixed train/eval split; every few-shot example must come from its train videos."""

    @property
    def model(self) -> str:
        return f"{self.base_model}-contributor" if self.tier == "contributor" else self.base_model


@dataclass(frozen=True)
class _Example:
    video_url: str
    reply: str
    sha256: str


class MuseSparkJudge:
    def __init__(
        self, settings: MuseSparkSettings, transport: ResponsesTransport | None = None
    ) -> None:
        self.settings = settings
        self.transport = transport or OpenAiTransport(settings.base_url)
        self.cache = ResponseCache(settings.cache_dir)
        self._examples = _load_examples(settings.few_shot, settings.split_file)

    @property
    def model_id(self) -> str:
        return self.settings.model

    def classify_clip(self, clip: Clip) -> ClipJudgement:
        return _parse(self.response(clip))

    def response(self, clip: Clip) -> list[Event]:
        """The clip's response as its raw event stream, usage included: from the cache, or else
        from one API call, cached once it parses as a judgement."""
        digest = input_hash(
            "clip",
            PROMPT,
            FORMAT,
            str(self.settings.input_fps),
            *(f"{e.sha256} {e.reply}" for e in self._examples),
            clip.frames_sha256,
        )
        model = self.settings.model
        cached = self.cache.get(model, PROMPT_VERSION, digest)
        if cached is not None:
            return cached
        request = self._request(_data_url(clip.video.read_bytes()))
        stream = list(self.transport.stream(request))
        _parse(stream)
        self.cache.put(model, PROMPT_VERSION, digest, stream)
        return stream

    def _request(self, clip_url: str) -> Request:
        """The prompt and the first clip in the first user turn, the shape T2 measured; each
        example's label replayed as an assistant turn after it; the clip to judge last."""
        turns = [(e.video_url, e.reply) for e in self._examples] + [(clip_url, None)]
        messages: list[dict[str, Any]] = []
        for n, (video_url, reply) in enumerate(turns):
            content: list[dict[str, Any]] = (
                [{"type": "input_text", "text": PROMPT}] if n == 0 else []
            )
            video: dict[str, Any] = {"type": "input_video", "video_url": video_url}
            if self.settings.input_fps is not None:
                video["fps"] = self.settings.input_fps
            content.append(video)
            messages.append({"type": "message", "role": "user", "content": content})
            if reply is not None:
                text = {"type": "output_text", "text": reply}
                messages.append({"type": "message", "role": "assistant", "content": [text]})
        return {"model": self.settings.model, "input": messages, "text": {"format": FORMAT}}


class RecordedJudge(MuseSparkJudge):
    """Muse Spark as recorded: replays the response cache by the production key and refuses a
    clip it has no response for. A response captured on the laptop is a test fixture as is."""

    def __init__(self, settings: MuseSparkSettings) -> None:
        super().__init__(settings, _NoApi())


class _NoApi:
    def stream(self, request: Request) -> Iterator[Event]:
        raise RecordingMissingError(f"no recorded {request['model']} response for this clip")


def _load_examples(examples: Sequence[FewShotExample], split_file: Path) -> list[_Example]:
    if not examples:
        return []
    train = set(json.loads(split_file.read_text())["train"])
    leaked = sorted({e.video_id for e in examples} - train)
    if leaked:
        raise FewShotLeakError(f"few-shot examples from outside the train split: {leaked}")
    loaded: list[_Example] = []
    for example in examples:
        data = example.clip.read_bytes()
        reply = ClipJudgement(label=example.label, confidence=1.0, rationale=example.rationale)
        loaded.append(_Example(_data_url(data), reply.model_dump_json(), input_hash(data)))
    return loaded


def _data_url(mp4: bytes) -> str:
    return f"data:video/mp4;base64,{base64.b64encode(mp4).decode()}"


def _parse(stream: Iterable[Event]) -> ClipJudgement:
    """The judgement in a completed stream's output text."""
    stream = list(stream)
    kinds = {event.get("type") for event in stream}
    if "response.completed" not in kinds:
        raise JudgeResponseError(f"incomplete Muse Spark response: {sorted(map(str, kinds))}")
    text = "".join(
        str(event["delta"]) for event in stream if event.get("type") == "response.output_text.delta"
    )
    try:
        return ClipJudgement.model_validate_json(text)
    except ValidationError as error:
        raise JudgeResponseError(f"malformed Muse Spark judgement {text!r}: {error}") from error
