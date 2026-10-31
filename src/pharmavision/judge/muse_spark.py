"""Muse Spark on the Meta Model API: the primary Judge backend.

One Responses request per clip (a rear candidate's, or a flagged QC window's), streamed, with
structured output. The clip goes inline as a base64 MP4 sampled at `input_fps`; few-shot
examples go ahead of a rear clip as earlier turns (a user turn with the example clip, then an
assistant turn replaying its label). Every response is cached by model id, prompt version and
the hash of the clip's frames (and of the prompt and examples), so a re-run makes no API call,
and the recorded Judge replays the same cache with no API at all.

The report agent is a loop of requests with function tools instead, each request cached by its
whole input."""

import base64
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..model_api import OpenAiTransport, Request, ResponsesTransport
from ..model_cache import Event, ResponseCache, input_hash
from . import (
    Clip,
    ClipJudgement,
    FlaggedWindow,
    RearLabel,
    ReportBrief,
    ReportSection,
    ReportTool,
    WindowVerdict,
)

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

REVIEW_PROMPT_VERSION = "qc-v1"
"""The QC review's request shape and wording: bump it when either changes."""

REVIEW_PROMPT = """\
This clip is a few seconds of a top-view camera over a mouse in an open-field arena, centred on
frames {first_frame}-{last_frame} of the video, which an automatic quality check flagged. The
green outline is the mouse's tracked outline.

The check found: {kind}: {reason}.
A hand detector looked at a few of the flagged frames: {hands}.

Decide what the flagged frames are worth for measuring the mouse's behaviour:
- keep: the tracking is right and nothing disturbs the mouse (a reflection, a dropping, a brief
  tracking glitch).
- exclude_window: leave the flagged frames out (a hand in the arena, the outline on something
  that is not the mouse), and keep the rest of the trial.
- exclude_trial: the whole trial is unusable (the mouse is handled or lifted out, escapes, or
  the arena is disturbed).
Answer with the verdict and one sentence of reason. Do not state any numbers in the reason."""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["keep", "exclude_window", "exclude_trial"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
REVIEW_FORMAT = {
    "type": "json_schema",
    "name": "qc_verdict",
    "schema": REVIEW_SCHEMA,
    "strict": True,
}


REPORT_PROMPT_VERSION = "report-v1"
"""The report agent's request shape, wording and tools: bump it when any of them changes."""

REPORT_PROMPT = """\
Write the methods and results section of a paper about an open-field study of mice, scored
automatically from top-view video by PharmaVision: SAM 3.1 tracked each mouse, and
deterministic Python code measured every metric and chose and ran every statistical test.
Read the study through the tools. They are your only source of facts: call each one you need.

Write Markdown with these headings, each once, in this order, and under each what it says:

{sections}

Rules:
- Every number you write must be a value a tool returned in a numeric field, as given or
  rounded (10.783 as 10.78). Never compute a number: no differences, ratios, percentages,
  sums or thresholds that no tool returned. Write p = 0.0016, not p < 0.01.
- Never copy a number out of a text field (a reason or a rationale) unless the same object's
  `reason_values` field holds it. Write numbers as digits, never as words.
- Write nothing that the tools do not hold. For the facts listed as left out, the report adds
  a placeholder for the scientist to fill: do not write them, and do not guess them.
- The pipeline makes no rear count, because its rear evidence did not agree with human raters
  per video. Say so; never count rears.
- Prose only under the headings: no tables, no numbered lists, no other headings."""

REPORT_REJECTION = """\
The number audit rejected that report. These numbers match no value the tools returned:

{failures}

Write the whole report again. Call the tools again for any value you need: use only values
they return, or leave the number out."""


class JudgeResponseError(RuntimeError):
    """Muse Spark answered with something other than a complete, well-formed answer."""


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
    report_max_turns: int = Field(default=8, ge=1)
    """Requests in one report draft: the agent's tool calls, then its text. Each costs input
    tokens for the whole conversation so far, so a draft that has not finished by then fails."""

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
        return _parse(self.response(clip), ClipJudgement)

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
        return self._respond(
            PROMPT_VERSION,
            digest,
            lambda: self._request(_data_url(clip.video.read_bytes())),
            lambda stream: _parse(stream, ClipJudgement),
        )

    def review_window(self, window: FlaggedWindow) -> WindowVerdict:
        """One request per flagged window: the prompt states what the rule and the hand prompts
        found, then the window's clip. No few-shot examples: they are rear clips."""
        prompt = _review_prompt(window)
        digest = input_hash(
            "qc", prompt, REVIEW_FORMAT, str(self.settings.input_fps), window.clip.frames_sha256
        )

        def request() -> Request:
            video = self._video(_data_url(window.clip.video.read_bytes()))
            content = [{"type": "input_text", "text": prompt}, video]
            return {
                "model": self.settings.model,
                "input": [{"type": "message", "role": "user", "content": content}],
                "text": {"format": REVIEW_FORMAT},
            }

        stream = self._respond(
            REVIEW_PROMPT_VERSION, digest, request, lambda s: _parse(s, WindowVerdict)
        )
        return _parse(stream, WindowVerdict)

    def write_report(self, brief: ReportBrief) -> str:
        """The agent loop: each request offers the tools, and each function call's output goes
        back in the next request, until a response writes text instead of calling a tool. Each
        request is cached by all of its input, which the tools' outputs decide, so a re-run
        replays every turn and the recorded Judge needs no API."""
        tools = {tool.name: tool for tool in brief.tools}
        offered = [_tool_spec(tool) for tool in brief.tools]
        conversation = _report_conversation(brief)
        for _ in range(self.settings.report_max_turns):
            request: Request = {
                "model": self.settings.model,
                "input": list(conversation),
                "tools": offered,
            }
            digest = input_hash("report", request)
            stream = self._respond(REPORT_PROMPT_VERSION, digest, lambda r=request: r, _check_turn)
            calls = _function_calls(stream)
            if not calls:
                return _completed_text(stream)
            for call in calls:
                tool = tools.get(call.name)
                output = tool.read() if tool else json.dumps({"error": f"no tool {call.name!r}"})
                conversation.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                )
                conversation.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": output}
                )
        raise JudgeResponseError(
            f"no report after {self.settings.report_max_turns} requests: the agent kept "
            "calling tools"
        )

    def _respond(
        self,
        prompt_version: str,
        digest: str,
        request: Callable[[], Request],
        check: Callable[[list[Event]], object],
    ) -> list[Event]:
        """The raw event stream: from the cache, or else from one API call, cached once `check`
        accepts it (it raises JudgeResponseError otherwise). `request` is built only on a
        miss."""
        model = self.settings.model
        cached = self.cache.get(model, prompt_version, digest)
        if cached is not None:
            return cached
        stream = list(self.transport.stream(request()))
        check(stream)
        self.cache.put(model, prompt_version, digest, stream)
        return stream

    def _video(self, video_url: str) -> dict[str, Any]:
        video: dict[str, Any] = {"type": "input_video", "video_url": video_url}
        if self.settings.input_fps is not None:
            video["fps"] = self.settings.input_fps
        return video

    def _request(self, clip_url: str) -> Request:
        """The prompt and the first clip in the first user turn, the shape T2 measured; each
        example's label replayed as an assistant turn after it; the clip to judge last."""
        turns = [(e.video_url, e.reply) for e in self._examples] + [(clip_url, None)]
        messages: list[dict[str, Any]] = []
        for n, (video_url, reply) in enumerate(turns):
            content: list[dict[str, Any]] = (
                [{"type": "input_text", "text": PROMPT}] if n == 0 else []
            )
            content.append(self._video(video_url))
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


def _review_prompt(window: FlaggedWindow) -> str:
    if window.hands:
        hands = "; ".join(
            f"{h.prompt!r} on frame {h.frame_index}, largest mask {h.area_cm2:.0f} cm²"
            for h in window.hands
        )
    else:
        hands = "nothing found"
    return REVIEW_PROMPT.format(
        first_frame=window.first_frame,
        last_frame=window.last_frame,
        kind=window.kind,
        reason=window.reason,
        hands=hands,
    )


@dataclass(frozen=True)
class _FunctionCall:
    call_id: str
    name: str
    arguments: str


_NO_ARGUMENTS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def _tool_spec(tool: ReportTool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": _NO_ARGUMENTS,
        "strict": True,
    }


def _report_conversation(brief: ReportBrief) -> list[dict[str, Any]]:
    """The prompt, then each rejected draft as the agent's turn and the audit's failures as the
    reply to it: the re-prompt."""
    sections = "\n\n".join(_section(section) for section in brief.sections)
    conversation = [_message("user", REPORT_PROMPT.format(sections=sections))]
    for draft in brief.rejected:
        failures = "\n".join(f"- {f.number!r} in: {f.sentence!r}" for f in draft.failures)
        conversation.append(_message("assistant", draft.text))
        conversation.append(_message("user", REPORT_REJECTION.format(failures=failures)))
    return conversation


def _section(section: ReportSection) -> str:
    lines = [f"## {section.heading}"]
    if section.covers:
        lines.append(f"Write: {section.covers}.")
    else:
        lines.append("Write nothing here: the tools hold nothing on it.")
    if section.to_fill:
        lines.append(f"Left out: {'; '.join(section.to_fill)}.")
    return "\n".join(lines)


def _message(role: Literal["user", "assistant"], text: str) -> dict[str, Any]:
    kind = "input_text" if role == "user" else "output_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _function_calls(stream: Iterable[Event]) -> list[_FunctionCall]:
    """The function calls in a stream, in order, from its finished output items."""
    calls: list[_FunctionCall] = []
    for event in stream:
        item = event.get("item")
        if event.get("type") != "response.output_item.done" or not isinstance(item, Mapping):
            continue
        fields = cast(Mapping[str, object], item)
        if fields.get("type") == "function_call":
            calls.append(
                _FunctionCall(
                    call_id=str(fields["call_id"]),
                    name=str(fields["name"]),
                    arguments=str(fields.get("arguments") or "{}"),
                )
            )
    return calls


def _check_turn(stream: list[Event]) -> None:
    """A report turn is complete and either calls a tool or writes text."""
    if not _function_calls(stream) and not _completed_text(stream).strip():
        raise JudgeResponseError("Muse Spark neither called a tool nor wrote the report")


def _completed_text(stream: Iterable[Event]) -> str:
    """A completed stream's output text, without its commentary: Muse Spark may send a message
    in `phase: "commentary"` (a remark to the user, seen live on Oct 31) beside its answer."""
    stream = list(stream)
    kinds = {event.get("type") for event in stream}
    if "response.completed" not in kinds:
        raise JudgeResponseError(f"incomplete Muse Spark response: {sorted(map(str, kinds))}")
    commentary = {
        str(fields.get("id"))
        for event in stream
        if isinstance(item := event.get("item"), Mapping)
        and (fields := cast(Mapping[str, object], item)).get("phase") == "commentary"
    }
    return "".join(
        str(event["delta"])
        for event in stream
        if event.get("type") == "response.output_text.delta"
        and event.get("item_id") not in commentary
    )


def _parse[A: BaseModel](stream: Iterable[Event], answer: type[A]) -> A:
    """The `answer` in a completed stream's output text."""
    text = _completed_text(stream)
    try:
        return answer.model_validate_json(text)
    except ValidationError as error:
        raise JudgeResponseError(f"malformed Muse Spark answer {text!r}: {error}") from error
