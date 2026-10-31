"""The single boundary to the "judges" layer: clips and tables in, labels and text out. A judge
never produces a number the report states unchecked: its confidence is its own, recorded as
given, and its report prose passes the number audit before it ships."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

RearLabel = Literal["rearing", "none"]
"""Binary, as T2 decided: Muse Spark could not tell supported from unsupported rears (κ 0.24 on
true rears, under the 0.4 bar; ARCHITECTURE open decision 3)."""


class ClipJudgement(BaseModel):
    """What the model said about one clip."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: RearLabel
    confidence: float = Field(ge=0, le=1)
    """The model's own confidence in its label, 0 to 1."""
    rationale: str


QcVerdict = Literal["keep", "exclude_window", "exclude_trial"]
"""What to do with a flagged window: keep it, leave its frames out, or leave the trial out."""


class WindowVerdict(BaseModel):
    """What the model said about one flagged window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: QcVerdict
    reason: str


class HandSighting(BaseModel):
    """A "hand" or "glove" prompt's largest mask on one sampled frame of a flagged window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    frame_index: int
    area_cm2: float


@dataclass(frozen=True)
class Clip:
    """A short MP4 cut from a run's video, and the hash of the frames it was encoded from. The
    hash is the clip's identity for the response cache: the MP4's bytes depend on the ffmpeg
    build that encoded it, so a response recorded on the laptop would never match in CI."""

    video: Path
    frames_sha256: str


@dataclass(frozen=True)
class FlaggedWindow:
    """A QC flag for the judge to review: the clip around its frames, what the rule found there,
    and what the hand prompts found on its sampled frames."""

    clip: Clip
    kind: str
    """The flag's kind, e.g. "second_mask"."""
    first_frame: int
    last_frame: int
    reason: str
    """The rule's own reason, as the QC artifact records it."""
    hands: tuple[HandSighting, ...]


@dataclass(frozen=True)
class ReportTool:
    """One tool the report agent may call. It takes no arguments and returns a stored table as
    JSON text: every tool reads, none writes, none computes a number."""

    name: str
    description: str
    read: Callable[[], str]


@dataclass(frozen=True)
class ReportSection:
    """One ARRIVE 2.0 Essential 10 item, under its own heading in the report."""

    heading: str
    covers: str
    """What the agent writes under it, from the tools; empty when the tools hold nothing for it
    and the agent writes nothing there."""
    to_fill: tuple[str, ...]
    """Facts the system cannot know: the report stage writes a `[TO FILL]` for each, never the
    agent."""


@dataclass(frozen=True)
class AuditFailure:
    """A number in a rejected draft that matches no value the tools returned."""

    number: str
    sentence: str


@dataclass(frozen=True)
class RejectedDraft:
    """A draft the number audit failed: the agent's text, and the numbers that failed it."""

    text: str
    failures: tuple[AuditFailure, ...]


@dataclass(frozen=True)
class ReportBrief:
    """What the report agent is asked to write, with what, and what it got wrong before."""

    sections: tuple[ReportSection, ...]
    tools: tuple[ReportTool, ...]
    rejected: tuple[RejectedDraft, ...] = ()
    """Earlier drafts of this report the audit failed, oldest first: the re-prompt."""


class Judge(Protocol):
    @property
    def model_id(self) -> str:
        """Recorded in the manifest so a run says which model judged it."""
        ...

    def write_report(self, brief: ReportBrief) -> str:
        """The report's Markdown, one `## <heading>` per section of `brief`, written from what
        the brief's tools return. Its numbers are audited after, never trusted."""
        ...

    def classify_clip(self, clip: Clip) -> ClipJudgement:
        """Label one short clip. It is never the full video."""
        ...

    def review_window(self, window: FlaggedWindow) -> WindowVerdict:
        """Decide what one flagged window's frames are worth. Its clip is never the full video."""
        ...
