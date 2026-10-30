"""The single boundary to the "judges" layer: clips in, labels and text out. A judge never
produces a number the report states; its confidence is its own, recorded as given."""

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


@dataclass(frozen=True)
class Clip:
    """A short MP4 cut from a run's video, and the hash of the frames it was encoded from. The
    hash is the clip's identity for the response cache: the MP4's bytes depend on the ffmpeg
    build that encoded it, so a response recorded on the laptop would never match in CI."""

    video: Path
    frames_sha256: str


class Judge(Protocol):
    @property
    def model_id(self) -> str:
        """Recorded in the manifest so a run says which model judged it."""
        ...

    def classify_clip(self, clip: Clip) -> ClipJudgement:
        """Label one short clip. It is never the full video."""
        ...
