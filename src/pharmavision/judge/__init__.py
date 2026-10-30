"""The single boundary to the "judges" layer: clips in, labels and text out. A judge never
produces a number the report states; its confidence is its own, recorded as given."""

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


class Judge(Protocol):
    @property
    def model_id(self) -> str:
        """Recorded in the manifest so a run says which model judged it."""
        ...

    def classify_clip(self, clip: Path) -> ClipJudgement:
        """Label one short MP4 clip. It is never the full video."""
        ...
