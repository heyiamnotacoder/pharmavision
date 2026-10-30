"""A stand-in Judge for tests that are not about the model: one fixed judgement for every clip."""

from dataclasses import dataclass, field
from pathlib import Path

from pharmavision.judge import ClipJudgement


@dataclass
class ScriptedJudge:
    """Answers every clip with `judgement`, and keeps the bytes of every clip it was shown."""

    judgement: ClipJudgement = field(
        default_factory=lambda: ClipJudgement(label="none", confidence=0.9, rationale="scripted")
    )
    model_id: str = "scripted"
    clips: list[bytes] = field(default_factory=list[bytes])

    def classify_clip(self, clip: Path) -> ClipJudgement:
        self.clips.append(clip.read_bytes())
        return self.judgement
