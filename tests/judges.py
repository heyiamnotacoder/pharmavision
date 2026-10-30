"""A stand-in Judge for tests that are not about the model: one fixed answer for every clip."""

from dataclasses import dataclass, field

from pharmavision.judge import Clip, ClipJudgement, FlaggedWindow, WindowVerdict


@dataclass(frozen=True)
class Reviewed:
    """A flagged window as the judge was shown it, its clip's bytes read before they are gone."""

    window: FlaggedWindow
    clip: bytes


@dataclass
class ScriptedJudge:
    """Answers every clip with `judgement` and every flagged window with `verdict`, and keeps
    the bytes of every clip it was shown."""

    judgement: ClipJudgement = field(
        default_factory=lambda: ClipJudgement(label="none", confidence=0.9, rationale="scripted")
    )
    verdict: WindowVerdict = field(
        default_factory=lambda: WindowVerdict(verdict="keep", reason="scripted")
    )
    model_id: str = "scripted"
    clips: list[bytes] = field(default_factory=list[bytes])
    windows: list[Reviewed] = field(default_factory=list[Reviewed])

    def classify_clip(self, clip: Clip) -> ClipJudgement:
        self.clips.append(clip.video.read_bytes())
        return self.judgement

    def review_window(self, window: FlaggedWindow) -> WindowVerdict:
        self.windows.append(Reviewed(window, window.clip.video.read_bytes()))
        return self.verdict
