"""Spend (NFR-4): what each live model call cost, from the usage the API reports with it, and
what a run of the eval videos would cost before it starts. A call served from the model cache
never reaches the transport, so it costs nothing and is not metered.

Prices: dev.meta.ai/docs/pricing-rate-limits, checked Oct 30, 2026 (ARCHITECTURE "Verified
facts"). A model with no price here is refused, never billed as free."""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from pharmavision.model_api import Request
from pharmavision.model_cache import Event
from pharmavision.segmenter.chunks import plan_chunks
from pharmavision.segmenter.meta_api import MetaApiSettings, SamTransport


class UnknownPriceError(KeyError):
    """No price is known for the model, so its spend cannot be logged or projected."""


class Price(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_m: float = 0.0
    cached_input_per_m: float = 0.0
    output_per_m: float = 0.0
    per_video_frame: float = 0.0
    per_image: float = 0.0
    min_video_frames: int = 0
    """Billed per video request, however few frames it holds."""


PRICES: dict[str, Price] = {
    "sam-3.1": Price(per_video_frame=0.20 / 1000, per_image=2.50 / 1000, min_video_frames=50),
    # No cached-input price is published for the standard tier: charged as full input.
    "muse-spark-1.3": Price(input_per_m=1.25, cached_input_per_m=1.25, output_per_m=4.25),
    "muse-spark-1.3-contributor": Price(
        input_per_m=0.10, cached_input_per_m=0.002, output_per_m=0.20
    ),
}


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cached_tokens: int = 0
    """Of `input_tokens`, those served from the prompt cache."""
    output_tokens: int = 0
    video_frames: int = 0
    images: int = 0

    @classmethod
    def from_response(cls, usage: Mapping[str, Any]) -> "Usage":
        """From a Responses `usage` block; Meta adds `video_frames_processed` and
        `images_processed` for SAM 3.1."""
        details = usage.get("input_tokens_details") or {}
        return cls(
            input_tokens=usage.get("input_tokens") or 0,
            cached_tokens=details.get("cached_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            video_frames=usage.get("video_frames_processed") or 0,
            images=usage.get("images_processed") or 0,
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            **{f: getattr(self, f) + getattr(other, f) for f in Usage.model_fields},
        )


def cost_usd(model: str, usage: Usage) -> float:
    """What one response with this usage cost."""
    price = PRICES.get(model)
    if price is None:
        raise UnknownPriceError(f"no price known for {model!r}; add it to eval.spend.PRICES")
    frames = max(usage.video_frames, price.min_video_frames) if usage.video_frames else 0
    return (
        (usage.input_tokens - usage.cached_tokens) * price.input_per_m / 1e6
        + usage.cached_tokens * price.cached_input_per_m / 1e6
        + usage.output_tokens * price.output_per_m / 1e6
        + frames * price.per_video_frame
        + usage.images * price.per_image
    )


class Call(BaseModel):
    """One live, billed response."""

    model_config = ConfigDict(frozen=True)

    model: str
    usage: Usage
    usd: float


class MeteredTransport:
    """Passes every call through to `inner` and keeps the usage of each completed response."""

    def __init__(self, inner: SamTransport) -> None:
        self.inner = inner
        self._calls: list[Call] = []

    def upload(self, video: Path) -> str:
        return self.inner.upload(video)

    def stream(self, request: Request) -> Iterator[Event]:
        for event in self.inner.stream(request):
            if event.get("type") == "response.completed":
                response = event.get("response")
                block = response.get("usage") if isinstance(response, Mapping) else None
                usage = Usage.from_response(block if isinstance(block, Mapping) else {})
                model = str(request["model"])
                self._calls.append(Call(model=model, usage=usage, usd=cost_usd(model, usage)))
            yield event

    def take(self) -> list[Call]:
        """The calls metered since the last `take`."""
        calls, self._calls = self._calls, []
        return calls


class ModelSpend(BaseModel):
    model_config = ConfigDict(frozen=True)

    calls: int
    usage: Usage
    usd: float


def by_model(calls: Sequence[Call]) -> dict[str, ModelSpend]:
    """Calls summed per model: what one video's log records."""
    out: dict[str, ModelSpend] = {}
    for call in calls:
        seen = out.get(call.model, ModelSpend(calls=0, usage=Usage(), usd=0.0))
        out[call.model] = ModelSpend(
            calls=seen.calls + 1, usage=seen.usage + call.usage, usd=seen.usd + call.usd
        )
    return out


class VideoPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    duration_s: float
    spent_usd: float | None = None
    """What its finished run cost, from its log; None while it is still to run."""


class ProjectedVideo(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    done: bool
    projected_usd: float
    spent_usd: float | None


class Projection(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: list[ProjectedVideo]

    @property
    def to_run(self) -> list[str]:
        return [r.video_id for r in self.rows if not r.done]

    @property
    def total_usd(self) -> float:
        return sum(r.projected_usd for r in self.rows)

    @property
    def spent_usd(self) -> float:
        return sum(r.spent_usd or 0.0 for r in self.rows)


def project(plans: Sequence[VideoPlan], fps: int, sam: MetaApiSettings | None = None) -> Projection:
    """SAM 3.1's price for the arena (one floor image) and the track (every frame at `fps`, in
    the chunks the segmenter would send) of each video still to run. An upper bound: responses
    already in the model cache cost nothing, and a floor found only on a later prompt costs one
    image more per prompt tried."""
    sam = sam or MetaApiSettings()
    rows: list[ProjectedVideo] = []
    for plan in plans:
        projected = 0.0
        if plan.spent_usd is None:
            n_frames = round(plan.duration_s * fps)
            chunks = plan_chunks(n_frames, sam.max_frames_per_request, sam.chunk_overlap_frames)
            projected = cost_usd(sam.model, Usage(images=1)) + sum(
                cost_usd(sam.model, Usage(video_frames=len(c))) for c in chunks
            )
        rows.append(
            ProjectedVideo(
                video_id=plan.video_id,
                done=plan.spent_usd is not None,
                projected_usd=projected,
                spent_usd=plan.spent_usd,
            )
        )
    return Projection(rows=rows)


def format_projection(projection: Projection) -> str:
    """The table printed before a run: each video's projected cost, or what it cost if done."""
    lines = [f"{'video':<9} {'status':<8} {'projected $':>11} {'spent $':>9}"]
    for row in projection.rows:
        spent = "-" if row.spent_usd is None else f"{row.spent_usd:.4f}"
        status = "done" if row.done else "to run"
        lines.append(f"{row.video_id:<9} {status:<8} {row.projected_usd:>11.4f} {spent:>9}")
    lines.append(
        f"{len(projection.to_run)} of {len(projection.rows)} videos to run: "
        f"projected ${projection.total_usd:.2f} at most (cached responses cost nothing); "
        f"${projection.spent_usd:.2f} spent so far"
    )
    return "\n".join(lines)
