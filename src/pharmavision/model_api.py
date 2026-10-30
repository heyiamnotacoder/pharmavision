"""The Meta Model API, which SAM 3.1 and Muse Spark share: an OpenAI-compatible Responses API,
streamed, so every response is a list of JSON events that the cache stores as they came."""

import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

from .model_cache import Event

Request = Mapping[str, Any]
"""A Responses request, untyped because `input_video` and SAM's `metadata` keys are Meta
extensions the OpenAI SDK's types lack."""


class ResponsesTransport(Protocol):
    """The one call every model backend makes. Tests replay recordings through this."""

    def stream(self, request: Request) -> Iterable[Event]:
        """Stream one Responses API request; yields every event as a JSON mapping."""
        ...


class OpenAiTransport:
    """The Model API is OpenAI-compatible. The client is made on first use, so a run served
    entirely from the cache needs no API key."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._client: Any = None

    def upload(self, video: Path) -> str:
        with video.open("rb") as f:
            return self._openai().files.create(purpose="user_data", file=f).id

    def stream(self, request: Request) -> Iterator[Event]:
        for event in self._openai().responses.create(stream=True, **request):
            yield event.model_dump(mode="json")

    def _openai(self) -> Any:
        if self._client is None:
            from dotenv import find_dotenv, load_dotenv
            from openai import OpenAI

            load_dotenv(find_dotenv(usecwd=True))
            self._client = OpenAI(base_url=self.base_url, api_key=os.environ["MODEL_API_KEY"])
        return self._client
