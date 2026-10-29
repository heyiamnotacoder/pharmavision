"""Model responses on disk, keyed by model id, prompt version and input hash, so a re-run with
the same inputs never pays twice and gives the same answer (NFR-2).

Each entry is the response's raw event stream as gzipped JSONL: the format T2 recorded, so a
response captured on the laptop is also a replayable test fixture."""

import gzip
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Event = Mapping[str, object]


def input_hash(*parts: bytes | str | Mapping[str, Any]) -> str:
    """Hash of everything that decides a model's answer. Mappings hash by sorted keys."""
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, Mapping):
            part = json.dumps(part, sort_keys=True)
        digest.update(part.encode() if isinstance(part, str) else part)
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResponseCache:
    directory: Path

    def get(self, model: str, prompt_version: str, input_digest: str) -> list[Event] | None:
        path = self._path(model, prompt_version, input_digest)
        if not path.exists():
            return None
        with gzip.open(path, "rt") as f:
            return [json.loads(line) for line in f]

    def put(
        self, model: str, prompt_version: str, input_digest: str, events: Sequence[Event]
    ) -> None:
        """Write atomically, so an interrupted run never leaves a truncated entry behind."""
        path = self._path(model, prompt_version, input_digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
        with gzip.open(partial, "wt") as f:
            f.writelines(json.dumps(event) + "\n" for event in events)
        partial.replace(path)

    def _path(self, model: str, prompt_version: str, input_digest: str) -> Path:
        return self.directory / model / prompt_version / f"{input_digest}.jsonl.gz"
