"""The numbers the report agent could have read: every numeric field of the tool outputs, and
the length of every list in them (a count of flags, exclusions or trials)."""

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from ..artifacts import KnownValue

type Json = bool | int | float | str | Sequence[Json] | Mapping[str, Json] | None


def known_values(tables: Mapping[str, BaseModel | Json]) -> list[KnownValue]:
    """Each table's numbers, sourced by their field path under the table's name. Numbers inside
    strings are not known values: a string may be model-written (a QC review's reason), and the
    audit must not take the model's word for a number."""
    known: list[KnownValue] = []
    for name, table in tables.items():
        dumped = table.model_dump(mode="json") if isinstance(table, BaseModel) else table
        _collect(name, dumped, known)
    return known


def _collect(path: str, value: Json, known: list[KnownValue]) -> None:
    match value:
        case None | bool() | str():
            pass
        case int() | float():
            if math.isfinite(value):
                known.append(KnownValue(source=path, value=float(value)))
        case Mapping():
            for key, item in value.items():
                _collect(f"{path}.{key}", item, known)
        case Sequence():
            known.append(KnownValue(source=f"len({path})", value=float(len(value))))
            for i, item in enumerate(value):
                _collect(f"{path}[{i}]", item, known)
