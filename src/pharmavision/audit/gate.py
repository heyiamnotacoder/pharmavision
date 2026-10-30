"""The blocking gate: each number in the prose against the known values."""

from collections.abc import Sequence

from ..artifacts import AuditedNumber, AuditSettings, KnownValue, NumberAudit
from .prose import WrittenNumber, sentence_at, sentences, written_numbers

_FLOAT_SLACK = 1e-9
"""Relative slack on the rounding bound, so 0.945 written as 0.95 is not lost to binary floats."""


def audit_numbers(prose: str, known: Sequence[KnownValue], settings: AuditSettings) -> NumberAudit:
    """Every number in `prose` matched against `known` (from `known_values`). The audit passes
    only when each one matches; with nothing known, any number fails (abstain, don't guess)."""
    spans = sentences(prose)
    numbers: list[AuditedNumber] = []
    for number in written_numbers(prose, settings.allowed_phrases):
        start, end = sentence_at(spans, number.span[0])
        numbers.append(
            AuditedNumber(
                text=number.text,
                value=number.value,
                percent=number.percent,
                span=number.span,
                sentence=prose[start:end],
                sentence_span=(start, end),
                source=_closest_match(number, known),
            )
        )
    return NumberAudit(settings=settings, numbers=numbers)


def _closest_match(number: WrittenNumber, known: Sequence[KnownValue]) -> str | None:
    """The source of the known value nearest the written number, of those that round to it;
    the first such on a tie."""
    bound = number.half_unit + _FLOAT_SLACK * max(number.value, number.half_unit)
    best: tuple[float, str] | None = None
    for k in known:
        distance = abs(abs(k.value) * (100 if number.percent else 1) - number.value)
        if distance <= bound and (best is None or distance < best[0]):
            best = (distance, k.source)
    return None if best is None else best[1]
