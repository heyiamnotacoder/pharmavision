"""The blocking gate: each number in the prose against the known values."""

from collections.abc import Sequence

from ..artifacts import AuditedNumber, AuditSettings, KnownValue, NumberAudit
from .prose import WrittenNumber, sentence_at, sentences, written_numbers


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
    """The source of the known value nearest the written number, of those that round to it. On a
    tie a field wins over a list's length, then the first one."""
    matches = [
        (error, k.source.startswith("len("), k.source)
        for k in known
        if (error := number.rounding_error(k.value)) is not None
    ]
    return min(matches, key=lambda m: m[:2])[2] if matches else None
