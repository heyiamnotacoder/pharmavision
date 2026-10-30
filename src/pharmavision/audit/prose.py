"""Finding the numbers in report prose, and the sentences they appear in."""

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")

_NUMBER = re.compile(
    r"""
    (?<![\w.])(?<![^\W\d]-)             # not part of an identifier (B07, OFT_16, OFT-16)
    (?P<mantissa>[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)
    (?:
        \s*[x\N{MULTIPLICATION SIGN}]\s*10                  # 1.6 x 10^-3, also as a times sign
        (?:\^(?P<caret>[-+\N{MINUS SIGN}]?[0-9]+)|(?P<superscript>[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))
      | [eE](?P<e>[-+\N{MINUS SIGN}]?[0-9]+)                 # 1.6e-3
    )?
    (?P<percent>\s?(?:%|per\s?cent\b))?
    """,
    re.VERBOSE,
)

_LINE_BLOCK = re.compile(r"[ \t]*(?:[#|*-]|[0-9]+\.\s)")
"""A Markdown heading, table row or list item: one line, a block of its own."""
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class WrittenNumber:
    text: str
    value: float
    """The magnitude, in the written units: signs are read as words ("decreased by") as often
    as minus signs, so they are not part of the number."""
    half_unit: float
    """Half a unit in the last written digit: how far the true value may be from the written
    one when it was rounded."""
    percent: bool
    span: tuple[int, int]


def written_numbers(prose: str, allowed_phrases: Sequence[str]) -> list[WrittenNumber]:
    """Every number in `prose`, in order, except those inside an allowed phrase."""
    searched = _blank_out(prose, allowed_phrases)
    return [_written(match) for match in _NUMBER.finditer(searched)]


def sentences(prose: str) -> list[tuple[int, int]]:
    """The span of each sentence in `prose`, and of each heading, table row and list item."""
    spans: list[tuple[int, int]] = []
    for block_start, block_end in _blocks(prose):
        start = block_start
        for brk in [*_SENTENCE_END.finditer(prose, block_start, block_end), None]:
            end = block_end if brk is None else brk.start()
            spans.append((start, end))
            if brk is not None:
                start = brk.end()
    return spans


def sentence_at(spans: list[tuple[int, int]], offset: int) -> tuple[int, int]:
    """The span in `spans` (from `sentences`) holding `offset`."""
    return spans[bisect.bisect_right([s for s, _ in spans], offset) - 1]


def _blocks(prose: str) -> list[tuple[int, int]]:
    """The span of each paragraph (lines up to a blank line) and each line-block, stripped."""
    blocks: list[tuple[int, int]] = []
    paragraph: tuple[int, int] | None = None
    offset = 0
    for line in prose.splitlines(keepends=True):
        text = line.strip()
        start = offset + line.index(text) if text else offset
        offset += len(line)
        if not text or _LINE_BLOCK.match(line):
            if paragraph is not None:
                blocks.append(paragraph)
            paragraph = None
            if text:
                blocks.append((start, start + len(text)))
        else:
            paragraph = (paragraph[0] if paragraph else start, start + len(text))
    if paragraph is not None:
        blocks.append(paragraph)
    return blocks


def _blank_out(prose: str, phrases: Sequence[str]) -> str:
    """`prose` with each allowed phrase replaced by spaces, so offsets stay the same."""
    for phrase in phrases:
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w|\.[0-9])", re.IGNORECASE)
        prose = pattern.sub(lambda m: " " * len(m.group()), prose)
    return prose


def _written(match: re.Match[str]) -> WrittenNumber:
    mantissa = match["mantissa"].replace(",", "")
    decimals = len(mantissa.partition(".")[2])
    power = match["caret"] or match["e"] or (match["superscript"] or "").translate(_SUPERSCRIPTS)
    exponent = int(power.replace("\N{MINUS SIGN}", "-")) if power else 0
    scale = 10.0**exponent
    text = match.group().strip()
    return WrittenNumber(
        text=text,
        value=float(mantissa) * scale,
        half_unit=0.5 * 10.0**-decimals * scale,
        percent=match["percent"] is not None,
        span=(match.start(), match.start() + len(text)),
    )
