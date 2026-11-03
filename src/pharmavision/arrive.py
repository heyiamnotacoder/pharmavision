"""The report's shape: the ARRIVE 2.0 Essential 10, one section each. The agent writes what the
tools hold; the facts the system cannot know are the stage's `[TO FILL]` placeholders, never the
agent's, so it cannot invent them. Pure: text in, text out."""

import re
from collections.abc import Sequence

from .artifacts import AuditedNumber
from .judge import ReportSection

TITLE = "# Methods and results"

SECTIONS: tuple[ReportSection, ...] = (
    ReportSection(
        "Study design",
        "the groups compared, which is the reference group, and the experimental unit (one "
        "trial is one animal tested once)",
        (),
    ),
    ReportSection(
        "Sample size",
        "the number of trials in each group and in total",
        ("how the sample size was decided",),
    ),
    ReportSection(
        "Inclusion and exclusion criteria",
        "the automatic quality-control rules, and every exclusion made, with its trial, its "
        "frames and its reason, and the number of trials analysed per group",
        ("criteria applied before testing, such as health checks",),
    ),
    ReportSection(
        "Randomisation",
        "",
        ("how animals were allocated to groups", "the order in which animals were tested"),
    ),
    ReportSection(
        "Blinding",
        "that every trial was scored automatically from video, identified by its blinding code",
        ("who knew the group allocation during the experiment and the analysis",),
    ),
    ReportSection(
        "Outcome measures",
        "each metric and how it was measured from the tracked animal",
        (),
    ),
    ReportSection(
        "Statistical methods",
        "the rule that chose each test and why it chose it, the tests, the effect sizes and "
        "their confidence intervals",
        (),
    ),
    ReportSection(
        "Experimental animals",
        "the species filmed",
        ("strain, sex, age, weight and source of the animals", "housing and husbandry"),
    ),
    ReportSection(
        "Experimental procedures",
        "the treatments and doses, the arena, the test duration, and how the video was "
        "recorded and analysed",
        (
            "route, volume and timing of drug administration",
            "time of day and place of testing",
        ),
    ),
    ReportSection(
        "Results",
        "for each metric: n, mean and SD per group, the test, its statistic, degrees of "
        "freedom and p, and the effect size with its interval; the low-confidence events; and "
        "that the pipeline makes no rear count, with the number of rear candidate windows",
        (),
    ),
)

UNWRITTEN = "nothing was written for this item"
STRUCK_FLAG = "**[failed the number audit]**"

_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+(?P<title>.*?)[ \t#]*$", re.MULTILINE)
_NUMBERING = re.compile(r"^[0-9]+[.)]?\s*")
_LINE_PREFIX = re.compile(r"[ \t]*(?:#+[ \t]+|>[ \t]*|[-*+][ \t]+|[0-9]+[.)][ \t]+)?")
_TABLE_ROW = re.compile(r"[ \t]*\|")
_CELL_BORDER = re.compile(r"(?<!\\)\|")
"""A table cell's border: a pipe not escaped."""


def placeholder(fact: str) -> str:
    return f"[TO FILL: {fact}]"


def assemble(text: str, sections: Sequence[ReportSection] = SECTIONS) -> str:
    """The report from the agent's Markdown: under each section's heading, the agent's text
    under its own heading of that name (a placeholder when it wrote none), then a placeholder
    per fact to fill. Text under any other heading, and the agent's text for a section it is
    not to write, is left out."""
    written = _by_heading(text)
    parts = [TITLE]
    for section in sections:
        parts.append(f"## {section.heading}")
        if section.covers:
            parts.append(written.get(_key(section.heading)) or placeholder(UNWRITTEN))
        parts.extend(placeholder(fact) for fact in section.to_fill)
    return "\n\n".join(parts) + "\n"


def strike(markdown: str, failed: Sequence[AuditedNumber]) -> str:
    """`markdown` with each sentence that holds a failed number struck out and flagged. A
    heading's, quote's or list item's marker stays outside the strike, and a table row (one
    sentence to the audit) is struck cell by cell, so each still parses as what it was."""
    spans = sorted({n.sentence_span for n in failed}, reverse=True)
    for start, end in spans:
        sentence = markdown[start:end]
        if _TABLE_ROW.match(sentence):
            struck = _strike_row(sentence)
        else:
            prefix = _LINE_PREFIX.match(sentence)
            cut = prefix.end() if prefix else 0
            struck = f"{sentence[:cut]}~~{sentence[cut:]}~~ {STRUCK_FLAG}"
        markdown = markdown[:start] + struck + markdown[end:]
    return markdown


def _strike_row(row: str) -> str:
    """A table row with each cell's text struck and the flag in its last cell: a strike across
    the borders would not parse, since GFM splits a row into cells before inline marks."""
    cells = _CELL_BORDER.split(row)
    filled = [i for i, cell in enumerate(cells) if i and cell.strip()]
    for i in filled:
        cell = cells[i]
        lead, trail = cell[: len(cell) - len(cell.lstrip())], cell[len(cell.rstrip()) :]
        text = cell.strip()
        flag = f" {STRUCK_FLAG}" if i == filled[-1] else ""
        cells[i] = f"{lead}~~{text}~~{flag}{trail}"
    return "|".join(cells)


def _by_heading(text: str) -> dict[str, str]:
    """The text under each heading, by its key; the first wins when a heading repeats."""
    headings = list(_HEADING.finditer(text))
    written: dict[str, str] = {}
    for heading, after in zip(headings, [*headings[1:], None], strict=True):
        body = text[heading.end() : after.start() if after else len(text)].strip()
        written.setdefault(_key(heading["title"]), body)
    return written


def _key(heading: str) -> str:
    """A heading as compared: case, numbering ("4. Randomisation") and a colon ignored."""
    return _NUMBERING.sub("", heading.strip()).rstrip(":").strip().casefold()
