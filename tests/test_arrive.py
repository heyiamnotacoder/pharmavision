"""`arrive.strike` at its seam: report Markdown and its failed numbers in, Markdown out that
still parses as the same blocks, with each failing sentence struck and flagged."""

from pharmavision.arrive import STRUCK_FLAG, strike
from pharmavision.artifacts import AuditSettings
from pharmavision.audit import audit_numbers


def struck(markdown: str) -> str:
    """`markdown` with every sentence holding a number struck: nothing is known."""
    return strike(markdown, audit_numbers(markdown, [], AuditSettings()).unmatched)


def test_a_table_row_is_struck_cell_by_cell_so_it_stays_a_table_row() -> None:
    table = "| Group | Distance |\n|---|---|\n| vehicle | 88.8 |\n"

    assert struck(table) == (
        f"| Group | Distance |\n|---|---|\n| ~~vehicle~~ | ~~88.8~~ {STRUCK_FLAG} |\n"
    )


def test_a_quote_or_list_marker_stays_outside_the_strike() -> None:
    assert struck("> Mice ran 12 cm.") == f"> ~~Mice ran 12 cm.~~ {STRUCK_FLAG}"
    assert struck("1) Mice ran 12 cm.") == f"1) ~~Mice ran 12 cm.~~ {STRUCK_FLAG}"
    assert struck("- Mice ran 12 cm.") == f"- ~~Mice ran 12 cm.~~ {STRUCK_FLAG}"
