"""The number audit: every number in report prose must match a value in the computed tables, or
the prose fails (NFR-1). Deterministic and pure, like stats/: the tool outputs and the prose
in, the audit out; no I/O, no model calls."""

from .gate import audit_numbers
from .known import known_values

__all__ = ["audit_numbers", "known_values"]
