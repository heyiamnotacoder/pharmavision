"""Stretches: maximal runs of consecutive frames that share a value."""

import itertools
from collections.abc import Iterable
from typing import NamedTuple


class Stretch[T](NamedTuple):
    value: T
    start: int
    length: int


def stretches[T](values: Iterable[T]) -> list[Stretch[T]]:
    """Consecutive equal values grouped, in order, with the index of the first one."""
    found: list[Stretch[T]] = []
    start = 0
    for value, group in itertools.groupby(values):
        length = sum(1 for _ in group)
        found.append(Stretch(value, start, length))
        start += length
    return found
