"""Pattern objects for compression pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from .types import Token


@dataclass(frozen=True)
class Pattern:
    subsequence: tuple[Token, ...]
    positions: tuple[int, ...]
    savings: int
    meta_dependencies: tuple[Token, ...] = ()


def compute_savings(length: int, count: int, extra_cost: int = 0) -> int:
    original = length * count
    compressed = 1 + length + count + extra_cost
    return original - compressed
