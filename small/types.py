"""Shared types for the LTSC implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

Token = Hashable
TokenSeq = Sequence[Token]


@dataclass(frozen=True)
class Candidate:
    subsequence: tuple[Token, ...]
    length: int
    positions: tuple[int, ...]
    priority: int = 0


@dataclass(frozen=True)
class Occurrence:
    start: int
    length: int
    subsequence: tuple[Token, ...]
    priority: int = 0


@dataclass(frozen=True)
class CompressionResult:
    compressed_tokens: list[Token]
    dictionary_tokens: list[Token]
    body_tokens: list[Token]
    dictionary_map: dict[Token, tuple[Token, ...]]
    meta_tokens_used: tuple[Token, ...]
    original_length: int
    compressed_length: int
    static_dictionary_id: str | None = None
