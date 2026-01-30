"""Shared types for the LTSC implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

Token = Hashable
TokenSeq = Sequence[Token]
Patch = tuple[int, Token]


@dataclass(frozen=True)
class Candidate:
    subsequence: tuple[Token, ...]
    length: int
    positions: tuple[int, ...]
    priority: int = 0
    patches: dict[int, tuple[Patch, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Occurrence:
    start: int
    length: int
    subsequence: tuple[Token, ...]
    priority: int = 0
    patches: tuple[Patch, ...] = ()


@dataclass(frozen=True)
class CompressionResult:
    original_tokens: tuple[Token, ...]
    compressed_tokens: list[Token]
    serialized_tokens: list[Token]
    dictionary_tokens: list[Token]
    body_tokens: list[Token]
    dictionary_map: dict[Token, tuple[Token, ...]]
    meta_tokens_used: tuple[Token, ...]
    original_length: int
    compressed_length: int
    static_dictionary_id: str | None = None
    metrics: object | None = None
    dictionary: object | None = None

    def verify(self, original_tokens: TokenSeq | None, config) -> None:
        from .compressor import decompress

        target = (
            list(original_tokens)
            if original_tokens is not None
            else list(self.original_tokens)
        )
        restored = decompress(self.serialized_tokens, config)
        if list(restored) != target:
            raise ValueError("Round-trip verification failed.")
