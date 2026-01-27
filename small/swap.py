"""Phase 2: subsequence swapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import CompressionConfig
from .types import Candidate, Token, TokenSeq
from .utils import generate_meta_token_pool, is_compressible


@dataclass(frozen=True)
class SwapResult:
    replacements: dict[int, tuple[int, Token]]
    dictionary_map: dict[Token, tuple[Token, ...]]
    meta_tokens_used: tuple[Token, ...]


def _positions_available(occupied: list[bool], start: int, length: int) -> bool:
    end = start + length
    return not any(occupied[start:end])


def perform_swaps(tokens: TokenSeq, candidates: Iterable[Candidate], config: CompressionConfig) -> SwapResult:
    occupied = [False] * len(tokens)
    replacements: dict[int, tuple[int, Token]] = {}
    dictionary_map: dict[Token, tuple[Token, ...]] = {}
    meta_tokens: list[Token] = []

    pool = generate_meta_token_pool(config, tokens)

    for candidate in candidates:
        available = [
            pos
            for pos in candidate.positions
            if _positions_available(occupied, pos, candidate.length)
        ]
        count = len(available)
        if not is_compressible(candidate.length, count):
            continue
        if not pool:
            break
        meta = pool.pop()
        dictionary_map[meta] = candidate.subsequence
        meta_tokens.append(meta)
        for pos in available:
            for idx in range(pos, pos + candidate.length):
                occupied[idx] = True
            replacements[pos] = (candidate.length, meta)

    return SwapResult(
        replacements=replacements,
        dictionary_map=dictionary_map,
        meta_tokens_used=tuple(meta_tokens),
    )
