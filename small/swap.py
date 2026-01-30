"""Phase 2: subsequence swapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import CompressionConfig
from .selection import select_occurrences
from .types import Candidate, Occurrence, Token, TokenSeq
from .utils import generate_meta_token_pool


@dataclass(frozen=True)
class SwapResult:
    replacements: dict[int, tuple[int, Token, tuple]]
    dictionary_map: dict[Token, tuple[Token, ...]]
    meta_tokens_used: tuple[Token, ...]


def perform_swaps(
    tokens: TokenSeq, candidates: Iterable[Candidate], config: CompressionConfig
) -> SwapResult:
    replacements: dict[int, tuple[int, Token, tuple]] = {}
    dictionary_map: dict[Token, tuple[Token, ...]] = {}
    meta_tokens: list[Token] = []

    pool = generate_meta_token_pool(config, tokens)

    # Pass tokens for semantic selection mode (needs context for embeddings)
    selection = select_occurrences(candidates, config, tokens=tokens)
    occurrences_by_subseq: dict[tuple[Token, ...], list[Occurrence]] = {}
    for occ in selection.selected:
        occurrences_by_subseq.setdefault(occ.subsequence, []).append(occ)

    for subseq, occs in occurrences_by_subseq.items():
        if not pool:
            break
        meta = pool.pop()
        dictionary_map[meta] = subseq
        meta_tokens.append(meta)
        for occ in occs:
            replacements[occ.start] = (occ.length, meta, occ.patches)

    return SwapResult(
        replacements=replacements,
        dictionary_map=dictionary_map,
        meta_tokens_used=tuple(meta_tokens),
    )
