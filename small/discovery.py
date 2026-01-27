"""Phase 1: subsequence discovery."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .config import CompressionConfig
from .types import Candidate, Token, TokenSeq
from .utils import is_compressible


def _non_overlapping_positions(positions: Iterable[int], length: int) -> tuple[int, ...]:
    selected: list[int] = []
    next_free = -1
    for pos in positions:
        if pos >= next_free:
            selected.append(pos)
            next_free = pos + length
    return tuple(selected)


def discover_candidates(tokens: TokenSeq, max_length: int, config: CompressionConfig | None = None) -> list[Candidate]:
    if max_length < 2:
        return []
    n = len(tokens)
    candidates: list[Candidate] = []
    extra_cost = 1 if config and config.dict_length_enabled else 0

    for length in range(max_length, 1, -1):
        if length > n:
            continue
        positions_by_subseq: dict[tuple[Token, ...], list[int]] = defaultdict(list)
        limit = n - length + 1
        for idx in range(limit):
            subseq = tuple(tokens[idx : idx + length])
            positions_by_subseq[subseq].append(idx)

        for subseq, positions in positions_by_subseq.items():
            non_overlapping = _non_overlapping_positions(positions, length)
            count = len(non_overlapping)
            if is_compressible(length, count, extra_cost=extra_cost):
                candidates.append(
                    Candidate(
                        subsequence=subseq,
                        length=length,
                        positions=non_overlapping,
                        priority=0,
                    )
                )

    return candidates
