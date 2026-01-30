"""Phase 1: subsequence discovery."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .config import CompressionConfig
from .types import Candidate, Token, TokenSeq
from .utils import is_compressible


def _non_overlapping_positions(
    positions: Iterable[int], length: int
) -> tuple[int, ...]:
    selected: list[int] = []
    next_free = -1
    for pos in positions:
        if pos >= next_free:
            selected.append(pos)
            next_free = pos + length
    return tuple(selected)


def _discover_for_length(
    tokens: TokenSeq, length: int, extra_cost: int
) -> list[Candidate]:
    n = len(tokens)
    if length > n or length < 2:
        return []
    positions_by_subseq: dict[tuple[Token, ...], list[int]] = defaultdict(list)
    limit = n - length + 1
    for idx in range(limit):
        subseq = tuple(tokens[idx : idx + length])
        positions_by_subseq[subseq].append(idx)
    candidates: list[Candidate] = []
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


def discover_candidates(
    tokens: TokenSeq, max_length: int, config: CompressionConfig | None = None
) -> list[Candidate]:
    min_length = config.min_subsequence_length if config else 2
    if max_length < min_length:
        return []
    candidates: list[Candidate] = []
    extra_cost = 1 if config and config.dict_length_enabled else 0

    for length in range(max_length, min_length - 1, -1):
        candidates.extend(_discover_for_length(tokens, length, extra_cost))

    return candidates


def discover_candidates_chunked(
    tokens: TokenSeq, config: CompressionConfig
) -> list[Candidate]:
    if config.chunk_size <= 0:
        return discover_candidates(tokens, config.max_subsequence_length, config)
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length
    if max_len < min_len:
        return []
    extra_cost = 1 if config.dict_length_enabled else 0
    n = len(tokens)
    positions_by_subseq: dict[tuple[Token, ...], list[int]] = defaultdict(list)
    overlap = max_len - 1

    for chunk_start in range(0, n, config.chunk_size):
        core_end = min(n, chunk_start + config.chunk_size)
        chunk_end = min(n, core_end + overlap)
        chunk_tokens = tokens[chunk_start:chunk_end]
        for length in range(max_len, min_len - 1, -1):
            if length > len(chunk_tokens):
                continue
            candidates = _discover_for_length(chunk_tokens, length, extra_cost)
            for cand in candidates:
                adjusted_positions = []
                for pos in cand.positions:
                    global_pos = chunk_start + pos
                    if global_pos < core_end:
                        adjusted_positions.append(global_pos)
                if adjusted_positions:
                    positions_by_subseq[cand.subsequence].extend(adjusted_positions)

    result: list[Candidate] = []
    for subseq, positions in positions_by_subseq.items():
        positions = sorted(set(positions))
        non_overlapping = _non_overlapping_positions(positions, len(subseq))
        if is_compressible(len(subseq), len(non_overlapping), extra_cost=extra_cost):
            result.append(
                Candidate(
                    subsequence=subseq,
                    length=len(subseq),
                    positions=non_overlapping,
                    priority=0,
                )
            )
    result.sort(key=lambda cand: cand.length, reverse=True)
    return result
