"""Parallel window-based discovery."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .config import CompressionConfig
from .discovery import _discover_for_length
from .types import Candidate, TokenSeq


def discover_candidates_parallel(tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length
    if max_len < min_len:
        return []
    extra_cost = 1 if config.dict_length_enabled else 0

    candidates: list[Candidate] = []
    lengths = list(range(max_len, min_len - 1, -1))
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_discover_for_length, tokens, length, extra_cost) for length in lengths]
        for fut in futures:
            candidates.extend(fut.result())
    return candidates
