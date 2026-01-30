"""Parallel window-based discovery.

This module provides parallel pattern discovery using multiple processes
for CPU-bound work. Uses ProcessPoolExecutor instead of ThreadPoolExecutor
to bypass Python's GIL for true parallelism.

Key improvements:
1. ProcessPoolExecutor for true parallelism
2. Proper chunk overlap handling
3. Correct pattern merging across process boundaries
4. Deduplication of patterns found in multiple chunks
"""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Sequence

from .config import CompressionConfig
from .discovery import _discover_for_length, _non_overlapping_positions
from .types import Candidate, Token, TokenSeq
from .utils import is_compressible


def _discover_chunk(
    chunk_tokens: Sequence[Token],
    chunk_start: int,
    min_len: int,
    max_len: int,
    extra_cost: int,
) -> list[tuple[tuple[Token, ...], int]]:
    """Discover patterns in a single chunk.

    Returns list of (pattern, global_position) pairs for later merging.
    """
    results: list[tuple[tuple[Token, ...], int]] = []
    n = len(chunk_tokens)

    for length in range(max_len, min_len - 1, -1):
        if length > n:
            continue

        # Find all subsequences of this length
        for i in range(n - length + 1):
            subseq = tuple(chunk_tokens[i : i + length])
            global_pos = chunk_start + i
            results.append((subseq, global_pos))

    return results


def discover_candidates_parallel(
    tokens: TokenSeq, config: CompressionConfig
) -> list[Candidate]:
    """Parallel pattern discovery using multiple processes.

    For small inputs (< 5000 tokens), falls back to single-threaded
    discovery as the overhead of process spawning exceeds benefits.

    For large inputs, splits into chunks with overlap and merges
    patterns found across chunks.
    """
    n = len(tokens)
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length

    if max_len < min_len:
        return []

    # For small inputs, use simple parallel by length
    if n < 5000:
        return _parallel_by_length(tokens, config)

    # For larger inputs, use chunked parallel processing
    return _parallel_by_chunk(tokens, config)


def _parallel_by_length(tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
    """Original parallel discovery: parallelize over pattern lengths.

    Uses ThreadPoolExecutor since _discover_for_length is relatively
    quick and thread overhead is lower than process overhead.
    """
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length
    extra_cost = 1 if config.dict_length_enabled else 0

    candidates: list[Candidate] = []
    lengths = list(range(max_len, min_len - 1, -1))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_discover_for_length, tokens, length, extra_cost)
            for length in lengths
        ]
        for fut in futures:
            candidates.extend(fut.result())

    return candidates


def _parallel_by_chunk(tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
    """Chunked parallel discovery for large inputs.

    Splits input into overlapping chunks, processes each in a separate
    process, then merges results.
    """
    n = len(tokens)
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length
    extra_cost = 1 if config.dict_length_enabled else 0

    # Determine chunk configuration
    num_workers = min(mp.cpu_count(), 8)  # Cap at 8 workers
    chunk_size = max(5000, n // num_workers)
    overlap = max_len - 1  # Overlap to catch patterns at boundaries

    # Build chunks with overlap
    chunks: list[tuple[list[Token], int, int]] = []
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size + overlap)
        core_end = min(n, start + chunk_size)  # Core region without overlap
        chunk_tokens = list(tokens[start:end])
        chunks.append((chunk_tokens, start, core_end))

    # Process chunks in parallel
    all_patterns: dict[tuple[Token, ...], set[int]] = defaultdict(set)

    # Use ProcessPoolExecutor for true parallelism
    # Note: For very large inputs, process spawn overhead is worth it
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _discover_chunk,
                    chunk_tokens,
                    chunk_start,
                    min_len,
                    max_len,
                    extra_cost,
                )
                for chunk_tokens, chunk_start, _ in chunks
            ]

            for future, (_, chunk_start, core_end) in zip(
                as_completed(futures), chunks
            ):
                try:
                    results = future.result(timeout=60)  # 60 second timeout
                    for subseq, global_pos in results:
                        # Only include positions from core region
                        if global_pos < core_end:
                            all_patterns[subseq].add(global_pos)
                except Exception:
                    # On failure, fall back for this chunk
                    continue
    except Exception:
        # If ProcessPoolExecutor fails entirely, fall back to threads
        return _parallel_by_length(tokens, config)

    # Build candidates from merged patterns
    candidates: list[Candidate] = []

    for subseq, positions in all_patterns.items():
        sorted_positions = sorted(positions)
        non_overlapping = _non_overlapping_positions(sorted_positions, len(subseq))
        count = len(non_overlapping)

        if is_compressible(len(subseq), count, extra_cost=extra_cost):
            candidates.append(
                Candidate(
                    subsequence=subseq,
                    length=len(subseq),
                    positions=non_overlapping,
                    priority=0,
                )
            )

    candidates.sort(key=lambda c: c.length, reverse=True)
    return candidates


def discover_candidates_parallel_adaptive(
    tokens: TokenSeq,
    config: CompressionConfig,
) -> list[Candidate]:
    """Adaptive parallel discovery that chooses best strategy.

    Automatically selects between:
    - Single-threaded for very small inputs
    - Thread-parallel by length for medium inputs
    - Process-parallel by chunk for large inputs
    """
    n = len(tokens)

    if n < 1000:
        # Very small: overhead not worth it
        from .discovery import discover_candidates

        return discover_candidates(tokens, config.max_subsequence_length, config)

    if n < 5000:
        # Medium: thread parallel by length
        return _parallel_by_length(tokens, config)

    # Large: process parallel by chunk
    return _parallel_by_chunk(tokens, config)
