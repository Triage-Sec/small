"""Suffix-array-based discovery for repeated substrings."""

from __future__ import annotations

from collections import defaultdict

from .config import CompressionConfig
from .suffix_array import build_suffix_array, lcp_intervals
from .suffix_array_fast import (
    build_suffix_array_fast,
    lcp_intervals_fast,
    should_use_fast_suffix_array,
)
from .types import Candidate, TokenSeq
from .utils import is_compressible


def _count_non_overlapping(positions: list[int], length: int) -> int:
    """Count non-overlapping occurrences in O(n) without building the full list."""
    count = 0
    next_free = -1
    for pos in positions:
        if pos >= next_free:
            count += 1
            next_free = pos + length
    return count


def _non_overlapping_positions(positions: list[int], length: int) -> list[int]:
    """Extract non-overlapping positions."""
    result: list[int] = []
    next_free = -1
    for pos in positions:
        if pos >= next_free:
            result.append(pos)
            next_free = pos + length
    return result


def discover_candidates_sa(
    tokens: TokenSeq, config: CompressionConfig
) -> list[Candidate]:
    """Discover compressible candidates using suffix array with maximal repeat optimization.

    Optimization: Instead of iterating through ALL lengths within each LCP interval,
    we work from longest to shortest and break early when we find a compressible length.
    This avoids generating massive candidate lists for highly repetitive input.

    Automatically uses numpy-accelerated suffix array for inputs > 1000 tokens.
    """
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length
    if max_len < min_len:
        return []

    n = len(tokens)
    extra_cost = 1 if config.dict_length_enabled else 0

    # Auto-select fast implementation for large inputs
    if should_use_fast_suffix_array(n):
        sa_fast = build_suffix_array_fast(tokens)
        # Convert numpy arrays to lists for uniform interface
        suffix_array = list(sa_fast.suffix_array)
        intervals = lcp_intervals_fast(sa_fast, min_len)
    else:
        sa = build_suffix_array(tokens)
        suffix_array = sa.suffix_array
        intervals = lcp_intervals(sa, min_len)

    # Track which patterns we've already found to avoid duplicates
    # Key insight: if we found "abcd" compressible, we may still want "ab" if it has
    # additional independent occurrences. But we process longest first for efficiency.
    seen_subseqs: set[tuple] = set()
    candidates: list[Candidate] = []

    for start, end, lcp_len in intervals:
        positions = sorted(suffix_array[idx] for idx in range(start, end + 1))
        length_limit = min(lcp_len, max_len)

        # Work from longest to shortest, finding maximal compressible lengths
        for length in range(length_limit, min_len - 1, -1):
            # Quick count check before building the subsequence
            non_overlap_count = _count_non_overlapping(positions, length)

            if not is_compressible(length, non_overlap_count, extra_cost=extra_cost):
                continue

            # Build the subsequence
            if positions[0] + length > n:
                continue
            subseq = tuple(tokens[positions[0] : positions[0] + length])

            if subseq in seen_subseqs:
                # Already found this pattern - but with different positions, merge them
                continue

            # Get actual non-overlapping positions
            non_overlapping = _non_overlapping_positions(positions, length)

            # Double-check compressibility with actual count
            if is_compressible(length, len(non_overlapping), extra_cost=extra_cost):
                seen_subseqs.add(subseq)
                candidates.append(
                    Candidate(
                        subsequence=subseq,
                        length=length,
                        positions=tuple(non_overlapping),
                        priority=0,
                    )
                )
                # Don't break - shorter patterns might have more occurrences
                # in different positions and still be valuable

    # Also collect shorter patterns that might appear in different contexts
    # Use a secondary pass with position tracking to find additional candidates
    positions_by_subseq: dict[tuple, set[int]] = defaultdict(set)

    for start, end, lcp_len in intervals:
        positions = sorted(suffix_array[idx] for idx in range(start, end + 1))
        length_limit = min(lcp_len, max_len)

        for length in range(min_len, length_limit + 1):
            if positions[0] + length > n:
                continue
            subseq = tuple(tokens[positions[0] : positions[0] + length])
            positions_by_subseq[subseq].update(positions)

    # Process aggregated positions for patterns we haven't seen
    for subseq, all_positions in positions_by_subseq.items():
        if subseq in seen_subseqs:
            continue

        sorted_positions = sorted(all_positions)
        non_overlapping = _non_overlapping_positions(sorted_positions, len(subseq))

        if is_compressible(len(subseq), len(non_overlapping), extra_cost=extra_cost):
            seen_subseqs.add(subseq)
            candidates.append(
                Candidate(
                    subsequence=subseq,
                    length=len(subseq),
                    positions=tuple(non_overlapping),
                    priority=0,
                )
            )

    candidates.sort(key=lambda cand: cand.length, reverse=True)
    return candidates
