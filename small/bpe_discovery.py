"""BPE-style iterative pattern discovery for maximum compression.

This module implements Byte-Pair Encoding style iterative merging, which often
finds better overall compression solutions than one-shot discovery approaches.

The algorithm:
1. Find the single best adjacent token pair to merge
2. Apply the merge (conceptually)
3. Repeat until no beneficial merges remain

This greedy iterative approach can discover hierarchical patterns that
fixed-length window approaches miss.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from .config import CompressionConfig
from .types import Candidate, Token, TokenSeq
from .utils import is_compressible


@dataclass(frozen=True)
class MergeCandidate:
    """Represents a potential pair merge."""

    pair: tuple[Token, Token]
    positions: tuple[int, ...]
    frequency: int
    savings: int


def _count_adjacent_pairs(tokens: TokenSeq) -> Counter[tuple[Token, Token]]:
    """Count all adjacent token pairs in O(n) time."""
    pairs: Counter[tuple[Token, Token]] = Counter()
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pairs[pair] += 1
    return pairs


def _find_pair_positions(tokens: TokenSeq, pair: tuple[Token, Token]) -> list[int]:
    """Find all non-overlapping positions of a pair."""
    positions: list[int] = []
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            positions.append(i)
            i += 2  # Skip to avoid overlap
        else:
            i += 1
    return positions


def _compute_merge_savings(length: int, count: int, extra_cost: int) -> int:
    """Compute net savings from compressing a pattern.

    Original tokens: length * count
    Compressed: 1 (meta-token) + length (definition) + count (references) + extra_cost
    Savings = original - compressed
    """
    original = length * count
    compressed = 1 + length + count + extra_cost
    return original - compressed


def discover_bpe_candidates(
    tokens: TokenSeq,
    config: CompressionConfig,
    max_iterations: int = 100,
) -> list[Candidate]:
    """Discover patterns using BPE-style iterative merging.

    This approach finds one best merge at a time, simulates the merge,
    then looks for the next best merge. Returns all discovered candidates.

    Uses proper index tracking through merge history to map positions
    back to the original sequence correctly.

    Args:
        tokens: Input token sequence
        config: Compression configuration
        max_iterations: Maximum number of merge iterations

    Returns:
        List of Candidate objects discovered through iterative merging
    """
    if len(tokens) < 2:
        return []

    extra_cost = 1 if config.dict_length_enabled else 0
    candidates: list[Candidate] = []
    working = list(tokens)
    iteration = 0

    # Track original indices for each position in the working sequence
    # Each element is a tuple of original indices that this position represents
    # Initially, each position maps to itself
    original_indices: list[tuple[int, ...]] = [(i,) for i in range(len(tokens))]

    while iteration < max_iterations and len(working) > 1:
        # Count all adjacent pairs
        pair_counts = _count_adjacent_pairs(working)

        if not pair_counts:
            break

        # Find best merge candidate
        best_pair: tuple[Token, Token] | None = None
        best_savings = 0
        best_positions: list[int] = []

        # Check most frequent pairs first for efficiency
        for pair, raw_count in pair_counts.most_common(100):
            positions = _find_pair_positions(working, pair)
            count = len(positions)

            if not is_compressible(2, count, extra_cost):
                continue

            savings = _compute_merge_savings(2, count, extra_cost)

            if savings > best_savings:
                best_savings = savings
                best_pair = pair
                best_positions = positions

        if best_pair is None or best_savings <= 0:
            break

        # Map positions back to original sequence using tracked indices
        original_positions = _map_positions_with_index_tracking(
            best_positions, original_indices, tokens
        )

        if original_positions:
            candidates.append(
                Candidate(
                    subsequence=best_pair,
                    length=2,
                    positions=tuple(original_positions),
                    priority=1,  # BPE candidates get slight priority bonus
                )
            )

        # Apply merge to working sequence and update index tracking
        placeholder = f"__BPE_{iteration}__"
        new_working: list[Token] = []
        new_indices: list[tuple[int, ...]] = []
        skip_next = False

        for i, token in enumerate(working):
            if skip_next:
                skip_next = False
                continue
            if i < len(working) - 1 and (working[i], working[i + 1]) == best_pair:
                new_working.append(placeholder)
                # Merge the original indices from both positions
                merged_indices = original_indices[i] + original_indices[i + 1]
                new_indices.append(merged_indices)
                skip_next = True
            else:
                new_working.append(token)
                new_indices.append(original_indices[i])

        working = new_working
        original_indices = new_indices
        iteration += 1

    return candidates


def _map_positions_with_index_tracking(
    positions: list[int],
    original_indices: list[tuple[int, ...]],
    original_tokens: TokenSeq,
) -> list[int]:
    """Map positions from merged sequence back to original using index tracking.

    Args:
        positions: Positions in the current (merged) working sequence
        original_indices: For each position in working sequence, tuple of original indices
        original_tokens: The original token sequence

    Returns:
        List of positions in the original sequence where the pattern starts
    """
    result: list[int] = []

    for pos in positions:
        if pos >= len(original_indices) or pos + 1 >= len(original_indices):
            continue

        # Get the first original index from the first merged position
        # This is where the pattern starts in the original sequence
        start_indices = original_indices[pos]
        if start_indices:
            original_start = start_indices[0]
            # Validate the position is within bounds
            if original_start < len(original_tokens) - 1:
                result.append(original_start)

    return result


def _map_positions_to_original(
    positions: list[int],
    merge_history: list[tuple[tuple[Token, Token], str]],
    original_tokens: TokenSeq,
) -> list[int]:
    """Legacy function for backward compatibility.

    Note: The new implementation uses _map_positions_with_index_tracking
    which provides correct position mapping through merges.
    """
    if not merge_history:
        return positions

    # For backward compatibility, filter to valid positions
    return [p for p in positions if p < len(original_tokens) - 1]


def discover_extended_bpe_candidates(
    tokens: TokenSeq,
    config: CompressionConfig,
) -> list[Candidate]:
    """Extended BPE: Find pairs then try to extend them to longer patterns.

    This combines BPE's pair-finding with extension to longer patterns,
    potentially finding better compression than pure BPE.
    """
    if len(tokens) < 2:
        return []

    extra_cost = 1 if config.dict_length_enabled else 0
    candidates: list[Candidate] = []

    # Phase 1: Find all compressible pairs
    pair_counts = _count_adjacent_pairs(tokens)
    frequent_pairs: list[tuple[Token, Token]] = []

    for pair, count in pair_counts.items():
        positions = _find_pair_positions(tokens, pair)
        if is_compressible(2, len(positions), extra_cost):
            frequent_pairs.append(pair)

    # Phase 2: Try to extend each pair to longer patterns
    seen_patterns: set[tuple[Token, ...]] = set()

    for pair in frequent_pairs:
        extended = _try_extend_pattern(tokens, pair, config)
        if extended and extended.subsequence not in seen_patterns:
            candidates.append(extended)
            seen_patterns.add(extended.subsequence)
        elif pair not in seen_patterns:
            # Add the pair itself if extension didn't help
            positions = _find_pair_positions(tokens, pair)
            if is_compressible(2, len(positions), extra_cost):
                candidates.append(
                    Candidate(
                        subsequence=pair,
                        length=2,
                        positions=tuple(positions),
                        priority=1,
                    )
                )
                seen_patterns.add(pair)

    candidates.sort(key=lambda c: (-c.length, -len(c.positions)))
    return candidates


def _try_extend_pattern(
    tokens: TokenSeq,
    seed: tuple[Token, ...],
    config: CompressionConfig,
) -> Candidate | None:
    """Try to extend a seed pattern to a longer compressible pattern.

    Greedily extends the pattern left or right while maintaining
    compressibility and improving savings.
    """
    extra_cost = 1 if config.dict_length_enabled else 0
    current = seed
    best_candidate: Candidate | None = None
    best_savings = 0

    # Get initial positions and savings
    positions = _find_pattern_positions(tokens, current)
    if not positions:
        return None

    current_savings = _compute_merge_savings(len(current), len(positions), extra_cost)
    if current_savings > 0:
        best_savings = current_savings
        best_candidate = Candidate(
            subsequence=current,
            length=len(current),
            positions=tuple(positions),
            priority=1,
        )

    # Try extending up to max_subsequence_length
    while len(current) < config.max_subsequence_length:
        # Try extending right
        right_extensions = _find_extensions(tokens, current, "right")
        # Try extending left
        left_extensions = _find_extensions(tokens, current, "left")

        best_ext_savings = best_savings
        best_extension: tuple[tuple[Token, ...], list[int]] | None = None

        # Evaluate right extensions
        for ext_token, ext_positions in right_extensions.items():
            new_pattern = current + (ext_token,)
            count = len(ext_positions)

            if not is_compressible(len(new_pattern), count, extra_cost):
                continue

            savings = _compute_merge_savings(len(new_pattern), count, extra_cost)
            if savings > best_ext_savings:
                best_ext_savings = savings
                best_extension = (new_pattern, ext_positions)

        # Evaluate left extensions
        for ext_token, ext_positions in left_extensions.items():
            new_pattern = (ext_token,) + current
            count = len(ext_positions)

            if not is_compressible(len(new_pattern), count, extra_cost):
                continue

            savings = _compute_merge_savings(len(new_pattern), count, extra_cost)
            if savings > best_ext_savings:
                best_ext_savings = savings
                best_extension = (new_pattern, ext_positions)

        if best_extension is None:
            break

        current, positions = best_extension
        best_savings = best_ext_savings
        best_candidate = Candidate(
            subsequence=current,
            length=len(current),
            positions=tuple(positions),
            priority=1,
        )

    return best_candidate


def _find_pattern_positions(tokens: TokenSeq, pattern: tuple[Token, ...]) -> list[int]:
    """Find all non-overlapping positions of a pattern."""
    positions: list[int] = []
    pattern_len = len(pattern)
    i = 0

    while i <= len(tokens) - pattern_len:
        if tuple(tokens[i : i + pattern_len]) == pattern:
            positions.append(i)
            i += pattern_len  # Non-overlapping
        else:
            i += 1

    return positions


def _find_extensions(
    tokens: TokenSeq,
    pattern: tuple[Token, ...],
    direction: str,
) -> dict[Token, list[int]]:
    """Find possible extensions of a pattern in the given direction.

    Returns a dict mapping extension token to list of positions where
    the extended pattern occurs.
    """
    extensions: dict[Token, list[int]] = {}
    pattern_len = len(pattern)

    for i in range(len(tokens) - pattern_len + 1):
        if tuple(tokens[i : i + pattern_len]) == pattern:
            if direction == "left" and i > 0:
                ext_token = tokens[i - 1]
                # Position for extended pattern starts at i-1
                extensions.setdefault(ext_token, []).append(i - 1)
            elif direction == "right" and i + pattern_len < len(tokens):
                ext_token = tokens[i + pattern_len]
                # Position for extended pattern starts at i
                extensions.setdefault(ext_token, []).append(i)

    # Filter to non-overlapping positions for each extension
    for ext_token in extensions:
        positions = extensions[ext_token]
        new_len = pattern_len + 1
        non_overlapping: list[int] = []
        next_free = -1
        for pos in sorted(positions):
            if pos >= next_free:
                non_overlapping.append(pos)
                next_free = pos + new_len
        extensions[ext_token] = non_overlapping

    return extensions
