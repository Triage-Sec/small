"""Pattern selection strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .config import CompressionConfig
from .types import Candidate, Occurrence
from .utils import is_compressible


@dataclass(frozen=True)
class SelectionResult:
    selected: list[Occurrence]


def _min_count_for_compressibility(length: int, extra_cost: int) -> int:
    """Compute minimum occurrence count required for a pattern to be compressible.

    Compressibility condition: length * count > 1 + length + count + extra_cost
    Solving for count: count > (1 + length + extra_cost) / (length - 1)
    """
    if length <= 1:
        return 1_000_000_000  # Effectively infinite
    return math.ceil((2 + length + extra_cost) / (length - 1))


def _compute_savings(length: int, count: int, extra_cost: int) -> int:
    """Compute net token savings for a pattern with given length and occurrence count."""
    if count == 0:
        return 0
    # Original tokens: length * count
    # Compressed: 1 (meta-token entry) + length (definition) + count (references) + extra_cost
    original = length * count
    compressed = 1 + length + count + extra_cost
    return max(0, original - compressed)


def _compute_marginal_savings(
    length: int, current_count: int, extra_cost: int
) -> float:
    """Compute marginal savings from adding one more occurrence of a pattern."""
    current_savings = _compute_savings(length, current_count, extra_cost)
    new_savings = _compute_savings(length, current_count + 1, extra_cost)
    return new_savings - current_savings


def _savings_density(occ: Occurrence) -> float:
    """Compute savings-density score for an occurrence.

    Higher values indicate better compression value per position consumed.
    """
    if occ.length <= 1:
        return 0.0
    # Savings per token if this occurrence is part of a compressed pattern
    pattern_savings = occ.length - 1  # Replace N tokens with 1 reference
    density = pattern_savings / occ.length
    return density + occ.priority * 0.1


def _build_occurrences(candidates: Iterable[Candidate]) -> list[Occurrence]:
    occurrences: list[Occurrence] = []
    for candidate in candidates:
        for pos in candidate.positions:
            occurrences.append(
                Occurrence(
                    start=pos,
                    length=candidate.length,
                    subsequence=candidate.subsequence,
                    priority=candidate.priority,
                    patches=candidate.patches.get(pos, ()),
                )
            )
    occurrences.sort(key=lambda occ: (occ.start + occ.length, occ.start))
    return occurrences


def _group_by_subsequence(
    occurrences: list[Occurrence],
) -> dict[tuple, list[Occurrence]]:
    grouped: dict[tuple, list[Occurrence]] = {}
    for occ in occurrences:
        grouped.setdefault(occ.subsequence, []).append(occ)
    return grouped


def _estimate_non_overlapping_count(occs: list[Occurrence]) -> int:
    """Estimate the maximum number of non-overlapping occurrences for a pattern.

    Uses a greedy sweep to count maximum non-overlapping positions.
    This provides a better estimate for amortized dictionary cost than raw count.
    """
    if not occs:
        return 0

    # Sort by start position
    sorted_occs = sorted(occs, key=lambda o: o.start)
    count = 0
    next_free = -1

    for occ in sorted_occs:
        if occ.start >= next_free:
            count += 1
            next_free = occ.start + occ.length

    return count


def _non_overlapping_with_compressibility(
    occurrences: list[Occurrence],
    config: CompressionConfig,
) -> list[Occurrence]:
    """Greedy selection using savings-density, enforcing compressibility during selection.

    Uses an iterative refinement approach:
    1. Pre-filter patterns that can't possibly achieve compressibility
    2. Greedily select non-overlapping occurrences
    3. Release positions from patterns that didn't achieve compressibility
    4. Repeat until stable (all selected patterns are compressible)
    """
    if not occurrences:
        return []

    extra_cost = 1 if config.dict_length_enabled else 0

    # Pre-compute minimum counts for each pattern length
    min_counts_cache: dict[int, int] = {}

    def get_min_count(length: int) -> int:
        if length not in min_counts_cache:
            min_counts_cache[length] = _min_count_for_compressibility(
                length, extra_cost
            )
        return min_counts_cache[length]

    # Group occurrences by subsequence
    subseq_to_occs: dict[tuple, list[Occurrence]] = {}
    for occ in occurrences:
        subseq_to_occs.setdefault(occ.subsequence, []).append(occ)

    # Filter out patterns that can never be compressible (not enough total occurrences)
    viable_subseqs: set[tuple] = set()
    for subseq, occs in subseq_to_occs.items():
        if len(occs) >= get_min_count(len(subseq)):
            viable_subseqs.add(subseq)

    # Filter occurrences to only viable patterns
    viable_occs = [occ for occ in occurrences if occ.subsequence in viable_subseqs]

    if not viable_occs:
        return []

    # Iterative refinement loop
    max_iterations = 10  # Prevent infinite loops
    for iteration in range(max_iterations):
        # Sort by savings-density (highest first), then by start position for stability
        sorted_occs = sorted(
            viable_occs, key=lambda o: (-_savings_density(o), o.start, o.length)
        )

        selected: list[Occurrence] = []
        occupied: set[int] = set()
        subseq_counts: dict[tuple, int] = {}

        for occ in sorted_occs:
            positions = set(range(occ.start, occ.start + occ.length))
            if positions & occupied:
                continue

            selected.append(occ)
            occupied |= positions
            subseq_counts[occ.subsequence] = subseq_counts.get(occ.subsequence, 0) + 1

        # Find patterns that achieved compressibility
        compressible_subseqs: set[tuple] = set()
        non_compressible_subseqs: set[tuple] = set()

        for subseq, count in subseq_counts.items():
            if is_compressible(len(subseq), count, extra_cost=extra_cost):
                compressible_subseqs.add(subseq)
            else:
                non_compressible_subseqs.add(subseq)

        # If all selected patterns are compressible, we're done
        if not non_compressible_subseqs:
            break

        # Remove non-compressible patterns from viable set and retry
        # This frees up positions for other patterns
        viable_subseqs -= non_compressible_subseqs
        viable_occs = [occ for occ in viable_occs if occ.subsequence in viable_subseqs]

        if not viable_occs:
            # No viable patterns left
            selected = []
            break

    # Final filter: only keep compressible patterns
    final_selected: list[Occurrence] = []
    for occ in selected:
        count = subseq_counts.get(occ.subsequence, 0)
        if is_compressible(occ.length, count, extra_cost=extra_cost):
            final_selected.append(occ)

    final_selected.sort(key=lambda occ: occ.start)
    return final_selected


def _weighted_interval_scheduling_with_savings(
    occurrences: list[Occurrence],
    config: CompressionConfig,
) -> list[Occurrence]:
    """Weighted interval scheduling with proper savings calculation accounting for dictionary overhead.

    Uses iterative refinement to handle compressibility constraints:
    1. Run DP to select optimal non-overlapping occurrences
    2. Check which patterns achieved compressibility
    3. Remove non-compressible patterns and re-run until stable
    """
    if not occurrences:
        return []

    extra_cost = 1 if config.dict_length_enabled else 0

    # Pre-filter patterns that can never be compressible
    subseq_to_occs: dict[tuple, list[Occurrence]] = {}
    for occ in occurrences:
        subseq_to_occs.setdefault(occ.subsequence, []).append(occ)

    viable_subseqs: set[tuple] = set()
    for subseq, occs_list in subseq_to_occs.items():
        min_count = _min_count_for_compressibility(len(subseq), extra_cost)
        if len(occs_list) >= min_count:
            viable_subseqs.add(subseq)

    viable_occs = [occ for occ in occurrences if occ.subsequence in viable_subseqs]

    if not viable_occs:
        return []

    # Iterative refinement loop
    max_iterations = 10
    for iteration in range(max_iterations):
        occs = sorted(viable_occs, key=lambda occ: (occ.start + occ.length, occ.start))

        if not occs:
            return []

        ends = [occ.start + occ.length for occ in occs]

        # p[i]: last index < i that doesn't overlap
        p: list[int] = []
        for i, occ in enumerate(occs):
            lo = 0
            hi = i - 1
            idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if ends[mid] <= occ.start:
                    idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            p.append(idx)

        # Compute weights using amortized dictionary cost
        # Group by subsequence to get better count estimates
        subseq_groups: dict[tuple, list[Occurrence]] = {}
        for occ in occs:
            subseq_groups.setdefault(occ.subsequence, []).append(occ)

        # Use non-overlapping estimate for amortized cost (more accurate than raw count)
        subseq_expected_counts: dict[tuple, int] = {}
        for subseq, group in subseq_groups.items():
            subseq_expected_counts[subseq] = _estimate_non_overlapping_count(group)

        weights: list[float] = []
        for occ in occs:
            expected_count = subseq_expected_counts[occ.subsequence]
            length = occ.length

            # Amortized dictionary cost per occurrence
            # Dictionary cost = 1 (meta-token) + length (definition) + extra_cost (length token if enabled)
            dict_cost_per_occ = (
                (1 + length + extra_cost) / expected_count
                if expected_count > 0
                else length
            )

            # Savings: original tokens replaced - (1 reference + amortized dict cost)
            # Each occurrence replaces `length` tokens with 1 reference token
            savings = length - 1 - dict_cost_per_occ
            weight = max(0, savings) + occ.priority * 0.5
            weights.append(weight)

        # DP over occurrences
        dp = [0.0] * len(occs)
        choose = [False] * len(occs)
        for i in range(len(occs)):
            take = weights[i] + (dp[p[i]] if p[i] >= 0 else 0)
            skip = dp[i - 1] if i > 0 else 0
            if take > skip:
                dp[i] = take
                choose[i] = True
            else:
                dp[i] = skip
                choose[i] = False

        # Reconstruct
        selected: list[Occurrence] = []
        i = len(occs) - 1
        while i >= 0:
            if choose[i]:
                selected.append(occs[i])
                i = p[i]
            else:
                i -= 1
        selected.reverse()

        # Check compressibility
        grouped = _group_by_subsequence(selected)
        compressible_subseqs: set[tuple] = set()
        non_compressible_subseqs: set[tuple] = set()

        for subseq, group_occs in grouped.items():
            if is_compressible(len(subseq), len(group_occs), extra_cost=extra_cost):
                compressible_subseqs.add(subseq)
            else:
                non_compressible_subseqs.add(subseq)

        # If all selected patterns are compressible, we're done
        if not non_compressible_subseqs:
            break

        # Remove non-compressible patterns and retry
        viable_subseqs -= non_compressible_subseqs
        viable_occs = [occ for occ in viable_occs if occ.subsequence in viable_subseqs]

    # Final filter
    final_selected: list[Occurrence] = []
    for subseq, group_occs in grouped.items():
        if is_compressible(len(subseq), len(group_occs), extra_cost=extra_cost):
            final_selected.extend(group_occs)

    final_selected.sort(key=lambda occ: occ.start)
    return final_selected


def _beam_search_with_savings(
    occurrences: list[Occurrence],
    width: int,
    config: CompressionConfig,
) -> list[Occurrence]:
    """Beam search with proper savings calculation and compressibility-aware scoring.

    Uses iterative refinement similar to other selection methods:
    1. Pre-filter patterns that can never be compressible
    2. Run beam search
    3. Remove patterns that didn't achieve compressibility and retry
    """
    if not occurrences:
        return []

    extra_cost = 1 if config.dict_length_enabled else 0

    # Pre-filter patterns that can never be compressible
    subseq_to_occs: dict[tuple, list[Occurrence]] = {}
    for occ in occurrences:
        subseq_to_occs.setdefault(occ.subsequence, []).append(occ)

    viable_subseqs: set[tuple] = set()
    for subseq, occs_list in subseq_to_occs.items():
        min_count = _min_count_for_compressibility(len(subseq), extra_cost)
        if len(occs_list) >= min_count:
            viable_subseqs.add(subseq)

    viable_occs = [occ for occ in occurrences if occ.subsequence in viable_subseqs]

    if not viable_occs:
        return []

    # Iterative refinement loop
    max_iterations = 10
    best_selected: list[Occurrence] = []
    best_counts: dict[tuple, int] = {}

    for iteration in range(max_iterations):
        occs = sorted(viable_occs, key=lambda occ: (occ.start, occ.length))

        if not occs:
            break

        # State: (score, last_end, selected, subseq_counts)
        initial_counts: dict[tuple, int] = {}
        states: list[tuple[float, int, list[Occurrence], dict[tuple, int]]] = [
            (0.0, -1, [], initial_counts.copy())
        ]

        for occ in occs:
            new_states: list[tuple[float, int, list[Occurrence], dict[tuple, int]]] = []
            for score, last_end, selected, subseq_counts in states:
                # Option 1: skip
                new_states.append((score, last_end, selected, subseq_counts))

                # Option 2: take (if non-overlapping)
                if occ.start >= last_end:
                    current_count = subseq_counts.get(occ.subsequence, 0)
                    marginal = _compute_marginal_savings(
                        occ.length, current_count, extra_cost
                    )

                    new_score = score + marginal + occ.priority * 0.5
                    new_selected = selected + [occ]
                    new_counts = subseq_counts.copy()
                    new_counts[occ.subsequence] = current_count + 1

                    new_states.append(
                        (new_score, occ.start + occ.length, new_selected, new_counts)
                    )

            # Keep top-k by score
            new_states.sort(key=lambda s: (s[0], -s[1]), reverse=True)
            states = new_states[: max(1, width)]

        # Select best state
        states.sort(key=lambda s: s[0], reverse=True)
        best_selected = states[0][2]
        best_counts = states[0][3]

        # Check compressibility
        compressible_subseqs: set[tuple] = set()
        non_compressible_subseqs: set[tuple] = set()

        for subseq, count in best_counts.items():
            if is_compressible(len(subseq), count, extra_cost=extra_cost):
                compressible_subseqs.add(subseq)
            else:
                non_compressible_subseqs.add(subseq)

        # If all selected patterns are compressible, we're done
        if not non_compressible_subseqs:
            break

        # Remove non-compressible patterns and retry
        viable_subseqs -= non_compressible_subseqs
        viable_occs = [occ for occ in viable_occs if occ.subsequence in viable_subseqs]

    # Final filter for compressibility
    final_selected: list[Occurrence] = []
    for occ in best_selected:
        count = best_counts.get(occ.subsequence, 0)
        if is_compressible(occ.length, count, extra_cost=extra_cost):
            final_selected.append(occ)

    final_selected.sort(key=lambda occ: occ.start)
    return final_selected


def select_occurrences(
    candidates: Iterable[Candidate],
    config: CompressionConfig,
    tokens: tuple | list | None = None,
) -> SelectionResult:
    """Select non-overlapping occurrences for compression.

    Selection modes:
    - greedy: Fast selection using savings-density heuristic
    - optimal: Weighted interval scheduling with proper savings
    - beam: Beam search balancing exploration and exploitation
    - ilp: Integer linear programming (requires scipy, see selection_ilp module)
    - semantic: Embedding-based selection (requires embedding provider)

    Args:
        candidates: Pattern candidates to select from
        config: Compression configuration
        tokens: Original token sequence (required for semantic mode)
    """
    # Convert to list once for modes that need it
    candidates_list = list(candidates)
    occurrences = _build_occurrences(candidates_list)

    if config.selection_mode == "greedy":
        selected = _non_overlapping_with_compressibility(occurrences, config)
    elif config.selection_mode == "optimal":
        selected = _weighted_interval_scheduling_with_savings(occurrences, config)
    elif config.selection_mode == "beam":
        selected = _beam_search_with_savings(occurrences, config.beam_width, config)
    elif config.selection_mode == "ilp":
        # Lazy import to avoid scipy dependency for basic usage
        try:
            from .selection_ilp import select_occurrences_ilp

            selected = select_occurrences_ilp(candidates_list, config)
        except ImportError:
            # Fall back to optimal if scipy not available
            selected = _weighted_interval_scheduling_with_savings(occurrences, config)
    elif config.selection_mode == "semantic":
        selected = _select_semantic_with_fallback(candidates_list, tokens, config)
    else:
        raise ValueError(f"Unsupported selection mode: {config.selection_mode}")

    return SelectionResult(selected=selected)


def _select_semantic_with_fallback(
    candidates: list[Candidate],
    tokens: tuple | list | None,
    config: CompressionConfig,
) -> list[Occurrence]:
    """Attempt semantic selection, falling back to optimal if provider unavailable."""
    import warnings

    # Check if provider is configured
    if not config.semantic_embedding_provider:
        warnings.warn(
            "selection_mode='semantic' requires semantic_embedding_provider. "
            "Falling back to 'optimal' selection.",
            RuntimeWarning,
        )
        occurrences = _build_occurrences(candidates)
        return _weighted_interval_scheduling_with_savings(occurrences, config)

    # Check if tokens provided
    if tokens is None:
        warnings.warn(
            "selection_mode='semantic' requires tokens for context extraction. "
            "Falling back to 'optimal' selection.",
            RuntimeWarning,
        )
        occurrences = _build_occurrences(candidates)
        return _weighted_interval_scheduling_with_savings(occurrences, config)

    # Try to create provider
    try:
        from .embeddings import get_provider_from_config

        provider = get_provider_from_config(
            config.semantic_embedding_provider,
            model=config.semantic_embedding_model,
        )

        if provider is None:
            warnings.warn(
                f"Could not initialize embedding provider '{config.semantic_embedding_provider}'. "
                "Check API key and package installation. Falling back to 'optimal' selection.",
                RuntimeWarning,
            )
            occurrences = _build_occurrences(candidates)
            return _weighted_interval_scheduling_with_savings(occurrences, config)

        # Use semantic selection
        from .selection_semantic import select_occurrences_semantic

        return select_occurrences_semantic(candidates, tokens, config, provider)

    except Exception as e:
        warnings.warn(
            f"Semantic selection failed: {e}. Falling back to 'optimal' selection.",
            RuntimeWarning,
        )
        occurrences = _build_occurrences(candidates)
        return _weighted_interval_scheduling_with_savings(occurrences, config)
