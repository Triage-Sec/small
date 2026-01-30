"""Semantic-aware pattern selection using embeddings.

This module provides selection that considers semantic context when choosing
which pattern occurrences to compress. Patterns appearing in semantically
similar contexts are better compression candidates (redundant information),
while patterns in diverse contexts carry unique meaning and should be
preserved more carefully.

Requires an embedding provider (OpenAI, Voyage, sentence-transformers, etc.).
Falls back to optimal selection if no provider is available.

Usage:
    from small.selection_semantic import select_occurrences_semantic
    from small.embeddings import create_provider

    provider = create_provider("openai")
    selected = select_occurrences_semantic(candidates, tokens, config, provider)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .config import CompressionConfig
from .types import Candidate, Occurrence, TokenSeq
from .utils import is_compressible

if TYPE_CHECKING:
    from .embeddings import EmbeddingProvider


@dataclass
class SemanticWeight:
    """Semantic weight for a pattern based on context similarity."""

    subsequence: tuple
    similarity_score: float  # Average pairwise similarity (0-1)
    diversity_score: float  # 1 - similarity_score
    weight_multiplier: float  # Computed weight multiplier for selection


def _build_occurrences(candidates: Iterable[Candidate]) -> list[Occurrence]:
    """Build flat list of occurrences from candidates."""
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


def _extract_context(
    tokens: TokenSeq,
    position: int,
    length: int,
    context_window: int,
) -> str:
    """Extract context window around an occurrence as text."""
    start = max(0, position - context_window)
    end = min(len(tokens), position + length + context_window)
    context_tokens = tokens[start:end]
    return " ".join(str(t) for t in context_tokens)


def _compute_pairwise_similarity(embeddings: list[list[float]]) -> float:
    """Compute average pairwise cosine similarity between embeddings."""
    if len(embeddings) < 2:
        return 1.0  # Single occurrence = fully similar to itself

    n = len(embeddings)
    total_sim = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Cosine similarity
            dot = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
            norm_i = math.sqrt(sum(a * a for a in embeddings[i]))
            norm_j = math.sqrt(sum(b * b for b in embeddings[j]))

            if norm_i > 0 and norm_j > 0:
                sim = dot / (norm_i * norm_j)
                total_sim += sim
                count += 1

    return total_sim / count if count > 0 else 1.0


def _compute_semantic_weights(
    candidates: list[Candidate],
    tokens: TokenSeq,
    provider: "EmbeddingProvider",
    context_window: int,
    similarity_threshold: float,
    diversity_penalty: float,
    batch_size: int = 32,
) -> dict[tuple, SemanticWeight]:
    """Compute semantic weights for each pattern based on context similarity.

    High context similarity = pattern is redundant = good to compress (high weight)
    Low context similarity = pattern carries unique info = compress carefully (low weight)
    """
    weights: dict[tuple, SemanticWeight] = {}

    for cand in candidates:
        if len(cand.positions) < 2:
            # Single occurrence - neutral weight
            weights[cand.subsequence] = SemanticWeight(
                subsequence=cand.subsequence,
                similarity_score=1.0,
                diversity_score=0.0,
                weight_multiplier=1.0,
            )
            continue

        # Extract context windows for each occurrence
        contexts: list[str] = []
        # Sample positions if too many (limit embedding calls)
        sample_positions = list(cand.positions[:min(len(cand.positions), batch_size)])

        for pos in sample_positions:
            context = _extract_context(tokens, pos, cand.length, context_window)
            contexts.append(context)

        # Get embeddings
        try:
            embeddings = provider.embed_batch(contexts)
        except Exception:
            # On embedding failure, use neutral weight
            weights[cand.subsequence] = SemanticWeight(
                subsequence=cand.subsequence,
                similarity_score=0.5,
                diversity_score=0.5,
                weight_multiplier=1.0,
            )
            continue

        # Compute pairwise similarity
        similarity = _compute_pairwise_similarity(embeddings)
        diversity = 1.0 - similarity

        # Compute weight multiplier:
        # High similarity (> threshold) = boost weight (good compression candidate)
        # Low similarity (< threshold) = reduce weight (preserve semantic info)
        if similarity >= similarity_threshold:
            # Good candidate - similarity above threshold
            # Scale from 1.0 to 1.5 based on how much above threshold
            bonus = (similarity - similarity_threshold) / (1.0 - similarity_threshold)
            weight_multiplier = 1.0 + bonus * 0.5
        else:
            # Diverse contexts - apply penalty
            # Scale penalty based on how far below threshold
            penalty_factor = (similarity_threshold - similarity) / similarity_threshold
            weight_multiplier = 1.0 - penalty_factor * diversity_penalty

        weights[cand.subsequence] = SemanticWeight(
            subsequence=cand.subsequence,
            similarity_score=similarity,
            diversity_score=diversity,
            weight_multiplier=max(0.1, weight_multiplier),  # Floor at 0.1
        )

    return weights


def _min_count_for_compressibility(length: int, extra_cost: int) -> int:
    """Compute minimum occurrence count for compressibility."""
    if length <= 1:
        return 1_000_000_000
    return math.ceil((2 + length + extra_cost) / (length - 1))


def _compute_savings(length: int, count: int, extra_cost: int) -> int:
    """Compute net token savings for a pattern."""
    if count == 0:
        return 0
    original = length * count
    compressed = 1 + length + count + extra_cost
    return max(0, original - compressed)


def _group_by_subsequence(
    occurrences: list[Occurrence],
) -> dict[tuple, list[Occurrence]]:
    """Group occurrences by their subsequence."""
    grouped: dict[tuple, list[Occurrence]] = {}
    for occ in occurrences:
        grouped.setdefault(occ.subsequence, []).append(occ)
    return grouped


def _estimate_non_overlapping_count(occs: list[Occurrence]) -> int:
    """Estimate maximum non-overlapping occurrences."""
    if not occs:
        return 0
    sorted_occs = sorted(occs, key=lambda o: o.start)
    count = 0
    next_free = -1
    for occ in sorted_occs:
        if occ.start >= next_free:
            count += 1
            next_free = occ.start + occ.length
    return count


def select_occurrences_semantic(
    candidates: Iterable[Candidate],
    tokens: TokenSeq,
    config: CompressionConfig,
    provider: "EmbeddingProvider",
) -> list[Occurrence]:
    """Select occurrences using semantic-aware weighted interval scheduling.

    Uses embedding similarity to weight patterns:
    - Patterns with similar contexts across occurrences get higher weights
    - Patterns with diverse contexts get lower weights (preserve semantic info)

    Args:
        candidates: Pattern candidates to select from
        tokens: Original token sequence
        config: Compression configuration
        provider: Embedding provider for context similarity computation

    Returns:
        List of selected Occurrence objects
    """
    candidates_list = list(candidates)
    if not candidates_list:
        return []

    extra_cost = 1 if config.dict_length_enabled else 0

    # Compute semantic weights for each pattern
    semantic_weights = _compute_semantic_weights(
        candidates_list,
        tokens,
        provider,
        context_window=config.semantic_context_window,
        similarity_threshold=config.semantic_similarity_threshold,
        diversity_penalty=config.semantic_diversity_penalty,
    )

    # Pre-filter patterns that can never be compressible
    subseq_to_occs: dict[tuple, list[int]] = {}
    all_positions: list[tuple[int, ...]] = []

    for cand in candidates_list:
        subseq_to_occs.setdefault(cand.subsequence, [])
        for pos in cand.positions:
            subseq_to_occs[cand.subsequence].append(len(all_positions))
            all_positions.append((pos, cand.length, cand.subsequence, cand.priority))

    viable_subseqs: set[tuple] = set()
    for subseq, pos_indices in subseq_to_occs.items():
        min_count = _min_count_for_compressibility(len(subseq), extra_cost)
        if len(pos_indices) >= min_count:
            viable_subseqs.add(subseq)

    # Build occurrences for viable patterns
    occurrences = _build_occurrences(
        cand for cand in candidates_list if cand.subsequence in viable_subseqs
    )

    if not occurrences:
        return []

    # Iterative refinement loop
    max_iterations = 10

    for iteration in range(max_iterations):
        occs = sorted(occurrences, key=lambda occ: (occ.start + occ.length, occ.start))

        if not occs:
            return []

        ends = [occ.start + occ.length for occ in occs]

        # p[i]: last index < i that doesn't overlap
        p: list[int] = []
        for i, occ in enumerate(occs):
            lo, hi, idx = 0, i - 1, -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if ends[mid] <= occ.start:
                    idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            p.append(idx)

        # Group by subsequence for count estimates
        subseq_groups: dict[tuple, list[Occurrence]] = {}
        for occ in occs:
            subseq_groups.setdefault(occ.subsequence, []).append(occ)

        subseq_expected_counts: dict[tuple, int] = {}
        for subseq, group in subseq_groups.items():
            subseq_expected_counts[subseq] = _estimate_non_overlapping_count(group)

        # Compute weights with semantic multiplier
        weights: list[float] = []
        for occ in occs:
            expected_count = subseq_expected_counts[occ.subsequence]
            length = occ.length

            # Amortized dictionary cost per occurrence
            dict_cost_per_occ = (
                (1 + length + extra_cost) / expected_count
                if expected_count > 0
                else length
            )

            # Base savings
            savings = length - 1 - dict_cost_per_occ

            # Apply semantic weight multiplier
            sem_weight = semantic_weights.get(occ.subsequence)
            if sem_weight:
                savings *= sem_weight.weight_multiplier

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

        if not non_compressible_subseqs:
            break

        # Remove non-compressible and retry
        viable_subseqs -= non_compressible_subseqs
        occurrences = [occ for occ in occurrences if occ.subsequence in viable_subseqs]

    # Final filter
    final_selected: list[Occurrence] = []
    for subseq, group_occs in grouped.items():
        if is_compressible(len(subseq), len(group_occs), extra_cost=extra_cost):
            final_selected.extend(group_occs)

    final_selected.sort(key=lambda occ: occ.start)
    return final_selected
