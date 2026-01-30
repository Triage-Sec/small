"""Pattern subsumption analysis for dictionary efficiency.

This module analyzes relationships between patterns to avoid redundant dictionary
entries. Pattern "abcd" subsumes patterns "ab", "bc", "cd", "abc", "bcd" - if we
select "abcd", we may not need the shorter patterns unless they have significant
independent occurrences.

Subsumption analysis helps:
1. Reduce dictionary size
2. Avoid selecting patterns that are fully covered by longer patterns
3. Identify patterns with independent value
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import CompressionConfig
from .types import Candidate, Token
from .utils import is_compressible


@dataclass
class SubsumptionGraph:
    """Graph of pattern subsumption relationships.

    Attributes:
        patterns: List of patterns (as tuples)
        subsumes: Mapping from pattern index to set of indices it subsumes
        subsumed_by: Mapping from pattern index to set of indices that subsume it
    """

    patterns: list[tuple[Token, ...]]
    subsumes: dict[int, set[int]] = field(default_factory=dict)
    subsumed_by: dict[int, set[int]] = field(default_factory=dict)


def build_subsumption_graph(candidates: list[Candidate]) -> SubsumptionGraph:
    """Build a directed graph showing which patterns subsume which others.

    Pattern A subsumes pattern B if B is a contiguous subsequence of A.

    Args:
        candidates: List of candidate patterns

    Returns:
        SubsumptionGraph with subsumption relationships
    """
    patterns = [c.subsequence for c in candidates]
    n = len(patterns)

    subsumes: dict[int, set[int]] = {i: set() for i in range(n)}
    subsumed_by: dict[int, set[int]] = {i: set() for i in range(n)}

    for i, longer in enumerate(patterns):
        longer_len = len(longer)

        # Check all shorter patterns
        for j, shorter in enumerate(patterns):
            if i == j:
                continue
            shorter_len = len(shorter)
            if shorter_len >= longer_len:
                continue

            # Check if shorter is a contiguous subsequence of longer
            for start in range(longer_len - shorter_len + 1):
                if longer[start : start + shorter_len] == shorter:
                    subsumes[i].add(j)
                    subsumed_by[j].add(i)
                    break

    return SubsumptionGraph(
        patterns=patterns,
        subsumes=subsumes,
        subsumed_by=subsumed_by,
    )


def find_maximal_patterns(candidates: list[Candidate]) -> list[Candidate]:
    """Find patterns that are not subsumed by any other pattern.

    These are the "maximal" patterns - longest patterns with no longer
    pattern that contains them.
    """
    if not candidates:
        return []

    graph = build_subsumption_graph(candidates)

    maximal: list[Candidate] = []
    for i, cand in enumerate(candidates):
        if not graph.subsumed_by[i]:
            # Not subsumed by anything
            maximal.append(cand)

    return maximal


def prune_subsumed_candidates(
    candidates: list[Candidate],
    config: CompressionConfig | None = None,
    min_independent_occurrences: int = 2,
) -> list[Candidate]:
    """Remove candidates that are fully subsumed by longer patterns.

    A pattern is kept if:
    1. It is not subsumed by any other pattern (maximal), OR
    2. It has significant independent occurrences not covered by subsuming patterns

    Args:
        candidates: List of candidate patterns
        config: Compression configuration (for compressibility check)
        min_independent_occurrences: Minimum independent occurrences to keep a subsumed pattern

    Returns:
        Pruned list of candidates
    """
    if not candidates:
        return []

    graph = build_subsumption_graph(candidates)
    extra_cost = 1 if config and config.dict_length_enabled else 0

    keep: list[Candidate] = []

    for i, cand in enumerate(candidates):
        if not graph.subsumed_by[i]:
            # Not subsumed by anything - always keep
            keep.append(cand)
            continue

        # Check if this pattern has positions not covered by subsuming patterns
        subsuming_positions: set[int] = set()

        for j in graph.subsumed_by[i]:
            subsuming_cand = candidates[j]
            subsuming_len = len(subsuming_cand.subsequence)
            cand_len = len(cand.subsequence)

            # For each position of the subsuming pattern, calculate which
            # positions of the shorter pattern are covered
            for pos in subsuming_cand.positions:
                # The shorter pattern could appear at multiple offsets within the longer
                for offset in range(subsuming_len - cand_len + 1):
                    if (
                        subsuming_cand.subsequence[offset : offset + cand_len]
                        == cand.subsequence
                    ):
                        subsuming_positions.add(pos + offset)

        # Find independent positions
        independent_positions = [
            p for p in cand.positions if p not in subsuming_positions
        ]

        if len(independent_positions) >= min_independent_occurrences:
            # Check if independent occurrences alone are compressible
            if is_compressible(
                len(cand.subsequence), len(independent_positions), extra_cost
            ):
                # Create new candidate with only independent positions
                keep.append(
                    Candidate(
                        subsequence=cand.subsequence,
                        length=cand.length,
                        positions=tuple(independent_positions),
                        priority=cand.priority,
                        patches=cand.patches,
                    )
                )

    return keep


def compute_coverage_score(
    candidates: list[Candidate],
    sequence_length: int,
) -> dict[int, float]:
    """Compute how much of the sequence each pattern covers.

    Returns a dict mapping candidate index to coverage ratio (0-1).
    """
    coverage: dict[int, float] = {}

    for i, cand in enumerate(candidates):
        total_covered = len(cand.positions) * cand.length
        coverage[i] = (
            min(1.0, total_covered / sequence_length) if sequence_length > 0 else 0.0
        )

    return coverage


def rank_by_independent_value(
    candidates: list[Candidate],
    config: CompressionConfig | None = None,
) -> list[tuple[Candidate, float]]:
    """Rank candidates by their independent compression value.

    Patterns that are fully subsumed get lower scores. Patterns with
    unique occurrences get higher scores.

    Returns list of (candidate, score) tuples, sorted by score descending.
    """
    if not candidates:
        return []

    graph = build_subsumption_graph(candidates)
    extra_cost = 1 if config and config.dict_length_enabled else 0

    results: list[tuple[Candidate, float]] = []

    for i, cand in enumerate(candidates):
        # Base score: compression savings
        length = len(cand.subsequence)
        count = len(cand.positions)
        base_savings = max(0, length * count - (1 + length + count + extra_cost))

        # Penalty for being subsumed
        subsumption_penalty = 0.0
        if graph.subsumed_by[i]:
            # Count what fraction of positions are covered by subsuming patterns
            covered_positions: set[int] = set()

            for j in graph.subsumed_by[i]:
                subsuming_cand = candidates[j]
                subsuming_len = len(subsuming_cand.subsequence)
                cand_len = len(cand.subsequence)

                for pos in subsuming_cand.positions:
                    for offset in range(subsuming_len - cand_len + 1):
                        if (
                            subsuming_cand.subsequence[offset : offset + cand_len]
                            == cand.subsequence
                        ):
                            covered_positions.add(pos + offset)

            coverage_ratio = (
                len(covered_positions & set(cand.positions)) / count if count > 0 else 0
            )
            subsumption_penalty = (
                base_savings * coverage_ratio * 0.8
            )  # 80% penalty for covered positions

        # Bonus for being maximal (not subsumed)
        maximal_bonus = base_savings * 0.1 if not graph.subsumed_by[i] else 0

        # Bonus for subsuming other patterns (indicates structural importance)
        subsumes_bonus = len(graph.subsumes[i]) * 0.5

        final_score = (
            base_savings - subsumption_penalty + maximal_bonus + subsumes_bonus
        )
        results.append((cand, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Remove duplicate candidates, merging their positions.

    If the same subsequence appears multiple times (e.g., from different
    discovery strategies), merge them into a single candidate.
    """
    if not candidates:
        return []

    by_subsequence: dict[tuple[Token, ...], list[Candidate]] = {}
    for cand in candidates:
        by_subsequence.setdefault(cand.subsequence, []).append(cand)

    deduplicated: list[Candidate] = []

    for subseq, group in by_subsequence.items():
        if len(group) == 1:
            deduplicated.append(group[0])
            continue

        # Merge positions from all candidates
        all_positions: set[int] = set()
        all_patches: dict[int, tuple] = {}
        max_priority = 0

        for cand in group:
            all_positions.update(cand.positions)
            all_patches.update(cand.patches)
            max_priority = max(max_priority, cand.priority)

        # Compute non-overlapping positions
        sorted_positions = sorted(all_positions)
        length = len(subseq)
        non_overlapping: list[int] = []
        next_free = -1

        for pos in sorted_positions:
            if pos >= next_free:
                non_overlapping.append(pos)
                next_free = pos + length

        deduplicated.append(
            Candidate(
                subsequence=subseq,
                length=length,
                positions=tuple(non_overlapping),
                priority=max_priority,
                patches=all_patches,
            )
        )

    return deduplicated
