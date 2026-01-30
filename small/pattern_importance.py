"""ML-based pattern importance scoring for intelligent compression.

This module provides scoring mechanisms to assess pattern importance,
enabling smarter compression decisions. Not all patterns are equally
important for downstream tasks - patterns in semantically critical
regions should be preserved more carefully.

Scoring approaches:
1. Embedding-based: Score patterns by context diversity
2. Positional: Earlier positions tend to be more important (instructions)
3. Composite: Weighted combination of multiple signals

Integration: Adjust Candidate.priority based on importance scores
before selection, influencing which patterns get compressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import Candidate, TokenSeq


class ImportanceScorer(Protocol):
    """Protocol for pattern importance scoring."""

    def score_patterns(
        self,
        tokens: TokenSeq,
        candidates: list[Candidate],
    ) -> list[float]:
        """Return importance scores for each candidate (higher = more important).

        Important patterns should be compressed more carefully (or not at all).
        Scores should be in range [0, 1].
        """
        ...


@dataclass
class EmbeddingImportanceScorer:
    """Score patterns based on their embedding context diversity.

    Intuition: Patterns that appear in semantically similar contexts
    are carrying redundant information (good to compress). Patterns
    that appear in diverse contexts carry unique information each time
    (important, compress carefully).

    Requires an EmbeddingProvider from small.embeddings.
    """

    provider: object  # EmbeddingProvider
    context_window: int = 5
    max_samples: int = 10

    def score_patterns(
        self,
        tokens: TokenSeq,
        candidates: list[Candidate],
    ) -> list[float]:
        """Score patterns by context diversity using embeddings."""
        if not candidates:
            return []

        try:
            import numpy as np
        except ImportError:
            # Return neutral scores if numpy unavailable
            return [0.5] * len(candidates)

        scores: list[float] = []

        for cand in candidates:
            # Get context windows around each occurrence
            context_texts: list[str] = []

            # Sample positions (up to max_samples)
            sample_positions = list(cand.positions[: self.max_samples])

            for pos in sample_positions:
                # Extract context window
                start = max(0, pos - self.context_window)
                end = min(len(tokens), pos + cand.length + self.context_window)
                context = " ".join(str(t) for t in tokens[start:end])
                context_texts.append(context)

            if len(context_texts) < 2:
                scores.append(0.5)  # Neutral score for single occurrence
                continue

            # Get embeddings
            try:
                embed_fn = getattr(self.provider, "embed_batch", None)
                if embed_fn is None:
                    scores.append(0.5)
                    continue
                embeddings = np.array(embed_fn(context_texts))
            except Exception:
                scores.append(0.5)
                continue

            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            # Compute average pairwise cosine distance (diversity)
            n = len(embeddings)
            total_dist = 0.0
            count = 0

            for i in range(n):
                for j in range(i + 1, n):
                    sim = np.dot(embeddings[i], embeddings[j])
                    total_dist += 1 - sim
                    count += 1

            avg_diversity = total_dist / count if count > 0 else 0

            # Higher diversity = higher importance
            # (pattern carries distinct info each time)
            scores.append(float(avg_diversity))

        # Normalize scores to [0, 1]
        if scores:
            min_s, max_s = min(scores), max(scores)
            if max_s > min_s:
                scores = [(s - min_s) / (max_s - min_s) for s in scores]

        return scores


@dataclass
class PositionalImportanceScorer:
    """Score patterns based on their position in the sequence.

    Intuition: Patterns at the beginning of a prompt are often more
    critical (system instructions, context setup) than patterns in
    the middle or end (examples, data).

    Uses exponential decay from start of sequence.
    """

    decay_rate: float = 2.0

    def score_patterns(
        self,
        tokens: TokenSeq,
        candidates: list[Candidate],
    ) -> list[float]:
        """Score patterns by position (earlier = more important)."""
        if not candidates:
            return []

        import math
        from typing import Callable

        def _math_exp(x: float) -> float:
            return math.exp(x)

        exp_fn: Callable[[float], float] = _math_exp
        try:
            import numpy as np

            def _np_exp(x: float) -> float:
                return float(np.exp(x))

            exp_fn = _np_exp
        except ImportError:
            pass

        n = len(tokens)
        scores: list[float] = []

        for cand in candidates:
            if not cand.positions:
                scores.append(0.5)
                continue

            # Average position of occurrences
            avg_pos = sum(cand.positions) / len(cand.positions)

            # Earlier positions get higher scores
            relative_pos = avg_pos / n if n > 0 else 1.0
            score = float(exp_fn(-self.decay_rate * relative_pos))
            scores.append(score)

        return scores


@dataclass
class FrequencyImportanceScorer:
    """Score patterns based on their frequency.

    Intuition: Very frequent patterns are often structural/boilerplate
    (good candidates for compression). Rare patterns may be more
    semantically significant.
    """

    def score_patterns(
        self,
        tokens: TokenSeq,
        candidates: list[Candidate],
    ) -> list[float]:
        """Score patterns by rarity (rarer = more important)."""
        if not candidates:
            return []

        # Find max frequency for normalization
        max_freq = max(len(c.positions) for c in candidates) if candidates else 1

        scores: list[float] = []
        for cand in candidates:
            freq = len(cand.positions)
            # Inverse frequency, normalized
            score = 1.0 - (freq / max_freq) if max_freq > 0 else 0.5
            scores.append(score)

        return scores


@dataclass
class LengthImportanceScorer:
    """Score patterns based on their length.

    Intuition: Longer patterns often represent meaningful structures
    (function signatures, repeated code blocks). Shorter patterns
    may be coincidental repetitions.
    """

    def score_patterns(
        self,
        tokens: TokenSeq,
        candidates: list[Candidate],
    ) -> list[float]:
        """Score patterns by length (longer = more important)."""
        if not candidates:
            return []

        max_len = max(c.length for c in candidates) if candidates else 1

        scores: list[float] = []
        for cand in candidates:
            # Longer patterns get higher importance
            score = cand.length / max_len if max_len > 0 else 0.5
            scores.append(score)

        return scores


@dataclass
class CompositeImportanceScorer:
    """Combine multiple importance signals with configurable weights.

    Default weights balance multiple signals for robust scoring.
    """

    scorers: list[tuple[ImportanceScorer, float]]  # (scorer, weight) pairs

    def score_patterns(
        self,
        tokens: TokenSeq,
        candidates: list[Candidate],
    ) -> list[float]:
        """Compute weighted combination of all scorer signals."""
        if not candidates:
            return []

        if not self.scorers:
            return [0.5] * len(candidates)

        combined = [0.0] * len(candidates)
        total_weight = sum(w for _, w in self.scorers)

        if total_weight == 0:
            return [0.5] * len(candidates)

        for scorer, weight in self.scorers:
            try:
                scores = scorer.score_patterns(tokens, candidates)
                for i, score in enumerate(scores):
                    combined[i] += score * weight / total_weight
            except Exception:
                # If a scorer fails, skip it
                continue

        return combined


def create_default_scorer() -> CompositeImportanceScorer:
    """Create a default composite scorer without ML dependencies.

    Uses positional, frequency, and length signals.
    """
    return CompositeImportanceScorer(
        scorers=[
            (PositionalImportanceScorer(decay_rate=2.0), 0.4),
            (FrequencyImportanceScorer(), 0.3),
            (LengthImportanceScorer(), 0.3),
        ]
    )


def create_embedding_scorer(provider: object) -> CompositeImportanceScorer:
    """Create a scorer that includes embedding-based importance.

    Args:
        provider: An EmbeddingProvider from small.embeddings
    """
    return CompositeImportanceScorer(
        scorers=[
            (EmbeddingImportanceScorer(provider=provider), 0.4),
            (PositionalImportanceScorer(decay_rate=2.0), 0.3),
            (FrequencyImportanceScorer(), 0.2),
            (LengthImportanceScorer(), 0.1),
        ]
    )


def adjust_candidate_priorities(
    candidates: list[Candidate],
    importance_scores: list[float],
    importance_weight: float = 0.5,
) -> list[Candidate]:
    """Adjust candidate priorities based on importance scores.

    Lower importance = higher compression priority (compress more aggressively)
    Higher importance = lower compression priority (preserve more carefully)

    Args:
        candidates: Original candidate list
        importance_scores: Scores from an ImportanceScorer (0-1 range)
        importance_weight: How much to weight importance in priority (0-1)

    Returns:
        New candidate list with adjusted priorities
    """
    if not candidates or not importance_scores:
        return candidates

    if len(candidates) != len(importance_scores):
        return candidates

    adjusted: list[Candidate] = []

    for cand, importance in zip(candidates, importance_scores):
        # Low importance = higher compression priority bonus
        # Map importance [0,1] to priority adjustment [5,0]
        priority_adjustment = int((1 - importance) * 5 * importance_weight)

        adjusted.append(
            Candidate(
                subsequence=cand.subsequence,
                length=cand.length,
                positions=cand.positions,
                priority=cand.priority + priority_adjustment,
                patches=cand.patches,
            )
        )

    return adjusted


def filter_high_importance_candidates(
    candidates: list[Candidate],
    importance_scores: list[float],
    threshold: float = 0.8,
) -> list[Candidate]:
    """Filter out candidates that are too important to compress.

    Args:
        candidates: Original candidate list
        importance_scores: Scores from an ImportanceScorer
        threshold: Importance threshold above which to exclude

    Returns:
        Filtered candidate list
    """
    if not candidates or not importance_scores:
        return candidates

    return [
        cand for cand, score in zip(candidates, importance_scores) if score < threshold
    ]
