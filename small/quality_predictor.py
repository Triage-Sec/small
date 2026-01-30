"""Predict compression quality impact on downstream tasks.

Before committing to compression, predict whether it will help or hurt
downstream task performance. This enables smart decisions about:
1. Whether to compress at all
2. How aggressively to compress
3. Which patterns to preserve

The predictor uses features extracted from the compression result to
estimate potential quality degradation. In production, this heuristic
model can be replaced with a trained classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .types import CompressionResult, Token, TokenSeq


@dataclass(frozen=True)
class QualityPrediction:
    """Prediction of compression quality impact.

    Attributes:
        predicted_degradation: Expected quality drop (0.0 = none, 1.0 = severe)
        confidence: Confidence in the prediction (0.0 - 1.0)
        recommendation: Action recommendation ("compress", "partial", "skip")
        suggested_max_ratio: Suggested maximum compression ratio to use
        risk_factors: List of identified risk factors
    """

    predicted_degradation: float
    confidence: float
    recommendation: str
    suggested_max_ratio: float
    risk_factors: tuple[str, ...]


@dataclass
class CompressionQualityPredictor:
    """Predict the impact of compression on downstream task quality.

    This model uses heuristics based on compression features to estimate
    potential quality impact. For production use, train a classifier on
    (original, compressed, task_score) triples.

    Attributes:
        model_path: Path to trained model (None for heuristic mode)
        task_type: Type of downstream task for tuning
        conservative: If True, be more conservative (prefer skip)
    """

    model_path: str | None = None
    task_type: str = "general"
    conservative: bool = False

    def predict(
        self,
        tokens: TokenSeq,
        proposed_result: CompressionResult,
    ) -> QualityPrediction:
        """Predict quality impact of proposed compression.

        Args:
            tokens: Original token sequence
            proposed_result: Proposed compression result to evaluate

        Returns:
            QualityPrediction with recommendation
        """
        features = self._extract_features(tokens, proposed_result)
        degradation, risk_factors = self._heuristic_predict(features)

        # Apply task-specific adjustments
        degradation = self._adjust_for_task(degradation, features)

        # Apply conservative mode
        if self.conservative:
            degradation = min(1.0, degradation * 1.3)

        # Generate recommendation
        recommendation, suggested_ratio = self._generate_recommendation(
            degradation, proposed_result
        )

        return QualityPrediction(
            predicted_degradation=degradation,
            confidence=0.7,  # Heuristic confidence
            recommendation=recommendation,
            suggested_max_ratio=suggested_ratio,
            risk_factors=tuple(risk_factors),
        )

    def _extract_features(
        self,
        tokens: TokenSeq,
        result: CompressionResult,
    ) -> dict[str, float]:
        """Extract features for quality prediction."""
        original_len = len(tokens)
        compressed_len = result.compressed_length

        features: dict[str, float] = {
            "original_length": float(original_len),
            "compressed_length": float(compressed_len),
            "compression_ratio": compressed_len / original_len
            if original_len > 0
            else 1.0,
            "num_meta_tokens": float(len(result.meta_tokens_used)),
            "num_patterns": float(len(result.dictionary_map)),
        }

        # Dictionary overhead ratio
        dict_len = len(result.dictionary_tokens)
        features["dict_overhead_ratio"] = (
            dict_len / compressed_len if compressed_len > 0 else 0
        )

        # Pattern statistics
        if result.dictionary_map:
            lengths = [len(seq) for seq in result.dictionary_map.values()]
            features["avg_pattern_length"] = sum(lengths) / len(lengths)
            features["max_pattern_length"] = float(max(lengths))
            features["min_pattern_length"] = float(min(lengths))

            # Variance in pattern lengths
            mean = features["avg_pattern_length"]
            variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
            features["pattern_length_variance"] = variance
        else:
            features["avg_pattern_length"] = 0.0
            features["max_pattern_length"] = 0.0
            features["min_pattern_length"] = 0.0
            features["pattern_length_variance"] = 0.0

        # Token diversity
        unique_original = len(set(tokens))
        unique_compressed = len(set(result.compressed_tokens))
        features["original_diversity"] = (
            unique_original / original_len if original_len > 0 else 1.0
        )
        features["compressed_diversity"] = (
            unique_compressed / compressed_len if compressed_len > 0 else 1.0
        )
        features["diversity_change"] = (
            features["compressed_diversity"] / features["original_diversity"]
            if features["original_diversity"] > 0
            else 1.0
        )

        # Pattern coverage
        total_replaced = sum(
            len(seq) * self._count_occurrences(result.body_tokens, meta)
            for meta, seq in result.dictionary_map.items()
        )
        features["coverage_ratio"] = (
            total_replaced / original_len if original_len > 0 else 0
        )

        return features

    def _count_occurrences(self, tokens: Sequence[Token], target: Token) -> int:
        """Count occurrences of a token in a sequence."""
        return sum(1 for t in tokens if t == target)

    def _heuristic_predict(
        self,
        features: dict[str, float],
    ) -> tuple[float, list[str]]:
        """Heuristic prediction of quality degradation."""
        degradation = 0.0
        risk_factors: list[str] = []

        ratio = features["compression_ratio"]

        # Very aggressive compression tends to hurt
        if ratio < 0.5:
            risk = 0.15 * (0.5 - ratio)
            degradation += risk
            risk_factors.append(f"Aggressive compression ({ratio:.1%})")

        if ratio < 0.3:
            risk = 0.2 * (0.3 - ratio)
            degradation += risk
            risk_factors.append("Very aggressive compression")

        # Too many short patterns indicate fragmentation
        avg_len = features["avg_pattern_length"]
        num_patterns = features["num_patterns"]
        if avg_len < 3 and num_patterns > 10:
            degradation += 0.03
            risk_factors.append("Many short patterns (fragmented)")

        # High dictionary overhead is wasteful and may confuse
        dict_overhead = features["dict_overhead_ratio"]
        if dict_overhead > 0.3:
            degradation += 0.02
            risk_factors.append(f"High dictionary overhead ({dict_overhead:.1%})")

        if dict_overhead > 0.5:
            degradation += 0.03
            risk_factors.append("Dictionary overhead exceeds 50%")

        # Large diversity drop may indicate loss of unique tokens
        div_change = features["diversity_change"]
        if div_change < 0.7:
            degradation += 0.02
            risk_factors.append("Significant diversity reduction")

        # Very high pattern length variance may indicate inconsistent compression
        if features["pattern_length_variance"] > 10:
            degradation += 0.01
            risk_factors.append("Inconsistent pattern lengths")

        # Small inputs may not benefit and overhead is proportionally higher
        if features["original_length"] < 100 and num_patterns > 5:
            degradation += 0.02
            risk_factors.append("Small input with many patterns")

        return min(1.0, degradation), risk_factors

    def _adjust_for_task(
        self,
        degradation: float,
        features: dict[str, float],
    ) -> float:
        """Adjust degradation based on task type."""
        multipliers = {
            "general": 1.0,
            "code": 1.5,  # Code is sensitive to token changes
            "policy": 1.3,  # Policy text has critical keywords
            "math": 1.4,  # Mathematical notation is precise
            "creative": 0.8,  # Creative tasks more tolerant
            "summarization": 0.9,
            "qa": 1.2,  # QA needs accurate retrieval
            "classification": 0.7,  # Classification is robust
        }

        multiplier = multipliers.get(self.task_type, 1.0)
        return min(1.0, degradation * multiplier)

    def _generate_recommendation(
        self,
        degradation: float,
        result: CompressionResult,
    ) -> tuple[str, float]:
        """Generate recommendation based on predicted degradation."""
        current_ratio = (
            result.compressed_length / result.original_length
            if result.original_length > 0
            else 1.0
        )

        if degradation < 0.02:
            return "compress", current_ratio

        if degradation < 0.05:
            # Suggest slightly less aggressive compression
            suggested = min(0.9, current_ratio + 0.1)
            return "partial", suggested

        if degradation < 0.10:
            # Suggest conservative compression
            suggested = min(0.95, current_ratio + 0.2)
            return "partial", suggested

        return "skip", 1.0

    def should_compress(
        self,
        tokens: TokenSeq,
        proposed_result: CompressionResult,
        max_degradation: float = 0.05,
    ) -> bool:
        """Simple boolean check if compression should proceed.

        Args:
            tokens: Original tokens
            proposed_result: Proposed compression result
            max_degradation: Maximum acceptable degradation

        Returns:
            True if compression is recommended
        """
        prediction = self.predict(tokens, proposed_result)
        return prediction.predicted_degradation <= max_degradation


def create_predictor(task_type: str = "general") -> CompressionQualityPredictor:
    """Create a quality predictor for the given task type.

    Task types:
    - general: Default balanced prediction
    - code: More conservative for code generation
    - policy: Conservative for policy/legal text
    - math: Conservative for mathematical content
    - creative: More lenient for creative tasks
    - summarization: Moderately lenient
    - qa: Conservative for Q&A
    - classification: Lenient for classification
    """
    return CompressionQualityPredictor(task_type=task_type)


def evaluate_compression_quality(
    tokens: TokenSeq,
    result: CompressionResult,
) -> dict[str, float]:
    """Evaluate compression quality without making predictions.

    Returns quality metrics that can be used for analysis.
    """
    predictor = CompressionQualityPredictor()
    features = predictor._extract_features(tokens, result)

    # Add some derived metrics
    features["effective_compression"] = (
        1.0 - features["compression_ratio"]
        if features["compression_ratio"] < 1.0
        else 0.0
    )

    features["efficiency"] = (
        features["effective_compression"] / features["dict_overhead_ratio"]
        if features["dict_overhead_ratio"] > 0
        else 0.0
    )

    return features
