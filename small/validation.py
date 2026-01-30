"""Configuration validation."""

from __future__ import annotations

from dataclasses import dataclass
from .config import CompressionConfig


@dataclass(frozen=True)
class ConfigWarning:
    message: str


def validate_config(config: CompressionConfig) -> list[ConfigWarning]:
    warnings: list[ConfigWarning] = []
    if config.min_subsequence_length < 2:
        raise ValueError("Minimum subsequence length must be >= 2.")
    if config.max_subsequence_length < config.min_subsequence_length:
        raise ValueError(
            "Maximum subsequence length must be >= minimum subsequence length."
        )
    if config.max_subsequence_length > 16:
        warnings.append(
            ConfigWarning("Maximum subsequence length above 16 is rarely beneficial.")
        )
    if config.meta_token_pool_size <= 0:
        raise ValueError("Meta-token pool size must be positive.")
    if config.hierarchical_max_depth < 1:
        raise ValueError("Hierarchical max depth must be >= 1.")
    if config.selection_mode not in {"greedy", "optimal", "beam", "ilp", "semantic"}:
        raise ValueError("Selection mode must be one of: greedy, optimal, beam, ilp, semantic.")
    if config.selection_mode == "beam" and config.beam_width < 1:
        raise ValueError("Beam width must be >= 1.")
    if (
        config.static_dictionary_min_confidence < 0
        or config.static_dictionary_min_confidence > 1
    ):
        raise ValueError("Static dictionary min confidence must be between 0 and 1.")
    if config.fuzzy_max_diff < 0:
        raise ValueError("Fuzzy max diff must be >= 0.")
    if config.discovery_mode not in {"suffix-array", "window"}:
        raise ValueError("Discovery mode must be one of: suffix-array, window.")
    if config.parallel_length_threshold < 0:
        raise ValueError("Parallel length threshold must be >= 0.")
    if config.chunk_size < 0:
        raise ValueError("Chunk size must be >= 0.")
    return warnings
