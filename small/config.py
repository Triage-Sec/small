"""Configuration for LTSC compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompressionConfig:
    # Core pattern discovery settings
    min_subsequence_length: int = 2
    max_subsequence_length: int = 8

    # Meta-token configuration
    meta_token_prefix: str = "<MT_"
    meta_token_suffix: str = ">"
    meta_token_pool_size: int = 500

    # Dictionary format settings
    dict_start_token: str = "<Dict>"
    dict_end_token: str = "</Dict>"
    dict_length_enabled: bool = True
    dict_length_prefix: str = "<Len:"
    dict_length_suffix: str = ">"

    # Hierarchical compression
    hierarchical_enabled: bool = True
    hierarchical_max_depth: int = 3
    hierarchical_min_improvement: float = 0.02  # Early stopping threshold

    # Selection algorithm settings
    # Modes: "greedy", "optimal", "beam", "ilp"
    selection_mode: str = "greedy"
    beam_width: int = 8
    ilp_time_limit: float = 1.0  # Timeout for ILP solver
    ilp_fallback_threshold: int = (
        2000  # Fallback to LP relaxation above this many occurrences
    )
    ilp_use_relaxation: bool = (
        True  # Use LP relaxation for larger inputs instead of beam search
    )

    # AST-aware compression (Python)
    ast_enabled: bool = True
    ast_priority_bonus: int = 2

    # Static dictionary settings
    static_dictionary_id: Optional[str] = None
    static_dictionary_auto: bool = True
    static_dictionary_min_confidence: float = 0.85
    static_dictionary_marker_prefix: str = "<StaticDict:"
    static_dictionary_marker_suffix: str = ">"
    static_dictionary_min_length: int = 2

    # Fuzzy matching settings
    fuzzy_enabled: bool = False
    fuzzy_max_diff: int = 1
    fuzzy_priority_bonus: int = 1
    patch_start_token: str = "<Patch>"
    patch_end_token: str = "</Patch>"
    patch_index_prefix: str = "<Idx:"
    patch_index_suffix: str = ">"

    # Metrics and logging
    metrics_enabled: bool = True
    metrics_jsonl_path: str | None = None
    combined_metrics_jsonl_path: str | None = None
    cache_stats: dict[str, int] | None = None
    cache_stats_source: object | None = None

    # Discovery algorithm settings
    # Modes: "suffix-array", "sliding-window", "bpe"
    discovery_mode: str = "suffix-array"
    parallel_discovery: bool = False
    parallel_length_threshold: int = 20000
    chunk_size: int = 200000

    # BPE-style discovery settings
    enable_bpe_discovery: bool = False
    bpe_max_iterations: int = 100

    # Subsumption analysis
    enable_subsumption_pruning: bool = True
    subsumption_min_independent: int = (
        2  # Min independent occurrences to keep subsumed pattern
    )

    # ML Integration: Pattern importance scoring
    use_importance_scoring: bool = False
    # Scorer types: "positional", "frequency", "length", "embedding", "composite"
    importance_scorer_type: str = "composite"
    importance_weight: float = 0.5  # How much importance affects priority
    importance_filter_threshold: float = 0.9  # Skip patterns above this importance

    # Adaptive region-aware compression
    enable_adaptive_regions: bool = False
    region_markers: dict[str, str] | None = None  # Custom marker -> region type mapping
    adaptive_system_fraction: float = (
        0.1  # Heuristic: fraction assumed to be system context
    )

    # Quality prediction
    enable_quality_prediction: bool = False
    quality_task_type: str = "general"  # Task type for quality prediction
    max_predicted_degradation: float = (
        0.05  # Skip compression if predicted degradation exceeds
    )
    quality_conservative: bool = False  # Be more conservative in quality predictions

    # Reproducibility and debugging
    rng_seed: Optional[int] = None
    verify: bool = False
