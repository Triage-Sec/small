"""Configuration for LTSC compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompressionConfig:
    max_subsequence_length: int = 6
    meta_token_prefix: str = "<MT_"
    meta_token_suffix: str = ">"
    meta_token_pool_size: int = 512
    dict_start_token: str = "<Dict>"
    dict_end_token: str = "</Dict>"
    dict_length_enabled: bool = True
    dict_length_prefix: str = "<Len:"
    dict_length_suffix: str = ">"
    hierarchical_enabled: bool = True
    hierarchical_max_depth: int = 3
    selection_mode: str = "greedy"
    beam_width: int = 8
    ast_enabled: bool = True
    ast_priority_bonus: int = 2
    static_dictionary_id: Optional[str] = None
    static_dictionary_auto: bool = True
    static_dictionary_min_confidence: float = 0.85
    static_dictionary_marker_prefix: str = "<StaticDict:"
    static_dictionary_marker_suffix: str = ">"
    static_dictionary_min_length: int = 2
    rng_seed: Optional[int] = None
    verify: bool = False
