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
    rng_seed: Optional[int] = None
    verify: bool = False
