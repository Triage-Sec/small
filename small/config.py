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
    rng_seed: Optional[int] = None
    verify: bool = False
