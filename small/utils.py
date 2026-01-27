"""Utility helpers for LTSC compression."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from .config import CompressionConfig
from .types import Token


def is_meta_token(token: Token, config: CompressionConfig) -> bool:
    if not isinstance(token, str):
        return False
    return token.startswith(config.meta_token_prefix) and token.endswith(config.meta_token_suffix)


def generate_meta_token_pool(config: CompressionConfig, existing: Iterable[Token]) -> list[str]:
    existing_set = set(existing)
    pool: list[str] = []
    for idx in range(config.meta_token_pool_size):
        token = f"{config.meta_token_prefix}{idx}{config.meta_token_suffix}"
        if token in existing_set:
            continue
        pool.append(token)
    rng = random.Random(config.rng_seed)
    rng.shuffle(pool)
    return pool


def is_compressible(length: int, count: int) -> bool:
    return length * count > 1 + length + count


def require_no_reserved_tokens(tokens: Sequence[Token], config: CompressionConfig) -> None:
    if config.dict_start_token in tokens or config.dict_end_token in tokens:
        raise ValueError("Dictionary delimiter token appears in input sequence.")
    for token in tokens:
        if is_meta_token(token, config):
            raise ValueError("Input sequence contains a meta-token pattern.")
