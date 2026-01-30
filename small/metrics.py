"""Compression metrics and logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from .config import CompressionConfig
from .types import Token
from .utils import is_meta_token


@dataclass(frozen=True)
class CompressionMetrics:
    compression_amount: float
    compression_ratio: float
    effective_savings: int
    candidates_discovered: int
    patterns_used: int
    avg_pattern_length: float
    avg_pattern_frequency: float
    dictionary_overhead_pct: float
    depth_utilization: dict[int, int]


def _depths(
    dictionary_map: dict[Token, tuple[Token, ...]], config: CompressionConfig
) -> dict[Token, int]:
    depths: dict[Token, int] = {}

    def depth(meta: Token) -> int:
        if meta in depths:
            return depths[meta]
        subseq = dictionary_map.get(meta, ())
        dep_depths = [depth(tok) for tok in subseq if is_meta_token(tok, config)]
        depths[meta] = 1 + (max(dep_depths) if dep_depths else 0)
        return depths[meta]

    for meta in dictionary_map:
        depth(meta)
    return depths


def compute_metrics(
    original_length: int,
    compressed_length: int,
    dictionary_tokens: Iterable[Token],
    dictionary_map: dict[Token, tuple[Token, ...]],
    body_tokens: Iterable[Token],
    candidates_discovered: int,
    config: CompressionConfig,
) -> CompressionMetrics:
    compression_amount = (
        (original_length - compressed_length) / original_length
        if original_length
        else 0.0
    )
    compression_ratio = compressed_length / original_length if original_length else 1.0
    effective_savings = original_length - compressed_length

    patterns_used = len(dictionary_map)
    lengths = [len(seq) for seq in dictionary_map.values()]
    avg_pattern_length = sum(lengths) / patterns_used if patterns_used else 0.0

    freq: list[int] = []
    body_list = list(body_tokens)
    for meta in dictionary_map:
        freq.append(sum(1 for token in body_list if token == meta))
    avg_pattern_frequency = sum(freq) / patterns_used if patterns_used else 0.0

    dict_len = len(list(dictionary_tokens))
    dictionary_overhead_pct = dict_len / compressed_length if compressed_length else 0.0

    depth_map = _depths(dictionary_map, config)
    depth_utilization: dict[int, int] = {}
    for depth in depth_map.values():
        depth_utilization[depth] = depth_utilization.get(depth, 0) + 1

    return CompressionMetrics(
        compression_amount=compression_amount,
        compression_ratio=compression_ratio,
        effective_savings=effective_savings,
        candidates_discovered=candidates_discovered,
        patterns_used=patterns_used,
        avg_pattern_length=avg_pattern_length,
        avg_pattern_frequency=avg_pattern_frequency,
        dictionary_overhead_pct=dictionary_overhead_pct,
        depth_utilization=depth_utilization,
    )


def log_metrics(metrics: CompressionMetrics) -> None:
    logger = logging.getLogger("small.metrics")
    logger.info(
        "compression_amount=%.4f compression_ratio=%.4f effective_savings=%d candidates=%d patterns_used=%d avg_len=%.2f avg_freq=%.2f dict_overhead=%.4f depths=%s",
        metrics.compression_amount,
        metrics.compression_ratio,
        metrics.effective_savings,
        metrics.candidates_discovered,
        metrics.patterns_used,
        metrics.avg_pattern_length,
        metrics.avg_pattern_frequency,
        metrics.dictionary_overhead_pct,
        metrics.depth_utilization,
    )


def log_cache_stats(stats: dict[str, int]) -> None:
    logger = logging.getLogger("small.metrics")
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    total = hits + misses
    hit_rate = (hits / total) if total else 0.0
    logger.info(
        "cache_sets=%d cache_hits=%d cache_misses=%d cache_evictions=%d cache_hit_rate=%.4f",
        stats.get("sets", 0),
        hits,
        misses,
        stats.get("evictions", 0),
        hit_rate,
    )
