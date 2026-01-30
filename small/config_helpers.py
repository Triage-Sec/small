"""Helpers for configuring compression with cache stats sources."""

from __future__ import annotations

from dataclasses import replace

from .config import CompressionConfig


def with_cache_stats_source(
    config: CompressionConfig, source: object
) -> CompressionConfig:
    """Return a new config with cache stats source set."""
    return replace(config, cache_stats_source=source)
