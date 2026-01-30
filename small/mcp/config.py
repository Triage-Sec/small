"""Configuration for MCP server.

Configuration can be set via environment variables or programmatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class MCPConfig:
    """Configuration for the MCP server.

    All settings can be overridden via environment variables prefixed with SMALL_MCP_.
    For example: SMALL_MCP_MAX_INPUT_TOKENS=50000

    Attributes:
        max_input_tokens: Maximum tokens allowed in a single compress request.
        max_text_length: Maximum text length (chars) for compress_text.
        metrics_dir: Directory for metrics storage. None disables persistence.
        metrics_file: Filename for JSONL metrics within metrics_dir.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        enable_benchmarks: Allow running benchmarks via MCP.
        default_min_length: Default minimum pattern length.
        default_max_length: Default maximum pattern length.
        verify_roundtrip: Enable round-trip verification by default.
        rate_limit_ops_per_min: Max operations per minute. 0 = unlimited.
        discovery_mode: Default discovery algorithm.
        selection_mode: Default selection algorithm.
    """

    max_input_tokens: int = 100_000
    max_text_length: int = 500_000
    metrics_dir: Path | None = field(default_factory=lambda: Path.home() / ".small")
    metrics_file: str = "mcp_metrics.jsonl"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_benchmarks: bool = True
    default_min_length: int = 2
    default_max_length: int = 16
    verify_roundtrip: bool = True
    rate_limit_ops_per_min: int = 0
    discovery_mode: str = "suffix-array"
    selection_mode: str = "greedy"

    # Pattern cache settings
    enable_pattern_cache: bool = True
    pattern_cache_max_patterns: int = 10_000
    pattern_cache_file: str = "pattern_cache.json"
    warm_start_top_k: int = 50

    @classmethod
    def from_env(cls) -> MCPConfig:
        """Create config from environment variables."""

        def get_env(key: str, default: str) -> str:
            return os.environ.get(f"SMALL_MCP_{key}", default)

        def get_env_int(key: str, default: int) -> int:
            val = os.environ.get(f"SMALL_MCP_{key}")
            return int(val) if val else default

        def get_env_bool(key: str, default: bool) -> bool:
            val = os.environ.get(f"SMALL_MCP_{key}")
            if val is None:
                return default
            return val.lower() in ("1", "true", "yes", "on")

        def get_env_path(key: str, default: Path | None) -> Path | None:
            val = os.environ.get(f"SMALL_MCP_{key}")
            if val is None:
                return default
            if val.lower() in ("", "none", "null", "false"):
                return None
            return Path(val)

        return cls(
            max_input_tokens=get_env_int("MAX_INPUT_TOKENS", 100_000),
            max_text_length=get_env_int("MAX_TEXT_LENGTH", 500_000),
            metrics_dir=get_env_path("METRICS_DIR", Path.home() / ".small"),
            metrics_file=get_env("METRICS_FILE", "mcp_metrics.jsonl"),
            log_level=get_env("LOG_LEVEL", "INFO"),  # type: ignore[arg-type]
            enable_benchmarks=get_env_bool("ENABLE_BENCHMARKS", True),
            default_min_length=get_env_int("DEFAULT_MIN_LENGTH", 2),
            default_max_length=get_env_int("DEFAULT_MAX_LENGTH", 16),
            verify_roundtrip=get_env_bool("VERIFY_ROUNDTRIP", True),
            rate_limit_ops_per_min=get_env_int("RATE_LIMIT_OPS_PER_MIN", 0),
            discovery_mode=get_env("DISCOVERY_MODE", "suffix-array"),
            selection_mode=get_env("SELECTION_MODE", "greedy"),
            enable_pattern_cache=get_env_bool("ENABLE_PATTERN_CACHE", True),
            pattern_cache_max_patterns=get_env_int("PATTERN_CACHE_MAX_PATTERNS", 10_000),
            pattern_cache_file=get_env("PATTERN_CACHE_FILE", "pattern_cache.json"),
            warm_start_top_k=get_env_int("WARM_START_TOP_K", 50),
        )

    @property
    def metrics_path(self) -> Path | None:
        """Full path to metrics file."""
        if self.metrics_dir is None:
            return None
        return self.metrics_dir / self.metrics_file

    @property
    def pattern_cache_path(self) -> Path | None:
        """Full path to pattern cache file."""
        if self.metrics_dir is None or not self.enable_pattern_cache:
            return None
        return self.metrics_dir / self.pattern_cache_file
