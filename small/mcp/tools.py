"""Tool definitions and handlers for MCP server.

Each tool has a definition (schema) and a handler function.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from ..compressor import compress, decompress
from ..config import CompressionConfig
from ..pattern_cache import PatternCache
from .metrics import MetricsStore, OperationMetrics

if TYPE_CHECKING:
    from .config import MCPConfig

logger = logging.getLogger(__name__)


# Model pricing (input tokens only) used for cost-savings estimates.
# Values are USD per 1,000,000 input tokens.
# Source: official vendor pricing pages, as of 2026-01-30.
MODEL_INPUT_USD_PER_MTOK: dict[str, float] = {
    # OpenAI
    "gpt-5.2-thinking": 1.75,
    # Google
    "gemini-3.0-pro": 2.00,
    "gemini-3.0-flash": 0.50,
    # Anthropic
    "claude-opus-4.5": 5.00,
}


def estimate_input_saved_usd(tokens_saved: int) -> dict[str, float]:
    """Estimate input-token cost savings for common models.

    Notes:
        Tokenization and billing may differ across model providers. This is best-effort
        based on the token counts you provide (or tiktoken when using compress_text/
        compress_context).
    """
    if tokens_saved <= 0:
        return {name: 0.0 for name in MODEL_INPUT_USD_PER_MTOK}

    return {
        name: round((usd_per_mtok / 1_000_000) * tokens_saved, 6)
        for name, usd_per_mtok in MODEL_INPUT_USD_PER_MTOK.items()
    }


def build_cost_estimate(*, tokens_saved: int, token_basis: str) -> dict[str, Any]:
    """Build a structured cost estimate payload."""
    return {
        "as_of": "2026-01-30",
        "token_basis": token_basis,
        "input_usd_per_million_tokens": MODEL_INPUT_USD_PER_MTOK,
        "estimated_input_saved_usd": estimate_input_saved_usd(tokens_saved),
    }


# Tool definitions (JSON Schema format for MCP)
TOOL_DEFINITIONS = [
    {
        "name": "compress_tokens",
        "description": (
            "Compress a sequence of LLM tokens using lossless LTSC compression. "
            "Returns compressed tokens and compression statistics. "
            "Use this to reduce context length when you detect repeated patterns in token sequences."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Array of token IDs to compress",
                },
                "min_length": {
                    "type": "integer",
                    "default": 2,
                    "description": "Minimum pattern length to consider (2-16)",
                },
                "max_length": {
                    "type": "integer",
                    "default": 16,
                    "description": "Maximum pattern length to consider (2-64)",
                },
                "selection_mode": {
                    "type": "string",
                    "enum": ["greedy", "optimal", "beam"],
                    "default": "greedy",
                    "description": "Pattern selection algorithm",
                },
            },
            "required": ["tokens"],
        },
    },
    {
        "name": "decompress_tokens",
        "description": (
            "Decompress a previously compressed token sequence. "
            "Guarantees lossless reconstruction of the original tokens."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Compressed token sequence to decompress",
                }
            },
            "required": ["tokens"],
        },
    },
    {
        "name": "analyze_compression",
        "description": (
            "Analyze how compressible a token sequence is without actually compressing. "
            "Returns potential savings, detected patterns, and a recommendation. "
            "Use this to decide if compression is worthwhile for a given input."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Token sequence to analyze",
                }
            },
            "required": ["tokens"],
        },
    },
    {
        "name": "compress_text",
        "description": (
            "Compress text by first tokenizing with tiktoken (cl100k_base for GPT-4), "
            "then applying LTSC compression. Returns compressed tokens and statistics. "
            "Convenient for compressing raw text without manual tokenization."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to compress",
                },
                "encoding": {
                    "type": "string",
                    "default": "cl100k_base",
                    "description": "Tiktoken encoding (cl100k_base, p50k_base, r50k_base)",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "compress_context",
        "description": (
            "Compress an LLM context window containing system prompt, conversation history, "
            "and/or retrieved documents. Optimized for typical context patterns. "
            "Returns compressed tokens ready for LLM inference with reduced costs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Full context window text to compress",
                },
                "encoding": {
                    "type": "string",
                    "default": "cl100k_base",
                    "description": "Tiktoken encoding",
                },
                "preserve_recent": {
                    "type": "integer",
                    "default": 0,
                    "description": "Number of recent tokens to preserve uncompressed (0 = compress all)",
                },
            },
            "required": ["context"],
        },
    },
    {
        "name": "get_session_metrics",
        "description": (
            "Get accumulated metrics for the current MCP session. "
            "Shows total tokens processed, compression savings, timing, and throughput. "
            "Useful for monitoring compression efficiency across multiple operations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_operations": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include detailed per-operation metrics",
                }
            },
        },
    },
    {
        "name": "get_historical_metrics",
        "description": (
            "Load historical metrics from all previous sessions. "
            "Useful for analyzing compression performance trends over time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 100,
                    "description": "Maximum number of operations to return",
                }
            },
        },
    },
    {
        "name": "run_benchmark",
        "description": (
            "Run a quick benchmark to test compression performance on different input types. "
            "Tests repeated patterns, code-like sequences, and random data. "
            "Returns timing and compression statistics for each test case."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "size": {
                    "type": "integer",
                    "default": 1000,
                    "description": "Approximate token count for test inputs",
                },
                "runs": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of runs per test case",
                },
            },
        },
    },
    {
        "name": "reset_session_metrics",
        "description": (
            "Reset the current session's accumulated metrics. "
            "Does not affect historical metrics file. "
            "Returns summary of the reset session."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_pattern_cache_stats",
        "description": (
            "Get statistics about the cross-document pattern cache. "
            "Shows cached patterns, hit rate, and cumulative savings from pattern reuse. "
            "The pattern cache learns from previous compressions to improve future ones."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "clear_pattern_cache",
        "description": (
            "Clear the cross-document pattern cache. "
            "Useful when switching to a completely different document corpus. "
            "Returns stats from the cleared cache."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class ToolHandlers:
    """Collection of tool handler functions."""

    def __init__(
        self,
        config: MCPConfig,
        metrics: MetricsStore,
        pattern_cache: PatternCache | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self.pattern_cache = pattern_cache

    def _validate_tokens(
        self,
        tokens: list[Any],
        max_tokens: int | None = None,
        *,
        allow_strings: bool = False,
    ) -> None:
        """Validate token input.

        Args:
            tokens: Token sequence to validate.
            max_tokens: Override for maximum token limit.
            allow_strings: If True, allow string tokens (for serialized/compressed sequences).
        """
        if not tokens:
            raise ValueError("tokens array cannot be empty")

        limit = max_tokens or self.config.max_input_tokens
        if len(tokens) > limit:
            raise ValueError(
                f"Input exceeds maximum token limit ({len(tokens)} > {limit}). "
                "Consider splitting into smaller chunks."
            )

        if allow_strings:
            # Serialized tokens can contain int or str (dictionary markers)
            if not all(isinstance(t, (int, str)) for t in tokens):
                raise TypeError("Tokens must be integers or strings")
        else:
            if not all(isinstance(t, int) for t in tokens):
                raise TypeError("All tokens must be integers")

    def _validate_text(self, text: str, max_length: int | None = None) -> None:
        """Validate text input."""
        if not text:
            raise ValueError("text cannot be empty")

        limit = max_length or self.config.max_text_length
        if len(text) > limit:
            raise ValueError(
                f"Text exceeds maximum length ({len(text)} > {limit}). "
                "Consider splitting into smaller chunks."
            )

    def _build_config(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        selection_mode: str | None = None,
        verify: bool | None = None,
    ) -> CompressionConfig:
        """Build compression config with defaults from MCP config."""
        return CompressionConfig(
            min_subsequence_length=min_length or self.config.default_min_length,
            max_subsequence_length=max_length or self.config.default_max_length,
            discovery_mode=self.config.discovery_mode,
            selection_mode=selection_mode or self.config.selection_mode,
            verify=verify if verify is not None else self.config.verify_roundtrip,
            pattern_cache=self.pattern_cache,
            warm_start_top_k=self.config.warm_start_top_k,
        )

    def compress_tokens(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle compress_tokens tool call."""
        tokens = params["tokens"]
        self._validate_tokens(tokens)

        config = self._build_config(
            min_length=params.get("min_length"),
            max_length=params.get("max_length"),
            selection_mode=params.get("selection_mode"),
        )

        start = time.perf_counter()
        result = compress(tokens, config)
        elapsed_ms = (time.perf_counter() - start) * 1000

        ratio = (
            result.compressed_length / result.original_length
            if result.original_length
            else 1.0
        )
        savings = (1 - ratio) * 100

        # Record metrics
        self.metrics.record(
            OperationMetrics(
                timestamp=datetime.now().isoformat(),
                operation="compress",
                input_tokens=result.original_length,
                output_tokens=result.compressed_length,
                compression_ratio=ratio,
                savings_percent=savings,
                patterns_found=len(result.dictionary_map),
                time_ms=elapsed_ms,
                success=True,
            )
        )

        return {
            "compressed_tokens": result.serialized_tokens,
            "original_length": result.original_length,
            "compressed_length": result.compressed_length,
            "compression_ratio": round(ratio, 4),
            "savings_percent": round(savings, 2),
            "patterns_found": len(result.dictionary_map),
            "time_ms": round(elapsed_ms, 2),
            "dictionary": {str(k): list(v) for k, v in result.dictionary_map.items()},
            "session": {
                "total_tokens_saved": self.metrics.session.total_tokens_saved,
                "total_operations": self.metrics.session.compress_operations,
            },
        }

    def decompress_tokens(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle decompress_tokens tool call."""
        tokens = params["tokens"]
        # Compressed/serialized tokens may contain string markers (e.g. <Dict>)
        self._validate_tokens(tokens, allow_strings=True)

        start = time.perf_counter()
        restored = decompress(tokens)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Record metrics
        self.metrics.record(
            OperationMetrics(
                timestamp=datetime.now().isoformat(),
                operation="decompress",
                input_tokens=len(tokens),
                output_tokens=len(restored),
                compression_ratio=len(tokens) / len(restored) if restored else 1.0,
                savings_percent=0,
                patterns_found=0,
                time_ms=elapsed_ms,
                success=True,
            )
        )

        return {
            "decompressed_tokens": restored,
            "compressed_length": len(tokens),
            "decompressed_length": len(restored),
            "time_ms": round(elapsed_ms, 2),
        }

    def analyze_compression(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle analyze_compression tool call."""
        tokens = params["tokens"]
        self._validate_tokens(tokens)

        start = time.perf_counter()
        config = self._build_config(verify=False)
        result = compress(tokens, config)
        elapsed_ms = (time.perf_counter() - start) * 1000

        ratio = (
            result.compressed_length / result.original_length
            if result.original_length
            else 1.0
        )
        savings = (1 - ratio) * 100

        # Extract pattern info
        patterns: list[dict[str, Any]] = [
            {"pattern": list(seq), "length": len(seq), "meta_token": meta}
            for meta, seq in result.dictionary_map.items()
        ]

        # Record metrics
        self.metrics.record(
            OperationMetrics(
                timestamp=datetime.now().isoformat(),
                operation="analyze",
                input_tokens=result.original_length,
                output_tokens=result.compressed_length,
                compression_ratio=ratio,
                savings_percent=savings,
                patterns_found=len(patterns),
                time_ms=elapsed_ms,
                success=True,
            )
        )

        # Recommendation logic
        if savings > 15:
            recommendation = "highly_recommended"
        elif savings > 5:
            recommendation = "recommended"
        elif savings > 1:
            recommendation = "marginal"
        else:
            recommendation = "not_recommended"

        return {
            "original_length": result.original_length,
            "potential_compressed_length": result.compressed_length,
            "potential_savings_percent": round(savings, 2),
            "patterns_detected": len(patterns),
            "top_patterns": sorted(
                patterns, key=lambda p: int(p["length"]), reverse=True
            )[:10],
            "recommendation": recommendation,
            "time_ms": round(elapsed_ms, 2),
        }

    def compress_text(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle compress_text tool call."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for compress_text. Install with: pip install tiktoken"
            )

        text = params["text"]
        self._validate_text(text)

        encoding_name = params.get("encoding", "cl100k_base")
        enc = tiktoken.get_encoding(encoding_name)

        start = time.perf_counter()
        tokens = enc.encode(text)
        tokenize_ms = (time.perf_counter() - start) * 1000

        compress_start = time.perf_counter()
        config = self._build_config()
        result = compress(tokens, config)
        compress_ms = (time.perf_counter() - compress_start) * 1000

        total_ms = tokenize_ms + compress_ms
        ratio = result.compressed_length / len(tokens) if tokens else 1.0
        savings = (1 - ratio) * 100

        # Record metrics
        self.metrics.record(
            OperationMetrics(
                timestamp=datetime.now().isoformat(),
                operation="compress_text",
                input_tokens=len(tokens),
                output_tokens=result.compressed_length,
                compression_ratio=ratio,
                savings_percent=savings,
                patterns_found=len(result.dictionary_map),
                time_ms=total_ms,
                success=True,
                metadata={"encoding": encoding_name, "text_length": len(text)},
            )
        )

        return {
            "original_text_length": len(text),
            "original_token_count": len(tokens),
            "compressed_token_count": result.compressed_length,
            "compression_ratio": round(ratio, 4),
            "savings_percent": round(savings, 2),
            "compressed_tokens": result.serialized_tokens,
            "patterns_found": len(result.dictionary_map),
            "timing": {
                "tokenize_ms": round(tokenize_ms, 2),
                "compress_ms": round(compress_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "session": {
                "total_tokens_saved": self.metrics.session.total_tokens_saved,
            },
        }

    def compress_context(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle compress_context tool call."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for compress_context. Install with: pip install tiktoken"
            )

        context = params["context"]
        self._validate_text(context)

        encoding_name = params.get("encoding", "cl100k_base")
        preserve_recent = params.get("preserve_recent", 0)
        enc = tiktoken.get_encoding(encoding_name)

        start = time.perf_counter()
        tokens = enc.encode(context)
        tokenize_ms = (time.perf_counter() - start) * 1000

        # Handle preserve_recent
        preserved: list[int] = []
        to_compress = tokens
        if preserve_recent > 0 and len(tokens) > preserve_recent:
            to_compress = tokens[:-preserve_recent]
            preserved = tokens[-preserve_recent:]

        compress_start = time.perf_counter()
        config = self._build_config()
        result = compress(to_compress, config)
        compress_ms = (time.perf_counter() - compress_start) * 1000

        # Combine compressed + preserved
        final_tokens = result.serialized_tokens + preserved
        total_ms = tokenize_ms + compress_ms

        original_len = len(tokens)
        compressed_len = len(final_tokens)
        ratio = compressed_len / original_len if original_len else 1.0
        savings = (1 - ratio) * 100

        # Record metrics
        self.metrics.record(
            OperationMetrics(
                timestamp=datetime.now().isoformat(),
                operation="compress_context",
                input_tokens=original_len,
                output_tokens=compressed_len,
                compression_ratio=ratio,
                savings_percent=savings,
                patterns_found=len(result.dictionary_map),
                time_ms=total_ms,
                success=True,
                metadata={
                    "encoding": encoding_name,
                    "context_length": len(context),
                    "preserved_tokens": len(preserved),
                },
            )
        )

        tokens_saved = original_len - compressed_len

        return {
            "original_context_length": len(context),
            "original_token_count": original_len,
            "compressed_token_count": compressed_len,
            "preserved_token_count": len(preserved),
            "compression_ratio": round(ratio, 4),
            "savings_percent": round(savings, 2),
            "tokens_saved": tokens_saved,
            "compressed_tokens": final_tokens,
            "patterns_found": len(result.dictionary_map),
            "timing": {
                "tokenize_ms": round(tokenize_ms, 2),
                "compress_ms": round(compress_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "cost_estimate": build_cost_estimate(
                tokens_saved=tokens_saved,
                token_basis=f"tiktoken:{encoding_name}",
            ),
        }

    def get_session_metrics(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle get_session_metrics tool call."""
        include_ops = params.get("include_operations", False)

        result = self.metrics.session.to_dict()

        # Add cost estimates
        if result["total_tokens_saved"] > 0:
            result["cost_estimate"] = build_cost_estimate(
                tokens_saved=result["total_tokens_saved"],
                token_basis="token counts reported by MCP operations",
            )

        if include_ops:
            result["operations"] = self.metrics.get_operations()

        return result

    def get_historical_metrics(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle get_historical_metrics tool call."""
        limit = params.get("limit", 100)
        return self.metrics.load_historical(limit=limit)

    def run_benchmark(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle run_benchmark tool call."""
        if not self.config.enable_benchmarks:
            raise ValueError("Benchmarks are disabled in server configuration")

        size = min(params.get("size", 1000), 10000)  # Cap at 10k for safety
        runs = min(params.get("runs", 3), 10)  # Cap at 10 runs

        COMMON_TOKENS = list(range(1000))

        def gen_repeated(n: int) -> list[int]:
            pattern = [random.choice(COMMON_TOKENS) for _ in range(6)]
            result: list[int] = []
            while len(result) < n:
                result.extend(pattern)
                result.extend(
                    [random.choice(COMMON_TOKENS) for _ in range(random.randint(1, 3))]
                )
            return result[:n]

        def gen_code(n: int) -> list[int]:
            indent = [100, 100, 100, 100]
            result: list[int] = []
            while len(result) < n:
                result.extend([200, 201, 202])
                result.extend([random.choice(COMMON_TOKENS) for _ in range(3)])
                for _ in range(5):
                    result.extend(indent)
                    result.extend([random.choice(COMMON_TOKENS) for _ in range(5)])
            return result[:n]

        def gen_random(n: int) -> list[int]:
            return [random.randint(0, 99999) for _ in range(n)]

        test_cases = [
            ("repeated_pattern", gen_repeated),
            ("code_like", gen_code),
            ("random", gen_random),
        ]

        results = []
        config = self._build_config()

        for name, generator in test_cases:
            times = []
            savings_list = []

            for _ in range(runs):
                tokens = generator(size)

                start = time.perf_counter()
                result = compress(tokens, config)
                elapsed = (time.perf_counter() - start) * 1000

                times.append(elapsed)
                ratio = result.compressed_length / result.original_length
                savings_list.append((1 - ratio) * 100)

            avg_time = sum(times) / len(times)
            results.append(
                {
                    "name": name,
                    "input_size": size,
                    "avg_time_ms": round(avg_time, 2),
                    "min_time_ms": round(min(times), 2),
                    "max_time_ms": round(max(times), 2),
                    "avg_savings_percent": round(
                        sum(savings_list) / len(savings_list), 1
                    ),
                    "tokens_per_second": round((size / avg_time) * 1000, 0),
                }
            )

        return {
            "benchmark_config": {"size": size, "runs": runs},
            "results": results,
            "summary": {
                "fastest": min(results, key=lambda r: r["avg_time_ms"])["name"],
                "best_compression": max(
                    results, key=lambda r: r["avg_savings_percent"]
                )["name"],
            },
        }

    def reset_session_metrics(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle reset_session_metrics tool call."""
        old_session = self.metrics.reset()
        return {"reset": True, "previous_session": old_session}

    def get_pattern_cache_stats(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle get_pattern_cache_stats tool call."""
        if self.pattern_cache is None:
            return {
                "enabled": False,
                "message": "Pattern cache is not enabled for this session",
            }
        stats = self.pattern_cache.stats()
        stats["enabled"] = True
        return stats

    def clear_pattern_cache(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle clear_pattern_cache tool call."""
        if self.pattern_cache is None:
            return {
                "cleared": False,
                "message": "Pattern cache is not enabled for this session",
            }
        old_stats = self.pattern_cache.clear()
        return {"cleared": True, "previous_stats": old_stats}


def create_tool_handlers(
    config: MCPConfig,
    metrics: MetricsStore,
    pattern_cache: PatternCache | None = None,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Create a mapping of tool names to handler functions."""
    handlers = ToolHandlers(config, metrics, pattern_cache)
    return {
        "compress_tokens": handlers.compress_tokens,
        "decompress_tokens": handlers.decompress_tokens,
        "analyze_compression": handlers.analyze_compression,
        "compress_text": handlers.compress_text,
        "compress_context": handlers.compress_context,
        "get_session_metrics": handlers.get_session_metrics,
        "get_historical_metrics": handlers.get_historical_metrics,
        "run_benchmark": handlers.run_benchmark,
        "reset_session_metrics": handlers.reset_session_metrics,
        "get_pattern_cache_stats": handlers.get_pattern_cache_stats,
        "clear_pattern_cache": handlers.clear_pattern_cache,
    }
