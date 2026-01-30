"""Metrics tracking for MCP compression operations.

Provides session-level statistics and persistent storage for analysis.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single compression/decompression operation."""

    timestamp: str
    operation: str  # compress, decompress, analyze, compress_text, compress_context
    input_tokens: int
    output_tokens: int
    compression_ratio: float
    savings_percent: float
    patterns_found: int
    time_ms: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SessionStats:
    """Accumulated statistics for the current MCP session."""

    session_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    total_operations: int = 0
    compress_operations: int = 0
    decompress_operations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens_saved: int = 0
    total_time_ms: float = 0.0
    errors: int = 0
    _ratios: list[float] = field(default_factory=list, repr=False)
    _savings: list[float] = field(default_factory=list, repr=False)

    @property
    def avg_compression_ratio(self) -> float:
        """Average compression ratio across all compress operations."""
        return sum(self._ratios) / len(self._ratios) if self._ratios else 1.0

    @property
    def avg_savings_percent(self) -> float:
        """Average savings percentage across all compress operations."""
        return sum(self._savings) / len(self._savings) if self._savings else 0.0

    @property
    def best_savings_percent(self) -> float:
        """Best (highest) savings percentage achieved."""
        return max(self._savings) if self._savings else 0.0

    @property
    def worst_savings_percent(self) -> float:
        """Worst (lowest) savings percentage."""
        return min(self._savings) if self._savings else 0.0

    @property
    def tokens_per_second(self) -> float:
        """Processing throughput in tokens per second."""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.total_input_tokens / self.total_time_ms) * 1000

    def record_compress(self, metrics: OperationMetrics) -> None:
        """Record a compression operation."""
        self.total_operations += 1
        self.compress_operations += 1
        self.total_time_ms += metrics.time_ms
        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens
        self.total_tokens_saved += metrics.input_tokens - metrics.output_tokens
        self._ratios.append(metrics.compression_ratio)
        self._savings.append(metrics.savings_percent)
        if not metrics.success:
            self.errors += 1

    def record_decompress(self, metrics: OperationMetrics) -> None:
        """Record a decompression operation."""
        self.total_operations += 1
        self.decompress_operations += 1
        self.total_time_ms += metrics.time_ms
        if not metrics.success:
            self.errors += 1

    def to_dict(self, include_internals: bool = False) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "total_operations": self.total_operations,
            "compress_operations": self.compress_operations,
            "decompress_operations": self.decompress_operations,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "total_time_ms": round(self.total_time_ms, 2),
            "errors": self.errors,
            "avg_compression_ratio": round(self.avg_compression_ratio, 4),
            "avg_savings_percent": round(self.avg_savings_percent, 2),
            "best_savings_percent": round(self.best_savings_percent, 2),
            "worst_savings_percent": round(self.worst_savings_percent, 2),
            "tokens_per_second": round(self.tokens_per_second, 0),
        }
        if include_internals:
            result["_ratios"] = self._ratios
            result["_savings"] = self._savings
        return result


class MetricsStore:
    """Thread-safe metrics storage with optional persistence."""

    def __init__(self, metrics_path: Path | None = None) -> None:
        """Initialize metrics store.

        Args:
            metrics_path: Path to JSONL file for persistent storage.
                         None disables persistence.
        """
        self._lock = threading.Lock()
        self._session = SessionStats()
        self._operations: list[OperationMetrics] = []
        self._metrics_path = metrics_path

        if metrics_path:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def session(self) -> SessionStats:
        """Current session statistics."""
        return self._session

    def record(self, metrics: OperationMetrics) -> None:
        """Record an operation's metrics."""
        with self._lock:
            self._operations.append(metrics)

            if metrics.operation in (
                "compress",
                "compress_text",
                "compress_context",
                "analyze",
            ):
                self._session.record_compress(metrics)
            elif metrics.operation == "decompress":
                self._session.record_decompress(metrics)

            self._persist(metrics)

    def _persist(self, metrics: OperationMetrics) -> None:
        """Persist metrics to file (called within lock)."""
        if self._metrics_path is None:
            return

        try:
            with open(self._metrics_path, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except OSError as e:
            logger.warning(f"Failed to persist metrics: {e}")

    def get_operations(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get recent operations from current session."""
        with self._lock:
            ops = self._operations[-limit:] if limit else self._operations
            return [m.to_dict() for m in ops]

    def load_historical(self, limit: int = 100) -> dict[str, Any]:
        """Load historical metrics from persistent storage."""
        if self._metrics_path is None or not self._metrics_path.exists():
            return {"operations": [], "total_count": 0}

        operations: list[dict[str, Any]] = []
        try:
            with open(self._metrics_path) as f:
                for line in f:
                    if line.strip():
                        operations.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load historical metrics: {e}")
            return {"error": str(e), "operations": [], "total_count": 0}

        # Compute aggregates
        compress_ops = [
            op
            for op in operations
            if op.get("operation") in ("compress", "compress_text", "compress_context")
        ]
        total_input = sum(op.get("input_tokens", 0) for op in compress_ops)
        total_output = sum(op.get("output_tokens", 0) for op in compress_ops)
        avg_savings = (
            sum(op.get("savings_percent", 0) for op in compress_ops) / len(compress_ops)
            if compress_ops
            else 0
        )

        return {
            "total_count": len(operations),
            "compress_operations": len(compress_ops),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens_saved": total_input - total_output,
            "avg_savings_percent": round(avg_savings, 2),
            "operations": operations[-limit:],
        }

    def reset(self) -> dict[str, Any]:
        """Reset session statistics. Returns previous session summary."""
        with self._lock:
            old_session = self._session.to_dict()
            self._session = SessionStats()
            self._operations.clear()
            return old_session
