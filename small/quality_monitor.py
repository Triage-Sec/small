"""Quality monitoring for compression operations.

Tracks compression quality metrics over time, computes rolling statistics,
and detects quality degradation. Designed for production use with:
- Thread-safe operations
- Low overhead (<1ms per record)
- Configurable rolling windows
- Baseline comparison
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Deque, Optional

from .types import CompressionResult


@dataclass(frozen=True)
class MonitoringConfig:
    """Configuration for quality monitoring.
    
    Attributes:
        enabled: Whether monitoring is active
        window_size: Number of records in rolling window
        window_duration_seconds: Time-based window (alternative to count)
        
        compression_ratio_warning: Warn if ratio exceeds this
        compression_ratio_critical: Critical alert if ratio exceeds this
        degradation_warning: Warn if predicted degradation exceeds this
        degradation_critical: Critical if predicted degradation exceeds this
        overhead_warning: Warn if dictionary overhead exceeds this
        overhead_critical: Critical if dictionary overhead exceeds this
        
        alert_cooldown_seconds: Minimum seconds between repeated alerts
        baseline_learning_window: Records to use for baseline learning
    """
    enabled: bool = True
    window_size: int = 1000
    window_duration_seconds: int = 3600  # 1 hour
    
    # Compression ratio thresholds (lower is better)
    compression_ratio_warning: float = 0.85
    compression_ratio_critical: float = 0.95
    
    # Predicted degradation thresholds
    degradation_warning: float = 0.03
    degradation_critical: float = 0.08
    
    # Dictionary overhead thresholds
    overhead_warning: float = 0.4
    overhead_critical: float = 0.6
    
    # Alert settings
    alert_cooldown_seconds: int = 300  # 5 minutes
    
    # Baseline settings
    baseline_learning_window: int = 100


@dataclass
class QualityRecord:
    """A single compression quality record.
    
    Attributes:
        timestamp: When the compression occurred
        compression_ratio: Compressed / original length
        predicted_degradation: Quality predictor output (0-1)
        dictionary_overhead: Dictionary size / compressed size
        patterns_used: Number of patterns in dictionary
        original_length: Original token count
        compressed_length: Compressed token count
        verification_passed: Whether round-trip verification passed
        latency_ms: Compression time in milliseconds (optional)
        context: Additional context (doc_id, etc.)
    """
    timestamp: float
    compression_ratio: float
    predicted_degradation: float
    dictionary_overhead: float
    patterns_used: int
    original_length: int
    compressed_length: int
    verification_passed: bool = True
    latency_ms: float = 0.0
    context: dict = field(default_factory=dict)
    
    @classmethod
    def from_result(
        cls,
        result: CompressionResult,
        predicted_degradation: float = 0.0,
        latency_ms: float = 0.0,
        verification_passed: bool = True,
        context: dict | None = None,
    ) -> "QualityRecord":
        """Create a QualityRecord from a CompressionResult."""
        original_len = result.original_length
        compressed_len = result.compressed_length
        dict_len = len(result.dictionary_tokens)
        
        return cls(
            timestamp=time.time(),
            compression_ratio=compressed_len / original_len if original_len > 0 else 1.0,
            predicted_degradation=predicted_degradation,
            dictionary_overhead=dict_len / compressed_len if compressed_len > 0 else 0.0,
            patterns_used=len(result.dictionary_map),
            original_length=original_len,
            compressed_length=compressed_len,
            verification_passed=verification_passed,
            latency_ms=latency_ms,
            context=context or {},
        )


@dataclass
class QualitySummary:
    """Aggregated quality statistics over a time window.
    
    Attributes:
        record_count: Number of records in window
        window_start: Start of time window
        window_end: End of time window
        
        compression_ratio_*: Statistics for compression ratio
        degradation_*: Statistics for predicted degradation
        overhead_*: Statistics for dictionary overhead
        
        verification_failure_rate: Fraction of failed verifications
        total_tokens_processed: Sum of original lengths
        total_tokens_saved: Sum of (original - compressed)
    """
    record_count: int
    window_start: datetime
    window_end: datetime
    
    # Compression ratio statistics
    compression_ratio_mean: float
    compression_ratio_p50: float
    compression_ratio_p95: float
    compression_ratio_p99: float
    compression_ratio_min: float
    compression_ratio_max: float
    
    # Degradation statistics
    degradation_mean: float
    degradation_p50: float
    degradation_p95: float
    degradation_p99: float
    degradation_max: float
    
    # Overhead statistics
    overhead_mean: float
    overhead_p95: float
    overhead_max: float
    
    # Patterns statistics
    patterns_mean: float
    patterns_max: int
    
    # Aggregate metrics
    verification_failure_rate: float
    total_tokens_processed: int
    total_tokens_saved: int
    avg_latency_ms: float


@dataclass
class QualityBaseline:
    """Baseline quality statistics for comparison.
    
    Learned from historical data to detect anomalies.
    """
    metric: str
    mean: float
    stddev: float
    p95: float
    sample_count: int
    learned_at: datetime
    
    def z_score(self, value: float) -> float:
        """Compute z-score for a value against this baseline."""
        if self.stddev == 0:
            return 0.0
        return (value - self.mean) / self.stddev
    
    def is_anomaly(self, value: float, threshold: float = 2.0) -> bool:
        """Check if value is anomalous (beyond threshold z-score)."""
        return abs(self.z_score(value)) > threshold


@dataclass
class HealthStatus:
    """Current health status of compression quality."""
    healthy: bool
    has_alerts: bool
    alerts: list  # List of QualityAlert
    summary: QualitySummary | None
    baselines: dict[str, QualityBaseline]
    checked_at: datetime


class QualityMonitor:
    """Monitor compression quality over time.
    
    Thread-safe implementation using a rolling window of records.
    Computes statistics, checks against thresholds, and manages baselines.
    
    Example:
        monitor = QualityMonitor(MonitoringConfig())
        
        # Record compressions
        for result in compression_results:
            monitor.record(result)
        
        # Check health
        health = monitor.check_health()
        if not health.healthy:
            for alert in health.alerts:
                print(f"Alert: {alert.message}")
        
        # Get summary
        summary = monitor.get_summary()
        print(f"P95 ratio: {summary.compression_ratio_p95}")
    """
    
    def __init__(self, config: MonitoringConfig | None = None):
        self.config = config or MonitoringConfig()
        self._records: Deque[QualityRecord] = deque(maxlen=self.config.window_size)
        self._lock = threading.RLock()
        self._baselines: dict[str, QualityBaseline] = {}
        self._alert_manager: Optional["AlertManager"] = None
        self._on_record_callbacks: list[Callable[[QualityRecord], None]] = []
    
    def record(
        self,
        result: CompressionResult,
        predicted_degradation: float = 0.0,
        latency_ms: float = 0.0,
        verification_passed: bool = True,
        context: dict | None = None,
    ) -> QualityRecord:
        """Record a compression result for monitoring.
        
        Args:
            result: The compression result to record
            predicted_degradation: Output from quality predictor
            latency_ms: Compression time in milliseconds
            verification_passed: Whether round-trip verification passed
            context: Additional context (e.g., doc_id)
            
        Returns:
            The created QualityRecord
        """
        if not self.config.enabled:
            return QualityRecord.from_result(result, predicted_degradation, latency_ms, verification_passed, context)
        
        record = QualityRecord.from_result(
            result,
            predicted_degradation=predicted_degradation,
            latency_ms=latency_ms,
            verification_passed=verification_passed,
            context=context,
        )
        
        with self._lock:
            self._records.append(record)
        
        # Fire callbacks
        for callback in self._on_record_callbacks:
            try:
                callback(record)
            except Exception:
                pass  # Don't let callbacks break monitoring
        
        return record
    
    def record_from_record(self, record: QualityRecord) -> None:
        """Add an existing QualityRecord directly."""
        if not self.config.enabled:
            return
        
        with self._lock:
            self._records.append(record)
    
    def get_records(
        self,
        window: timedelta | None = None,
        max_records: int | None = None,
    ) -> list[QualityRecord]:
        """Get records from the window.
        
        Args:
            window: Time window (None = all records)
            max_records: Maximum records to return
            
        Returns:
            List of QualityRecord objects
        """
        with self._lock:
            records = list(self._records)
        
        if window is not None:
            cutoff = time.time() - window.total_seconds()
            records = [r for r in records if r.timestamp >= cutoff]
        
        if max_records is not None:
            records = records[-max_records:]
        
        return records
    
    def get_summary(self, window: timedelta | None = None) -> QualitySummary | None:
        """Compute aggregated statistics for the window.
        
        Args:
            window: Time window (None = use configured window)
            
        Returns:
            QualitySummary or None if no records
        """
        if window is None:
            window = timedelta(seconds=self.config.window_duration_seconds)
        
        records = self.get_records(window=window)
        
        if not records:
            return None
        
        # Extract metrics
        ratios = [r.compression_ratio for r in records]
        degradations = [r.predicted_degradation for r in records]
        overheads = [r.dictionary_overhead for r in records]
        patterns = [r.patterns_used for r in records]
        latencies = [r.latency_ms for r in records if r.latency_ms > 0]
        
        # Compute percentiles
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_data) else f
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
        
        # Verification stats
        failures = sum(1 for r in records if not r.verification_passed)
        
        return QualitySummary(
            record_count=len(records),
            window_start=datetime.fromtimestamp(min(r.timestamp for r in records)),
            window_end=datetime.fromtimestamp(max(r.timestamp for r in records)),
            
            compression_ratio_mean=statistics.mean(ratios),
            compression_ratio_p50=percentile(ratios, 50),
            compression_ratio_p95=percentile(ratios, 95),
            compression_ratio_p99=percentile(ratios, 99),
            compression_ratio_min=min(ratios),
            compression_ratio_max=max(ratios),
            
            degradation_mean=statistics.mean(degradations),
            degradation_p50=percentile(degradations, 50),
            degradation_p95=percentile(degradations, 95),
            degradation_p99=percentile(degradations, 99),
            degradation_max=max(degradations),
            
            overhead_mean=statistics.mean(overheads),
            overhead_p95=percentile(overheads, 95),
            overhead_max=max(overheads),
            
            patterns_mean=statistics.mean(patterns),
            patterns_max=max(patterns),
            
            verification_failure_rate=failures / len(records),
            total_tokens_processed=sum(r.original_length for r in records),
            total_tokens_saved=sum(r.original_length - r.compressed_length for r in records),
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        )
    
    def check_health(self) -> HealthStatus:
        """Check current quality health against thresholds.
        
        Returns:
            HealthStatus with alerts if thresholds exceeded
        """
        from .quality_alerts import AlertManager, check_thresholds
        
        summary = self.get_summary()
        alerts = []
        
        if summary is not None:
            alerts = check_thresholds(summary, self.config)
        
        return HealthStatus(
            healthy=len(alerts) == 0,
            has_alerts=len(alerts) > 0,
            alerts=alerts,
            summary=summary,
            baselines=dict(self._baselines),
            checked_at=datetime.now(),
        )
    
    def learn_baseline(self, metric: str) -> QualityBaseline | None:
        """Learn a baseline for a metric from current records.
        
        Args:
            metric: Metric name ("compression_ratio", "degradation", "overhead")
            
        Returns:
            QualityBaseline or None if insufficient data
        """
        records = self.get_records(max_records=self.config.baseline_learning_window)
        
        if len(records) < 10:  # Minimum records for baseline
            return None
        
        # Extract values based on metric
        if metric == "compression_ratio":
            values = [r.compression_ratio for r in records]
        elif metric == "degradation":
            values = [r.predicted_degradation for r in records]
        elif metric == "overhead":
            values = [r.dictionary_overhead for r in records]
        else:
            return None
        
        baseline = QualityBaseline(
            metric=metric,
            mean=statistics.mean(values),
            stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
            p95=sorted(values)[int(len(values) * 0.95)],
            sample_count=len(values),
            learned_at=datetime.now(),
        )
        
        with self._lock:
            self._baselines[metric] = baseline
        
        return baseline
    
    def get_baseline(self, metric: str) -> QualityBaseline | None:
        """Get the current baseline for a metric."""
        return self._baselines.get(metric)
    
    def clear(self) -> None:
        """Clear all records and baselines."""
        with self._lock:
            self._records.clear()
            self._baselines.clear()
    
    def on_record(self, callback: Callable[[QualityRecord], None]) -> None:
        """Register a callback to be called on each record.
        
        Args:
            callback: Function to call with each QualityRecord
        """
        self._on_record_callbacks.append(callback)
    
    @property
    def record_count(self) -> int:
        """Current number of records in the window."""
        with self._lock:
            return len(self._records)


# Global monitor instance for convenience
_global_monitor: QualityMonitor | None = None


def get_global_monitor() -> QualityMonitor:
    """Get the global QualityMonitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = QualityMonitor()
    return _global_monitor


def set_global_monitor(monitor: QualityMonitor) -> None:
    """Set the global QualityMonitor instance."""
    global _global_monitor
    _global_monitor = monitor
