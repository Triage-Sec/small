"""Export quality metrics in various formats.

Supports:
- Prometheus exposition format
- ASCII text reports
- JSONL for log aggregation
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .quality_monitor import QualityBaseline, QualityMonitor, QualitySummary


def export_prometheus(
    summary: "QualitySummary",
    prefix: str = "small_compression",
    labels: dict[str, str] | None = None,
) -> str:
    """Export summary as Prometheus exposition format.
    
    Args:
        summary: Quality summary to export
        prefix: Metric name prefix
        labels: Additional labels to add to all metrics
        
    Returns:
        Prometheus format string
    """
    lines: list[str] = []
    label_str = _format_labels(labels) if labels else ""
    
    # Helper to add metric
    def add_metric(name: str, value: float, help_text: str, metric_type: str = "gauge"):
        full_name = f"{prefix}_{name}"
        lines.append(f"# HELP {full_name} {help_text}")
        lines.append(f"# TYPE {full_name} {metric_type}")
        lines.append(f"{full_name}{label_str} {value}")
    
    # Compression ratio metrics
    add_metric(
        "compression_ratio_mean",
        summary.compression_ratio_mean,
        "Mean compression ratio (compressed/original)",
    )
    add_metric(
        "compression_ratio_p50",
        summary.compression_ratio_p50,
        "P50 compression ratio",
    )
    add_metric(
        "compression_ratio_p95",
        summary.compression_ratio_p95,
        "P95 compression ratio",
    )
    add_metric(
        "compression_ratio_p99",
        summary.compression_ratio_p99,
        "P99 compression ratio",
    )
    
    # Degradation metrics
    add_metric(
        "degradation_mean",
        summary.degradation_mean,
        "Mean predicted quality degradation",
    )
    add_metric(
        "degradation_p95",
        summary.degradation_p95,
        "P95 predicted quality degradation",
    )
    add_metric(
        "degradation_max",
        summary.degradation_max,
        "Max predicted quality degradation",
    )
    
    # Overhead metrics
    add_metric(
        "dictionary_overhead_mean",
        summary.overhead_mean,
        "Mean dictionary overhead ratio",
    )
    add_metric(
        "dictionary_overhead_p95",
        summary.overhead_p95,
        "P95 dictionary overhead ratio",
    )
    
    # Pattern metrics
    add_metric(
        "patterns_mean",
        summary.patterns_mean,
        "Mean patterns per compression",
    )
    add_metric(
        "patterns_max",
        float(summary.patterns_max),
        "Max patterns in single compression",
    )
    
    # Aggregate metrics
    add_metric(
        "verification_failure_rate",
        summary.verification_failure_rate,
        "Rate of verification failures",
    )
    add_metric(
        "tokens_processed_total",
        float(summary.total_tokens_processed),
        "Total tokens processed",
        metric_type="counter",
    )
    add_metric(
        "tokens_saved_total",
        float(summary.total_tokens_saved),
        "Total tokens saved by compression",
        metric_type="counter",
    )
    add_metric(
        "record_count",
        float(summary.record_count),
        "Number of compression records in window",
    )
    
    # Latency
    if summary.avg_latency_ms > 0:
        add_metric(
            "latency_ms_avg",
            summary.avg_latency_ms,
            "Average compression latency in milliseconds",
        )
    
    return "\n".join(lines) + "\n"


def _format_labels(labels: dict[str, str]) -> str:
    """Format labels for Prometheus."""
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


def export_summary_ascii(summary: "QualitySummary") -> str:
    """Export summary as ASCII text report.
    
    Args:
        summary: Quality summary to export
        
    Returns:
        ASCII formatted report
    """
    lines: list[str] = []
    
    lines.append("=" * 70)
    lines.append("COMPRESSION QUALITY SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Window: {summary.window_start.strftime('%Y-%m-%d %H:%M:%S')} - "
                 f"{summary.window_end.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Records: {summary.record_count}")
    lines.append("")
    
    # Compression ratio section
    lines.append("-" * 70)
    lines.append("COMPRESSION RATIO (lower is better)")
    lines.append("-" * 70)
    lines.append(f"  Mean:  {summary.compression_ratio_mean:.4f}  ({summary.compression_ratio_mean:.2%})")
    lines.append(f"  P50:   {summary.compression_ratio_p50:.4f}")
    lines.append(f"  P95:   {summary.compression_ratio_p95:.4f}")
    lines.append(f"  P99:   {summary.compression_ratio_p99:.4f}")
    lines.append(f"  Min:   {summary.compression_ratio_min:.4f}")
    lines.append(f"  Max:   {summary.compression_ratio_max:.4f}")
    lines.append("")
    
    # Degradation section
    lines.append("-" * 70)
    lines.append("PREDICTED DEGRADATION (lower is better)")
    lines.append("-" * 70)
    lines.append(f"  Mean:  {summary.degradation_mean:.4f}")
    lines.append(f"  P50:   {summary.degradation_p50:.4f}")
    lines.append(f"  P95:   {summary.degradation_p95:.4f}")
    lines.append(f"  P99:   {summary.degradation_p99:.4f}")
    lines.append(f"  Max:   {summary.degradation_max:.4f}")
    lines.append("")
    
    # Overhead section
    lines.append("-" * 70)
    lines.append("DICTIONARY OVERHEAD (lower is better)")
    lines.append("-" * 70)
    lines.append(f"  Mean:  {summary.overhead_mean:.4f}  ({summary.overhead_mean:.2%})")
    lines.append(f"  P95:   {summary.overhead_p95:.4f}")
    lines.append(f"  Max:   {summary.overhead_max:.4f}")
    lines.append("")
    
    # Patterns section
    lines.append("-" * 70)
    lines.append("PATTERNS")
    lines.append("-" * 70)
    lines.append(f"  Mean:  {summary.patterns_mean:.2f}")
    lines.append(f"  Max:   {summary.patterns_max}")
    lines.append("")
    
    # Aggregate section
    lines.append("-" * 70)
    lines.append("AGGREGATES")
    lines.append("-" * 70)
    lines.append(f"  Tokens Processed:     {summary.total_tokens_processed:,}")
    lines.append(f"  Tokens Saved:         {summary.total_tokens_saved:,}")
    savings_pct = (summary.total_tokens_saved / summary.total_tokens_processed * 100
                  if summary.total_tokens_processed > 0 else 0)
    lines.append(f"  Overall Savings:      {savings_pct:.2f}%")
    lines.append(f"  Verification Failures: {summary.verification_failure_rate:.4f}  "
                 f"({summary.verification_failure_rate:.2%})")
    if summary.avg_latency_ms > 0:
        lines.append(f"  Avg Latency:          {summary.avg_latency_ms:.2f} ms")
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def export_baseline_ascii(baseline: "QualityBaseline") -> str:
    """Export baseline as ASCII text.
    
    Args:
        baseline: Quality baseline to export
        
    Returns:
        ASCII formatted baseline info
    """
    lines: list[str] = []
    
    lines.append(f"Baseline: {baseline.metric}")
    lines.append("-" * 40)
    lines.append(f"  Mean:         {baseline.mean:.4f}")
    lines.append(f"  Std Dev:      {baseline.stddev:.4f}")
    lines.append(f"  P95:          {baseline.p95:.4f}")
    lines.append(f"  Sample Count: {baseline.sample_count}")
    lines.append(f"  Learned At:   {baseline.learned_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)


def export_health_ascii(
    monitor: "QualityMonitor",
) -> str:
    """Export full health report as ASCII.
    
    Args:
        monitor: Quality monitor to report on
        
    Returns:
        ASCII formatted health report
    """
    from .quality_alerts import format_alerts_ascii
    
    lines: list[str] = []
    health = monitor.check_health()
    
    lines.append("=" * 70)
    lines.append("COMPRESSION QUALITY HEALTH REPORT")
    lines.append(f"Generated: {health.checked_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    
    # Status
    status = "HEALTHY" if health.healthy else "DEGRADED"
    lines.append(f"Status: {status}")
    lines.append(f"Records in Window: {monitor.record_count}")
    lines.append("")
    
    # Summary
    if health.summary:
        lines.append(export_summary_ascii(health.summary))
        lines.append("")
    else:
        lines.append("No compression data available.")
        lines.append("")
    
    # Alerts
    if health.alerts:
        lines.append(format_alerts_ascii(health.alerts))
    else:
        lines.append("No active alerts.")
    lines.append("")
    
    # Baselines
    if health.baselines:
        lines.append("-" * 70)
        lines.append("BASELINES")
        lines.append("-" * 70)
        for metric, baseline in health.baselines.items():
            lines.append(export_baseline_ascii(baseline))
            lines.append("")
    
    return "\n".join(lines)


def export_jsonl(summary: "QualitySummary", extra: dict | None = None) -> str:
    """Export summary as JSONL line.
    
    Args:
        summary: Quality summary to export
        extra: Additional fields to include
        
    Returns:
        Single JSONL line
    """
    data = {
        "timestamp": int(time.time()),
        "window_start": summary.window_start.isoformat(),
        "window_end": summary.window_end.isoformat(),
        "record_count": summary.record_count,
        
        "compression_ratio_mean": summary.compression_ratio_mean,
        "compression_ratio_p50": summary.compression_ratio_p50,
        "compression_ratio_p95": summary.compression_ratio_p95,
        "compression_ratio_p99": summary.compression_ratio_p99,
        "compression_ratio_min": summary.compression_ratio_min,
        "compression_ratio_max": summary.compression_ratio_max,
        
        "degradation_mean": summary.degradation_mean,
        "degradation_p50": summary.degradation_p50,
        "degradation_p95": summary.degradation_p95,
        "degradation_p99": summary.degradation_p99,
        "degradation_max": summary.degradation_max,
        
        "overhead_mean": summary.overhead_mean,
        "overhead_p95": summary.overhead_p95,
        "overhead_max": summary.overhead_max,
        
        "patterns_mean": summary.patterns_mean,
        "patterns_max": summary.patterns_max,
        
        "verification_failure_rate": summary.verification_failure_rate,
        "total_tokens_processed": summary.total_tokens_processed,
        "total_tokens_saved": summary.total_tokens_saved,
        "avg_latency_ms": summary.avg_latency_ms,
    }
    
    if extra:
        data.update(extra)
    
    return json.dumps(data)


def export_histogram_ascii(
    values: list[float],
    title: str = "Histogram",
    bins: int = 10,
    width: int = 50,
) -> str:
    """Export a list of values as an ASCII histogram.
    
    Args:
        values: Values to plot
        title: Histogram title
        bins: Number of bins
        width: Width of the histogram bars
        
    Returns:
        ASCII histogram
    """
    if not values:
        return f"{title}: No data"
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        return f"{title}: All values = {min_val:.4f}"
    
    # Compute bin edges and counts
    bin_width = (max_val - min_val) / bins
    counts = [0] * bins
    
    for v in values:
        idx = min(int((v - min_val) / bin_width), bins - 1)
        counts[idx] += 1
    
    max_count = max(counts)
    
    # Build histogram
    lines: list[str] = []
    lines.append(title)
    lines.append("-" * (width + 20))
    
    for i, count in enumerate(counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_len
        lines.append(f"{bin_start:8.4f} - {bin_end:8.4f} | {bar} ({count})")
    
    lines.append("-" * (width + 20))
    lines.append(f"Total: {len(values)}  Min: {min_val:.4f}  Max: {max_val:.4f}")
    
    return "\n".join(lines)
