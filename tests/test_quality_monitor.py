"""Tests for quality monitoring system."""

import time
from datetime import datetime, timedelta

import pytest

from small.config import CompressionConfig
from small.compressor import compress
from small.quality_monitor import (
    MonitoringConfig,
    QualityBaseline,
    QualityMonitor,
    QualityRecord,
    QualitySummary,
    HealthStatus,
    get_global_monitor,
    set_global_monitor,
)
from small.quality_alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    QualityAlert,
    check_thresholds,
    format_alerts_ascii,
)
from small.quality_export import (
    export_prometheus,
    export_summary_ascii,
    export_health_ascii,
    export_jsonl,
    export_histogram_ascii,
)


# ============================================================================
# MonitoringConfig Tests
# ============================================================================


class TestMonitoringConfig:
    """Tests for MonitoringConfig."""

    def test_default_config(self):
        config = MonitoringConfig()
        assert config.enabled is True
        assert config.window_size == 1000
        assert config.compression_ratio_warning == 0.85
        assert config.compression_ratio_critical == 0.95

    def test_custom_config(self):
        config = MonitoringConfig(
            window_size=500,
            compression_ratio_warning=0.80,
        )
        assert config.window_size == 500
        assert config.compression_ratio_warning == 0.80


# ============================================================================
# QualityRecord Tests
# ============================================================================


class TestQualityRecord:
    """Tests for QualityRecord."""

    def test_record_creation(self):
        record = QualityRecord(
            timestamp=time.time(),
            compression_ratio=0.75,
            predicted_degradation=0.02,
            dictionary_overhead=0.25,
            patterns_used=5,
            original_length=100,
            compressed_length=75,
        )
        assert record.compression_ratio == 0.75
        assert record.patterns_used == 5

    def test_record_from_result(self):
        tokens = list("the quick brown fox " * 10)
        config = CompressionConfig(hierarchical_enabled=False)
        result = compress(tokens, config)
        
        record = QualityRecord.from_result(
            result,
            predicted_degradation=0.01,
            latency_ms=5.0,
            context={"test": True},
        )
        
        assert record.original_length == result.original_length
        assert record.compressed_length == result.compressed_length
        assert record.predicted_degradation == 0.01
        assert record.latency_ms == 5.0
        assert record.context == {"test": True}


# ============================================================================
# QualityMonitor Tests
# ============================================================================


class TestQualityMonitor:
    """Tests for QualityMonitor."""

    def test_monitor_creation(self):
        monitor = QualityMonitor()
        assert monitor.config.enabled is True
        assert monitor.record_count == 0

    def test_record_compression_result(self):
        monitor = QualityMonitor()
        tokens = list("the quick brown fox " * 10)
        config = CompressionConfig(hierarchical_enabled=False)
        result = compress(tokens, config)
        
        record = monitor.record(result)
        
        assert monitor.record_count == 1
        assert record.original_length == len(tokens)

    def test_multiple_records(self):
        monitor = QualityMonitor()
        
        for i in range(10):
            tokens = list(f"pattern {i} " * (5 + i))
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result)
        
        assert monitor.record_count == 10

    def test_window_size_limit(self):
        config = MonitoringConfig(window_size=5)
        monitor = QualityMonitor(config)
        
        for i in range(10):
            tokens = list(f"test {i} " * 5)
            comp_config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, comp_config)
            monitor.record(result)
        
        # Should only keep last 5
        assert monitor.record_count == 5

    def test_get_records_all(self):
        monitor = QualityMonitor()
        
        for i in range(5):
            tokens = list(f"pattern {i} " * 5)
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result)
        
        records = monitor.get_records()
        assert len(records) == 5

    def test_get_records_with_max(self):
        monitor = QualityMonitor()
        
        for i in range(10):
            tokens = list(f"pattern {i} " * 5)
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result)
        
        records = monitor.get_records(max_records=3)
        assert len(records) == 3

    def test_get_summary(self):
        monitor = QualityMonitor()
        
        for i in range(10):
            tokens = list(f"the quick brown fox " * (5 + i))
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result, predicted_degradation=0.01 * i)
        
        summary = monitor.get_summary()
        
        assert summary is not None
        assert summary.record_count == 10
        assert 0 <= summary.compression_ratio_mean <= 1
        assert summary.compression_ratio_p50 >= 0
        assert summary.compression_ratio_p95 >= 0
        assert summary.total_tokens_processed > 0

    def test_get_summary_empty(self):
        monitor = QualityMonitor()
        summary = monitor.get_summary()
        assert summary is None

    def test_check_health(self):
        monitor = QualityMonitor()
        
        # Add some good records
        for i in range(5):
            tokens = list("the quick brown fox " * 10)
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result, predicted_degradation=0.01)
        
        health = monitor.check_health()
        
        assert isinstance(health, HealthStatus)
        assert health.summary is not None
        assert health.checked_at is not None

    def test_learn_baseline(self):
        monitor = QualityMonitor(MonitoringConfig(baseline_learning_window=20))
        
        # Add enough records
        for i in range(20):
            tokens = list("the quick brown fox " * 10)
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result)
        
        baseline = monitor.learn_baseline("compression_ratio")
        
        assert baseline is not None
        assert baseline.metric == "compression_ratio"
        assert baseline.mean >= 0
        assert baseline.sample_count == 20

    def test_learn_baseline_insufficient_data(self):
        monitor = QualityMonitor()
        
        # Only 5 records
        for i in range(5):
            tokens = list(f"test {i} " * 5)
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result)
        
        baseline = monitor.learn_baseline("compression_ratio")
        assert baseline is None  # Needs at least 10 records

    def test_on_record_callback(self):
        monitor = QualityMonitor()
        records_received = []
        
        def callback(record):
            records_received.append(record)
        
        monitor.on_record(callback)
        
        tokens = list("test pattern " * 5)
        config = CompressionConfig(hierarchical_enabled=False)
        result = compress(tokens, config)
        monitor.record(result)
        
        assert len(records_received) == 1

    def test_clear(self):
        monitor = QualityMonitor()
        
        for i in range(5):
            tokens = list(f"test {i} " * 5)
            config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, config)
            monitor.record(result)
        
        assert monitor.record_count == 5
        monitor.clear()
        assert monitor.record_count == 0

    def test_disabled_monitoring(self):
        config = MonitoringConfig(enabled=False)
        monitor = QualityMonitor(config)
        
        tokens = list("test pattern " * 5)
        comp_config = CompressionConfig(hierarchical_enabled=False)
        result = compress(tokens, comp_config)
        monitor.record(result)
        
        # Should not record when disabled
        assert monitor.record_count == 0


# ============================================================================
# QualityBaseline Tests
# ============================================================================


class TestQualityBaseline:
    """Tests for QualityBaseline."""

    def test_z_score(self):
        baseline = QualityBaseline(
            metric="compression_ratio",
            mean=0.7,
            stddev=0.1,
            p95=0.85,
            sample_count=100,
            learned_at=datetime.now(),
        )
        
        # Value at mean should have z-score 0
        assert baseline.z_score(0.7) == 0.0
        
        # Value 1 std dev above mean
        assert abs(baseline.z_score(0.8) - 1.0) < 0.01
        
        # Value 2 std devs below mean
        assert abs(baseline.z_score(0.5) - (-2.0)) < 0.01

    def test_is_anomaly(self):
        baseline = QualityBaseline(
            metric="compression_ratio",
            mean=0.7,
            stddev=0.1,
            p95=0.85,
            sample_count=100,
            learned_at=datetime.now(),
        )
        
        # Normal values (within 2 std devs)
        assert not baseline.is_anomaly(0.7)
        assert not baseline.is_anomaly(0.8)
        
        # Anomalous values (beyond 2 std devs)
        assert baseline.is_anomaly(0.95, threshold=2.0)
        assert baseline.is_anomaly(0.45, threshold=2.0)


# ============================================================================
# AlertRule Tests
# ============================================================================


class TestAlertRule:
    """Tests for AlertRule."""

    def test_rule_no_alert(self):
        rule = AlertRule(
            metric="compression_ratio",
            warning_threshold=0.85,
            critical_threshold=0.95,
            comparison="gt",
        )
        
        alert = rule.check(0.7)  # Below warning
        assert alert is None

    def test_rule_warning_alert(self):
        rule = AlertRule(
            metric="compression_ratio",
            warning_threshold=0.85,
            critical_threshold=0.95,
            comparison="gt",
        )
        
        alert = rule.check(0.88)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_rule_critical_alert(self):
        rule = AlertRule(
            metric="compression_ratio",
            warning_threshold=0.85,
            critical_threshold=0.95,
            comparison="gt",
        )
        
        alert = rule.check(0.97)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_rule_less_than_comparison(self):
        rule = AlertRule(
            metric="compression_savings",
            warning_threshold=0.2,
            critical_threshold=0.1,
            comparison="lt",
        )
        
        # No alert (good savings)
        alert = rule.check(0.3)
        assert alert is None
        
        # Warning (low savings)
        alert = rule.check(0.15)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        
        # Critical (very low savings)
        alert = rule.check(0.05)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL


# ============================================================================
# AlertManager Tests
# ============================================================================


class TestAlertManager:
    """Tests for AlertManager."""

    def test_manager_creation(self):
        manager = AlertManager()
        assert len(manager.rules) > 0  # Default rules

    def test_custom_rules(self):
        rules = [
            AlertRule(
                metric="compression_ratio_mean",
                warning_threshold=0.80,
                critical_threshold=0.90,
            )
        ]
        manager = AlertManager(rules=rules)
        assert len(manager.rules) == 1

    def test_subscribe_callback(self):
        manager = AlertManager(cooldown_seconds=0)
        alerts_received = []
        
        def callback(alert):
            alerts_received.append(alert)
        
        manager.subscribe(callback)
        
        # Create a summary that triggers alerts
        monitor = QualityMonitor()
        # Add records with bad compression
        for i in range(5):
            record = QualityRecord(
                timestamp=time.time(),
                compression_ratio=0.95,  # Bad ratio
                predicted_degradation=0.1,
                dictionary_overhead=0.5,
                patterns_used=3,
                original_length=100,
                compressed_length=95,
            )
            monitor.record_from_record(record)
        
        summary = monitor.get_summary()
        manager.check(summary)
        
        assert len(alerts_received) > 0

    def test_cooldown(self):
        manager = AlertManager(cooldown_seconds=60)
        
        # Create bad summary
        monitor = QualityMonitor()
        for i in range(5):
            record = QualityRecord(
                timestamp=time.time(),
                compression_ratio=0.95,
                predicted_degradation=0.1,
                dictionary_overhead=0.5,
                patterns_used=3,
                original_length=100,
                compressed_length=95,
            )
            monitor.record_from_record(record)
        
        summary = monitor.get_summary()
        
        # First check - should get alerts
        alerts1 = manager.check(summary)
        
        # Second check - should be in cooldown
        alerts2 = manager.check(summary)
        
        assert len(alerts1) > 0
        assert len(alerts2) == 0  # Cooldown

    def test_clear_cooldowns(self):
        manager = AlertManager(cooldown_seconds=60)
        
        monitor = QualityMonitor()
        for i in range(5):
            record = QualityRecord(
                timestamp=time.time(),
                compression_ratio=0.95,
                predicted_degradation=0.1,
                dictionary_overhead=0.5,
                patterns_used=3,
                original_length=100,
                compressed_length=95,
            )
            monitor.record_from_record(record)
        
        summary = monitor.get_summary()
        
        alerts1 = manager.check(summary)
        manager.clear_cooldowns()
        alerts2 = manager.check(summary)
        
        # Both should have alerts after clearing cooldowns
        assert len(alerts1) > 0
        assert len(alerts2) > 0


# ============================================================================
# Export Tests
# ============================================================================


class TestExports:
    """Tests for export functions."""

    def _make_summary(self) -> QualitySummary:
        """Create a test summary."""
        monitor = QualityMonitor()
        for i in range(10):
            record = QualityRecord(
                timestamp=time.time(),
                compression_ratio=0.7 + i * 0.02,
                predicted_degradation=0.01 + i * 0.005,
                dictionary_overhead=0.25 + i * 0.01,
                patterns_used=5 + i,
                original_length=100 + i * 10,
                compressed_length=70 + i * 8,
                latency_ms=5.0 + i,
            )
            monitor.record_from_record(record)
        return monitor.get_summary()

    def test_export_prometheus(self):
        summary = self._make_summary()
        output = export_prometheus(summary)
        
        # Should be valid Prometheus format
        assert "# HELP" in output
        assert "# TYPE" in output
        assert "small_compression_compression_ratio_mean" in output
        assert "small_compression_degradation_mean" in output
        assert "gauge" in output

    def test_export_prometheus_with_labels(self):
        summary = self._make_summary()
        output = export_prometheus(summary, labels={"env": "test", "service": "small"})
        
        assert 'env="test"' in output
        assert 'service="small"' in output

    def test_export_summary_ascii(self):
        summary = self._make_summary()
        output = export_summary_ascii(summary)
        
        # Should be clean ASCII
        assert "COMPRESSION QUALITY SUMMARY" in output
        assert "COMPRESSION RATIO" in output
        assert "PREDICTED DEGRADATION" in output
        assert "=" in output
        assert "-" in output
        # No unicode/emoji
        assert all(ord(c) < 128 for c in output)

    def test_export_health_ascii(self):
        monitor = QualityMonitor()
        for i in range(5):
            record = QualityRecord(
                timestamp=time.time(),
                compression_ratio=0.7,
                predicted_degradation=0.01,
                dictionary_overhead=0.25,
                patterns_used=5,
                original_length=100,
                compressed_length=70,
            )
            monitor.record_from_record(record)
        
        output = export_health_ascii(monitor)
        
        assert "HEALTH REPORT" in output
        assert "Status:" in output
        # No unicode/emoji
        assert all(ord(c) < 128 for c in output)

    def test_export_jsonl(self):
        import json
        
        summary = self._make_summary()
        output = export_jsonl(summary)
        
        # Should be valid JSON
        data = json.loads(output)
        assert "compression_ratio_mean" in data
        assert "timestamp" in data

    def test_export_histogram_ascii(self):
        values = [0.5, 0.6, 0.65, 0.7, 0.7, 0.7, 0.75, 0.8, 0.85, 0.9]
        output = export_histogram_ascii(values, title="Compression Ratios")
        
        assert "Compression Ratios" in output
        assert "#" in output  # Histogram bars
        assert "Total:" in output
        # No unicode/emoji
        assert all(ord(c) < 128 for c in output)

    def test_format_alerts_ascii(self):
        alerts = [
            QualityAlert(
                severity=AlertSeverity.WARNING,
                metric="compression_ratio",
                current_value=0.88,
                threshold=0.85,
                message="Test warning",
                timestamp=datetime.now(),
            ),
            QualityAlert(
                severity=AlertSeverity.CRITICAL,
                metric="degradation",
                current_value=0.15,
                threshold=0.08,
                message="Test critical",
                timestamp=datetime.now(),
            ),
        ]
        
        output = format_alerts_ascii(alerts)
        
        assert "QUALITY ALERTS" in output
        assert "WARNING" in output
        assert "CRITICAL" in output
        # No unicode/emoji
        assert all(ord(c) < 128 for c in output)


# ============================================================================
# Global Monitor Tests
# ============================================================================


class TestGlobalMonitor:
    """Tests for global monitor functions."""

    def test_get_global_monitor(self):
        monitor = get_global_monitor()
        assert isinstance(monitor, QualityMonitor)

    def test_set_global_monitor(self):
        custom_config = MonitoringConfig(window_size=500)
        custom_monitor = QualityMonitor(custom_config)
        
        set_global_monitor(custom_monitor)
        
        retrieved = get_global_monitor()
        assert retrieved.config.window_size == 500


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full monitoring pipeline."""

    def test_full_monitoring_pipeline(self):
        """Test complete monitoring flow."""
        # Setup
        config = MonitoringConfig(
            window_size=100,
            compression_ratio_warning=0.90,
            compression_ratio_critical=0.98,
        )
        monitor = QualityMonitor(config)
        manager = AlertManager(cooldown_seconds=0)
        
        alerts_received = []
        manager.subscribe(lambda a: alerts_received.append(a))
        
        # Simulate compression operations
        for i in range(20):
            tokens = list("the quick brown fox jumps over the lazy dog " * (3 + i))
            comp_config = CompressionConfig(hierarchical_enabled=False)
            result = compress(tokens, comp_config)
            monitor.record(result, predicted_degradation=0.01 * i)
        
        # Get summary
        summary = monitor.get_summary()
        assert summary is not None
        assert summary.record_count == 20
        
        # Check health
        health = monitor.check_health()
        assert isinstance(health, HealthStatus)
        
        # Check alerts
        manager.check(summary)
        
        # Export
        prometheus_output = export_prometheus(summary)
        assert "small_compression" in prometheus_output
        
        ascii_output = export_summary_ascii(summary)
        assert "COMPRESSION QUALITY SUMMARY" in ascii_output
        
        # Learn baseline
        baseline = monitor.learn_baseline("compression_ratio")
        assert baseline is not None

    def test_thread_safety(self):
        """Test thread-safe recording."""
        import threading
        
        monitor = QualityMonitor(MonitoringConfig(window_size=1000))
        errors = []
        
        def record_worker(worker_id):
            try:
                for i in range(10):
                    record = QualityRecord(
                        timestamp=time.time(),
                        compression_ratio=0.7,
                        predicted_degradation=0.01,
                        dictionary_overhead=0.25,
                        patterns_used=5,
                        original_length=100,
                        compressed_length=70,
                        context={"worker": worker_id, "iteration": i},
                    )
                    monitor.record_from_record(record)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=record_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert monitor.record_count == 50  # 5 workers * 10 records
