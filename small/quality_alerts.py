"""Alert system for compression quality monitoring.

Detects quality degradation and dispatches alerts based on
configurable thresholds and rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .quality_monitor import MonitoringConfig, QualitySummary


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class QualityAlert:
    """A quality alert triggered by threshold violation.
    
    Attributes:
        severity: Alert severity (info, warning, critical)
        metric: Which metric triggered the alert
        current_value: Current value of the metric
        threshold: Threshold that was exceeded
        message: Human-readable alert message
        timestamp: When the alert was generated
        context: Additional context
    """
    severity: AlertSeverity
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    context: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.metric}: {self.message}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class AlertRule:
    """A rule for generating alerts.
    
    Attributes:
        metric: Metric to check
        warning_threshold: Threshold for warning alerts
        critical_threshold: Threshold for critical alerts
        comparison: "gt" (greater than) or "lt" (less than)
        message_template: Template for alert message
    """
    metric: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "gt"  # "gt" or "lt"
    message_template: str = "{metric} is {value:.3f} (threshold: {threshold:.3f})"
    
    def check(self, value: float) -> QualityAlert | None:
        """Check value against thresholds.
        
        Returns:
            QualityAlert if threshold exceeded, None otherwise
        """
        if self.comparison == "gt":
            if value >= self.critical_threshold:
                return self._make_alert(AlertSeverity.CRITICAL, value, self.critical_threshold)
            if value >= self.warning_threshold:
                return self._make_alert(AlertSeverity.WARNING, value, self.warning_threshold)
        else:  # lt
            if value <= self.critical_threshold:
                return self._make_alert(AlertSeverity.CRITICAL, value, self.critical_threshold)
            if value <= self.warning_threshold:
                return self._make_alert(AlertSeverity.WARNING, value, self.warning_threshold)
        return None
    
    def _make_alert(
        self, 
        severity: AlertSeverity, 
        value: float, 
        threshold: float
    ) -> QualityAlert:
        """Create an alert."""
        message = self.message_template.format(
            metric=self.metric,
            value=value,
            threshold=threshold,
        )
        return QualityAlert(
            severity=severity,
            metric=self.metric,
            current_value=value,
            threshold=threshold,
            message=message,
            timestamp=datetime.now(),
        )


# Default alert rules
DEFAULT_RULES = [
    AlertRule(
        metric="compression_ratio_mean",
        warning_threshold=0.85,
        critical_threshold=0.95,
        comparison="gt",
        message_template="Mean compression ratio {value:.2%} exceeds {threshold:.2%}",
    ),
    AlertRule(
        metric="compression_ratio_p95",
        warning_threshold=0.90,
        critical_threshold=0.98,
        comparison="gt",
        message_template="P95 compression ratio {value:.2%} exceeds {threshold:.2%}",
    ),
    AlertRule(
        metric="degradation_mean",
        warning_threshold=0.03,
        critical_threshold=0.08,
        comparison="gt",
        message_template="Mean predicted degradation {value:.3f} exceeds {threshold:.3f}",
    ),
    AlertRule(
        metric="degradation_p95",
        warning_threshold=0.05,
        critical_threshold=0.10,
        comparison="gt",
        message_template="P95 predicted degradation {value:.3f} exceeds {threshold:.3f}",
    ),
    AlertRule(
        metric="overhead_mean",
        warning_threshold=0.40,
        critical_threshold=0.60,
        comparison="gt",
        message_template="Mean dictionary overhead {value:.2%} exceeds {threshold:.2%}",
    ),
    AlertRule(
        metric="verification_failure_rate",
        warning_threshold=0.01,
        critical_threshold=0.05,
        comparison="gt",
        message_template="Verification failure rate {value:.2%} exceeds {threshold:.2%}",
    ),
]


class AlertManager:
    """Manage alert rules and dispatch alerts.
    
    Handles:
    - Checking rules against summaries
    - Alert cooldowns to prevent spam
    - Callback dispatch for alert handling
    
    Example:
        manager = AlertManager()
        manager.subscribe(lambda alert: print(alert))
        
        alerts = manager.check(summary)
        for alert in alerts:
            logging.warning(str(alert))
    """
    
    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        cooldown_seconds: int = 300,
    ):
        self.rules = rules if rules is not None else list(DEFAULT_RULES)
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time: dict[str, float] = {}
        self._subscribers: list[Callable[[QualityAlert], None]] = []
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self.rules.append(rule)
    
    def remove_rule(self, metric: str) -> None:
        """Remove rules for a metric."""
        self.rules = [r for r in self.rules if r.metric != metric]
    
    def subscribe(self, callback: Callable[[QualityAlert], None]) -> None:
        """Subscribe to receive alerts.
        
        Args:
            callback: Function to call with each alert
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[QualityAlert], None]) -> None:
        """Unsubscribe from alerts."""
        self._subscribers = [s for s in self._subscribers if s != callback]
    
    def check(self, summary: "QualitySummary") -> list[QualityAlert]:
        """Check all rules against a summary.
        
        Args:
            summary: Quality summary to check
            
        Returns:
            List of triggered alerts (respecting cooldowns)
        """
        alerts: list[QualityAlert] = []
        now = time.time()
        
        for rule in self.rules:
            # Get value from summary
            value = getattr(summary, rule.metric, None)
            if value is None:
                continue
            
            # Check rule
            alert = rule.check(value)
            if alert is None:
                continue
            
            # Check cooldown
            last_time = self._last_alert_time.get(rule.metric, 0)
            if now - last_time < self.cooldown_seconds:
                continue
            
            # Record and dispatch
            self._last_alert_time[rule.metric] = now
            alerts.append(alert)
            
            # Dispatch to subscribers
            for subscriber in self._subscribers:
                try:
                    subscriber(alert)
                except Exception:
                    pass  # Don't let subscriber errors break alerting
        
        return alerts
    
    def clear_cooldowns(self) -> None:
        """Clear all alert cooldowns."""
        self._last_alert_time.clear()


def check_thresholds(
    summary: "QualitySummary",
    config: "MonitoringConfig",
) -> list[QualityAlert]:
    """Check summary against config thresholds.
    
    Convenience function that creates rules from config and checks.
    
    Args:
        summary: Quality summary to check
        config: Monitoring config with thresholds
        
    Returns:
        List of triggered alerts
    """
    rules = [
        AlertRule(
            metric="compression_ratio_mean",
            warning_threshold=config.compression_ratio_warning,
            critical_threshold=config.compression_ratio_critical,
            comparison="gt",
            message_template="Mean compression ratio {value:.2%} exceeds {threshold:.2%}",
        ),
        AlertRule(
            metric="degradation_mean",
            warning_threshold=config.degradation_warning,
            critical_threshold=config.degradation_critical,
            comparison="gt",
            message_template="Mean predicted degradation {value:.3f} exceeds {threshold:.3f}",
        ),
        AlertRule(
            metric="overhead_mean",
            warning_threshold=config.overhead_warning,
            critical_threshold=config.overhead_critical,
            comparison="gt",
            message_template="Mean dictionary overhead {value:.2%} exceeds {threshold:.2%}",
        ),
    ]
    
    alerts: list[QualityAlert] = []
    for rule in rules:
        value = getattr(summary, rule.metric, None)
        if value is not None:
            alert = rule.check(value)
            if alert is not None:
                alerts.append(alert)
    
    return alerts


def format_alerts_ascii(alerts: list[QualityAlert]) -> str:
    """Format alerts as ASCII text.
    
    Args:
        alerts: List of alerts to format
        
    Returns:
        ASCII formatted string
    """
    if not alerts:
        return "No alerts."
    
    lines = []
    lines.append("=" * 60)
    lines.append("QUALITY ALERTS")
    lines.append("=" * 60)
    
    for alert in alerts:
        severity = alert.severity.value.upper()
        lines.append(f"[{severity}] {alert.metric}")
        lines.append(f"  Value:     {alert.current_value:.4f}")
        lines.append(f"  Threshold: {alert.threshold:.4f}")
        lines.append(f"  Message:   {alert.message}")
        lines.append(f"  Time:      {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-" * 60)
    
    return "\n".join(lines)
