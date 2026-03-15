"""
PULSE Dashboard — report generation and coordination.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.pulse.metrics import MetricStore
from argus.pulse.probes import DebateProbe
from argus.pulse.anomaly_detector import AnomalyDetector, FailureTaxonomy, AnomalyReport

logger = logging.getLogger(__name__)


@dataclass
class PULSEConfig:
    """Configuration for PULSE dashboard."""
    export_format: str = "html"  # html, json, stdout
    output_dir: str = "./pulse_reports"
    anomaly_z_threshold: float = 2.5
    enable_token_tracking: bool = True
    enable_accuracy_tracking: bool = True


@dataclass
class DashboardReport:
    """Full operational intelligence report."""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    stage_summary: dict[str, Any] = field(default_factory=dict)
    anomalies: list[AnomalyReport] = field(default_factory=list)
    failure_taxonomy: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "metrics": self.metrics_snapshot,
            "stages": self.stage_summary,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "failures": self.failure_taxonomy,
            "recommendations": self.recommendations,
        }


class PULSEDashboard:
    """
    Operational Intelligence Dashboard.

    Coordinates metric collection, anomaly detection, and reporting.

    Example:
        >>> pulse = PULSEDashboard()
        >>> pulse.start()
        >>> # ... run debates with pulse.probe ...
        >>> report = pulse.generate_report()
        >>> pulse.export_report(report)
    """

    def __init__(self, config: Optional[PULSEConfig] = None):
        self.config = config or PULSEConfig()
        self.store = MetricStore()
        self.probe = DebateProbe(store=self.store)
        self.anomaly_detector = AnomalyDetector(z_threshold=self.config.anomaly_z_threshold)
        self.failure_taxonomy = FailureTaxonomy()
        self._started = False

    def start(self) -> None:
        """Start monitoring."""
        self._started = True
        self.store.gauge("dashboard_active").set(1)
        logger.info("PULSE dashboard started")

    def stop(self) -> None:
        """Stop monitoring."""
        self._started = False
        self.store.gauge("dashboard_active").set(0)

    def record_failure(self, error_message: str) -> str:
        """Record and classify a failure."""
        category = self.failure_taxonomy.classify(error_message)
        self.store.counter("failures_total").inc()
        self.store.counter(f"failure_{category}").inc()
        return category

    def generate_report(self) -> DashboardReport:
        """Generate a full operational intelligence report."""
        # Metrics snapshot
        metrics = self.store.snapshot()

        # Stage summary
        stages = self.probe.stage_summary()

        # Anomaly detection
        anomalies = self.anomaly_detector.check(self.store)

        # Failure taxonomy
        failures = self.failure_taxonomy.to_dict()

        # Recommendations
        recommendations = self._generate_recommendations(anomalies, failures)

        report = DashboardReport(
            metrics_snapshot=metrics,
            stage_summary=stages,
            anomalies=anomalies,
            failure_taxonomy=failures,
            recommendations=recommendations,
        )

        logger.info(
            f"PULSE report: {len(anomalies)} anomalies, "
            f"{self.failure_taxonomy.total_failures} total failures"
        )
        return report

    def export_report(
        self,
        report: Optional[DashboardReport] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Export report to file."""
        report = report or self.generate_report()
        from pathlib import Path

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.export_format == "json":
            path = output_path or str(output_dir / "pulse_report.json")
            with open(path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        else:
            path = output_path or str(output_dir / "pulse_report.html")
            from argus.pulse.visualization import generate_dashboard_html
            html = generate_dashboard_html(report)
            with open(path, "w") as f:
                f.write(html)

        logger.info(f"PULSE report exported to {path}")
        return path

    def _generate_recommendations(
        self,
        anomalies: list[AnomalyReport],
        failures: dict[str, Any],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        # Latency recommendations
        for anomaly in anomalies:
            if "latency" in anomaly.metric_name and anomaly.severity == "critical":
                recs.append(
                    f"⚠️ Critical latency spike in {anomaly.metric_name}: "
                    f"consider caching or reducing LLM calls."
                )

        # Failure recommendations
        failure_counts = failures.get("counts", {})
        if failure_counts.get("LLM_TIMEOUT", 0) > 3:
            recs.append("🔄 Multiple LLM timeouts detected. Consider increasing timeout or using a faster model.")
        if failure_counts.get("LLM_RATE_LIMIT", 0) > 2:
            recs.append("🚫 Rate limiting detected. Implement request queuing or reduce parallel calls.")
        if failure_counts.get("EVIDENCE_EMPTY", 0) > 5:
            recs.append("📄 Frequent empty evidence sets. Review document ingestion pipeline.")

        # Token usage recommendations
        total_input = self.store.counter("total_input_tokens").value
        total_output = self.store.counter("total_output_tokens").value
        if total_input > 100000:
            recs.append(f"💰 High token usage ({total_input:.0f} input). Consider chunking optimization.")

        if not recs:
            recs.append("✅ All systems operating within normal parameters.")

        return recs
