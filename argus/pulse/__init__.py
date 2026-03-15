"""
PULSE — Operational Intelligence Dashboard for ARGUS.

Always-on monitoring pipeline: latency histograms, LLM token metering,
accuracy trends, failure-mode taxonomies. Embeds lightweight probes
inside each debate stage.

Example:
    >>> from argus.pulse import PULSEDashboard, PULSEConfig
    >>> pulse = PULSEDashboard(config=PULSEConfig(export_format='html'))
    >>> pulse.start()
    >>> # ... run debates ...
    >>> pulse.generate_report()
"""

from argus.pulse.metrics import MetricStore, Metric, Counter, Histogram, Gauge
from argus.pulse.probes import DebateProbe, ProbeResult, LatencyTracker
from argus.pulse.anomaly_detector import AnomalyDetector, FailureTaxonomy, AnomalyReport
from argus.pulse.dashboard import PULSEDashboard, PULSEConfig, DashboardReport
from argus.pulse.orchestrator import PULSEOrchestrator, PULSEResult
from argus.pulse.visualization import (
    plot_latency_histogram,
    plot_token_usage,
    plot_accuracy_trend,
    plot_failure_taxonomy,
)

__all__ = [
    "MetricStore", "Metric", "Counter", "Histogram", "Gauge",
    "DebateProbe", "ProbeResult", "LatencyTracker",
    "AnomalyDetector", "FailureTaxonomy", "AnomalyReport",
    "PULSEDashboard", "PULSEConfig", "DashboardReport",
    "PULSEOrchestrator", "PULSEResult",
    "plot_latency_histogram", "plot_token_usage",
    "plot_accuracy_trend", "plot_failure_taxonomy",
]
