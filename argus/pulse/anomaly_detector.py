"""
Anomaly Detector & Failure Taxonomy for PULSE.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np

from argus.pulse.metrics import MetricStore

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Report of detected anomaly."""
    metric_name: str = ""
    anomaly_type: str = "spike"  # spike, degradation, pattern_change
    severity: str = "warning"   # info, warning, critical
    current_value: float = 0.0
    threshold: float = 0.0
    z_score: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "type": self.anomaly_type,
            "severity": self.severity,
            "value": round(self.current_value, 4),
            "z_score": round(self.z_score, 4),
            "description": self.description,
        }


class FailureTaxonomy:
    """
    Classifies debate failures into categories.

    Categories:
        LLM_TIMEOUT: LLM calls exceeding timeout
        LLM_RATE_LIMIT: Rate limiting errors
        EVIDENCE_EMPTY: No evidence found
        PROPAGATION_DIVERGENCE: C-DAG propagation diverged
        VERDICT_ABSTAIN: Agent abstained from verdict
        UNKNOWN: Unclassified failures
    """

    CATEGORIES = [
        "LLM_TIMEOUT", "LLM_RATE_LIMIT", "EVIDENCE_EMPTY",
        "PROPAGATION_DIVERGENCE", "VERDICT_ABSTAIN", "UNKNOWN",
    ]

    ERROR_PATTERNS: dict[str, list[str]] = {
        "LLM_TIMEOUT": ["timeout", "timed out", "deadline exceeded"],
        "LLM_RATE_LIMIT": ["rate limit", "429", "too many requests"],
        "EVIDENCE_EMPTY": ["no evidence", "empty evidence", "no chunks"],
        "PROPAGATION_DIVERGENCE": ["diverge", "nan", "overflow", "inf"],
        "VERDICT_ABSTAIN": ["abstain", "undetermined", "insufficient"],
    }

    def __init__(self):
        self._counts: dict[str, int] = {cat: 0 for cat in self.CATEGORIES}
        self._examples: dict[str, list[str]] = {cat: [] for cat in self.CATEGORIES}

    def classify(self, error_message: str) -> str:
        """Classify an error into a failure category."""
        msg_lower = error_message.lower()
        for category, patterns in self.ERROR_PATTERNS.items():
            if any(p in msg_lower for p in patterns):
                self._counts[category] += 1
                if len(self._examples[category]) < 5:
                    self._examples[category].append(error_message[:200])
                return category

        self._counts["UNKNOWN"] += 1
        if len(self._examples["UNKNOWN"]) < 5:
            self._examples["UNKNOWN"].append(error_message[:200])
        return "UNKNOWN"

    @property
    def total_failures(self) -> int:
        return sum(self._counts.values())

    def distribution(self) -> dict[str, float]:
        """Get failure distribution as percentages."""
        total = max(self.total_failures, 1)
        return {cat: count / total for cat, count in self._counts.items() if count > 0}

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_failures": self.total_failures,
            "counts": {k: v for k, v in self._counts.items() if v > 0},
            "distribution": self.distribution(),
        }


class AnomalyDetector:
    """
    Detects anomalies in operational metrics using z-score analysis.

    Monitors latency spikes, degradation trends, and unusual patterns.

    Example:
        >>> detector = AnomalyDetector(z_threshold=2.5)
        >>> anomalies = detector.check(store)
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        min_samples: int = 10,
    ):
        self.z_threshold = z_threshold
        self.min_samples = min_samples

    def check(self, store: MetricStore) -> list[AnomalyReport]:
        """Check all histograms for anomalies."""
        anomalies: list[AnomalyReport] = []
        snapshot = store.snapshot()

        for name, hist_data in snapshot.get("histograms", {}).items():
            hist = store.histogram(name)
            if hist.count < self.min_samples:
                continue

            values = np.array(hist.values)
            mean = values.mean()
            std = values.std()

            if std < 1e-10:
                continue

            # Check latest value
            latest = values[-1]
            z_score = (latest - mean) / std

            if abs(z_score) > self.z_threshold:
                severity = "critical" if abs(z_score) > self.z_threshold * 1.5 else "warning"
                anomaly_type = "spike" if z_score > 0 else "drop"

                anomalies.append(AnomalyReport(
                    metric_name=name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    current_value=latest,
                    threshold=mean + self.z_threshold * std,
                    z_score=z_score,
                    description=(
                        f"{name}: {anomaly_type} detected "
                        f"(z={z_score:.2f}, value={latest:.2f}, "
                        f"mean={mean:.2f})"
                    ),
                ))

            # Trend detection (last 20% vs previous)
            if len(values) >= 20:
                split = int(len(values) * 0.8)
                old_mean = values[:split].mean()
                new_mean = values[split:].mean()
                if old_mean > 0 and (new_mean / old_mean) > 1.5:
                    anomalies.append(AnomalyReport(
                        metric_name=name,
                        anomaly_type="degradation",
                        severity="warning",
                        current_value=new_mean,
                        threshold=old_mean * 1.5,
                        z_score=(new_mean - old_mean) / max(std, 1e-10),
                        description=f"{name}: degradation trend ({old_mean:.2f} → {new_mean:.2f})",
                    ))

        return anomalies
