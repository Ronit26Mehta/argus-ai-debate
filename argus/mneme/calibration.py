"""
Calibration History — rolling Brier Score monitoring per domain.

Detects degrading calibration and generates drift reports.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Report on calibration drift."""
    domain: str = ""
    direction: str = "stable"  # improving, degrading, stable
    current_brier: float = 0.0
    baseline_brier: float = 0.0
    magnitude: float = 0.0
    recommendation: str = ""
    window_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "direction": self.direction,
            "current_brier": round(self.current_brier, 4),
            "baseline_brier": round(self.baseline_brier, 4),
            "magnitude": round(self.magnitude, 4),
            "recommendation": self.recommendation,
        }


class CalibrationHistory:
    """
    Rolling Brier Score tracker for a single domain.

    Brier Score = (1/N) Σ (prediction - outcome)²
    Lower = better calibrated

    Example:
        >>> cal = CalibrationHistory(domain="clinical")
        >>> cal.record(predicted=0.8, actual=True)
        >>> cal.record(predicted=0.7, actual=False)
        >>> print(cal.brier_score)  # 0.245
    """

    def __init__(
        self,
        domain: str = "general",
        window_size: int = 100,
    ):
        self.domain = domain
        self.window_size = window_size
        self._predictions: deque[float] = deque(maxlen=window_size)
        self._outcomes: deque[float] = deque(maxlen=window_size)

    def record(self, predicted: float, actual: bool) -> None:
        """Record a prediction and outcome."""
        self._predictions.append(predicted)
        self._outcomes.append(1.0 if actual else 0.0)

    @property
    def brier_score(self) -> float:
        """Compute current Brier Score."""
        if not self._predictions:
            return 0.25  # Default (random)
        preds = np.array(self._predictions)
        outcomes = np.array(self._outcomes)
        return float(np.mean((preds - outcomes) ** 2))

    @property
    def num_observations(self) -> int:
        return len(self._predictions)

    @property
    def is_well_calibrated(self) -> bool:
        """Brier Score < 0.15 is considered well-calibrated."""
        return self.brier_score < 0.15

    def rolling_brier(self, window: int = 20) -> list[float]:
        """Compute rolling Brier Score."""
        if len(self._predictions) < window:
            return [self.brier_score]
        preds = list(self._predictions)
        outcomes = list(self._outcomes)
        scores = []
        for i in range(window, len(preds) + 1):
            p = np.array(preds[i - window:i])
            o = np.array(outcomes[i - window:i])
            scores.append(float(np.mean((p - o) ** 2)))
        return scores


class CalibrationDriftMonitor:
    """
    Monitors calibration drift across domains.

    Compares recent Brier Score against baseline to detect
    degradation in agent calibration quality.

    Example:
        >>> monitor = CalibrationDriftMonitor()
        >>> drift = monitor.check_drift("clinical", cal_history)
        >>> if drift.direction == "degrading":
        ...     print(f"Recalibrate! {drift.recommendation}")
    """

    def __init__(
        self,
        threshold: float = 0.05,
        baseline_window: int = 50,
    ):
        self.threshold = threshold
        self.baseline_window = baseline_window
        self._baselines: dict[str, float] = {}

    def set_baseline(self, domain: str, brier_score: float) -> None:
        self._baselines[domain] = brier_score

    def check_drift(
        self,
        domain: str,
        history: CalibrationHistory,
    ) -> DriftReport:
        """Check for calibration drift."""
        current = history.brier_score
        baseline = self._baselines.get(domain, 0.25)

        delta = current - baseline

        if abs(delta) < self.threshold:
            direction = "stable"
            recommendation = "No action needed."
        elif delta > 0:
            direction = "degrading"
            recommendation = (
                f"Brier Score increased by {delta:.3f}. "
                f"Recommend recalibration for domain '{domain}'."
            )
        else:
            direction = "improving"
            recommendation = f"Calibration improved by {abs(delta):.3f}."

        return DriftReport(
            domain=domain,
            direction=direction,
            current_brier=current,
            baseline_brier=baseline,
            magnitude=abs(delta),
            recommendation=recommendation,
            window_size=history.num_observations,
        )

    def monitor_all(
        self,
        histories: dict[str, CalibrationHistory],
    ) -> list[DriftReport]:
        """Monitor drift across all domains."""
        reports = []
        for domain, history in histories.items():
            report = self.check_drift(domain, history)
            reports.append(report)
            if report.direction == "degrading":
                logger.warning(f"Calibration drift: {report.recommendation}")
        return reports
