"""
Belief Drift Detector for CHRONOS.

Runs PELT (Pruned Exact Linear Time) change-point detection on the
TemporalPosterior series. For each detected inflection point, identifies
evidence nodes whose temporal_weight changed most sharply in the
surrounding window, creating causal attributions.

Novel algorithm: Causal attribution tracing C-DAG evidence to posterior
inflection points — has no equivalent in any existing multi-agent debate
or Bayesian reasoning framework.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np

from argus.chronos.temporal_posterior import TemporalPosterior, PosteriorSnapshot
from argus.chronos.temporal_cdag import TemporalCDAG, TemporalEvidence

logger = logging.getLogger(__name__)


@dataclass
class CausalAttribution:
    """
    Links an inflection point to its causal evidence.

    Attributes:
        evidence_id: ID of the evidence node
        evidence_text: Text of the evidence (truncated)
        delta_weight: Change in temporal weight around inflection
        impact_score: Normalised impact on posterior shift
        category: Evidence category
        direction: 'strengthened' or 'weakened'
    """
    evidence_id: str
    evidence_text: str
    delta_weight: float
    impact_score: float
    category: str
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_text": self.evidence_text[:100],
            "delta_weight": round(self.delta_weight, 6),
            "impact_score": round(self.impact_score, 4),
            "category": self.category,
            "direction": self.direction,
        }


@dataclass
class InflectionPoint:
    """
    A detected change-point in the posterior series.

    Attributes:
        time: When the inflection occurred
        index: Index in the time series
        posterior_before: Mean posterior before inflection
        posterior_after: Mean posterior after inflection
        magnitude: Absolute change in posterior
        direction: 'up' or 'down'
        causal_attributions: Evidence causing this shift
        confidence: Detection confidence (0-1)
    """
    time: datetime
    index: int
    posterior_before: float
    posterior_after: float
    magnitude: float
    direction: str
    causal_attributions: list[CausalAttribution] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def summary(self) -> str:
        """Human-readable summary of the inflection."""
        causes = ", ".join(
            ca.evidence_text[:40] for ca in self.causal_attributions[:3]
        )
        return (
            f"Belief shifted from {self.posterior_before:.2f} to "
            f"{self.posterior_after:.2f} ({self.direction}) on "
            f"{self.time.strftime('%Y-%m-%d')}"
            + (f" primarily due to: {causes}" if causes else "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time.isoformat(),
            "index": self.index,
            "posterior_before": round(self.posterior_before, 4),
            "posterior_after": round(self.posterior_after, 4),
            "magnitude": round(self.magnitude, 4),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "causal_attributions": [ca.to_dict() for ca in self.causal_attributions],
            "summary": self.summary,
        }


@dataclass
class BeliefDriftReport:
    """
    Complete belief drift analysis report.

    Attributes:
        proposition_id: ID of the proposition analysed
        proposition_text: Text of the proposition
        prior: Prior probability
        current_posterior: Latest posterior
        trend: Overall trend direction
        volatility: Posterior volatility
        inflections: Detected inflection points
        total_drift: Total cumulative drift
        num_evidence: Total evidence nodes
    """
    proposition_id: str
    proposition_text: str
    prior: float
    current_posterior: float
    trend: str
    volatility: float
    inflections: list[InflectionPoint] = field(default_factory=list)
    total_drift: float = 0.0
    num_evidence: int = 0

    @property
    def num_inflections(self) -> int:
        return len(self.inflections)

    @property
    def major_inflections(self) -> list[InflectionPoint]:
        """Inflections with magnitude > 0.1."""
        return [ip for ip in self.inflections if ip.magnitude > 0.1]

    def narrative(self) -> str:
        """Generate a prose narrative of the drift report."""
        lines = [
            f"Belief Drift Report for: \"{self.proposition_text[:80]}\"",
            f"Prior: {self.prior:.2f} → Current: {self.current_posterior:.2f}",
            f"Trend: {self.trend}, Volatility: {self.volatility:.4f}",
            f"Detected {self.num_inflections} inflection point(s).",
        ]
        for i, ip in enumerate(self.inflections, 1):
            lines.append(f"\n  Inflection {i}: {ip.summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition_id": self.proposition_id,
            "proposition_text": self.proposition_text[:200],
            "prior": self.prior,
            "current_posterior": self.current_posterior,
            "trend": self.trend,
            "volatility": round(self.volatility, 6),
            "total_drift": round(self.total_drift, 4),
            "num_evidence": self.num_evidence,
            "inflections": [ip.to_dict() for ip in self.inflections],
        }


class BeliefDriftDetector:
    """
    Detects inflection points in temporal posterior series using
    PELT (Pruned Exact Linear Time) change-point detection.

    For each detected inflection, traces back through the TemporalCDAG
    to identify which evidence nodes caused the shift.

    Example:
        >>> detector = BeliefDriftDetector(min_magnitude=0.05)
        >>> inflections = detector.detect_inflections(temporal_posterior)
        >>> for ip in inflections:
        ...     print(ip.summary)
    """

    def __init__(
        self,
        min_magnitude: float = 0.05,
        window_size: int = 3,
        penalty_factor: float = 3.0,
    ):
        """
        Args:
            min_magnitude: Minimum posterior change to qualify as inflection
            window_size: Window (in time points) around inflection for analysis
            penalty_factor: PELT penalty factor (higher = fewer change-points)
        """
        self.min_magnitude = min_magnitude
        self.window_size = window_size
        self.penalty_factor = penalty_factor

    def detect_inflections(
        self,
        temporal_posterior: TemporalPosterior,
    ) -> list[InflectionPoint]:
        """
        Detect inflection points in the posterior series.

        Uses a PELT-inspired algorithm: scans with a sliding window
        and detects points where the mean shifts significantly.

        Args:
            temporal_posterior: The temporal posterior series

        Returns:
            List of detected InflectionPoints
        """
        values = temporal_posterior.values
        times = temporal_posterior.times
        n = len(values)

        if n < 3:
            return []

        # Compute change-point costs using cumulative sum approach
        arr = np.array(values)
        change_points = self._pelt_detect(arr)

        inflections = []
        for cp_idx in change_points:
            if cp_idx <= 0 or cp_idx >= n:
                continue

            # Compute before/after averages
            start = max(0, cp_idx - self.window_size)
            end = min(n, cp_idx + self.window_size + 1)

            before_vals = arr[start:cp_idx]
            after_vals = arr[cp_idx:end]

            if len(before_vals) == 0 or len(after_vals) == 0:
                continue

            mean_before = float(before_vals.mean())
            mean_after = float(after_vals.mean())
            magnitude = abs(mean_after - mean_before)

            if magnitude < self.min_magnitude:
                continue

            direction = "up" if mean_after > mean_before else "down"

            # Detection confidence based on magnitude and sample size
            confidence = min(1.0, magnitude * 5.0) * min(
                1.0, len(before_vals) * len(after_vals) / 4.0
            )

            inflection = InflectionPoint(
                time=times[cp_idx],
                index=cp_idx,
                posterior_before=mean_before,
                posterior_after=mean_after,
                magnitude=magnitude,
                direction=direction,
                confidence=confidence,
            )
            inflections.append(inflection)

        return inflections

    def _pelt_detect(self, data: np.ndarray) -> list[int]:
        """
        PELT-inspired change-point detection.

        Uses penalised cost optimisation to find optimal segmentation.

        Args:
            data: 1D array of posterior values

        Returns:
            List of change-point indices
        """
        n = len(data)
        if n < 3:
            return []

        # Penalty proportional to log(n)
        penalty = self.penalty_factor * math.log(max(n, 2))

        # Dynamic programming for optimal segmentation
        # cost[i] = minimum cost for data[0:i]
        cost = np.full(n + 1, np.inf)
        cost[0] = 0.0
        last_change = np.zeros(n + 1, dtype=int)

        for i in range(1, n + 1):
            for j in range(0, i):
                segment = data[j:i]
                segment_cost = self._segment_cost(segment)
                total = cost[j] + segment_cost + penalty
                if total < cost[i]:
                    cost[i] = total
                    last_change[i] = j

        # Trace back change-points
        change_points = []
        idx = n
        while idx > 0:
            cp = last_change[idx]
            if cp > 0:
                change_points.append(cp)
            idx = cp

        change_points.sort()
        return change_points

    @staticmethod
    def _segment_cost(segment: np.ndarray) -> float:
        """
        Compute cost of a segment (sum of squared residuals from mean).

        Args:
            segment: 1D array of values

        Returns:
            Cost value (lower = more homogeneous)
        """
        if len(segment) <= 1:
            return 0.0
        mean = segment.mean()
        return float(np.sum((segment - mean) ** 2))

    def trace_cause(
        self,
        inflection: InflectionPoint,
        tcdag: TemporalCDAG,
    ) -> list[CausalAttribution]:
        """
        Trace the cause of an inflection point back to evidence nodes.

        For each evidence node, computes the change in temporal weight
        around the inflection time and ranks by impact.

        Args:
            inflection: The inflection point to trace
            tcdag: The temporal C-DAG containing evidence

        Returns:
            List of CausalAttributions sorted by impact
        """
        attributions = []
        time_before = inflection.time - timedelta(days=15)
        time_after = inflection.time + timedelta(days=15)

        for ev in tcdag.evidence:
            decay_fn = tcdag.get_decay_fn(ev.category)

            # Weight before and after inflection
            w_before = ev.temporal_weight(time_before, decay_fn)
            w_after = ev.temporal_weight(time_after, decay_fn)

            delta = w_after - w_before
            if abs(delta) < 1e-6:
                continue

            direction = "strengthened" if delta > 0 else "weakened"

            attributions.append(CausalAttribution(
                evidence_id=ev.evidence_id,
                evidence_text=ev.text[:100] if ev.text else "",
                delta_weight=delta,
                impact_score=abs(delta),
                category=ev.category.value,
                direction=direction,
            ))

        # Sort by impact (descending)
        attributions.sort(key=lambda a: a.impact_score, reverse=True)

        # Normalise impact scores
        total = sum(a.impact_score for a in attributions)
        if total > 0:
            for a in attributions:
                a.impact_score /= total

        return attributions

    def generate_drift_report(
        self,
        temporal_posterior: TemporalPosterior,
        tcdag: Optional[TemporalCDAG] = None,
    ) -> BeliefDriftReport:
        """
        Generate a complete drift analysis report.

        Args:
            temporal_posterior: The temporal posterior series
            tcdag: Optional temporal C-DAG for causal attribution

        Returns:
            BeliefDriftReport with inflections and attributions
        """
        inflections = self.detect_inflections(temporal_posterior)

        # Add causal attributions if C-DAG available
        if tcdag is not None:
            for ip in inflections:
                ip.causal_attributions = self.trace_cause(ip, tcdag)

        # Compute total drift
        values = temporal_posterior.values
        total_drift = sum(
            abs(values[i] - values[i - 1])
            for i in range(1, len(values))
        ) if len(values) > 1 else 0.0

        latest = temporal_posterior.latest
        report = BeliefDriftReport(
            proposition_id=temporal_posterior.proposition_id,
            proposition_text=temporal_posterior.proposition_text,
            prior=temporal_posterior.prior,
            current_posterior=latest.posterior if latest else 0.5,
            trend=temporal_posterior.trend_direction,
            volatility=temporal_posterior.volatility,
            inflections=inflections,
            total_drift=total_drift,
            num_evidence=tcdag.num_evidence if tcdag else 0,
        )

        logger.info(
            f"Drift report: {report.num_inflections} inflection(s), "
            f"trend={report.trend}, volatility={report.volatility:.4f}"
        )

        return report
