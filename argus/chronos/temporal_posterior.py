"""
Temporal Posterior — time-indexed series of Bayesian posterior estimates.

Provides TemporalPosterior (a time-series of posterior values with credible
intervals), CredibleBand (upper/lower bounds at a percentile), and
PosteriorSnapshot (a single point with confidence bands).
"""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PosteriorSnapshot:
    """
    A single posterior value at a specific time point.

    Attributes:
        time: Timestamp
        posterior: Posterior probability
        lower_95: Lower 95% credible bound
        upper_95: Upper 95% credible bound
        lower_50: Lower 50% credible bound
        upper_50: Upper 50% credible bound
        num_evidence: Number of evidence nodes active at this time
    """
    time: datetime
    posterior: float
    lower_95: float = 0.0
    upper_95: float = 1.0
    lower_50: float = 0.25
    upper_50: float = 0.75
    num_evidence: int = 0

    def contains(self, value: float, level: float = 0.95) -> bool:
        """Check if value falls within credible interval."""
        if level >= 0.95:
            return self.lower_95 <= value <= self.upper_95
        return self.lower_50 <= value <= self.upper_50


@dataclass
class CredibleBand:
    """
    Credible interval band across the full time series.

    Attributes:
        level: Confidence level (e.g., 0.95, 0.50)
        times: Timestamp for each point
        lower: Lower bound at each time
        upper: Upper bound at each time
    """
    level: float
    times: list[datetime] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)

    @property
    def width(self) -> list[float]:
        """Width of the band at each time point."""
        return [u - l for u, l in zip(self.upper, self.lower)]

    @property
    def mean_width(self) -> float:
        """Average band width across time."""
        w = self.width
        return sum(w) / max(len(w), 1)


class TemporalPosterior:
    """
    Time-indexed series of posterior estimates with credible intervals.

    Contains the full temporal evolution of a posterior probability,
    enabling drift detection, trend analysis, and temporal confidence
    assessment.

    Example:
        >>> tp = TemporalPosterior(proposition_id="prop_001")
        >>> tp.add_point(datetime(2023, 1, 1), 0.65, n_evidence=5)
        >>> tp.add_point(datetime(2023, 7, 1), 0.58, n_evidence=7)
        >>> print(tp.at("2023-01"))     # 0.65
        >>> print(tp.trend_direction)   # 'declining'
    """

    def __init__(
        self,
        proposition_id: str = "",
        proposition_text: str = "",
        prior: float = 0.5,
    ):
        self.proposition_id = proposition_id
        self.proposition_text = proposition_text
        self.prior = prior
        self.snapshots: list[PosteriorSnapshot] = []

    def add_point(
        self,
        time: datetime,
        posterior: float,
        num_evidence: int = 0,
        uncertainty: float = 0.1,
    ) -> None:
        """
        Add a posterior data point with auto-computed credible intervals.

        Uses Beta distribution approximation for credible intervals.
        With more evidence, intervals tighten.

        Args:
            time: Timestamp
            posterior: Posterior probability
            num_evidence: Number of active evidence nodes
            uncertainty: Base uncertainty scale
        """
        # Scale uncertainty by evidence count (more evidence = tighter)
        scale = uncertainty / max(math.sqrt(num_evidence + 1), 1.0)

        # 95% credible interval
        lower_95 = max(0.0, posterior - 1.96 * scale)
        upper_95 = min(1.0, posterior + 1.96 * scale)

        # 50% credible interval
        lower_50 = max(0.0, posterior - 0.674 * scale)
        upper_50 = min(1.0, posterior + 0.674 * scale)

        snapshot = PosteriorSnapshot(
            time=time,
            posterior=posterior,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_50=lower_50,
            upper_50=upper_50,
            num_evidence=num_evidence,
        )
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.time)

    def at(self, time_str: str) -> Optional[float]:
        """
        Get posterior at a specific time (nearest match).

        Args:
            time_str: Time string (e.g., '2023-01', '2023-06-15')

        Returns:
            Posterior value or None if no data
        """
        if not self.snapshots:
            return None

        # Parse various formats
        for fmt in ("%Y-%m", "%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y"):
            try:
                target = datetime.strptime(time_str, fmt)
                break
            except ValueError:
                continue
        else:
            return None

        # Find nearest snapshot
        best = min(
            self.snapshots,
            key=lambda s: abs((s.time - target).total_seconds()),
        )
        return best.posterior

    def get_snapshot_at(self, time_str: str) -> Optional[PosteriorSnapshot]:
        """Get full snapshot at a specific time (nearest match)."""
        if not self.snapshots:
            return None
        for fmt in ("%Y-%m", "%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y"):
            try:
                target = datetime.strptime(time_str, fmt)
                break
            except ValueError:
                continue
        else:
            return None
        return min(
            self.snapshots,
            key=lambda s: abs((s.time - target).total_seconds()),
        )

    @property
    def times(self) -> list[datetime]:
        """All timestamps."""
        return [s.time for s in self.snapshots]

    @property
    def values(self) -> list[float]:
        """All posterior values."""
        return [s.posterior for s in self.snapshots]

    @property
    def latest(self) -> Optional[PosteriorSnapshot]:
        """Most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    @property
    def earliest(self) -> Optional[PosteriorSnapshot]:
        """Earliest snapshot."""
        return self.snapshots[0] if self.snapshots else None

    @property
    def trend_direction(self) -> str:
        """
        Determine overall trend direction.

        Returns:
            'rising', 'declining', 'stable', or 'insufficient_data'
        """
        if len(self.snapshots) < 3:
            return "insufficient_data"

        values = self.values
        n = len(values)

        # Simple linear regression slope
        x = np.arange(n, dtype=float)
        y = np.array(values)

        x_mean = x.mean()
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if abs(denominator) < 1e-10:
            return "stable"

        slope = numerator / denominator

        if slope > 0.01:
            return "rising"
        elif slope < -0.01:
            return "declining"
        return "stable"

    @property
    def volatility(self) -> float:
        """
        Compute volatility (standard deviation of posterior changes).

        Returns:
            Volatility score (0 = completely stable)
        """
        if len(self.snapshots) < 2:
            return 0.0
        values = self.values
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]
        return float(np.std(changes))

    def get_credible_band(self, level: float = 0.95) -> CredibleBand:
        """
        Extract credible band at specified level.

        Args:
            level: Confidence level (0.50 or 0.95)

        Returns:
            CredibleBand object
        """
        band = CredibleBand(level=level)
        for snap in self.snapshots:
            band.times.append(snap.time)
            if level >= 0.95:
                band.lower.append(snap.lower_95)
                band.upper.append(snap.upper_95)
            else:
                band.lower.append(snap.lower_50)
                band.upper.append(snap.upper_50)
        return band

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "proposition_id": self.proposition_id,
            "prior": self.prior,
            "num_points": len(self.snapshots),
            "trend": self.trend_direction,
            "volatility": self.volatility,
            "latest_posterior": self.latest.posterior if self.latest else None,
            "snapshots": [
                {
                    "time": s.time.isoformat(),
                    "posterior": s.posterior,
                    "lower_95": s.lower_95,
                    "upper_95": s.upper_95,
                }
                for s in self.snapshots
            ],
        }
