"""
Population Posterior — full belief distribution over possible verdicts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BeliefDistribution:
    """
    Statistical summary of population beliefs.

    Attributes:
        mean: Mean belief
        median: Median belief
        std: Standard deviation
        skew: Skewness (negative = pessimistic skew)
        kurtosis: Kurtosis (>3 = heavy-tailed)
        q25: 25th percentile
        q75: 75th percentile
        iqr: Interquartile range
        min_belief: Minimum belief
        max_belief: Maximum belief
    """
    mean: float = 0.5
    median: float = 0.5
    std: float = 0.0
    skew: float = 0.0
    kurtosis: float = 3.0
    q25: float = 0.25
    q75: float = 0.75
    iqr: float = 0.5
    min_belief: float = 0.0
    max_belief: float = 1.0

    @classmethod
    def from_beliefs(cls, beliefs: list[float]) -> "BeliefDistribution":
        if not beliefs:
            return cls()
        arr = np.array(beliefs)
        from scipy import stats as scipy_stats
        try:
            skew_val = float(scipy_stats.skew(arr))
            kurt_val = float(scipy_stats.kurtosis(arr, fisher=False))
        except Exception:
            skew_val = 0.0
            kurt_val = 3.0
        return cls(
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            std=float(arr.std()),
            skew=skew_val,
            kurtosis=kurt_val,
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
            iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            min_belief=float(arr.min()),
            max_belief=float(arr.max()),
        )


class PopulationPosterior:
    """
    Full distribution over verdicts from population-scale debate.

    Aggregates individual micro-debate results into a population-level
    belief distribution with rich statistical summaries.

    Example:
        >>> pp = PopulationPosterior()
        >>> pp.add_beliefs([0.65, 0.72, 0.41, 0.83, 0.55])
        >>> print(pp.mean)      # 0.632
        >>> print(pp.std)       # ~0.15
        >>> print(pp.quartile(0.25))  # bottom quartile beliefs
    """

    def __init__(self, proposition_text: str = ""):
        self.proposition_text = proposition_text
        self._beliefs: list[float] = []
        self._persona_ids: list[str] = []
        self._distribution: Optional[BeliefDistribution] = None

    def add_belief(self, belief: float, persona_id: str = "") -> None:
        """Add a single belief."""
        self._beliefs.append(belief)
        self._persona_ids.append(persona_id)
        self._distribution = None

    def add_beliefs(
        self,
        beliefs: list[float],
        persona_ids: Optional[list[str]] = None,
    ) -> None:
        """Add multiple beliefs."""
        self._beliefs.extend(beliefs)
        if persona_ids:
            self._persona_ids.extend(persona_ids)
        else:
            self._persona_ids.extend(["" for _ in beliefs])
        self._distribution = None

    @property
    def distribution(self) -> BeliefDistribution:
        if self._distribution is None:
            self._distribution = BeliefDistribution.from_beliefs(self._beliefs)
        return self._distribution

    @property
    def mean(self) -> float:
        return self.distribution.mean

    @property
    def std(self) -> float:
        return self.distribution.std

    @property
    def median(self) -> float:
        return self.distribution.median

    @property
    def beliefs(self) -> list[float]:
        return self._beliefs.copy()

    @property
    def persona_ids(self) -> list[str]:
        return self._persona_ids.copy()

    @property
    def size(self) -> int:
        return len(self._beliefs)

    def quartile(self, q: float) -> list[float]:
        """Get beliefs at a specific quartile."""
        arr = np.array(sorted(self._beliefs))
        n = len(arr)
        if q <= 0.25:
            return arr[:n // 4].tolist()
        elif q >= 0.75:
            return arr[3 * n // 4:].tolist()
        else:
            start = int(n * max(0, q - 0.125))
            end = int(n * min(1, q + 0.125))
            return arr[start:end].tolist()

    def histogram(self, bins: int = 20) -> tuple[list[float], list[float]]:
        """Compute histogram for visualization."""
        arr = np.array(self._beliefs)
        counts, edges = np.histogram(arr, bins=bins, range=(0, 1))
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        return centers, counts.tolist()

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "median": round(self.median, 4),
            "distribution": {
                "skew": round(self.distribution.skew, 4),
                "kurtosis": round(self.distribution.kurtosis, 4),
                "iqr": round(self.distribution.iqr, 4),
            },
        }
