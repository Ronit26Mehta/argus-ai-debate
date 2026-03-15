"""
Emergent Consensus Detector — analyses population posteriors for structural features.

Key metrics:
    - PolarisationIndex: Jensen-Shannon Divergence between quartiles
    - Bimodality detection
    - Variance/skew analysis
    - Minority dissent cluster identification
"""

from __future__ import annotations

import math
import logging
from typing import Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DissentCluster:
    """
    A minority cluster with distinct beliefs.

    Attributes:
        cluster_id: Cluster identifier
        size: Number of personas in cluster
        mean_belief: Average belief in cluster
        std_belief: Standard deviation of beliefs
        persona_ids: IDs of personas in cluster
    """
    cluster_id: int = 0
    size: int = 0
    mean_belief: float = 0.5
    std_belief: float = 0.0
    persona_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "mean_belief": round(self.mean_belief, 4),
            "std_belief": round(self.std_belief, 4),
        }


@dataclass
class PolarisationIndex:
    """
    Measures population polarisation using Jensen-Shannon Divergence.

    PI = 0 → Strong consensus
    PI 0.1-0.3 → Moderate disagreement
    PI > 0.5 → Genuine controversy

    Attributes:
        value: The polarisation index (0-1)
        q1_mean: Mean belief of bottom quartile
        q4_mean: Mean belief of top quartile
        interpretation: Human-readable interpretation
    """
    value: float = 0.0
    q1_mean: float = 0.0
    q4_mean: float = 0.0
    interpretation: str = "unknown"

    @staticmethod
    def interpret(pi: float) -> str:
        if pi < 0.1:
            return "strong_consensus"
        elif pi < 0.3:
            return "moderate_disagreement"
        elif pi < 0.5:
            return "significant_controversy"
        else:
            return "genuine_controversy"


class EmergentConsensusDetector:
    """
    Analyses PopulationPosterior for structural features.

    Detects:
        - Bimodality (genuine controversy)
        - High variance (high uncertainty)
        - Skew (asymmetric evidence)
        - Polarisation (quartile divergence)
        - Minority dissent clusters

    Example:
        >>> detector = EmergentConsensusDetector()
        >>> pi = detector.compute_polarisation_index(beliefs)
        >>> print(pi.interpretation)  # 'moderate_disagreement'
    """

    def __init__(
        self,
        num_bins: int = 20,
        cluster_threshold: float = 0.15,
    ):
        self.num_bins = num_bins
        self.cluster_threshold = cluster_threshold

    def compute_polarisation_index(
        self,
        beliefs: list[float],
    ) -> PolarisationIndex:
        """
        Compute Jensen-Shannon Polarisation Index.

        Splits population into top and bottom quartiles by belief,
        then computes JSD between the two sub-distributions.

        Args:
            beliefs: List of posterior beliefs from all personas

        Returns:
            PolarisationIndex with value and interpretation
        """
        if len(beliefs) < 4:
            return PolarisationIndex(value=0.0, interpretation="insufficient_data")

        arr = np.array(sorted(beliefs))
        n = len(arr)
        q1 = arr[:n // 4]
        q4 = arr[3 * n // 4:]

        # Build distributions (histograms)
        bins = np.linspace(0, 1, self.num_bins + 1)
        p_q1, _ = np.histogram(q1, bins=bins, density=True)
        p_q4, _ = np.histogram(q4, bins=bins, density=True)

        # Normalise to probability distributions
        p_q1 = p_q1.astype(float) + 1e-10
        p_q4 = p_q4.astype(float) + 1e-10
        p_q1 /= p_q1.sum()
        p_q4 /= p_q4.sum()

        # Jensen-Shannon Divergence
        m = 0.5 * (p_q1 + p_q4)
        kl_q1_m = np.sum(p_q1 * np.log(p_q1 / m))
        kl_q4_m = np.sum(p_q4 * np.log(p_q4 / m))
        jsd = 0.5 * (kl_q1_m + kl_q4_m)

        # Normalise to [0, 1]
        pi_value = min(1.0, jsd / math.log(2))

        return PolarisationIndex(
            value=pi_value,
            q1_mean=float(q1.mean()),
            q4_mean=float(q4.mean()),
            interpretation=PolarisationIndex.interpret(pi_value),
        )

    def detect_bimodality(self, beliefs: list[float]) -> bool:
        """
        Detect bimodality using Hartigan's dip test approximation.

        Args:
            beliefs: Population beliefs

        Returns:
            True if distribution appears bimodal
        """
        if len(beliefs) < 10:
            return False

        arr = np.array(beliefs)
        hist, _ = np.histogram(arr, bins=self.num_bins)

        # Simple bimodality check: look for valley between two peaks
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((i, hist[i]))

        return len(peaks) >= 2

    def detect_dissent_clusters(
        self,
        beliefs: list[float],
        persona_ids: list[str],
    ) -> list[DissentCluster]:
        """
        Identify minority dissent clusters.

        Uses simple threshold-based clustering: groups of personas
        whose beliefs are far from the population mean.

        Args:
            beliefs: Population beliefs
            persona_ids: Corresponding persona IDs

        Returns:
            List of DissentCluster objects
        """
        if len(beliefs) < 5:
            return []

        arr = np.array(beliefs)
        mean = arr.mean()
        std = arr.std()

        clusters = []

        # Find personas significantly below mean (pessimist cluster)
        pessimists_mask = arr < (mean - self.cluster_threshold)
        if pessimists_mask.sum() >= 2:
            pess_beliefs = arr[pessimists_mask]
            pess_ids = [pid for pid, m in zip(persona_ids, pessimists_mask) if m]
            clusters.append(DissentCluster(
                cluster_id=0,
                size=len(pess_ids),
                mean_belief=float(pess_beliefs.mean()),
                std_belief=float(pess_beliefs.std()),
                persona_ids=pess_ids,
            ))

        # Find personas significantly above mean (optimist cluster)
        optimists_mask = arr > (mean + self.cluster_threshold)
        if optimists_mask.sum() >= 2:
            opt_beliefs = arr[optimists_mask]
            opt_ids = [pid for pid, m in zip(persona_ids, optimists_mask) if m]
            clusters.append(DissentCluster(
                cluster_id=1,
                size=len(opt_ids),
                mean_belief=float(opt_beliefs.mean()),
                std_belief=float(opt_beliefs.std()),
                persona_ids=opt_ids,
            ))

        return clusters

    def classify_consensus(
        self,
        beliefs: list[float],
    ) -> str:
        """
        Classify the consensus type.

        Returns one of:
            'STRONG_CONSENSUS'
            'SUPPORTED'
            'CONTESTED'
            'POLARISED'
            'INSUFFICIENT_DATA'
        """
        if len(beliefs) < 3:
            return "INSUFFICIENT_DATA"

        arr = np.array(beliefs)
        mean = arr.mean()
        std = arr.std()

        pi = self.compute_polarisation_index(beliefs)

        if pi.value > 0.5:
            return "POLARISED"
        elif std < 0.05 and mean > 0.6:
            return "STRONG_CONSENSUS"
        elif mean > 0.55:
            return "SUPPORTED"
        elif mean < 0.45:
            return "REJECTED"
        else:
            return "CONTESTED"
