"""
Expertise Profile — per-domain Bayesian competence tracking.

Tracks agent performance by domain with Bayesian beta-distribution
updates after each debate verdict check.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BayesianUpdate:
    """A single Bayesian update record."""
    domain: str = ""
    correct: bool = True
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    posterior_alpha: float = 1.0
    posterior_beta: float = 1.0
    timestamp: str = ""


@dataclass
class DomainCompetence:
    """
    Per-domain competence tracked via Beta distribution.

    Alpha = correct predictions, Beta = incorrect predictions.
    Competence = Alpha / (Alpha + Beta) (expected value of Beta distribution).
    """
    domain: str = "general"
    alpha: float = 2.0  # Prior pseudo-count (successes)
    beta: float = 2.0   # Prior pseudo-count (failures)
    total_evaluations: int = 0

    @property
    def competence(self) -> float:
        """Current competence = E[Beta(α, β)]"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """Confidence inversely proportional to variance."""
        total = self.alpha + self.beta
        variance = (self.alpha * self.beta) / (total**2 * (total + 1))
        return 1.0 - min(1.0, math.sqrt(variance) * 4)

    def update(self, correct: bool, weight: float = 1.0) -> BayesianUpdate:
        """Update competence with new observation."""
        prior_a, prior_b = self.alpha, self.beta
        if correct:
            self.alpha += weight
        else:
            self.beta += weight
        self.total_evaluations += 1
        return BayesianUpdate(
            domain=self.domain, correct=correct,
            prior_alpha=prior_a, prior_beta=prior_b,
            posterior_alpha=self.alpha, posterior_beta=self.beta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "competence": round(self.competence, 4),
            "confidence": round(self.confidence, 4),
            "alpha": round(self.alpha, 2),
            "beta": round(self.beta, 2),
            "evaluations": self.total_evaluations,
        }


class ExpertiseProfile:
    """
    Tracks agent expertise across all domains.

    Example:
        >>> profile = ExpertiseProfile(agent_id="moderator-001")
        >>> profile.record_outcome("clinical", correct=True)
        >>> profile.record_outcome("finance", correct=False)
        >>> print(profile.competence_in("clinical"))  # ~0.6
    """

    def __init__(self, agent_id: str = ""):
        self.agent_id = agent_id
        self._domains: dict[str, DomainCompetence] = {}
        self._updates: list[BayesianUpdate] = []

    def get_domain(self, domain: str) -> DomainCompetence:
        """Get or create domain competence."""
        if domain not in self._domains:
            self._domains[domain] = DomainCompetence(domain=domain)
        return self._domains[domain]

    def competence_in(self, domain: str) -> float:
        """Get competence in a specific domain."""
        return self.get_domain(domain).competence

    def record_outcome(
        self,
        domain: str,
        correct: bool,
        weight: float = 1.0,
    ) -> BayesianUpdate:
        """Record debate outcome and update competence."""
        dc = self.get_domain(domain)
        update = dc.update(correct, weight)
        self._updates.append(update)
        logger.debug(
            f"Agent {self.agent_id}: {domain} competence = "
            f"{dc.competence:.3f} ({'correct' if correct else 'incorrect'})"
        )
        return update

    @property
    def domains(self) -> list[str]:
        return list(self._domains.keys())

    @property
    def overall_competence(self) -> float:
        """Weighted average competence across all domains."""
        if not self._domains:
            return 0.5
        total_evals = sum(dc.total_evaluations for dc in self._domains.values())
        if total_evals == 0:
            return 0.5
        weighted = sum(
            dc.competence * dc.total_evaluations
            for dc in self._domains.values()
        )
        return weighted / total_evals

    @property
    def strongest_domain(self) -> Optional[str]:
        if not self._domains:
            return None
        return max(
            self._domains.keys(),
            key=lambda d: self._domains[d].competence,
        )

    @property
    def weakest_domain(self) -> Optional[str]:
        if not self._domains:
            return None
        return min(
            self._domains.keys(),
            key=lambda d: self._domains[d].competence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "overall_competence": round(self.overall_competence, 4),
            "domains": {d: dc.to_dict() for d, dc in self._domains.items()},
        }
