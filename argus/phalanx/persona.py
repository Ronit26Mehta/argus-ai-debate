"""
Epistemic Persona — lightweight virtual analyst with cognitive priors.

Each persona has a distinct prior distribution, domain expertise weighting,
and set of active cognitive biases. Personas participate in micro-debates
as part of the PHALANX population-scale simulation.
"""

from __future__ import annotations

import uuid
import math
import random
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class PriorDistribution:
    """
    A persona's prior belief distribution.

    Attributes:
        mean: Central prior probability
        std: Uncertainty around the prior
        skew: Skew direction (-1 to 1, negative = pessimistic)
    """
    mean: float = 0.5
    std: float = 0.15
    skew: float = 0.0

    def sample(self) -> float:
        """Sample a prior from the distribution."""
        value = random.gauss(self.mean, self.std)
        if self.skew != 0:
            value += self.skew * abs(random.gauss(0, self.std * 0.5))
        return max(0.01, min(0.99, value))

    @classmethod
    def optimistic(cls) -> "PriorDistribution":
        return cls(mean=0.65, std=0.12, skew=0.2)

    @classmethod
    def pessimistic(cls) -> "PriorDistribution":
        return cls(mean=0.35, std=0.12, skew=-0.2)

    @classmethod
    def neutral(cls) -> "PriorDistribution":
        return cls(mean=0.5, std=0.15, skew=0.0)

    @classmethod
    def from_archetype(cls, archetype: str) -> "PriorDistribution":
        archetypes = {
            "optimistic": cls.optimistic,
            "pessimistic": cls.pessimistic,
            "neutral": cls.neutral,
            "cautious": lambda: cls(mean=0.4, std=0.1, skew=-0.1),
            "aggressive": lambda: cls(mean=0.6, std=0.2, skew=0.15),
        }
        factory = archetypes.get(archetype, cls.neutral)
        return factory()


@dataclass
class ExpertiseProfile:
    """
    Domain expertise weighting for a persona.

    Attributes:
        domain_weights: Mapping of domain name to expertise (0-1)
        overall_competence: General reasoning competence (0-1)
    """
    domain_weights: dict[str, float] = field(default_factory=dict)
    overall_competence: float = 0.5

    def expertise_in(self, domain: str) -> float:
        """Get expertise level in a specific domain."""
        return self.domain_weights.get(domain, self.overall_competence)

    def weight_evidence(self, domain: str, base_weight: float) -> float:
        """Weight evidence by domain expertise."""
        expertise = self.expertise_in(domain)
        return base_weight * (0.5 + 0.5 * expertise)

    @classmethod
    def generalist(cls) -> "ExpertiseProfile":
        return cls(overall_competence=0.5)

    @classmethod
    def specialist(cls, domain: str, level: float = 0.9) -> "ExpertiseProfile":
        return cls(
            domain_weights={domain: level},
            overall_competence=0.4,
        )

    @classmethod
    def random_profile(cls, domains: Optional[list[str]] = None) -> "ExpertiseProfile":
        """Generate random expertise profile."""
        domains = domains or ["clinical", "finance", "policy", "technology"]
        weights = {d: random.betavariate(2, 5) for d in domains}
        return cls(
            domain_weights=weights,
            overall_competence=random.betavariate(3, 3),
        )


class EpistemicPersona:
    """
    A lightweight virtual analyst with distinct cognitive characteristics.

    Each persona combines a prior distribution, expertise profile, and
    set of cognitive biases to produce a unique perspective on evidence.

    Attributes:
        persona_id: Unique persona identifier
        name: Display name
        prior_distribution: How this persona forms prior beliefs
        expertise: Domain expertise profile
        bias_strengths: Mapping of bias name to strength (0-1)
        initial_prior: Sampled prior for current debate
    """

    def __init__(
        self,
        persona_id: Optional[str] = None,
        name: str = "",
        prior_distribution: Optional[PriorDistribution] = None,
        expertise: Optional[ExpertiseProfile] = None,
        bias_strengths: Optional[dict[str, float]] = None,
    ):
        self.persona_id = persona_id or f"persona_{uuid.uuid4().hex[:8]}"
        self.name = name or f"Analyst-{self.persona_id[-4:]}"
        self.prior_distribution = prior_distribution or PriorDistribution.neutral()
        self.expertise = expertise or ExpertiseProfile.generalist()
        self.bias_strengths = bias_strengths or {}
        self.initial_prior = self.prior_distribution.sample()
        self._current_belief = self.initial_prior

    @property
    def current_belief(self) -> float:
        return self._current_belief

    @current_belief.setter
    def current_belief(self, value: float) -> None:
        self._current_belief = max(0.01, min(0.99, value))

    def get_bias_strength(self, bias_name: str) -> float:
        """Get strength of a specific bias (0 = no bias)."""
        return self.bias_strengths.get(bias_name, 0.0)

    def update_belief(self, log_odds_delta: float) -> float:
        """
        Update belief using log-odds addition.

        Args:
            log_odds_delta: Change in log-odds

        Returns:
            New posterior belief
        """
        p = max(0.001, min(0.999, self._current_belief))
        current_lo = math.log(p / (1 - p))
        new_lo = current_lo + log_odds_delta
        new_lo = max(-500, min(500, new_lo))
        self._current_belief = 1.0 / (1.0 + math.exp(-new_lo))
        return self._current_belief

    def reset(self) -> None:
        """Reset belief to initial prior."""
        self.initial_prior = self.prior_distribution.sample()
        self._current_belief = self.initial_prior

    @classmethod
    def random_persona(
        cls,
        domains: Optional[list[str]] = None,
        bias_types: Optional[list[str]] = None,
    ) -> "EpistemicPersona":
        """Generate a random persona with random biases."""
        bias_types = bias_types or [
            "confirmation", "anchoring", "availability",
            "authority", "recency",
        ]
        archetypes = ["optimistic", "pessimistic", "neutral", "cautious", "aggressive"]

        bias_strengths = {
            bt: random.betavariate(2, 5)
            for bt in random.sample(bias_types, k=random.randint(1, len(bias_types)))
        }

        return cls(
            prior_distribution=PriorDistribution.from_archetype(
                random.choice(archetypes)
            ),
            expertise=ExpertiseProfile.random_profile(domains),
            bias_strengths=bias_strengths,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "initial_prior": round(self.initial_prior, 4),
            "current_belief": round(self._current_belief, 4),
            "bias_strengths": {k: round(v, 4) for k, v in self.bias_strengths.items()},
            "overall_competence": self.expertise.overall_competence,
        }
