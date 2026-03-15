"""
Population management — spawning and sampling of persona populations.
"""

from __future__ import annotations

import random
import logging
from typing import Optional
from dataclasses import dataclass, field

from argus.phalanx.persona import EpistemicPersona, PriorDistribution, ExpertiseProfile
from argus.phalanx.bias_engine import CognitiveBiasEngine

logger = logging.getLogger(__name__)


@dataclass
class PopulationSampler:
    """
    Configures how personas are sampled for the population.

    Attributes:
        sampling_strategy: 'empirical', 'random', or 'custom'
        archetype_distribution: Distribution over persona archetypes
        bias_prevalence: Probability of each bias being active
        domains: Available domain specializations
    """
    sampling_strategy: str = "empirical"
    archetype_distribution: dict[str, float] = field(default_factory=lambda: {
        "optimistic": 0.25,
        "pessimistic": 0.25,
        "neutral": 0.30,
        "cautious": 0.10,
        "aggressive": 0.10,
    })
    bias_prevalence: dict[str, float] = field(default_factory=lambda: {
        "confirmation": 0.60,
        "anchoring": 0.40,
        "availability": 0.35,
        "authority": 0.30,
        "recency": 0.45,
    })
    domains: list[str] = field(default_factory=lambda: [
        "clinical", "finance", "policy", "technology", "science",
    ])

    def sample_archetype(self) -> str:
        """Sample an archetype based on distribution."""
        archetypes = list(self.archetype_distribution.keys())
        weights = list(self.archetype_distribution.values())
        return random.choices(archetypes, weights=weights, k=1)[0]

    def sample_biases(self) -> dict[str, float]:
        """Sample biases based on prevalence."""
        biases = {}
        for bias_name, prevalence in self.bias_prevalence.items():
            if random.random() < prevalence:
                biases[bias_name] = random.betavariate(2, 5)
        return biases


class PersonaPopulation:
    """
    Manages a population of N EpistemicPersona instances.

    Enables spawning, accessing, and analyzing populations for
    the PHALANX simulation.

    Example:
        >>> pop = PersonaPopulation(size=200)
        >>> pop.spawn()
        >>> beliefs = pop.all_beliefs
        >>> print(f"Mean belief: {sum(beliefs)/len(beliefs):.3f}")
    """

    def __init__(
        self,
        size: int = 100,
        sampler: Optional[PopulationSampler] = None,
    ):
        self.size = size
        self.sampler = sampler or PopulationSampler()
        self.personas: list[EpistemicPersona] = []
        self._spawned = False

    def spawn(self) -> None:
        """Spawn the full population based on sampler configuration."""
        self.personas = []
        for i in range(self.size):
            archetype = self.sampler.sample_archetype()
            biases = self.sampler.sample_biases()

            persona = EpistemicPersona(
                name=f"Pop-{i:04d}",
                prior_distribution=PriorDistribution.from_archetype(archetype),
                expertise=ExpertiseProfile.random_profile(self.sampler.domains),
                bias_strengths=biases,
            )
            self.personas.append(persona)

        self._spawned = True
        logger.info(f"Spawned population of {self.size} personas")

    def reset_all(self) -> None:
        """Reset all personas to new priors."""
        for persona in self.personas:
            persona.reset()

    @property
    def all_beliefs(self) -> list[float]:
        """All current beliefs in the population."""
        return [p.current_belief for p in self.personas]

    @property
    def all_priors(self) -> list[float]:
        """All initial priors."""
        return [p.initial_prior for p in self.personas]

    @property
    def mean_belief(self) -> float:
        beliefs = self.all_beliefs
        return sum(beliefs) / max(len(beliefs), 1)

    @property
    def belief_std(self) -> float:
        import numpy as np
        return float(np.std(self.all_beliefs)) if self.personas else 0.0

    def get_quartile(self, q: float) -> list[EpistemicPersona]:
        """Get personas in a belief quartile (0.25 = bottom, 0.75 = top)."""
        sorted_personas = sorted(self.personas, key=lambda p: p.current_belief)
        n = len(sorted_personas)
        if q <= 0.25:
            return sorted_personas[:n // 4]
        elif q >= 0.75:
            return sorted_personas[3 * n // 4:]
        else:
            start = int(n * (q - 0.125))
            end = int(n * (q + 0.125))
            return sorted_personas[start:end]

    def bias_summary(self) -> dict[str, float]:
        """Average bias strength across population."""
        from collections import defaultdict
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for p in self.personas:
            for bias, strength in p.bias_strengths.items():
                totals[bias] += strength
                counts[bias] += 1
        return {
            bias: totals[bias] / counts[bias]
            for bias in totals
        }
