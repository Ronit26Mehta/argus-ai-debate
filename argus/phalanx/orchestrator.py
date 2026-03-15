"""
PHALANX Orchestrator — Population-Scale Debate.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.phalanx.persona import EpistemicPersona
from argus.phalanx.population import PersonaPopulation, PopulationSampler
from argus.phalanx.runner import ParallelPersonaRunner, MicroDebateResult
from argus.phalanx.consensus import EmergentConsensusDetector, PolarisationIndex, DissentCluster
from argus.phalanx.posterior import PopulationPosterior

logger = logging.getLogger(__name__)


@dataclass
class PHALANXConfig:
    """Configuration for PHALANX population-scale debate."""
    population_size: int = 100
    bias_sampling: str = "empirical"
    parallel_workers: int = 4
    consensus_threshold: float = 0.1
    num_evaluation_rounds: int = 3


@dataclass
class PHALANXResult:
    """Result from a PHALANX population-scale debate."""
    base_result: Any = None
    population_posterior: Optional[PopulationPosterior] = None
    polarisation_index: Optional[PolarisationIndex] = None
    consensus_type: str = "UNKNOWN"
    minority_dissent_clusters: list[DissentCluster] = field(default_factory=list)
    micro_debates: list[MicroDebateResult] = field(default_factory=list)
    config: Optional[PHALANXConfig] = None

    @property
    def verdict(self) -> Any:
        return self.base_result.verdict if self.base_result else None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.population_posterior:
            result["population_posterior"] = self.population_posterior.to_dict()
        if self.polarisation_index:
            result["polarisation_index"] = {
                "value": round(self.polarisation_index.value, 4),
                "interpretation": self.polarisation_index.interpretation,
            }
        result["consensus_type"] = self.consensus_type
        result["num_dissent_clusters"] = len(self.minority_dissent_clusters)
        result["num_micro_debates"] = len(self.micro_debates)
        return result


class PHALANXOrchestrator:
    """
    Population-Scale Epistemic Simulation Orchestrator.

    Spawns N EpistemicPersona instances with cognitive biases, runs
    parallel micro-debates, and analyses population-level consensus.
    """

    def __init__(
        self,
        base: Optional[Any] = None,
        config: Optional[PHALANXConfig] = None,
        **kwargs: Any,
    ):
        self.base = base
        self.config = config or PHALANXConfig(**{
            k: v for k, v in kwargs.items()
            if k in PHALANXConfig.__dataclass_fields__
        })
        self._runner = ParallelPersonaRunner(
            parallel_workers=self.config.parallel_workers,
            num_evaluation_rounds=self.config.num_evaluation_rounds,
        )
        self._consensus_detector = EmergentConsensusDetector()

    def debate(
        self,
        proposition: str,
        prior: float = 0.5,
        evidence_items: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> PHALANXResult:
        """
        Run a population-scale debate.

        Args:
            proposition: Proposition text
            prior: Base prior (personas will sample around it)
            evidence_items: Evidence to evaluate
            **kwargs: Passed to base orchestrator

        Returns:
            PHALANXResult with population posterior and polarisation
        """
        # Generate default evidence if not provided
        if evidence_items is None:
            import random
            evidence_items = [
                {
                    "confidence": random.uniform(0.4, 0.9),
                    "relevance": random.uniform(0.5, 1.0),
                    "polarity": random.choice([1, -1]),
                    "recency": random.uniform(0.3, 1.0),
                    "prestige": random.uniform(0.3, 0.9),
                    "alignment": random.uniform(0.2, 0.8),
                }
                for _ in range(5)
            ]

        # Spawn population
        population = PersonaPopulation(
            size=self.config.population_size,
            sampler=PopulationSampler(sampling_strategy=self.config.bias_sampling),
        )
        population.spawn()

        # Run micro-debates
        micro_results = self._runner.run(
            population.personas, evidence_items, proposition,
        )

        # Build population posterior
        pop_posterior = PopulationPosterior(proposition_text=proposition)
        for mr in micro_results:
            pop_posterior.add_belief(mr.final_posterior, mr.persona_id)

        # Analyse consensus
        beliefs = pop_posterior.beliefs
        persona_ids = pop_posterior.persona_ids

        pi = self._consensus_detector.compute_polarisation_index(beliefs)
        consensus_type = self._consensus_detector.classify_consensus(beliefs)
        dissent = self._consensus_detector.detect_dissent_clusters(beliefs, persona_ids)

        # Run base debate if available
        base_result = None
        if self.base is not None:
            try:
                base_result = self.base.debate(proposition, prior=prior, **kwargs)
            except Exception as e:
                logger.warning(f"Base debate failed: {e}")

        result = PHALANXResult(
            base_result=base_result,
            population_posterior=pop_posterior,
            polarisation_index=pi,
            consensus_type=consensus_type,
            minority_dissent_clusters=dissent,
            micro_debates=micro_results,
            config=self.config,
        )

        logger.info(
            f"PHALANX debate complete: {self.config.population_size} personas, "
            f"PI={pi.value:.3f} ({pi.interpretation}), "
            f"consensus={consensus_type}"
        )

        return result

    def __or__(self, other: Any) -> Any:
        return other
