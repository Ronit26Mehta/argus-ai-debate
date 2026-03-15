"""
Parallel Persona Runner — runs micro-debates for each persona.
"""

from __future__ import annotations

import math
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from argus.phalanx.persona import EpistemicPersona
from argus.phalanx.bias_engine import CognitiveBiasEngine

logger = logging.getLogger(__name__)


@dataclass
class MicroDebateResult:
    """
    Result of a single persona's micro-debate.

    Attributes:
        persona_id: Persona who ran the debate
        initial_prior: Starting prior
        final_posterior: Final posterior after evidence evaluation
        evidence_weights: Modified weights applied
        biases_active: Which biases were active
        rounds_completed: Number of evaluation rounds
    """
    persona_id: str
    initial_prior: float
    final_posterior: float
    evidence_weights: list[float] = field(default_factory=list)
    biases_active: list[str] = field(default_factory=list)
    rounds_completed: int = 1

    @property
    def belief_shift(self) -> float:
        return self.final_posterior - self.initial_prior


class ParallelPersonaRunner:
    """
    Runs micro-debates in parallel for all personas in a population.

    Each persona evaluates evidence through their biased lens,
    producing an individual posterior. The collection of posteriors
    forms the PopulationPosterior.

    Example:
        >>> runner = ParallelPersonaRunner(parallel_workers=8)
        >>> results = runner.run(population, evidence_items)
    """

    def __init__(
        self,
        parallel_workers: int = 4,
        num_evaluation_rounds: int = 3,
    ):
        self.parallel_workers = parallel_workers
        self.num_evaluation_rounds = num_evaluation_rounds
        self.bias_engine = CognitiveBiasEngine()

    def run(
        self,
        personas: list[EpistemicPersona],
        evidence_items: list[dict[str, Any]],
        proposition: str = "",
    ) -> list[MicroDebateResult]:
        """
        Run micro-debates for all personas.

        Args:
            personas: List of personas to run debates for
            evidence_items: Evidence to evaluate (list of dicts with
                'confidence', 'relevance', 'polarity', 'recency', 'prestige')
            proposition: Proposition text

        Returns:
            List of MicroDebateResult for each persona
        """
        results = []

        if self.parallel_workers <= 1 or len(personas) < 10:
            # Sequential for small populations
            for persona in personas:
                result = self._run_single(persona, evidence_items)
                results.append(result)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = {
                    executor.submit(self._run_single, p, evidence_items): p
                    for p in personas
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        persona = futures[future]
                        logger.warning(
                            f"Micro-debate failed for {persona.persona_id}: {e}"
                        )
                        results.append(MicroDebateResult(
                            persona_id=persona.persona_id,
                            initial_prior=persona.initial_prior,
                            final_posterior=persona.initial_prior,
                        ))

        logger.info(
            f"Completed {len(results)} micro-debates "
            f"(mean posterior: {sum(r.final_posterior for r in results)/max(len(results),1):.3f})"
        )
        return results

    def _run_single(
        self,
        persona: EpistemicPersona,
        evidence_items: list[dict[str, Any]],
    ) -> MicroDebateResult:
        """Run a single persona's micro-debate."""
        persona.reset()

        evidence_weights = []
        biases_active = list(persona.bias_strengths.keys())

        for round_num in range(self.num_evaluation_rounds):
            for ev in evidence_items:
                base_confidence = ev.get("confidence", 0.5)
                relevance = ev.get("relevance", 1.0)
                polarity = ev.get("polarity", 1)
                recency = ev.get("recency", 0.5)
                prestige = ev.get("prestige", 0.5)
                alignment = ev.get("alignment", 0.5)

                base_weight = base_confidence * relevance

                # Apply biases
                biased_weight = self.bias_engine.apply_all_biases(
                    base_weight=base_weight,
                    bias_strengths=persona.bias_strengths,
                    persona_prior=persona.current_belief,
                    evidence_alignment=alignment,
                    evidence_recency=recency,
                    source_prestige=prestige,
                    posterior_delta=abs(persona.current_belief - persona.initial_prior),
                )

                evidence_weights.append(biased_weight)

                # Compute log-odds contribution
                eff = max(0.001, min(0.999, biased_weight))
                llr = math.log(eff / (1.0 - eff))
                sign = float(polarity)

                # Dampen by round number (later rounds have less impact)
                dampen = 1.0 / (1.0 + round_num * 0.3)
                persona.update_belief(sign * llr * dampen * 0.3)

        return MicroDebateResult(
            persona_id=persona.persona_id,
            initial_prior=persona.initial_prior,
            final_posterior=persona.current_belief,
            evidence_weights=evidence_weights,
            biases_active=biases_active,
            rounds_completed=self.num_evaluation_rounds,
        )
