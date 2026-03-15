"""
Cognitive Bias Engine for PHALANX.

Implements cognitive biases as differentiable weight modifiers applied
during evidence evaluation. This is NOT symbolic simulation — biases
are numeric weight functions applied directly to confidence scores.

Biases:
    CONFIRMATION:  Increases weight of evidence aligned with persona's prior
    ANCHORING:     Applies inertia resisting large posterior updates
    AVAILABILITY:  Amplifies weight of recent/vivid evidence
    AUTHORITY:     Boosts weight of evidence from high-prestige sources
    RECENCY:       Over-weights the most recent evidence
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Optional, Any, Callable
from dataclasses import dataclass


class CognitiveBias(str, Enum):
    """Supported cognitive bias types."""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    AUTHORITY = "authority"
    RECENCY = "recency"

    @classmethod
    def all_biases(cls) -> list["CognitiveBias"]:
        return list(cls)


@dataclass
class BiasWeightFn:
    """
    A bias weight modification function.

    Encapsulates a bias type with its parameters for application
    to evidence weights.

    Attributes:
        bias_type: Type of cognitive bias
        strength: Bias strength (0 = no bias, 1 = max bias)
        params: Additional bias-specific parameters
    """
    bias_type: CognitiveBias
    strength: float = 0.3
    params: dict[str, float] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self.strength = max(0.0, min(1.0, self.strength))


class CognitiveBiasEngine:
    """
    Applies cognitive biases as quantitative weight modifiers.

    The first multi-agent debate system to model cognitive bias as
    first-class numeric components of reasoning, applied directly
    to evidence evaluation weights.

    Example:
        >>> engine = CognitiveBiasEngine()
        >>> biased_weight = engine.apply_bias(
        ...     base_weight=0.8,
        ...     bias_type=CognitiveBias.CONFIRMATION,
        ...     persona_prior=0.7,
        ...     evidence_alignment=0.9,
        ...     bias_strength=0.5,
        ... )
    """

    def __init__(self):
        self._bias_fns: dict[CognitiveBias, Callable] = {
            CognitiveBias.CONFIRMATION: self._apply_confirmation,
            CognitiveBias.ANCHORING: self._apply_anchoring,
            CognitiveBias.AVAILABILITY: self._apply_availability,
            CognitiveBias.AUTHORITY: self._apply_authority,
            CognitiveBias.RECENCY: self._apply_recency,
        }

    def apply_bias(
        self,
        base_weight: float,
        bias_type: CognitiveBias,
        bias_strength: float = 0.3,
        persona_prior: float = 0.5,
        evidence_alignment: float = 0.5,
        evidence_recency: float = 0.5,
        source_prestige: float = 0.5,
        posterior_delta: float = 0.0,
    ) -> float:
        """
        Apply a single bias to an evidence weight.

        Args:
            base_weight: Original evidence weight
            bias_type: Type of bias to apply
            bias_strength: Strength of the bias (0-1)
            persona_prior: Persona's current prior belief (0-1)
            evidence_alignment: How aligned evidence is with prior (0-1)
            evidence_recency: How recent the evidence is (0-1)
            source_prestige: Prestige of the evidence source (0-1)
            posterior_delta: Magnitude of posterior change

        Returns:
            Modified weight after bias application
        """
        fn = self._bias_fns.get(bias_type)
        if fn is None:
            return base_weight

        return fn(
            base_weight=base_weight,
            strength=bias_strength,
            persona_prior=persona_prior,
            evidence_alignment=evidence_alignment,
            evidence_recency=evidence_recency,
            source_prestige=source_prestige,
            posterior_delta=posterior_delta,
        )

    def apply_all_biases(
        self,
        base_weight: float,
        bias_strengths: dict[str, float],
        persona_prior: float = 0.5,
        evidence_alignment: float = 0.5,
        evidence_recency: float = 0.5,
        source_prestige: float = 0.5,
        posterior_delta: float = 0.0,
    ) -> float:
        """
        Apply all active biases sequentially.

        Args:
            base_weight: Original evidence weight
            bias_strengths: Mapping of bias name to strength
            Other args: Evidence attributes for bias computation

        Returns:
            Weight after all biases applied
        """
        weight = base_weight

        for bias_name, strength in bias_strengths.items():
            try:
                bias_type = CognitiveBias(bias_name)
            except ValueError:
                continue

            if strength > 0:
                weight = self.apply_bias(
                    base_weight=weight,
                    bias_type=bias_type,
                    bias_strength=strength,
                    persona_prior=persona_prior,
                    evidence_alignment=evidence_alignment,
                    evidence_recency=evidence_recency,
                    source_prestige=source_prestige,
                    posterior_delta=posterior_delta,
                )

        return max(0.0, min(2.0, weight))

    @staticmethod
    def _apply_confirmation(
        base_weight: float,
        strength: float,
        persona_prior: float,
        evidence_alignment: float,
        **kwargs: Any,
    ) -> float:
        """
        Confirmation bias: increases weight of evidence aligned with prior.

        biased_weight = base × (1 + strength × alignment)
        where alignment = cosine-like proximity to persona's prior.
        """
        # Higher alignment = evidence matches prior expectations
        modifier = 1.0 + strength * evidence_alignment
        return base_weight * modifier

    @staticmethod
    def _apply_anchoring(
        base_weight: float,
        strength: float,
        posterior_delta: float,
        **kwargs: Any,
    ) -> float:
        """
        Anchoring bias: applies inertia resisting large posterior updates.

        biased_weight = base × exp(-strength × |delta|)
        Large changes are dampened by the anchoring effect.
        """
        inertia = math.exp(-strength * abs(posterior_delta))
        return base_weight * inertia

    @staticmethod
    def _apply_availability(
        base_weight: float,
        strength: float,
        evidence_recency: float,
        **kwargs: Any,
    ) -> float:
        """
        Availability bias: amplifies weight of recent/vivid evidence.

        biased_weight = base × (1 + strength × recency_score)
        """
        modifier = 1.0 + strength * evidence_recency
        return base_weight * modifier

    @staticmethod
    def _apply_authority(
        base_weight: float,
        strength: float,
        source_prestige: float,
        **kwargs: Any,
    ) -> float:
        """
        Authority bias: boosts weight of evidence from high-prestige sources.

        biased_weight = base × (1 + strength × prestige)
        """
        modifier = 1.0 + strength * source_prestige
        return base_weight * modifier

    @staticmethod
    def _apply_recency(
        base_weight: float,
        strength: float,
        evidence_recency: float,
        **kwargs: Any,
    ) -> float:
        """
        Recency bias: over-weights the most recent evidence.

        Stronger than availability — uses exponential scaling.
        biased_weight = base × (1 + strength × recency²)
        """
        modifier = 1.0 + strength * (evidence_recency ** 2)
        return base_weight * modifier

    @staticmethod
    def random_bias_set(
        num_biases: int = 2,
        strength_range: tuple[float, float] = (0.1, 0.6),
    ) -> dict[str, float]:
        """Generate random bias set for a persona."""
        all_biases = CognitiveBias.all_biases()
        selected = random.sample(all_biases, k=min(num_biases, len(all_biases)))
        return {
            b.value: random.uniform(*strength_range)
            for b in selected
        }
