"""
Epistemic Precedent Injector — injects prior verdicts as evidence.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.verichain.node import TruthNode

logger = logging.getLogger(__name__)


@dataclass
class InjectionPlan:
    """Plan for injecting precedents into a debate."""
    precedents: list[TruthNode] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    prior_adjustment: float = 0.0
    evidence_texts: list[str] = field(default_factory=list)

    @property
    def num_precedents(self) -> int:
        return len(self.precedents)


class EpistemicPrecedentInjector:
    """
    Injects prior verdicts as evidence into new debates.

    Converts VERICHAIN precedents into evidence nodes that influence
    the new debate's prior and evidence set.

    Example:
        >>> injector = EpistemicPrecedentInjector()
        >>> plan = injector.plan_injection(precedents, "new proposition")
    """

    def __init__(
        self,
        max_precedents: int = 5,
        authority_threshold: float = 0.3,
        prior_influence: float = 0.2,
    ):
        self.max_precedents = max_precedents
        self.authority_threshold = authority_threshold
        self.prior_influence = prior_influence

    def plan_injection(
        self,
        precedents: list[tuple[TruthNode, float]],
        proposition: str = "",
    ) -> InjectionPlan:
        """Create an injection plan from retrieved precedents."""
        plan = InjectionPlan()

        for node, score in precedents[:self.max_precedents]:
            if node.authority_score < self.authority_threshold:
                continue

            plan.precedents.append(node)
            plan.scores.append(score)

            # Create evidence text
            evidence = (
                f"[VERICHAIN Precedent] Previous debate on "
                f"\"{node.proposition[:80]}\" concluded "
                f"{node.current_verdict} (P={node.current_posterior:.2f}, "
                f"authority={node.authority_score:.2f}, "
                f"cited {node.citation_count} times)"
            )
            plan.evidence_texts.append(evidence)

            # Adjust prior based on precedent
            weight = score * node.authority_score * self.prior_influence
            if node.current_posterior > 0.5:
                plan.prior_adjustment += weight * 0.1
            else:
                plan.prior_adjustment -= weight * 0.1

        # Clamp prior adjustment
        plan.prior_adjustment = max(-0.15, min(0.15, plan.prior_adjustment))

        logger.info(
            f"Injection plan: {plan.num_precedents} precedents, "
            f"prior adjustment={plan.prior_adjustment:+.3f}"
        )
        return plan
