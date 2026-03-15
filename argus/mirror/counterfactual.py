"""
CounterfactualChallenger — what-if analysis via sensitivity derivatives.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.mirror.graph import ConsequenceGraph, SensitivityScore

logger = logging.getLogger(__name__)


@dataclass
class PivotalNode:
    """A pivotal consequence node with counterfactual analysis."""
    node_id: str = ""
    text: str = ""
    sensitivity: float = 0.0
    probability_if_true: float = 0.5
    probability_if_false: float = 0.1
    marginal: float = 0.3
    category: str = ""


@dataclass
class CounterfactualReport:
    """Full counterfactual analysis report."""
    proposition: str = ""
    root_posterior: float = 0.5
    pivotal_nodes: list[PivotalNode] = field(default_factory=list)
    max_consequence_swing: float = 0.0
    most_sensitive_category: str = ""

    def narrative(self) -> str:
        lines = [
            f"Counterfactual Analysis for: \"{self.proposition[:80]}\"",
            f"Root posterior: {self.root_posterior:.3f}",
            f"Max consequence swing: {self.max_consequence_swing:.3f}",
            "",
        ]
        for i, pn in enumerate(self.pivotal_nodes[:5], 1):
            lines.append(
                f"  {i}. {pn.text[:70]}  "
                f"[P(if T)={pn.probability_if_true:.2f}, "
                f"P(if F)={pn.probability_if_false:.2f}, "
                f"Δ={pn.sensitivity:.3f}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition[:200],
            "root_posterior": round(self.root_posterior, 4),
            "max_swing": round(self.max_consequence_swing, 4),
            "most_sensitive_category": self.most_sensitive_category,
            "pivotal_nodes": [
                {"text": pn.text[:100], "sensitivity": round(pn.sensitivity, 4),
                 "p_if_true": round(pn.probability_if_true, 4),
                 "p_if_false": round(pn.probability_if_false, 4)}
                for pn in self.pivotal_nodes
            ],
        }


class CounterfactualChallenger:
    """
    Computes counterfactual analysis: what happens if the verdict flips?

    For each consequence node, computes the swing in probability if
    the root verdict were to change from TRUE to FALSE.

    Example:
        >>> challenger = CounterfactualChallenger()
        >>> report = challenger.analyse(consequence_graph)
        >>> print(report.narrative())
    """

    def __init__(self, min_sensitivity: float = 0.1):
        self.min_sensitivity = min_sensitivity

    def analyse(
        self,
        graph: ConsequenceGraph,
    ) -> CounterfactualReport:
        """Run full counterfactual analysis."""
        graph.compute_marginals()
        sensitivities = graph.compute_sensitivities()
        pivotal_nodes: list[PivotalNode] = []
        max_swing = 0.0

        for node in graph.all_nodes:
            sens = sensitivities.get(node.node_id)
            if not sens or abs(sens.dp_droot) < self.min_sensitivity:
                continue

            marginal = graph._marginals.get(node.node_id, 0.5)
            pn = PivotalNode(
                node_id=node.node_id,
                text=node.text,
                sensitivity=sens.dp_droot,
                probability_if_true=node.conditional_probability,
                probability_if_false=node.inverse_probability,
                marginal=marginal,
                category=node.category,
            )
            pivotal_nodes.append(pn)
            max_swing = max(max_swing, abs(sens.dp_droot))

        pivotal_nodes.sort(key=lambda p: abs(p.sensitivity), reverse=True)

        # Find most sensitive category
        cat_sens: dict[str, float] = {}
        for pn in pivotal_nodes:
            cat_sens[pn.category] = cat_sens.get(pn.category, 0) + abs(pn.sensitivity)
        most_sensitive = max(cat_sens, key=cat_sens.get) if cat_sens else ""

        report = CounterfactualReport(
            proposition=graph.proposition,
            root_posterior=graph.root_posterior,
            pivotal_nodes=pivotal_nodes,
            max_consequence_swing=max_swing,
            most_sensitive_category=most_sensitive,
        )

        logger.info(
            f"Counterfactual: {len(pivotal_nodes)} pivotal nodes, "
            f"max swing={max_swing:.3f}"
        )
        return report
