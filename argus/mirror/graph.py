"""
ConsequenceGraph — forward-inference DAG of consequences.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.mirror.inference_agent import ConsequenceNode

logger = logging.getLogger(__name__)


@dataclass
class ConsequenceProbability:
    """Computed probability for a consequence considering full graph."""
    node_id: str = ""
    marginal_probability: float = 0.5
    conditional_probability: float = 0.5
    sensitivity: float = 0.0


@dataclass
class SensitivityScore:
    """Sensitivity of a consequence to the root verdict."""
    node_id: str = ""
    text: str = ""
    dp_droot: float = 0.0  # dP(consequence)/dP(root)
    is_pivotal: bool = False


class ConsequenceGraph:
    """
    Forward-inference DAG of downstream consequences.

    Computes marginal probabilities through the graph and
    identifies pivotal nodes with highest sensitivity.

    Example:
        >>> graph = ConsequenceGraph(root_verdict="Supported", root_posterior=0.73)
        >>> graph.add_consequences(opportunity_consequences)
        >>> graph.add_consequences(risk_consequences)
        >>> graph.compute_marginals()
        >>> pivotal = graph.get_pivotal_nodes(k=3)
    """

    def __init__(
        self,
        root_verdict: str = "",
        root_posterior: float = 0.5,
        proposition: str = "",
    ):
        self.root_verdict = root_verdict
        self.root_posterior = root_posterior
        self.proposition = proposition
        self._nodes: dict[str, ConsequenceNode] = {}
        self._marginals: dict[str, float] = {}
        self._sensitivities: dict[str, float] = {}

    def add_consequence(self, node: ConsequenceNode) -> None:
        self._nodes[node.node_id] = node

    def add_consequences(self, nodes: list[ConsequenceNode]) -> None:
        for node in nodes:
            self.add_consequence(node)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def all_nodes(self) -> list[ConsequenceNode]:
        return list(self._nodes.values())

    def compute_marginals(self) -> dict[str, float]:
        """
        Compute marginal P(consequence) for each node.

        P(C) = P(C|root=T) × P(root) + P(C|root=F) × (1-P(root))
        """
        self._marginals = {}
        for node_id, node in self._nodes.items():
            marginal = (
                node.conditional_probability * self.root_posterior
                + node.inverse_probability * (1 - self.root_posterior)
            )
            self._marginals[node_id] = marginal
        return self._marginals

    def compute_sensitivities(self) -> dict[str, SensitivityScore]:
        """
        Compute dP(consequence)/dP(root) for each node.

        This is the analytical sensitivity of each consequence to
        the root verdict.
        """
        results: dict[str, SensitivityScore] = {}

        for node_id, node in self._nodes.items():
            # dP(C)/dP(root) = P(C|T) - P(C|F)
            dp = node.conditional_probability - node.inverse_probability
            self._sensitivities[node_id] = dp

            results[node_id] = SensitivityScore(
                node_id=node_id,
                text=node.text,
                dp_droot=dp,
                is_pivotal=abs(dp) > 0.3,
            )

        return results

    def get_pivotal_nodes(self, k: int = 5) -> list[SensitivityScore]:
        """Get top-k most sensitive (pivotal) nodes."""
        senses = self.compute_sensitivities()
        sorted_scores = sorted(
            senses.values(), key=lambda s: abs(s.dp_droot), reverse=True,
        )
        return sorted_scores[:k]

    def get_by_category(self, category: str) -> list[ConsequenceNode]:
        return [n for n in self._nodes.values() if n.category == category]

    @property
    def categories(self) -> list[str]:
        return list(set(n.category for n in self._nodes.values()))

    def to_dict(self) -> dict[str, Any]:
        if not self._marginals:
            self.compute_marginals()
        return {
            "root_verdict": self.root_verdict,
            "root_posterior": round(self.root_posterior, 4),
            "num_consequences": self.num_nodes,
            "categories": self.categories,
            "nodes": [
                {**n.to_dict(), "marginal_p": round(self._marginals.get(n.node_id, 0.5), 4)}
                for n in self._nodes.values()
            ],
        }
