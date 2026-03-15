"""
FRACTAL Orchestrator — hierarchical proposition debate.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.fractal.decomposer import PropositionDecomposer, PropositionTree
from argus.fractal.classifier import LogicalRelationshipClassifier
from argus.fractal.runner import ParallelDebateRunner, LeafDebateResult
from argus.fractal.aggregator import HierarchicalBayesianAggregator

logger = logging.getLogger(__name__)


@dataclass
class FRACTALConfig:
    """Configuration for FRACTAL."""
    max_depth: int = 3
    max_children: int = 5
    parallel_workers: int = 4
    default_prior: float = 0.5


@dataclass
class FRACTALResult:
    """Result from FRACTAL debate."""
    base_result: Any = None
    proposition_tree: Optional[PropositionTree] = None
    root_posterior: float = 0.5
    leaf_results: dict[str, LeafDebateResult] = field(default_factory=dict)
    num_leaves: int = 0
    max_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "root_posterior": round(self.root_posterior, 4),
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
        }
        if self.proposition_tree:
            result["tree"] = self.proposition_tree.to_dict()
        result["leaf_results"] = {
            k: v.to_dict() for k, v in self.leaf_results.items()
        }
        return result


class FRACTALOrchestrator:
    """
    Hierarchical Proposition Decomposition Orchestrator.

    Decomposes complex propositions, runs parallel leaf debates,
    and aggregates with relationship-aware Bayesian logic.
    """

    def __init__(
        self,
        base: Optional[Any] = None,
        config: Optional[FRACTALConfig] = None,
        **kwargs: Any,
    ):
        self.base = base
        self.config = config or FRACTALConfig(**{
            k: v for k, v in kwargs.items()
            if k in FRACTALConfig.__dataclass_fields__
        })
        self._decomposer = PropositionDecomposer(
            max_depth=self.config.max_depth,
            max_children=self.config.max_children,
        )
        self._classifier = LogicalRelationshipClassifier()
        self._runner = ParallelDebateRunner(
            base_orchestrator=base,
            parallel_workers=self.config.parallel_workers,
            default_prior=self.config.default_prior,
        )
        self._aggregator = HierarchicalBayesianAggregator()

    def debate(self, proposition: str, **kwargs: Any) -> FRACTALResult:
        """Run a FRACTAL hierarchical debate."""
        # Step 1: Decompose
        tree = self._decomposer.decompose(proposition)

        # Step 2: Classify relationships
        for node in tree.all_nodes:
            if node.parent_id:
                parent = tree.get_node(node.parent_id)
                if parent:
                    rel = self._classifier.classify(parent.text, node.text)
                    node.relationship_to_parent = rel.value

        # Step 3: Run leaf debates
        leaf_results = self._runner.run_leaf_debates(tree, **kwargs)

        # Step 4: Aggregate
        root_posterior = self._aggregator.aggregate(tree, leaf_results)

        result = FRACTALResult(
            proposition_tree=tree,
            root_posterior=root_posterior,
            leaf_results=leaf_results,
            num_leaves=len(tree.leaves),
            max_depth=tree.max_depth,
        )

        logger.info(
            f"FRACTAL complete: {tree.num_nodes} nodes, "
            f"depth {tree.max_depth}, root P={root_posterior:.4f}"
        )
        return result
