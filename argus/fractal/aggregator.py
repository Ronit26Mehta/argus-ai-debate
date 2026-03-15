"""
Hierarchical Bayesian Aggregator — relationship-aware bottom-up aggregation.

Novel: Different aggregation rules for different logical relationships:
    NECESSARY → AND (product): P(parent) = ∏ P(child_i)
    SUFFICIENT → OR (noisy-or): P(parent) = 1 - ∏ (1 - P(child_i))
    CONTRIBUTING → Weighted Bayesian: P(parent) = Σ w_i × P(child_i)
    INDEPENDENT → Geometric mean: P(parent) = (∏ P(child_i))^(1/n)
"""

from __future__ import annotations

import math
import logging
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass

from argus.fractal.decomposer import PropositionTree, PropositionNode
from argus.fractal.runner import LeafDebateResult
from argus.fractal.classifier import RelationshipType

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Aggregation strategy types."""
    AND_PRODUCT = "and_product"
    OR_NOISY = "or_noisy"
    WEIGHTED_BAYESIAN = "weighted_bayesian"
    GEOMETRIC_MEAN = "geometric_mean"

    @classmethod
    def from_relationship(cls, rel: str) -> "AggregationStrategy":
        mapping = {
            "necessary": cls.AND_PRODUCT,
            "sufficient": cls.OR_NOISY,
            "contributing": cls.WEIGHTED_BAYESIAN,
            "independent": cls.GEOMETRIC_MEAN,
        }
        return mapping.get(rel, cls.WEIGHTED_BAYESIAN)


class HierarchicalBayesianAggregator:
    """
    Bottom-up aggregation of leaf posteriors through the tree.

    Uses relationship-aware aggregation rules. Each node's posterior
    is computed from its children's posteriors based on their logical
    relationship to the parent.

    Example:
        >>> agg = HierarchicalBayesianAggregator()
        >>> root_posterior = agg.aggregate(tree, leaf_results)
    """

    def __init__(
        self,
        default_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_BAYESIAN,
        necessary_dampening: float = 0.95,
    ):
        self.default_strategy = default_strategy
        self.necessary_dampening = necessary_dampening

    def aggregate(
        self,
        tree: PropositionTree,
        leaf_results: dict[str, LeafDebateResult],
    ) -> float:
        """
        Aggregate leaf posteriors bottom-up to compute root posterior.

        Args:
            tree: PropositionTree
            leaf_results: Leaf debate results

        Returns:
            Root posterior probability
        """
        # Set leaf posteriors
        for leaf in tree.leaves:
            result = leaf_results.get(leaf.node_id)
            if result:
                leaf.posterior = result.posterior
            else:
                leaf.posterior = 0.5

        # Bottom-up aggregation
        root_posterior = self._aggregate_recursive(tree, tree.root_id)
        tree.root.posterior = root_posterior

        logger.info(f"Aggregated root posterior: {root_posterior:.4f}")
        return root_posterior

    def _aggregate_recursive(
        self,
        tree: PropositionTree,
        node_id: str,
    ) -> float:
        """Recursively aggregate from leaves to root."""
        node = tree.get_node(node_id)
        if not node:
            return 0.5

        if node.is_leaf:
            return node.posterior if node.posterior is not None else 0.5

        # Get children posteriors
        children = tree.get_children(node_id)
        child_posteriors = []
        child_relationships = []

        for child in children:
            p = self._aggregate_recursive(tree, child.node_id)
            child.posterior = p
            child_posteriors.append(p)
            child_relationships.append(child.relationship_to_parent)

        if not child_posteriors:
            return 0.5

        # Determine aggregation strategy from majority relationship
        rel_counts: dict[str, int] = {}
        for rel in child_relationships:
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
        majority_rel = max(rel_counts, key=rel_counts.get)
        strategy = AggregationStrategy.from_relationship(majority_rel)

        # Apply relationship-specific aggregation
        return self._apply_strategy(strategy, child_posteriors)

    def _apply_strategy(
        self,
        strategy: AggregationStrategy,
        posteriors: list[float],
    ) -> float:
        """Apply aggregation strategy to child posteriors."""
        if not posteriors:
            return 0.5

        if strategy == AggregationStrategy.AND_PRODUCT:
            # NECESSARY: P(parent) = ∏ P(child_i) dampened
            result = 1.0
            for p in posteriors:
                result *= max(0.01, p)
            # Dampening to avoid overly harsh AND gates
            return result ** (self.necessary_dampening / len(posteriors))

        elif strategy == AggregationStrategy.OR_NOISY:
            # SUFFICIENT: P(parent) = 1 - ∏ (1 - P(child_i))
            result = 1.0
            for p in posteriors:
                result *= (1.0 - max(0.01, min(0.99, p)))
            return 1.0 - result

        elif strategy == AggregationStrategy.GEOMETRIC_MEAN:
            # INDEPENDENT: Geometric mean
            log_sum = sum(math.log(max(0.01, p)) for p in posteriors)
            return math.exp(log_sum / len(posteriors))

        else:
            # CONTRIBUTING: Weighted Bayesian (uniform weights)
            return sum(posteriors) / len(posteriors)
