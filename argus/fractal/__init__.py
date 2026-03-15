"""
FRACTAL — Hierarchical Proposition Decomposition for ARGUS.

Decomposes complex propositions into sub-proposition trees, runs
parallel leaf debates, and aggregates results using relationship-aware
Bayesian aggregation.

Example:
    >>> from argus.fractal import FRACTALOrchestrator
    >>> fractal = FRACTALOrchestrator(base=rdc)
    >>> result = fractal.debate('AI will exceed human intelligence by 2040')
    >>> print(result.proposition_tree)
"""

from argus.fractal.decomposer import PropositionDecomposer, PropositionTree, PropositionNode
from argus.fractal.classifier import LogicalRelationshipClassifier, RelationshipType
from argus.fractal.runner import ParallelDebateRunner, LeafDebateResult
from argus.fractal.aggregator import HierarchicalBayesianAggregator, AggregationStrategy
from argus.fractal.orchestrator import FRACTALOrchestrator, FRACTALConfig, FRACTALResult
from argus.fractal.visualization import plot_proposition_tree, export_tree_html

__all__ = [
    "PropositionDecomposer", "PropositionTree", "PropositionNode",
    "LogicalRelationshipClassifier", "RelationshipType",
    "ParallelDebateRunner", "LeafDebateResult",
    "HierarchicalBayesianAggregator", "AggregationStrategy",
    "FRACTALOrchestrator", "FRACTALConfig", "FRACTALResult",
    "plot_proposition_tree", "export_tree_html",
]
