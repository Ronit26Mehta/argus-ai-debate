"""
Conceptual Debate Graph (C-DAG) module for ARGUS.

This module implements the argumentation layer with:
    - Graph data structure for debate representation
    - Node types (Proposition, Evidence, Rebuttal, Finding, Assumption)
    - Edge types (supports, attacks, refines)
    - Signed influence propagation for scoring
"""

from argus.cdag.nodes import (
    Proposition,
    Evidence,
    Rebuttal,
    Finding,
    Assumption,
    NodeStatus,
)
from argus.cdag.edges import (
    Edge,
    EdgeType,
    EdgePolarity,
)
from argus.cdag.graph import CDAG
from argus.cdag.propagation import (
    propagate_influence,
    compute_posterior,
    PropagationConfig,
)

__all__ = [
    # Nodes
    "Proposition",
    "Evidence",
    "Rebuttal",
    "Finding",
    "Assumption",
    "NodeStatus",
    # Edges
    "Edge",
    "EdgeType",
    "EdgePolarity",
    # Graph
    "CDAG",
    # Propagation
    "propagate_influence",
    "compute_posterior",
    "PropagationConfig",
]
