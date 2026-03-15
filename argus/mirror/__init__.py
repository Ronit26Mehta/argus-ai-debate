"""
MIRROR — Consequence Inference Graph for ARGUS.

After debate reaches a verdict, spawns two ConsequenceInferenceAgents
that project downstream implications. Builds a forward-lookinig DAG
of consequences with conditional probabilities. CounterfactualChallenger
computes sensitivity: dP(consequence)/dP(root_verdict).

Example:
    >>> from argus.mirror import MIRROROrchestrator
    >>> mirror = MIRROROrchestrator(base=rdc)
    >>> result = mirror.debate('Ban single-use plastics')
    >>> print(result.consequence_graph.pivotal_nodes)
"""

from argus.mirror.inference_agent import ConsequenceInferenceAgent, ConsequenceNode
from argus.mirror.graph import ConsequenceGraph, ConsequenceProbability, SensitivityScore
from argus.mirror.counterfactual import CounterfactualChallenger, PivotalNode, CounterfactualReport
from argus.mirror.orchestrator import MIRROROrchestrator, MIRRORConfig, MIRRORResult
from argus.mirror.visualization import plot_consequence_graph, export_consequence_html

__all__ = [
    "ConsequenceInferenceAgent", "ConsequenceNode",
    "ConsequenceGraph", "ConsequenceProbability", "SensitivityScore",
    "CounterfactualChallenger", "PivotalNode", "CounterfactualReport",
    "MIRROROrchestrator", "MIRRORConfig", "MIRRORResult",
    "plot_consequence_graph", "export_consequence_html",
]
