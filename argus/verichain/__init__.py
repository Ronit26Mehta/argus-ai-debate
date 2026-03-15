"""
VERICHAIN — Cross-Debate Truth Network for ARGUS.

Persistent, signed registry of debate verdicts. Each verdict becomes a
TruthNode with hash-chain integrity. VERICHAINRetriever uses semantic
search over past verdicts to inject epistemic precedent into new debates.

Example:
    >>> from argus.verichain import VERICHAINRegistry
    >>> registry = VERICHAINRegistry(backend='sqlite')
    >>> registry.register_verdict(proposition, verdict, posterior)
    >>> precedents = registry.retrieve_precedents("new related question")
"""

from argus.verichain.node import TruthNode, TruthNodeBuilder, NodeVersion
from argus.verichain.registry import VERICHAINRegistry, RegistryBackend
from argus.verichain.retriever import VERICHAINRetriever, SemanticSearch, PrecedentScorer
from argus.verichain.injector import EpistemicPrecedentInjector, InjectionPlan
from argus.verichain.integrity import ChainVerifier, HashChain, TamperDetector

__all__ = [
    "TruthNode", "TruthNodeBuilder", "NodeVersion",
    "VERICHAINRegistry", "RegistryBackend",
    "VERICHAINRetriever", "SemanticSearch", "PrecedentScorer",
    "EpistemicPrecedentInjector", "InjectionPlan",
    "ChainVerifier", "HashChain", "TamperDetector",
]
