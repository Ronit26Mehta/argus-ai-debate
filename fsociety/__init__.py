"""
fsociety — Federated Security Operations & Cyber Intelligence Engine Through Yield.

A multi-agent VAPT intelligence terminal powered by ARGUS.
"Hello, friend."

Usage:
    >>> from fsociety import FsocietyEngine
    >>> engine = FsocietyEngine()
    >>> engine.scan(path="/path/to/codebase")
"""

__version__ = "1.0.0"
__author__ = "Ronit Mehta"
__license__ = "MIT"
__tagline__ = "Hello, friend."

from fsociety.config import FsocietyConfig
from fsociety.vkg import VulnerabilityKnowledgeGraph
from fsociety.orchestrator import VAPTOrchestrator
from fsociety.models import (
    SeverityLevel,
    FindingStatus,
    VulnerabilityNode,
    ExploitChainNode,
    AttackSurfaceNode,
)

__all__ = [
    "__version__",
    "FsocietyConfig",
    "VulnerabilityKnowledgeGraph",
    "VAPTOrchestrator",
    "SeverityLevel",
    "FindingStatus",
    "VulnerabilityNode",
    "ExploitChainNode",
    "AttackSurfaceNode",
]
