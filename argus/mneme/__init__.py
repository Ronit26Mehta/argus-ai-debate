"""
MNEME — Persistent Agent Memory & Expertise Evolution for ARGUS.

Transforms ARGUS agents from stateless into self-improving by adding:
    - KnowledgeReservoir (vector store with recency decay)
    - ExpertiseProfile (per-domain Bayesian competence tracking)
    - CalibrationHistory (rolling Brier Score monitoring)

Example:
    >>> from argus.mneme import MNEMEPlugin, MNEMEConfig
    >>> plugin = MNEMEPlugin(backend='sqlite', db_path='./memory.db')
    >>> orchestrator = RDCOrchestrator(plugins=[plugin])
"""

from argus.mneme.reservoir import KnowledgeReservoir, ReservoirEntry, DecayFunction
from argus.mneme.expertise import ExpertiseProfile, DomainCompetence, BayesianUpdate
from argus.mneme.calibration import CalibrationHistory, CalibrationDriftMonitor, DriftReport
from argus.mneme.plugin import MNEMEPlugin, MNEMEConfig, AgentMemoryState

__all__ = [
    "KnowledgeReservoir", "ReservoirEntry", "DecayFunction",
    "ExpertiseProfile", "DomainCompetence", "BayesianUpdate",
    "CalibrationHistory", "CalibrationDriftMonitor", "DriftReport",
    "MNEMEPlugin", "MNEMEConfig", "AgentMemoryState",
]
