"""
MNEME Plugin — pluggable memory extension for RDCOrchestrator.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.mneme.reservoir import KnowledgeReservoir, ReservoirEntry
from argus.mneme.expertise import ExpertiseProfile
from argus.mneme.calibration import CalibrationHistory, CalibrationDriftMonitor

logger = logging.getLogger(__name__)


@dataclass
class AgentMemoryState:
    """Per-agent memory state."""
    agent_id: str = ""
    reservoir: Optional[KnowledgeReservoir] = None
    expertise: Optional[ExpertiseProfile] = None
    calibration_histories: dict[str, CalibrationHistory] = field(default_factory=dict)


@dataclass
class MNEMEConfig:
    """Configuration for MNEME plugin."""
    backend: str = "memory"  # memory, sqlite, postgres, qdrant
    db_path: str = "./mneme_memory.db"
    dsn: str = "postgresql://localhost/mneme"
    qdrant_url: str = "http://localhost:6333"
    max_entries: int = 5000
    reservoir_decay_days: float = 90.0
    enable_calibration: bool = True
    enable_expertise: bool = True


class MNEMEPlugin:
    """
    Pluggable persistent memory for RDCOrchestrator agents.

    Adds KnowledgeReservoir, ExpertiseProfile, and CalibrationHistory
    to each agent in the debate.

    Example:
        >>> from argus.mneme import MNEMEPlugin, MNEMEConfig
        >>> plugin = MNEMEPlugin(
        ...     config=MNEMEConfig(backend='sqlite', db_path='./memory.db')
        ... )
        >>> # Integrate with orchestrator
        >>> plugin.on_debate_start("proposition text", agents=["mod", "spec"])
        >>> plugin.record_outcome("spec", "clinical", correct=True, predicted=0.8)
    """

    def __init__(
        self,
        config: Optional[MNEMEConfig] = None,
        **kwargs: Any,
    ):
        self.config = config or MNEMEConfig(**{
            k: v for k, v in kwargs.items()
            if k in MNEMEConfig.__dataclass_fields__
        })
        self._agents: dict[str, AgentMemoryState] = {}
        self._drift_monitor = CalibrationDriftMonitor()
        self._backend: Optional[Any] = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize storage backend."""
        if self.config.backend == "sqlite":
            from argus.mneme.backends.sqlite import SQLiteMemoryBackend
            self._backend = SQLiteMemoryBackend(self.config.db_path)
        elif self.config.backend == "postgres":
            from argus.mneme.backends.postgres import PostgreSQLMemoryBackend
            self._backend = PostgreSQLMemoryBackend(self.config.dsn)
        elif self.config.backend == "qdrant":
            from argus.mneme.backends.qdrant import QdrantMemoryBackend
            self._backend = QdrantMemoryBackend(self.config.qdrant_url)
        # else: in-memory only

    def get_agent_state(self, agent_id: str) -> AgentMemoryState:
        """Get or create agent memory state."""
        if agent_id not in self._agents:
            from argus.mneme.reservoir import DecayFunction
            self._agents[agent_id] = AgentMemoryState(
                agent_id=agent_id,
                reservoir=KnowledgeReservoir(
                    max_entries=self.config.max_entries,
                    decay=DecayFunction(half_life_days=self.config.reservoir_decay_days),
                ),
                expertise=ExpertiseProfile(agent_id=agent_id),
            )
        return self._agents[agent_id]

    def on_debate_start(
        self,
        proposition: str,
        agents: Optional[list[str]] = None,
    ) -> dict[str, list[ReservoirEntry]]:
        """
        Called at debate start — retrieves relevant memories for agents.

        Returns mapping of agent_id to relevant past knowledge.
        """
        agents = agents or list(self._agents.keys())
        relevant: dict[str, list[ReservoirEntry]] = {}

        for agent_id in agents:
            state = self.get_agent_state(agent_id)
            results = state.reservoir.retrieve(proposition, top_k=5)
            relevant[agent_id] = [entry for entry, _ in results]

        return relevant

    def on_debate_end(
        self,
        proposition: str,
        verdict: str,
        domain: str = "general",
        confidence: float = 0.5,
        agents: Optional[list[str]] = None,
    ) -> None:
        """Called at debate end — stores new knowledge."""
        agents = agents or list(self._agents.keys())

        for agent_id in agents:
            state = self.get_agent_state(agent_id)

            entry = ReservoirEntry(
                text=f"Verdict on '{proposition[:100]}': {verdict}",
                domain=domain,
                confidence=confidence,
                proposition_text=proposition,
                verdict=verdict,
            )
            state.reservoir.store(entry)

            # Persist to backend
            if self._backend and hasattr(self._backend, "save_entry"):
                self._backend.save_entry(entry)

    def record_outcome(
        self,
        agent_id: str,
        domain: str,
        correct: bool,
        predicted: float = 0.5,
    ) -> None:
        """Record debate outcome for expertise and calibration."""
        state = self.get_agent_state(agent_id)

        if self.config.enable_expertise:
            state.expertise.record_outcome(domain, correct)

        if self.config.enable_calibration:
            if domain not in state.calibration_histories:
                state.calibration_histories[domain] = CalibrationHistory(domain=domain)
            state.calibration_histories[domain].record(predicted, correct)

    def get_drift_reports(self, agent_id: str) -> list[Any]:
        """Get calibration drift reports for an agent."""
        state = self.get_agent_state(agent_id)
        return self._drift_monitor.monitor_all(state.calibration_histories)

    def summary(self) -> dict[str, Any]:
        return {
            "num_agents": len(self._agents),
            "backend": self.config.backend,
            "agents": {
                aid: {
                    "reservoir_size": s.reservoir.size if s.reservoir else 0,
                    "overall_competence": (
                        s.expertise.overall_competence if s.expertise else 0.5
                    ),
                }
                for aid, s in self._agents.items()
            },
        }
