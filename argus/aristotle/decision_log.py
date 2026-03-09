"""
Decision Logger — ORCHESTRATOR_DECISION event management.

Every structural decision ARISTOTLE makes during debate execution is
recorded in the ARGUS ProvenanceLedger as an ``ORCHESTRATOR_DECISION``
event with a fully-typed rationale, so the orchestrator itself is
governed, logged, and auditable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from argus.aristotle.models import OrchestratorDecisionType

if TYPE_CHECKING:
    from argus.provenance.ledger import ProvenanceLedger

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Writes ``ORCHESTRATOR_DECISION`` events to the ARGUS ProvenanceLedger.

    Each decision carries:
        - decision_type     (:class:`OrchestratorDecisionType`)
        - rationale         (human-readable reason)
        - affected_agents   (list of agent IDs)
        - round_num         (current debate round)
        - metadata          (extra context — posteriors, thresholds, etc.)
    """

    def __init__(self, ledger: "ProvenanceLedger"):
        self.ledger = ledger
        self._decisions: list[dict[str, Any]] = []

    def log(
        self,
        decision_type: OrchestratorDecisionType,
        rationale: str,
        round_num: int = 0,
        affected_agents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Record an ORCHESTRATOR_DECISION event.

        Returns the constructed event dict.
        """
        from argus.provenance.ledger import EventType

        attrs = {
            "decision_type": decision_type.value,
            "rationale": rationale,
            "round_num": round_num,
            "affected_agents": affected_agents or [],
            **(metadata or {}),
        }

        event = self.ledger.record(
            event_type=EventType.AGENT_ACTION,
            agent_id="ARISTOTLE",
            activity_id=f"orchestrator_decision_{decision_type.value}",
            attributes=attrs,
        )

        record = {
            "decision_type": decision_type.value,
            "rationale": rationale,
            "round_num": round_num,
            "affected_agents": affected_agents or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._decisions.append(record)

        logger.info(
            "ORCHESTRATOR_DECISION [round %d]: %s — %s",
            round_num, decision_type.value, rationale,
        )
        return record

    @property
    def decisions(self) -> list[dict[str, Any]]:
        """All decisions logged during this session."""
        return list(self._decisions)

    def query_by_type(
        self, decision_type: OrchestratorDecisionType,
    ) -> list[dict[str, Any]]:
        """Return decisions of a specific type."""
        return [
            d for d in self._decisions
            if d["decision_type"] == decision_type.value
        ]

    def export(self) -> list[dict[str, Any]]:
        """Export all decisions for audit."""
        return list(self._decisions)
