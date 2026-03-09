"""
ARISTOTLE data models — shared across all five layers.

Every structured artefact produced or consumed by ARISTOTLE is defined here
so that layers communicate via well-typed, serialisable objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


# ── helpers ────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uid(prefix: str = "ar") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ── Enums ──────────────────────────────────────────────────────────────

class DebateType(str, Enum):
    """Six canonical debate types classified by L1."""
    BINARY_FACTUAL = "binary_factual"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    NORMATIVE = "normative"
    DEFINITIONAL = "definitional"


class RefuterIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class JuryArchitecture(str, Enum):
    BAYESIAN = "bayesian"
    DELIBERATIVE = "deliberative"
    VALIDITY_WEIGHTED = "validity_weighted"


class VerdictLevel(str, Enum):
    STRONG_SUPPORT = "Strong Evidence Supports"
    MODERATE_SUPPORT = "Contested with Moderate Evidence"
    INSUFFICIENT = "Insufficient Evidence to Conclude"
    EVIDENCE_AGAINST = "Evidence Actively Against"


class MonitorEventType(str, Enum):
    SESSION_START = "session_start"
    SPECIALIST_INSTANTIATED = "specialist_instantiated"
    SUPPORT_EVIDENCE = "support_evidence"
    ATTACK_EVIDENCE = "attack_evidence"
    REBUTTAL_FILED = "rebuttal_filed"
    BAYESIAN_UPDATE = "bayesian_update"
    ROUND_COMPLETE = "round_complete"
    INTERVENTION_CONVERGENCE = "intervention_convergence"
    INTERVENTION_STAGNATION = "intervention_stagnation"
    INTERVENTION_DRIFT = "intervention_drift"
    INTERVENTION_CONFIDENCE_GUARD = "intervention_confidence_guard"
    INTERVENTION_EVIDENCE_DROUGHT = "intervention_evidence_drought"
    INTERVENTION_BUDGET_OVERFLOW = "intervention_budget_overflow"
    INTERVENTION_DEADLOCK = "intervention_deadlock"
    AGENT_REPLACED = "agent_replaced"
    VERDICT_RENDERED = "verdict_rendered"
    DEBATE_COMPLETE = "debate_complete"


class OrchestratorDecisionType(str, Enum):
    EARLY_TERMINATION = "early_termination"
    STAGNATION_INJECTION = "stagnation_injection"
    DRIFT_CORRECTION = "drift_correction"
    CONFIDENCE_RESET = "confidence_reset"
    AGENT_REPLACEMENT = "agent_replacement"
    BUDGET_ACCELERATION = "budget_acceleration"
    DEADLOCK_RESOLUTION = "deadlock_resolution"


# ── Layer 1 outputs ───────────────────────────────────────────────────

@dataclass
class StancePosition:
    """A single position in the stance space."""
    label: str
    description: str
    estimated_support: float = 0.5  # 0-1


@dataclass
class SubClaim:
    """An atomic sub-proposition decomposed from the user query."""
    id: str = field(default_factory=lambda: _uid("sc"))
    text: str = ""
    is_primary: bool = False
    depends_on: list[str] = field(default_factory=list)


@dataclass
class LiteratureProbe:
    """Result of the rapid n=10 literature scan."""
    num_support: int = 0
    num_against: int = 0
    num_partial: int = 0
    total_scanned: int = 0
    summaries: list[str] = field(default_factory=list)
    recency_weight: float = 1.0


@dataclass
class DebateFrame:
    """
    Complete output of Layer 1 (framing.py).

    Captures every analytical operation: domain, debate type, controversy,
    evidence density, stance space, claim decomposition, and calibrated prior.
    """
    id: str = field(default_factory=lambda: _uid("df"))
    raw_query: str = ""

    # (a) domain classification
    primary_domain: str = "general"
    secondary_domains: list[str] = field(default_factory=list)

    # (b) debate type
    debate_type: DebateType = DebateType.BINARY_FACTUAL

    # (c) controversy level 0-1
    controversy_score: float = 0.5
    controversy_signals: list[str] = field(default_factory=list)

    # (d) evidence density
    evidence_density: str = "moderate"  # sparse | moderate | rich
    evidence_density_score: float = 0.5

    # (e) stance space
    stance_positions: list[StancePosition] = field(default_factory=list)

    # (f) claim decomposition
    sub_claims: list[SubClaim] = field(default_factory=list)
    primary_proposition: str = ""

    # (g) calibrated prior
    prior_probability: float = 0.5
    literature_probe: Optional[LiteratureProbe] = None

    # metadata
    framing_narrative: str = ""
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "raw_query": self.raw_query,
            "primary_domain": self.primary_domain,
            "secondary_domains": self.secondary_domains,
            "debate_type": self.debate_type.value,
            "controversy_score": self.controversy_score,
            "evidence_density": self.evidence_density,
            "evidence_density_score": self.evidence_density_score,
            "stance_positions": [
                {"label": s.label, "description": s.description,
                 "estimated_support": s.estimated_support}
                for s in self.stance_positions
            ],
            "sub_claims": [
                {"id": c.id, "text": c.text, "is_primary": c.is_primary}
                for c in self.sub_claims
            ],
            "primary_proposition": self.primary_proposition,
            "prior_probability": self.prior_probability,
            "framing_narrative": self.framing_narrative,
            "created_at": self.created_at.isoformat(),
        }


# ── Layer 2 outputs ───────────────────────────────────────────────────

@dataclass
class AgentSpec:
    """Blueprint for a single specialist or refuter agent."""
    id: str = field(default_factory=lambda: _uid("ag"))
    name: str = ""
    role: str = "specialist"  # specialist | refuter
    domain_mandate: str = ""
    evidence_source_priority: list[str] = field(default_factory=list)
    epistemic_prior: float = 0.5
    evidence_type_focus: list[str] = field(default_factory=list)
    persona_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "domain_mandate": self.domain_mandate,
            "evidence_source_priority": self.evidence_source_priority,
            "epistemic_prior": self.epistemic_prior,
            "evidence_type_focus": self.evidence_type_focus,
            "persona_description": self.persona_description,
        }


@dataclass
class JurySpec:
    """Specification for the jury system."""
    architecture: JuryArchitecture = JuryArchitecture.BAYESIAN
    size: int = 3
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "size": self.size,
            "rationale": self.rationale,
        }


@dataclass
class TopologySpec:
    """
    Complete output of Layer 2 (topology.py).

    Everything needed to instantiate and run the debate: agents, jury,
    round count, cost estimate.
    """
    id: str = field(default_factory=lambda: _uid("ts"))
    frame_id: str = ""

    specialists: list[AgentSpec] = field(default_factory=list)
    refuters: list[AgentSpec] = field(default_factory=list)
    jury: JurySpec = field(default_factory=JurySpec)
    refuter_intensity: RefuterIntensity = RefuterIntensity.MEDIUM

    estimated_rounds: int = 3
    round_reasoning: str = ""
    max_rounds: int = 8
    min_rounds: int = 1

    estimated_tokens: int = 0
    estimated_runtime_seconds: float = 0.0
    budget_cap_tokens: int = 100_000

    created_at: datetime = field(default_factory=_utcnow)

    @property
    def agent_count(self) -> int:
        return len(self.specialists) + len(self.refuters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "frame_id": self.frame_id,
            "specialists": [s.to_dict() for s in self.specialists],
            "refuters": [r.to_dict() for r in self.refuters],
            "jury": self.jury.to_dict(),
            "refuter_intensity": self.refuter_intensity.value,
            "estimated_rounds": self.estimated_rounds,
            "round_reasoning": self.round_reasoning,
            "max_rounds": self.max_rounds,
            "estimated_tokens": self.estimated_tokens,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "budget_cap_tokens": self.budget_cap_tokens,
            "created_at": self.created_at.isoformat(),
        }

    def preview_card_text(self) -> str:
        """Plain-text topology preview for the chat interface."""
        lines = [
            "DEBATE TOPOLOGY — ARISTOTLE's Proposed Plan",
            "─" * 50,
            f"SPECIALISTS ({len(self.specialists)}):",
        ]
        for i, s in enumerate(self.specialists, 1):
            lines.append(
                f"  {i}. {s.name} — {s.domain_mandate} [Prior: {s.epistemic_prior:.2f}]"
            )
            if s.persona_description:
                lines.append(f"     {s.persona_description}")

        lines.append("")
        lines.append(
            f"REFUTERS: {len(self.refuters)} "
            f"({self.refuter_intensity.value.upper()} intensity)"
        )
        lines.append("")
        lines.append(
            f"JURY: {self.jury.size}-member {self.jury.architecture.value.replace('_', ' ').title()} Jury"
        )
        if self.jury.rationale:
            lines.append(f"  ({self.jury.rationale})")

        lines.append("")
        lines.append(
            f"ROUNDS: {self.estimated_rounds} estimated "
            f"({self.round_reasoning})"
        )
        lines.append(
            f"COST ESTIMATE: ~{self.estimated_tokens:,} tokens | "
            f"~{self.estimated_runtime_seconds:.0f}s runtime"
        )
        return "\n".join(lines)


# ── Layer 3 events ────────────────────────────────────────────────────

@dataclass
class MonitorEvent:
    """
    A single event emitted by the execution monitor (L3).
    Consumed by the interface (L4) for live visualisation updates.
    """
    id: str = field(default_factory=lambda: _uid("ev"))
    event_type: MonitorEventType = MonitorEventType.SESSION_START
    round_num: int = 0
    timestamp: datetime = field(default_factory=_utcnow)

    # Context — not every field is populated for every event type
    agent_id: str = ""
    agent_name: str = ""
    node_id: str = ""
    node_text: str = ""
    edge_id: str = ""
    posterior: float = 0.0
    evid_q: float = 0.0
    polarity: int = 0  # 1 support, -1 attack
    intervention_type: str = ""
    intervention_reason: str = ""
    chat_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "round_num": self.round_num,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "node_id": self.node_id,
            "posterior": self.posterior,
            "evid_q": self.evid_q,
            "polarity": self.polarity,
            "intervention_type": self.intervention_type,
            "chat_message": self.chat_message,
            "details": self.details,
        }


# ── Layer 5 outputs ───────────────────────────────────────────────────

@dataclass
class MetricReport:
    """A single metric with its score and plain-language interpretation."""
    name: str = ""
    full_name: str = ""
    score: float = 0.0
    interpretation: str = ""


@dataclass
class SynthesisResult:
    """
    Complete output of Layer 5 (synthesis.py).

    The user-facing result package delivered in the chat pane.
    """
    id: str = field(default_factory=lambda: _uid("sr"))
    frame_id: str = ""

    # verdict
    verdict_label: VerdictLevel = VerdictLevel.INSUFFICIENT
    jury_confidence: float = 0.5
    convergence_round: int = 0

    # reasoning narrative
    reasoning_narrative: str = ""

    # dissent log
    dissent_log: str = ""

    # What Could Change This
    what_could_change: list[str] = field(default_factory=list)

    # provenance
    evidence_provenance_map: dict[str, str] = field(default_factory=dict)

    # metric dashboard
    metrics: list[MetricReport] = field(default_factory=list)

    # raw data for export
    num_rounds: int = 0
    num_evidence: int = 0
    num_rebuttals: int = 0
    duration_seconds: float = 0.0

    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "frame_id": self.frame_id,
            "verdict_label": self.verdict_label.value,
            "jury_confidence": self.jury_confidence,
            "convergence_round": self.convergence_round,
            "reasoning_narrative": self.reasoning_narrative,
            "dissent_log": self.dissent_log,
            "what_could_change": self.what_could_change,
            "metrics": [
                {"name": m.name, "score": m.score, "interpretation": m.interpretation}
                for m in self.metrics
            ],
            "num_rounds": self.num_rounds,
            "num_evidence": self.num_evidence,
            "num_rebuttals": self.num_rebuttals,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat(),
        }

    def chat_card(self) -> str:
        """Formatted text for the left chat pane."""
        lines = [
            "━" * 50,
            f"VERDICT: {self.verdict_label.value.upper()}",
            f"Jury Confidence: {self.jury_confidence:.0%} | "
            f"Convergence: Round {self.convergence_round}",
            "━" * 50,
            "",
            self.reasoning_narrative,
        ]
        if self.dissent_log:
            lines += ["", "MINORITY VIEW:", self.dissent_log]
        if self.what_could_change:
            lines += ["", "WHAT COULD CHANGE THIS VERDICT:"]
            for i, item in enumerate(self.what_could_change, 1):
                lines.append(f"  {i}. {item}")
        if self.metrics:
            lines += ["", "METRIC DASHBOARD:"]
            for m in self.metrics:
                lines.append(f"  {m.name} = {m.score:.2f} — {m.interpretation}")
        return "\n".join(lines)
