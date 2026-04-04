"""
HANNIBAL data models — shared across all HANNIBAL components.

Every structured artefact produced or consumed by the Hierarchical Adversarial
Network for Nested Intelligence Battles and Logic is defined here so that
components communicate via well-typed, serialisable objects.

Design constraint: sequential execution on i3 / 8 GB RAM.
All skirmishes run one-at-a-time, single LLM instance reused.
"""

from __future__ import annotations

import uuid
import hashlib
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


# ── helpers ────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uid(prefix: str = "hn") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════

class ForceType(str, Enum):
    """Types of forces in a HANNIBAL campaign."""
    PROPOSITION = "proposition"        # PF — fights to prove TRUE
    OPPOSITION = "opposition"          # OF — fights to prove FALSE
    FACTION_1 = "faction_1"            # FF-1
    FACTION_2 = "faction_2"            # FF-2
    FACTION_3 = "faction_3"            # FF-3

    @property
    def abbreviation(self) -> str:
        _MAP = {
            "proposition": "PF",
            "opposition": "OF",
            "faction_1": "FF-1",
            "faction_2": "FF-2",
            "faction_3": "FF-3",
        }
        return _MAP.get(self.value, self.value[:2].upper())

    @property
    def display_name(self) -> str:
        _MAP = {
            "proposition": "Proposition Force",
            "opposition": "Opposition Force",
            "faction_1": "Faction Force 1",
            "faction_2": "Faction Force 2",
            "faction_3": "Faction Force 3",
        }
        return _MAP.get(self.value, self.value.replace("_", " ").title())

    @property
    def color_hex(self) -> str:
        _MAP = {
            "proposition": "#2ECC71",   # Emerald green
            "opposition": "#E74C3C",    # Crimson red
            "faction_1": "#3498DB",     # Blue
            "faction_2": "#E67E22",     # Orange
            "faction_3": "#9B59B6",     # Purple
        }
        return _MAP.get(self.value, "#555555")


class MilitaryRole(str, Enum):
    """Tactical roles within a Force."""
    COMMANDER = "commander"
    VANGUARD = "vanguard"
    FLANKING = "flanking"
    INTELLIGENCE_OFFICER = "intelligence_officer"
    RESERVE = "reserve"

    @property
    def abbreviation(self) -> str:
        _MAP = {
            "commander": "C",
            "vanguard": "V",
            "flanking": "F",
            "intelligence_officer": "IO",
            "reserve": "R",
        }
        return _MAP.get(self.value, self.value[0].upper())

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


class TournamentNodeType(str, Enum):
    """Types of nodes in the Tournament Tree."""
    SKIRMISH = "skirmish"
    ENGAGEMENT = "engagement"
    THEATRE = "theatre"
    CAMPAIGN_THEATRE = "campaign_theatre"
    CAMPAIGN_ROOT = "campaign_root"


class CampaignPhase(str, Enum):
    """Phases of a HANNIBAL campaign."""
    ANALYSIS = "analysis"
    DEPLOYMENT = "deployment"
    BATTLE = "battle"
    RESOLUTION = "resolution"
    ARMISTICE = "armistice"
    COMPLETE = "complete"


class VictoryStrength(str, Enum):
    """Campaign victory strength classification."""
    CONTESTED = "Contested"    # < 0.25
    NARROW = "Narrow"          # 0.25 – 0.50
    CLEAR = "Clear"            # 0.50 – 0.75
    DECISIVE = "Decisive"      # > 0.75

    @staticmethod
    def from_score(score: float) -> "VictoryStrength":
        if score < 0.25:
            return VictoryStrength.CONTESTED
        elif score < 0.50:
            return VictoryStrength.NARROW
        elif score < 0.75:
            return VictoryStrength.CLEAR
        else:
            return VictoryStrength.DECISIVE


class ArmisticeOption(str, Enum):
    """Options when the Armistice Protocol fires."""
    NARROW_VERDICT = "narrow_verdict"
    REDIRECT_AGORA = "redirect_agora"
    REDIRECT_ARISTOTLE = "redirect_aristotle"


class CampaignVerdictLabel(str, Enum):
    """Possible verdict labels for the campaign."""
    SUPPORTED = "Supported"
    CHALLENGED = "Challenged"
    QUALIFIED = "Qualified"
    ENCIRCLEMENT_CONCLUSION = "Encirclement Conclusion"


class PolarityStructure(str, Enum):
    """Polarity structure of a proposition."""
    BIPOLAR = "bipolar"
    TRIPOLAR = "tripolar"
    QUADRUPOLAR = "quadrupolar"

    @property
    def force_count(self) -> int:
        _MAP = {"bipolar": 2, "tripolar": 3, "quadrupolar": 4}
        return _MAP.get(self.value, 2)


class CampaignLogEventType(str, Enum):
    """Event types for the Campaign Log (Field Manual)."""
    FORCE_DEPLOYED = "force_deployed"
    SKIRMISH_INITIATED = "skirmish_initiated"
    EVIDENCE_SUBMITTED = "evidence_submitted"
    SKIRMISH_ADJUDICATED = "skirmish_adjudicated"
    ENGAGEMENT_RESOLVED = "engagement_resolved"
    THEATRE_DECIDED = "theatre_decided"
    RESERVE_DEPLOYED = "reserve_deployed"
    COMMANDER_DIRECTIVE = "commander_directive"
    FIELD_MARSHAL_RULING = "field_marshal_ruling"
    FORCE_POSTERIOR_UPDATE = "force_posterior_update"
    CANNAE_COMPUTATION = "cannae_computation"
    CAMPAIGN_RESOLVED = "campaign_resolved"
    ARMISTICE_TRIGGERED = "armistice_triggered"
    LOG_SEALED = "log_sealed"


class SkirmishPhase(str, Enum):
    """Phases within a single skirmish."""
    SCOPE_DECLARATION = "scope_declaration"
    INITIAL_DEPLOYMENT = "initial_deployment"
    COUNTEROFFENSIVE = "counteroffensive"
    FINAL_STRIKE = "final_strike"
    ADJUDICATION = "adjudication"


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HannibalSessionConfig:
    """User-tunable configuration for a HANNIBAL campaign.

    Designed for sequential execution on constrained hardware
    (i3 processor, 8 GB RAM). All skirmishes run one-at-a-time.
    """
    # Token budget (0 = unlimited)
    token_budget: int = 0

    # Skirmish config
    max_skirmish_rounds: int = 3        # 1–3 exchange rounds per skirmish
    high_depth_threshold: float = 0.65  # depth above this triggers 3rd round

    # Force sizing
    min_force_size: int = 3             # minimum agents per force
    max_force_size: int = 7             # capped for RAM constraint
    max_total_agents: int = 14          # total across all forces (i3 safe)

    # Tournament
    max_tree_height: int = 4            # max tree height (capped for perf)
    max_engagements_per_theatre: int = 3  # cap engagements (perf)
    max_skirmishes_per_engagement: int = 2  # cap skirmishes (perf)

    # Execution mode — always sequential on constrained hardware
    sequential_execution: bool = True

    # Armistice threshold
    armistice_threshold: float = 0.20

    # Confidence threshold for skirmish victory
    confidence_threshold: float = 0.05

    @property
    def is_unlimited_budget(self) -> bool:
        return self.token_budget <= 0


# ══════════════════════════════════════════════════════════════════════
# PDA Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EpistemicDepthScore:
    """Three-axis epistemic depth scoring for a proposition."""
    factual: float = 0.5       # 0.0–1.0
    normative: float = 0.5     # 0.0–1.0
    inferential: float = 0.5   # 0.0–1.0

    @property
    def aggregate(self) -> float:
        """Weighted aggregate: PDS = 0.4×F + 0.35×N + 0.25×I"""
        return (self.factual * 0.4 +
                self.normative * 0.35 +
                self.inferential * 0.25)

    @property
    def tree_height(self) -> int:
        """Determine tournament tree height from depth score."""
        agg = self.aggregate
        if agg < 0.35:
            return 2   # Quick Battle: Skirmish → Campaign
        elif agg < 0.65:
            return 3   # Skirmish → Engagement → Campaign
        elif agg < 0.85:
            return 4   # Skirmish → Engagement → Theatre → Campaign
        else:
            return 4   # Capped at 4 for i3 constraint (spec allows 5)

    def to_dict(self) -> dict[str, float]:
        return {
            "factual": self.factual,
            "normative": self.normative,
            "inferential": self.inferential,
            "aggregate": self.aggregate,
            "tree_height": self.tree_height,
        }


@dataclass
class TheatreSpec:
    """Specification for a single Theatre of operations."""
    id: str = field(default_factory=lambda: _uid("th"))
    name: str = ""
    topical_scope: str = ""
    engagement_count: int = 2
    epistemic_importance: float = 1.0  # weight for theatre-level scoring

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "topical_scope": self.topical_scope,
            "engagement_count": self.engagement_count,
            "epistemic_importance": self.epistemic_importance,
        }


@dataclass
class BattleMap:
    """Complete pre-battle specification produced by the PDA.

    Everything needed to construct the Tournament Tree and deploy Forces.
    """
    id: str = field(default_factory=lambda: _uid("bm"))
    proposition: str = ""

    # Polarity analysis
    polarity: PolarityStructure = PolarityStructure.BIPOLAR
    force_designations: list[ForceType] = field(default_factory=lambda: [
        ForceType.PROPOSITION, ForceType.OPPOSITION,
    ])

    # Depth analysis
    depth_score: EpistemicDepthScore = field(default_factory=EpistemicDepthScore)

    # Theatre specification
    theatres: list[TheatreSpec] = field(default_factory=list)

    # Tournament tree structure
    tree_height: int = 3
    estimated_skirmish_count: int = 4
    estimated_total_agents: int = 10

    # Force sizing
    force_sizes: dict[str, int] = field(default_factory=dict)

    # CANNAE activation flag
    cannae_activated: bool = False

    # Faction positions (for multipolar only)
    faction_positions: dict[str, str] = field(default_factory=dict)
    
    # Dynamic faction names mapping ForceType.value -> str
    faction_names: dict[str, str] = field(default_factory=dict)

    created_at: datetime = field(default_factory=_utcnow)

    @property
    def num_forces(self) -> int:
        return len(self.force_designations)

    @property
    def num_theatres(self) -> int:
        return len(self.theatres)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "polarity": self.polarity.value,
            "force_designations": [f.value for f in self.force_designations],
            "depth_score": self.depth_score.to_dict(),
            "theatres": [t.to_dict() for t in self.theatres],
            "tree_height": self.tree_height,
            "estimated_skirmish_count": self.estimated_skirmish_count,
            "estimated_total_agents": self.estimated_total_agents,
            "force_sizes": self.force_sizes,
            "cannae_activated": self.cannae_activated,
            "faction_positions": self.faction_positions,
            "faction_names": self.faction_names,
            "num_forces": self.num_forces,
            "num_theatres": self.num_theatres,
            "created_at": self.created_at.isoformat(),
        }

    def preview_card_text(self) -> str:
        """Plain-text Battle Map preview for the War Room interface."""
        lines = [
            "BATTLE MAP — HANNIBAL's Campaign Plan",
            "─" * 55,
            f"PROPOSITION: {self.proposition[:120]}",
            f"POLARITY: {self.polarity.value.upper()} ({self.num_forces} forces)",
            f"DEPTH SCORE: factual={self.depth_score.factual:.2f} | "
            f"normative={self.depth_score.normative:.2f} | "
            f"inferential={self.depth_score.inferential:.2f} "
            f"(aggregate={self.depth_score.aggregate:.2f})",
            f"TREE HEIGHT: {self.tree_height}",
            "",
            "FORCES:",
        ]
        for ft in self.force_designations:
            size = self.force_sizes.get(ft.value, 0)
            lines.append(f"  {ft.abbreviation} — {ft.display_name} ({size} agents)")
            if ft.value in self.faction_positions:
                lines.append(f"     Position: {self.faction_positions[ft.value]}")

        lines.append("")
        lines.append(f"THEATRES ({self.num_theatres}):")
        for i, th in enumerate(self.theatres, 1):
            lines.append(
                f"  Theatre {i}: {th.name} — {th.topical_scope[:80]}"
                f" [{th.engagement_count} engagements]"
            )

        cannae_str = "YES ⚔" if self.cannae_activated else "NO"
        lines.append("")
        lines.append(f"CANNAE ENGINE: {cannae_str}")
        lines.append(f"ESTIMATED SKIRMISHES: {self.estimated_skirmish_count}")
        lines.append(f"TOTAL AGENTS: {self.estimated_total_agents}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Force Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AgentRoleSpec:
    """Blueprint for a single agent within a Force."""
    id: str = field(default_factory=lambda: _uid("ag"))
    name: str = ""
    role: MilitaryRole = MilitaryRole.VANGUARD
    force_type: ForceType = ForceType.PROPOSITION
    domain_expertise: str = ""
    epistemic_prior: float = 0.5
    evidence_source_priority: list[str] = field(default_factory=list)
    evidence_type_focus: list[str] = field(default_factory=list)
    persona_description: str = ""
    assigned_theatre_id: str = ""  # for vanguards
    is_deployed: bool = True       # False for reserves

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "role_abbr": self.role.abbreviation,
            "force_type": self.force_type.value,
            "domain_expertise": self.domain_expertise,
            "epistemic_prior": self.epistemic_prior,
            "evidence_source_priority": self.evidence_source_priority,
            "evidence_type_focus": self.evidence_type_focus,
            "persona_description": self.persona_description,
            "assigned_theatre_id": self.assigned_theatre_id,
            "is_deployed": self.is_deployed,
        }


@dataclass
class ForceSpec:
    """Complete specification for a deployed Force."""
    id: str = field(default_factory=lambda: _uid("fs"))
    force_type: ForceType = ForceType.PROPOSITION
    force_name: str = ""
    position_description: str = ""
    agents: list[AgentRoleSpec] = field(default_factory=list)
    force_prior: float = 0.5
    force_posterior: float = 0.5

    @property
    def commander(self) -> Optional[AgentRoleSpec]:
        for a in self.agents:
            if a.role == MilitaryRole.COMMANDER:
                return a
        return None

    @property
    def vanguards(self) -> list[AgentRoleSpec]:
        return [a for a in self.agents if a.role == MilitaryRole.VANGUARD]

    @property
    def flanking_agents(self) -> list[AgentRoleSpec]:
        return [a for a in self.agents if a.role == MilitaryRole.FLANKING]

    @property
    def intelligence_officer(self) -> Optional[AgentRoleSpec]:
        for a in self.agents:
            if a.role == MilitaryRole.INTELLIGENCE_OFFICER:
                return a
        return None

    @property
    def reserves(self) -> list[AgentRoleSpec]:
        return [a for a in self.agents if a.role == MilitaryRole.RESERVE]

    @property
    def deployed_agents(self) -> list[AgentRoleSpec]:
        return [a for a in self.agents if a.is_deployed]

    @property
    def force_size(self) -> int:
        return len(self.agents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "force_type": self.force_type.value,
            "force_name": self.force_name or self.force_type.display_name,
            "force_type_display": self.force_type.display_name,
            "position_description": self.position_description,
            "agents": [a.to_dict() for a in self.agents],
            "force_prior": self.force_prior,
            "force_posterior": self.force_posterior,
            "force_size": self.force_size,
        }


@dataclass
class CommanderDirective:
    """A tactical order issued by a Force Commander."""
    id: str = field(default_factory=lambda: _uid("cd"))
    force_type: ForceType = ForceType.PROPOSITION
    directive_text: str = ""
    target_agent_ids: list[str] = field(default_factory=list)
    priority_theatre_id: str = ""
    deploy_reserve: bool = False
    reserve_agent_id: str = ""
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "force_type": self.force_type.value,
            "directive_text": self.directive_text,
            "target_agent_ids": self.target_agent_ids,
            "priority_theatre_id": self.priority_theatre_id,
            "deploy_reserve": self.deploy_reserve,
            "created_at": self.created_at.isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════
# Evidence & Battle Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceItem:
    """A single piece of evidence deployed in a skirmish."""
    id: str = field(default_factory=lambda: _uid("ei"))
    agent_id: str = ""
    agent_name: str = ""
    force_type: ForceType = ForceType.PROPOSITION
    claim_text: str = ""
    source_reference: str = ""
    evid_q: float = 0.5            # EVID-Q quality score 0.0–1.0
    confidence: float = 0.5
    relevance: float = 0.5
    polarity_strength: float = 1.0   # how strongly it supports/attacks
    is_counter_evidence: bool = False
    skirmish_round: int = 1          # which round submitted
    submitted_at: datetime = field(default_factory=_utcnow)

    @property
    def effective_weight(self) -> float:
        """EVID-Q × Confidence × Relevance × Polarity_strength."""
        return self.evid_q * self.confidence * self.relevance * self.polarity_strength

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "force_type": self.force_type.value,
            "claim_text": self.claim_text,
            "source_reference": self.source_reference,
            "evid_q": self.evid_q,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "polarity_strength": self.polarity_strength,
            "effective_weight": self.effective_weight,
            "is_counter_evidence": self.is_counter_evidence,
            "skirmish_round": self.skirmish_round,
            "submitted_at": self.submitted_at.isoformat(),
        }


@dataclass
class SkirmishResult:
    """Result of a single skirmish adjudication."""
    skirmish_id: str = ""
    winner_force: ForceType = ForceType.PROPOSITION
    loser_force: ForceType = ForceType.OPPOSITION
    ecs_winner: float = 0.0
    ecs_loser: float = 0.0
    confidence_score: float = 0.0   # |ECS_A - ECS_B| / max(ECS_A, ECS_B)
    is_draw: bool = False
    evidence_a: list[EvidenceItem] = field(default_factory=list)
    evidence_b: list[EvidenceItem] = field(default_factory=list)
    decisive_evidence_ids: list[str] = field(default_factory=list)
    adjudication_summary: str = ""
    rounds_played: int = 2

    @property
    def posterior_delta(self) -> float:
        """ECS delta for force posterior update."""
        return self.confidence_score if not self.is_draw else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skirmish_id": self.skirmish_id,
            "winner_force": self.winner_force.value,
            "loser_force": self.loser_force.value,
            "ecs_winner": self.ecs_winner,
            "ecs_loser": self.ecs_loser,
            "confidence_score": self.confidence_score,
            "is_draw": self.is_draw,
            "evidence_a": [e.to_dict() for e in self.evidence_a],
            "evidence_b": [e.to_dict() for e in self.evidence_b],
            "decisive_evidence_ids": self.decisive_evidence_ids,
            "adjudication_summary": self.adjudication_summary,
            "rounds_played": self.rounds_played,
        }


@dataclass
class EngagementResult:
    """Result of an engagement (aggregation of skirmishes)."""
    engagement_id: str = ""
    winner_force: ForceType = ForceType.PROPOSITION
    margin: float = 0.0
    skirmish_results: list[SkirmishResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engagement_id": self.engagement_id,
            "winner_force": self.winner_force.value,
            "margin": self.margin,
            "skirmish_count": len(self.skirmish_results),
        }


@dataclass
class TheatreResult:
    """Result of a theatre (aggregation of engagements)."""
    theatre_id: str = ""
    theatre_name: str = ""
    winner_force: ForceType = ForceType.PROPOSITION
    theatre_score: float = 0.0
    engagement_results: list[EngagementResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theatre_id": self.theatre_id,
            "theatre_name": self.theatre_name,
            "winner_force": self.winner_force.value,
            "theatre_score": self.theatre_score,
            "engagement_count": len(self.engagement_results),
        }


@dataclass
class ForcePosteriorUpdate:
    """A single update to a Force's posterior belief."""
    force_type: ForceType = ForceType.PROPOSITION
    prior_value: float = 0.5
    new_value: float = 0.5
    delta: float = 0.0
    trigger_skirmish_id: str = ""
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_type": self.force_type.value,
            "prior_value": self.prior_value,
            "new_value": self.new_value,
            "delta": self.delta,
            "trigger_skirmish_id": self.trigger_skirmish_id,
            "timestamp": self.timestamp.isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════
# Tournament Tree Node Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TournamentNode:
    """A single node in the Tournament Tree."""
    id: str = field(default_factory=lambda: _uid("tn"))
    node_type: TournamentNodeType = TournamentNodeType.SKIRMISH
    parent_id: str = ""
    child_ids: list[str] = field(default_factory=list)
    label: str = ""

    # Results (filled as battle progresses)
    winner_force: Optional[ForceType] = None
    confidence: float = 0.0
    margin: float = 0.0
    is_resolved: bool = False

    # Skirmish-specific
    force_a_type: Optional[ForceType] = None
    force_b_type: Optional[ForceType] = None
    agent_a_id: str = ""
    agent_b_id: str = ""
    topic_scope: str = ""

    # Engagement-specific
    theatre_id: str = ""
    topic_cluster: str = ""

    # Theatre-specific
    topical_scope: str = ""
    engagement_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "label": self.label,
            "winner_force": self.winner_force.value if self.winner_force else None,
            "confidence": self.confidence,
            "margin": self.margin,
            "is_resolved": self.is_resolved,
            "force_a_type": self.force_a_type.value if self.force_a_type else None,
            "force_b_type": self.force_b_type.value if self.force_b_type else None,
            "topic_scope": self.topic_scope,
            "topical_scope": self.topical_scope,
        }


# ══════════════════════════════════════════════════════════════════════
# Campaign Log Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CampaignLogEntry:
    """A single entry in the Campaign Log (Field Manual)."""
    id: str = field(default_factory=lambda: _uid("cl"))
    event_type: CampaignLogEventType = CampaignLogEventType.FORCE_DEPLOYED
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            data = f"{self.id}:{self.event_type.value}:{self.content}:{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
        }

    def to_field_manual_line(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        label = self.event_type.value.upper().replace("_", " ")
        return f"[{ts}] [{label}] {self.content}"


# ══════════════════════════════════════════════════════════════════════
# Result Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CampaignVerdict:
    """The primary output of a HANNIBAL campaign."""
    verdict_label: CampaignVerdictLabel = CampaignVerdictLabel.SUPPORTED
    winning_force: ForceType = ForceType.PROPOSITION
    position_description: str = ""
    campaign_strength_score: float = 0.0
    campaign_strength_label: VictoryStrength = VictoryStrength.CONTESTED
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_label": self.verdict_label.value,
            "winning_force": self.winning_force.value,
            "position_description": self.position_description,
            "campaign_strength_score": self.campaign_strength_score,
            "campaign_strength_label": self.campaign_strength_label.value,
            "narrative": self.narrative,
        }


@dataclass
class CampaignMinorityRecord:
    """The losing Force's strongest surviving arguments."""
    losing_force: ForceType = ForceType.OPPOSITION
    surviving_arguments: list[str] = field(default_factory=list)
    sustained_challenges: list[str] = field(default_factory=list)
    conditions_to_prevail: list[str] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "losing_force": self.losing_force.value,
            "surviving_arguments": self.surviving_arguments,
            "sustained_challenges": self.sustained_challenges,
            "conditions_to_prevail": self.conditions_to_prevail,
            "narrative": self.narrative,
        }


@dataclass
class ForcePerformanceScorecard:
    """Per-Force performance metrics."""
    force_type: ForceType = ForceType.PROPOSITION
    skirmishes_won: int = 0
    skirmishes_lost: int = 0
    skirmishes_drawn: int = 0
    engagements_won: int = 0
    evidence_submitted: int = 0
    avg_evid_q: float = 0.0
    flanking_attack_success_rate: float = 0.0
    reserve_deployments: int = 0
    commander_interventions: int = 0
    battle_efficiency_score: float = 0.0  # BES

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_type": self.force_type.value,
            "force_display": self.force_type.display_name,
            "skirmishes_won": self.skirmishes_won,
            "skirmishes_lost": self.skirmishes_lost,
            "skirmishes_drawn": self.skirmishes_drawn,
            "engagements_won": self.engagements_won,
            "evidence_submitted": self.evidence_submitted,
            "avg_evid_q": self.avg_evid_q,
            "flanking_attack_success_rate": self.flanking_attack_success_rate,
            "reserve_deployments": self.reserve_deployments,
            "commander_interventions": self.commander_interventions,
            "battle_efficiency_score": self.battle_efficiency_score,
        }


@dataclass
class DecisiveEvidenceRecord:
    """Top evidence items that most determined the campaign outcome."""
    items: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "count": len(self.items),
        }


@dataclass
class HannibalResult:
    """Complete output of a HANNIBAL campaign — all nine result components.

    Components:
        1. Campaign Verdict
        2. Battle Map Summary
        3. Campaign Minority Record
        4. Force Performance Scorecards
        5. Decisive Evidence Record
        6. Encirclement Report (CANNAE only)
        7. What Would Change This Verdict
        8. Armistice Record
        9. Campaign Log (Field Manual)
    """
    id: str = field(default_factory=lambda: _uid("hr"))
    proposition: str = ""
    battle_map_id: str = ""

    # 1. Campaign Verdict
    verdict: CampaignVerdict = field(default_factory=CampaignVerdict)

    # 2. Battle Map Summary (serialized tree state)
    battle_map_summary: dict[str, Any] = field(default_factory=dict)

    # 3. Campaign Minority Record
    minority_record: CampaignMinorityRecord = field(default_factory=CampaignMinorityRecord)

    # 4. Force Performance Scorecards
    scorecards: list[ForcePerformanceScorecard] = field(default_factory=list)

    # 5. Decisive Evidence Record
    decisive_evidence: DecisiveEvidenceRecord = field(default_factory=DecisiveEvidenceRecord)

    # 6. Encirclement Report (CANNAE campaigns only)
    encirclement_report: dict[str, Any] = field(default_factory=dict)

    # 7. What Would Change This Verdict
    what_would_change: list[str] = field(default_factory=list)

    # 8. Armistice Record
    armistice_fired: bool = False
    armistice_option: Optional[ArmisticeOption] = None
    armistice_details: str = ""

    # 9. Campaign Log
    campaign_log: list[CampaignLogEntry] = field(default_factory=list)
    log_seal_hash: str = ""

    # Force posterior history
    force_posterior_history: dict[str, list[float]] = field(default_factory=dict)

    # Session metadata
    num_skirmishes: int = 0
    num_engagements: int = 0
    num_theatres: int = 0
    total_evidence: int = 0
    duration_seconds: float = 0.0
    total_tokens_used: int = 0

    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "battle_map_id": self.battle_map_id,
            "verdict": self.verdict.to_dict(),
            "battle_map_summary": self.battle_map_summary,
            "minority_record": self.minority_record.to_dict(),
            "scorecards": [s.to_dict() for s in self.scorecards],
            "decisive_evidence": self.decisive_evidence.to_dict(),
            "encirclement_report": self.encirclement_report,
            "what_would_change": self.what_would_change,
            "armistice_fired": self.armistice_fired,
            "armistice_option": self.armistice_option.value if self.armistice_option else None,
            "armistice_details": self.armistice_details,
            "campaign_log": [e.to_dict() for e in self.campaign_log],
            "log_seal_hash": self.log_seal_hash,
            "force_posterior_history": self.force_posterior_history,
            "num_skirmishes": self.num_skirmishes,
            "num_engagements": self.num_engagements,
            "num_theatres": self.num_theatres,
            "total_evidence": self.total_evidence,
            "duration_seconds": self.duration_seconds,
            "total_tokens_used": self.total_tokens_used,
            "created_at": self.created_at.isoformat(),
        }

    def chat_card(self) -> str:
        """Formatted text for the War Room command pane."""
        v = self.verdict
        strength = v.campaign_strength_label.value
        lines = [
            "━" * 55,
            f"CAMPAIGN VERDICT: {v.verdict_label.value.upper()}",
            f"Winner: {v.winning_force.display_name} | "
            f"Strength: {strength} ({v.campaign_strength_score:.0%})",
            f"Skirmishes: {self.num_skirmishes} | "
            f"Evidence: {self.total_evidence} | "
            f"Duration: {self.duration_seconds:.0f}s",
            "━" * 55,
            "",
            v.narrative,
        ]
        if self.minority_record.narrative:
            lines += ["", "CAMPAIGN MINORITY RECORD:", self.minority_record.narrative]
        if self.scorecards:
            lines += ["", "FORCE PERFORMANCE:"]
            for sc in self.scorecards:
                w, l = sc.skirmishes_won, sc.skirmishes_lost
                lines.append(
                    f"  {sc.force_type.abbreviation} — "
                    f"W:{w} L:{l} | Evidence: {sc.evidence_submitted} | "
                    f"BES: {sc.battle_efficiency_score:.3f}"
                )
        if self.what_would_change:
            lines += ["", "WHAT WOULD CHANGE THIS VERDICT:"]
            for i, item in enumerate(self.what_would_change, 1):
                lines.append(f"  {i}. {item}")
        if self.armistice_fired:
            lines += [
                "",
                f"⚠ ARMISTICE PROTOCOL FIRED: {self.armistice_option.value if self.armistice_option else 'N/A'}",
                f"  {self.armistice_details}",
            ]
        return "\n".join(lines)
