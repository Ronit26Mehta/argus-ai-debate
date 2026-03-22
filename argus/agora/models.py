"""
AGORA data models — shared across all AGORA components.

Every structured artefact produced or consumed by AGORA is defined here
so that components communicate via well-typed, serialisable objects.

The Autonomous Governed Open Reasoning Assembly (AGORA) is a procedurally
governed, dynamically composed, real-time deliberative body for multi-agent
debate within the ARGUS framework.
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


def _uid(prefix: str = "ag") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════

class SenatorCategory(str, Enum):
    """Eight epistemic categories for AGORA senators."""
    DOMAIN_EXPERT = "domain_expert"                # DE
    ADVERSARIAL_CHALLENGER = "adversarial_challenger"  # AC
    SYNTHESIS_AGENT = "synthesis_agent"              # SA
    NORMATIVE_ANALYST = "normative_analyst"          # NA
    HISTORICAL_CONTEXTUALIST = "historical_contextualist"  # HC
    DEVILS_ADVOCATE = "devils_advocate"              # DA
    EPISTEMIC_AUDITOR = "epistemic_auditor"          # EA
    CROSS_DOMAIN_INTEGRATOR = "cross_domain_integrator"   # CDI

    @property
    def abbreviation(self) -> str:
        _MAP = {
            "domain_expert": "DE",
            "adversarial_challenger": "AC",
            "synthesis_agent": "SA",
            "normative_analyst": "NA",
            "historical_contextualist": "HC",
            "devils_advocate": "DA",
            "epistemic_auditor": "EA",
            "cross_domain_integrator": "CDI",
        }
        return _MAP.get(self.value, self.value[:2].upper())

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


class SessionPhase(str, Enum):
    """Five formal phases of an AGORA session."""
    OPENING_STATEMENTS = "opening_statements"
    EVIDENCE_SUBMISSION = "evidence_submission"
    CROSS_EXAMINATION = "cross_examination"
    DELIBERATIVE_SYNTHESIS = "deliberative_synthesis"
    CLOSING_AND_VERDICT = "closing_and_verdict"

    @property
    def phase_number(self) -> int:
        _ORDER = {
            "opening_statements": 1,
            "evidence_submission": 2,
            "cross_examination": 3,
            "deliberative_synthesis": 4,
            "closing_and_verdict": 5,
        }
        return _ORDER[self.value]


class StoppingTrigger(str, Enum):
    """Five user-governed stopping triggers."""
    FULL_EVIDENCE = "full_evidence"        # All senators submitted minimum evidence
    TIME_BOUNDARY = "time_boundary"        # Max session duration reached
    CONVERGENCE = "convergence"            # Posterior variance below threshold
    USER_DEMAND = "user_demand"            # User pressed the gavel
    QUORUM_FAILURE = "quorum_failure"      # Minimum participation not met
    UNBOUNDED = "unbounded"               # Let agents complete all phases naturally


class EvidencePolarity(str, Enum):
    """Polarity of evidence relative to the proposition."""
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    QUALIFIES = "qualifies"


class DocketEvidenceType(str, Enum):
    """Type classification for docket evidence items."""
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    HISTORICAL = "historical"
    THEORETICAL = "theoretical"
    ANECDOTAL = "anecdotal"
    LEGAL = "legal"
    EXPERIMENTAL = "experimental"


class ChallengeType(str, Enum):
    """Type of formal evidence challenge."""
    CLAIM = "claim"           # The assertion is wrong
    SOURCE = "source"         # The source is unreliable or misrepresented
    INFERENCE = "inference"   # The relationship between evidence and proposition is invalid
    CONFIDENCE = "confidence" # The confidence score is inflated


class ChallengeOutcome(str, Enum):
    """Outcome of a formal challenge ruled by the Epistemic Auditor."""
    SUSTAINED = "sustained"     # Challenge upheld — evidence downweighted
    OVERRULED = "overruled"     # Challenge rejected — evidence upgraded
    NOTED = "noted"             # Noted for synthesis — original weight with flag


class VerdictLabel(str, Enum):
    """Possible verdict labels for the majority opinion."""
    SUPPORTED = "Supported"
    CHALLENGED = "Challenged"
    INDETERMINATE = "Indeterminate"
    QUALIFIED = "Qualified"


class RecordEntryType(str, Enum):
    """Entry types for the Senate Record (Hansard)."""
    SENATOR_STATEMENT = "senator_statement"
    EVIDENCE_SUBMISSION = "evidence_submission"
    CHALLENGE_ISSUED = "challenge_issued"
    CHALLENGE_REPLY = "challenge_reply"
    EA_RULING = "ea_ruling"
    POINT_OF_ORDER = "point_of_order"
    SOCRATIC_ACTION = "socratic_action"
    FLOOR_TIME_EVENT = "floor_time_event"
    COALITION_DETECTED = "coalition_detected"
    PHASE_TRANSITION = "phase_transition"
    QUORUM_UPDATE = "quorum_update"
    STOPPING_TRIGGER = "stopping_trigger"
    FINAL_POSITIONS = "final_positions"
    MAJORITY_OPINION = "majority_opinion"
    MINORITY_REPORT = "minority_report"
    RECORD_SEALED = "record_sealed"


class ControversyAxis(str, Enum):
    """Three axes of controversy scoring."""
    EMPIRICAL = "empirical"
    NORMATIVE = "normative"
    EPISTEMIC = "epistemic"


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AgoraSessionConfig:
    """User-tunable configuration for an AGORA session.

    Controls stopping triggers, round limits, time boundaries,
    convergence thresholds, and senate sizing.
    """
    # Stopping triggers — which ones are active
    active_triggers: list[StoppingTrigger] = field(default_factory=lambda: [
        StoppingTrigger.UNBOUNDED,
    ])

    # Round configuration — rounds per deliberation phase
    max_rounds: int = 5                  # max rounds within evidence/cross-exam phases
    min_evidence_per_senator: int = 3    # minimum evidence items before FULL_EVIDENCE trigger

    # Time boundary (seconds) — 0 or None means unbounded
    time_limit_seconds: Optional[float] = None

    # Convergence threshold (posterior standard deviation)
    convergence_threshold: float = 0.15

    # Senate sizing overrides (None = auto-calculate)
    min_senators: int = 7
    max_senators: int = 25

    # Coalition detection threshold
    coalition_similarity_threshold: float = 0.75

    # Quorum — minimum fraction of senators that must participate
    quorum_fraction: float = 0.60

    # Token budget cap (0 = unlimited)
    token_budget: int = 0

    @property
    def is_unbounded(self) -> bool:
        """Check if session runs until natural completion (no time limit)."""
        return (
            StoppingTrigger.UNBOUNDED in self.active_triggers
            or self.time_limit_seconds is None
            or self.time_limit_seconds <= 0
        )


# ══════════════════════════════════════════════════════════════════════
# Senate Specification Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ControversyVector:
    """Three-axis controversy profile for a proposition."""
    empirical: float = 0.5    # 0.0–1.0 — is the factual record contested?
    normative: float = 0.5    # 0.0–1.0 — do reasonable people disagree on values?
    epistemic: float = 0.5    # 0.0–1.0 — is the evidence uncertain or incomplete?

    @property
    def aggregate(self) -> float:
        """Aggregate controversy score (weighted average)."""
        return (self.empirical * 0.4 + self.normative * 0.3 + self.epistemic * 0.3)

    def to_dict(self) -> dict[str, float]:
        return {
            "empirical": self.empirical,
            "normative": self.normative,
            "epistemic": self.epistemic,
            "aggregate": self.aggregate,
        }


@dataclass
class StancePosition:
    """A single position in the stance space."""
    label: str = ""
    description: str = ""
    estimated_support: float = 0.5


@dataclass
class EvidenceLandscape:
    """Assessment of available evidence types for the proposition."""
    available_types: list[DocketEvidenceType] = field(default_factory=list)
    density: str = "moderate"  # sparse | moderate | rich
    density_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_types": [t.value for t in self.available_types],
            "density": self.density,
            "density_score": self.density_score,
        }


@dataclass
class SenatorSpec:
    """Blueprint for a single AGORA senator.

    Contains everything needed to instantiate and configure one
    deliberating agent within the Senate.
    """
    id: str = field(default_factory=lambda: _uid("sn"))
    name: str = ""
    category: SenatorCategory = SenatorCategory.DOMAIN_EXPERT
    domain_expertise: str = ""
    prior_position: float = 0.5              # 0.0–1.0 prior probability
    evidence_gathering_mandate: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    deliberative_temperament: str = "measured"  # aggressive | measured | cautious
    floor_time_budget: int = 5               # number of actions allowed
    cross_exam_authority: list[str] = field(default_factory=list)  # senator IDs can challenge
    persona_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "category_abbreviation": self.category.abbreviation,
            "domain_expertise": self.domain_expertise,
            "prior_position": self.prior_position,
            "evidence_gathering_mandate": self.evidence_gathering_mandate,
            "evidence_sources": self.evidence_sources,
            "deliberative_temperament": self.deliberative_temperament,
            "floor_time_budget": self.floor_time_budget,
            "persona_description": self.persona_description,
        }


@dataclass
class SenateSpec:
    """Senate Composition Specification — fully specifies every senator.

    Output of the Senate Generation Engine. Displayed to the user as a
    Senate Preview Card before the session begins.
    """
    id: str = field(default_factory=lambda: _uid("ss"))
    proposition: str = ""
    primary_domain: str = "general"
    secondary_domains: list[str] = field(default_factory=list)
    controversy: ControversyVector = field(default_factory=ControversyVector)
    evidence_landscape: EvidenceLandscape = field(default_factory=EvidenceLandscape)
    stance_space: list[StancePosition] = field(default_factory=list)

    senators: list[SenatorSpec] = field(default_factory=list)
    n_calculation_reasoning: str = ""

    estimated_tokens: int = 0
    estimated_runtime_seconds: float = 0.0
    created_at: datetime = field(default_factory=_utcnow)

    @property
    def senate_size(self) -> int:
        return len(self.senators)

    @property
    def category_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for s in self.senators:
            dist[s.category.value] = dist.get(s.category.value, 0) + 1
        return dist

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "primary_domain": self.primary_domain,
            "secondary_domains": self.secondary_domains,
            "controversy": self.controversy.to_dict(),
            "evidence_landscape": self.evidence_landscape.to_dict(),
            "stance_space": [
                {"label": s.label, "description": s.description,
                 "estimated_support": s.estimated_support}
                for s in self.stance_space
            ],
            "senators": [s.to_dict() for s in self.senators],
            "senate_size": self.senate_size,
            "category_distribution": self.category_distribution,
            "n_calculation_reasoning": self.n_calculation_reasoning,
            "estimated_tokens": self.estimated_tokens,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "created_at": self.created_at.isoformat(),
        }

    def preview_card_text(self) -> str:
        """Plain-text Senate Preview Card for the interface."""
        lines = [
            "SENATE COMPOSITION — AGORA's Proposed Assembly",
            "─" * 55,
            f"PROPOSITION: {self.proposition[:100]}",
            f"DOMAIN: {self.primary_domain}",
            f"CONTROVERSY: empirical={self.controversy.empirical:.2f} | "
            f"normative={self.controversy.normative:.2f} | "
            f"epistemic={self.controversy.epistemic:.2f}",
            f"SENATE SIZE: {self.senate_size} senators",
            "",
        ]

        # Group by category
        by_category: dict[str, list[SenatorSpec]] = {}
        for s in self.senators:
            by_category.setdefault(s.category.display_name, []).append(s)

        for cat_name, members in by_category.items():
            lines.append(f"{cat_name.upper()} ({len(members)}):")
            for i, m in enumerate(members, 1):
                lines.append(
                    f"  {i}. {m.name} — {m.domain_expertise} "
                    f"[Prior: {m.prior_position:.2f}]"
                )
                if m.persona_description:
                    lines.append(f"     {m.persona_description}")
            lines.append("")

        dist = self.category_distribution
        lines.append("BALANCE CHECK:")
        total = max(self.senate_size, 1)
        for cat, count in sorted(dist.items()):
            pct = count / total * 100
            lines.append(f"  {cat}: {count} ({pct:.0f}%)")

        lines.append("")
        lines.append(f"N REASONING: {self.n_calculation_reasoning}")
        lines.append(
            f"COST ESTIMATE: ~{self.estimated_tokens:,} tokens | "
            f"~{self.estimated_runtime_seconds:.0f}s runtime"
        )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Evidence Docket Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DocketItem:
    """A single evidence submission in the formal docket.

    Every piece of evidence has a submission ID, submitting senator,
    timestamp, domain classification, and challenge status.
    """
    id: str = field(default_factory=lambda: _uid("ev"))
    senator_id: str = ""
    senator_name: str = ""
    senator_category: SenatorCategory = SenatorCategory.DOMAIN_EXPERT

    claim_text: str = ""
    polarity: EvidencePolarity = EvidencePolarity.SUPPORTS
    source_reference: str = ""
    source_type: str = ""  # primary_research, secondary_analysis, etc.
    confidence_score: float = 0.5
    evidence_type: DocketEvidenceType = DocketEvidenceType.QUALITATIVE
    relationship_to_prior: str = ""  # corroborates / contradicts / extends

    # Dynamic Evidence Weight (calculated)
    dew_score: float = 0.5

    # Challenge status
    is_challenged: bool = False
    challenge_outcome: Optional[ChallengeOutcome] = None

    # Cross-corroboration
    corroborating_items: list[str] = field(default_factory=list)  # other DocketItem IDs

    # Timestamps
    submitted_at: datetime = field(default_factory=_utcnow)

    # Provenance hash
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.senator_id}:{self.claim_text}:{self.submitted_at.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def evidence_id_display(self) -> str:
        """Display ID with category prefix (e.g., DE-07)."""
        cat_abbr = self.senator_category.abbreviation
        return f"{cat_abbr}-{self.id[-4:]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "evidence_id_display": self.evidence_id_display,
            "senator_id": self.senator_id,
            "senator_name": self.senator_name,
            "senator_category": self.senator_category.value,
            "claim_text": self.claim_text,
            "polarity": self.polarity.value,
            "source_reference": self.source_reference,
            "source_type": self.source_type,
            "confidence_score": self.confidence_score,
            "evidence_type": self.evidence_type.value,
            "relationship_to_prior": self.relationship_to_prior,
            "dew_score": self.dew_score,
            "is_challenged": self.is_challenged,
            "challenge_outcome": self.challenge_outcome.value if self.challenge_outcome else None,
            "corroborating_items": self.corroborating_items,
            "submitted_at": self.submitted_at.isoformat(),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class Challenge:
    """A formal challenge to a docket evidence item."""
    id: str = field(default_factory=lambda: _uid("ch"))
    challenger_id: str = ""
    challenger_name: str = ""
    target_evidence_id: str = ""
    challenge_type: ChallengeType = ChallengeType.CLAIM
    challenge_argument: str = ""
    counter_evidence_id: Optional[str] = None  # optional supporting counter-evidence

    # Reply from submitting senator
    reply_text: str = ""
    reply_submitted: bool = False

    # EA ruling
    outcome: Optional[ChallengeOutcome] = None
    ea_reasoning: str = ""

    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "challenger_id": self.challenger_id,
            "challenger_name": self.challenger_name,
            "target_evidence_id": self.target_evidence_id,
            "challenge_type": self.challenge_type.value,
            "challenge_argument": self.challenge_argument,
            "counter_evidence_id": self.counter_evidence_id,
            "reply_text": self.reply_text,
            "reply_submitted": self.reply_submitted,
            "outcome": self.outcome.value if self.outcome else None,
            "ea_reasoning": self.ea_reasoning,
            "created_at": self.created_at.isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════
# Coalition Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CoalitionInfo:
    """A detected coalition of senators with shared epistemic premises."""
    id: str = field(default_factory=lambda: _uid("co"))
    name: str = ""                           # generated descriptor
    member_ids: list[str] = field(default_factory=list)
    member_names: list[str] = field(default_factory=list)
    shared_premise: str = ""
    epistemic_independence_score: float = 0.0  # 0.0 = fully dependent, 1.0 = fully independent
    similarity_score: float = 0.0
    detected_at: datetime = field(default_factory=_utcnow)

    @property
    def size(self) -> int:
        return len(self.member_ids)

    @property
    def is_low_independence(self) -> bool:
        """Flag if coalition strength may be artifact of shared evidence."""
        return self.epistemic_independence_score < 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "member_ids": self.member_ids,
            "member_names": self.member_names,
            "shared_premise": self.shared_premise,
            "epistemic_independence_score": self.epistemic_independence_score,
            "similarity_score": self.similarity_score,
            "is_low_independence": self.is_low_independence,
            "size": self.size,
            "detected_at": self.detected_at.isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════
# Senate Record Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SenateRecordEntry:
    """A single entry in the Senate Record (Hansard)."""
    id: str = field(default_factory=lambda: _uid("re"))
    entry_type: RecordEntryType = RecordEntryType.SOCRATIC_ACTION
    phase: SessionPhase = SessionPhase.OPENING_STATEMENTS
    round_num: int = 0
    senator_id: str = ""
    senator_name: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entry_type": self.entry_type.value,
            "phase": self.phase.value,
            "round_num": self.round_num,
            "senator_id": self.senator_id,
            "senator_name": self.senator_name,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_hansard_line(self) -> str:
        """Format this entry as a Hansard-style line."""
        ts = self.timestamp.strftime("%H:%M:%S")
        phase_num = self.phase.phase_number
        prefix = f"[{ts}] P{phase_num}/R{self.round_num}"
        type_label = self.entry_type.value.upper().replace("_", " ")

        if self.senator_name:
            return f"{prefix} [{type_label}] {self.senator_name}: {self.content}"
        return f"{prefix} [{type_label}] {self.content}"


# ══════════════════════════════════════════════════════════════════════
# Result Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MajorityOpinion:
    """The primary verdict of the AGORA session."""
    verdict_label: VerdictLabel = VerdictLabel.INDETERMINATE
    posterior_probability: float = 0.5
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    narrative: str = ""
    key_supporting_evidence: list[str] = field(default_factory=list)  # DocketItem IDs
    majority_coalition_ids: list[str] = field(default_factory=list)
    majority_coalition_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_label": self.verdict_label.value,
            "posterior_probability": self.posterior_probability,
            "confidence_interval": list(self.confidence_interval),
            "narrative": self.narrative,
            "key_supporting_evidence": self.key_supporting_evidence,
            "majority_coalition_ids": self.majority_coalition_ids,
            "majority_coalition_names": self.majority_coalition_names,
        }


@dataclass
class MinorityReport:
    """The minority position document — a first-class output."""
    minority_claim: str = ""
    supporting_evidence_ids: list[str] = field(default_factory=list)
    sustained_challenges: list[str] = field(default_factory=list)  # Challenge IDs
    why_majority_insufficient: str = ""
    what_would_change: list[str] = field(default_factory=list)
    minority_senator_ids: list[str] = field(default_factory=list)
    minority_senator_names: list[str] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "minority_claim": self.minority_claim,
            "supporting_evidence_ids": self.supporting_evidence_ids,
            "sustained_challenges": self.sustained_challenges,
            "why_majority_insufficient": self.why_majority_insufficient,
            "what_would_change": self.what_would_change,
            "minority_senator_ids": self.minority_senator_ids,
            "minority_senator_names": self.minority_senator_names,
            "narrative": self.narrative,
        }


@dataclass
class SenatorScorecard:
    """Per-senator performance metrics."""
    senator_id: str = ""
    senator_name: str = ""
    category: SenatorCategory = SenatorCategory.DOMAIN_EXPERT
    floor_time_used: int = 0
    floor_time_budget: int = 5
    evidence_submitted: int = 0
    challenges_issued: int = 0
    challenges_received: int = 0
    challenges_sustained: int = 0
    challenges_overruled: int = 0
    points_of_order: int = 0
    position_trajectory: list[float] = field(default_factory=list)  # position per round
    epistemic_contribution_score: float = 0.0  # ECS — novel metric

    def to_dict(self) -> dict[str, Any]:
        return {
            "senator_id": self.senator_id,
            "senator_name": self.senator_name,
            "category": self.category.value,
            "floor_time_used": self.floor_time_used,
            "floor_time_budget": self.floor_time_budget,
            "floor_time_pct": (self.floor_time_used / max(self.floor_time_budget, 1)) * 100,
            "evidence_submitted": self.evidence_submitted,
            "challenges_issued": self.challenges_issued,
            "challenges_received": self.challenges_received,
            "challenges_sustained": self.challenges_sustained,
            "challenges_overruled": self.challenges_overruled,
            "points_of_order": self.points_of_order,
            "position_trajectory": self.position_trajectory,
            "epistemic_contribution_score": self.epistemic_contribution_score,
        }


@dataclass
class AgoraResult:
    """Complete output of an AGORA session — all nine result components.

    Components:
        1. Majority Opinion
        2. Minority Report
        3. Coalition Map
        4. Evidence Docket Summary
        5. Position Trajectory Map
        6. Senator Performance Scorecards
        7. What Would Change This
        8. Senate Record
        9. Quorum Certificate
    """
    id: str = field(default_factory=lambda: _uid("ar"))
    proposition: str = ""
    senate_spec_id: str = ""

    # 1. Majority Opinion
    majority_opinion: MajorityOpinion = field(default_factory=MajorityOpinion)

    # 2. Minority Report
    minority_report: MinorityReport = field(default_factory=MinorityReport)

    # 3. Coalition Map
    coalitions: list[CoalitionInfo] = field(default_factory=list)

    # 4. Evidence Docket Summary
    docket_items: list[DocketItem] = field(default_factory=list)

    # 5. Position Trajectory Map (senator_id -> list of positions per round)
    position_trajectories: dict[str, list[float]] = field(default_factory=dict)

    # 6. Senator Scorecards
    scorecards: list[SenatorScorecard] = field(default_factory=list)

    # 7. What Would Change This
    what_would_change: list[str] = field(default_factory=list)

    # 8. Senate Record entries
    senate_record_entries: list[SenateRecordEntry] = field(default_factory=list)

    # 9. Quorum Certificate
    quorum_met: bool = True
    quorum_fraction_achieved: float = 1.0
    quorum_details: str = ""

    # Session metadata
    num_rounds: int = 0
    num_evidence: int = 0
    num_challenges: int = 0
    num_senators: int = 0
    duration_seconds: float = 0.0
    total_tokens_used: int = 0
    stopping_trigger_fired: Optional[StoppingTrigger] = None

    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "senate_spec_id": self.senate_spec_id,
            "majority_opinion": self.majority_opinion.to_dict(),
            "minority_report": self.minority_report.to_dict(),
            "coalitions": [c.to_dict() for c in self.coalitions],
            "docket_items": [d.to_dict() for d in self.docket_items],
            "position_trajectories": self.position_trajectories,
            "scorecards": [s.to_dict() for s in self.scorecards],
            "what_would_change": self.what_would_change,
            "senate_record_entries": [e.to_dict() for e in self.senate_record_entries],
            "quorum_met": self.quorum_met,
            "quorum_fraction_achieved": self.quorum_fraction_achieved,
            "quorum_details": self.quorum_details,
            "num_rounds": self.num_rounds,
            "num_evidence": self.num_evidence,
            "num_challenges": self.num_challenges,
            "num_senators": self.num_senators,
            "duration_seconds": self.duration_seconds,
            "total_tokens_used": self.total_tokens_used,
            "stopping_trigger_fired": (
                self.stopping_trigger_fired.value if self.stopping_trigger_fired else None
            ),
            "created_at": self.created_at.isoformat(),
        }

    def chat_card(self) -> str:
        """Formatted text for the chat/result pane."""
        lines = [
            "━" * 55,
            f"VERDICT: {self.majority_opinion.verdict_label.value.upper()}",
            f"Posterior: {self.majority_opinion.posterior_probability:.0%} | "
            f"Senators: {self.num_senators} | "
            f"Rounds: {self.num_rounds}",
            "━" * 55,
            "",
            self.majority_opinion.narrative,
        ]
        if self.minority_report.narrative:
            lines += ["", "MINORITY REPORT:", self.minority_report.narrative]
        if self.coalitions:
            lines += ["", "COALITIONS DETECTED:"]
            for c in self.coalitions:
                indep = "⚠ LOW INDEPENDENCE" if c.is_low_independence else "✓ Independent"
                lines.append(
                    f"  • {c.name} ({c.size} members, {indep})"
                )
        if self.what_would_change:
            lines += ["", "WHAT WOULD CHANGE THIS VERDICT:"]
            for i, item in enumerate(self.what_would_change, 1):
                lines.append(f"  {i}. {item}")
        if self.scorecards:
            lines += ["", "SENATOR PERFORMANCE (Top ECS):"]
            sorted_cards = sorted(
                self.scorecards,
                key=lambda s: s.epistemic_contribution_score,
                reverse=True,
            )
            for sc in sorted_cards[:5]:
                lines.append(
                    f"  {sc.senator_name} ({sc.category.abbreviation}) — "
                    f"ECS: {sc.epistemic_contribution_score:.2f} | "
                    f"Evidence: {sc.evidence_submitted}"
                )
        lines.append("")
        lines.append(
            f"Session: {self.duration_seconds:.0f}s | "
            f"{self.num_evidence} evidence items | "
            f"{self.num_challenges} challenges | "
            f"Quorum: {'✓' if self.quorum_met else '✗'} ({self.quorum_fraction_achieved:.0%})"
        )
        if self.stopping_trigger_fired:
            lines.append(f"Stopped by: {self.stopping_trigger_fired.value}")
        return "\n".join(lines)
