"""
Procedural Rules Engine — governs the formal conduct of an AGORA session.

Implements five sub-engines that enforce parliamentary-style order during
the deliberation, preventing abuse, redundancy, and ensuring every
senator gets a fair hearing.

Sub-engines:
    1. PhaseManager       — 5-phase lifecycle with round tracking
    2. FloorTimeEngine    — budget allocations per senator
    3. FilibusterDetector — redundancy check via text similarity
    4. PointOfOrderHandler — procedural motions
    5. QuorumEngine       — participation tracking
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from argus.agora.models import (
    AgoraSessionConfig,
    SessionPhase,
    StoppingTrigger,
    SenateSpec,
    SenateRecordEntry,
    RecordEntryType,
    _utcnow,
    _uid,
)

logger = logging.getLogger(__name__)

# ── stopword set for filibuster similarity ────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "between", "out", "off", "over", "under",
    "again", "then", "once", "when", "where", "why", "how", "all", "both",
    "each", "more", "most", "other", "some", "such", "no", "not", "only",
    "so", "than", "too", "very", "just", "and", "but", "or", "if", "that",
    "this", "it", "its", "about", "up",
})


def _keyword_set(text: str) -> set[str]:
    """Extract meaningful keywords from text."""
    return {w.lower() for w in text.split() if w.lower() not in _STOPWORDS and len(w) > 2}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two keyword sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / max(len(union), 1)


# ══════════════════════════════════════════════════════════════════════
# 1. Phase Manager
# ══════════════════════════════════════════════════════════════════════

_PHASE_ORDER = [
    SessionPhase.OPENING_STATEMENTS,
    SessionPhase.EVIDENCE_SUBMISSION,
    SessionPhase.CROSS_EXAMINATION,
    SessionPhase.DELIBERATIVE_SYNTHESIS,
    SessionPhase.CLOSING_AND_VERDICT,
]

# Rounds per phase (defaults — can be overridden)
_DEFAULT_ROUNDS_PER_PHASE = {
    SessionPhase.OPENING_STATEMENTS: 1,     # Each senator makes one opening
    SessionPhase.EVIDENCE_SUBMISSION: 5,     # Main evidence-gathering rounds
    SessionPhase.CROSS_EXAMINATION: 3,       # Challenge and re-challenge
    SessionPhase.DELIBERATIVE_SYNTHESIS: 2,  # Integration rounds
    SessionPhase.CLOSING_AND_VERDICT: 1,     # Final statements + verdict
}


class PhaseManager:
    """Manages the 5-phase session lifecycle.

    Each phase contains a configurable number of rounds.
    The ``advance_round()`` method returns True when the current phase
    is exhausted and ``advance_phase()`` should be called.
    """

    def __init__(self, config: AgoraSessionConfig):
        self.config = config
        self.current_phase: SessionPhase = SessionPhase.OPENING_STATEMENTS
        self.phase_index: int = 0
        self.current_round: int = 0
        self.total_rounds_elapsed: int = 0
        self.session_start_time: float = time.time()

        # Compute rounds per phase
        self._rounds_per_phase: dict[SessionPhase, int] = dict(_DEFAULT_ROUNDS_PER_PHASE)
        # Override evidence and cross-exam phases with config.max_rounds
        self._rounds_per_phase[SessionPhase.EVIDENCE_SUBMISSION] = config.max_rounds
        self._rounds_per_phase[SessionPhase.CROSS_EXAMINATION] = max(2, config.max_rounds // 2)

    @property
    def max_rounds_in_phase(self) -> int:
        """Max rounds for the current phase."""
        return self._rounds_per_phase.get(self.current_phase, 1)

    @property
    def is_final_phase(self) -> bool:
        return self.phase_index >= len(_PHASE_ORDER) - 1

    @property
    def is_session_complete(self) -> bool:
        return self.phase_index >= len(_PHASE_ORDER)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.session_start_time

    def advance_round(self) -> bool:
        """Advance to next round within current phase.

        Returns True if phase is exhausted (all rounds done).
        """
        self.current_round += 1
        self.total_rounds_elapsed += 1
        return self.current_round >= self.max_rounds_in_phase

    def advance_phase(self) -> SessionPhase | None:
        """Move to the next phase. Returns the new phase, or None if complete."""
        self.phase_index += 1
        self.current_round = 0
        if self.phase_index < len(_PHASE_ORDER):
            self.current_phase = _PHASE_ORDER[self.phase_index]
            return self.current_phase
        return None

    def check_time_boundary(self) -> bool:
        """Check if time boundary stopping trigger is hit.

        Returns True if the session should stop due to time.
        Returns False if unbounded or within time limit.
        """
        if self.config.is_unbounded:
            return False  # Unbounded session — agents run until natural completion
        if self.config.time_limit_seconds is None:
            return False
        return self.elapsed_seconds >= self.config.time_limit_seconds

    def make_transition_record(self, new_phase: SessionPhase) -> SenateRecordEntry:
        """Generate a Senate Record entry for a phase transition."""
        return SenateRecordEntry(
            entry_type=RecordEntryType.PHASE_TRANSITION,
            phase=new_phase,
            round_num=self.current_round,
            content=(
                f"Phase transition: {self.current_phase.value} → {new_phase.value}. "
                f"Total elapsed rounds: {self.total_rounds_elapsed}. "
                f"Session time: {self.elapsed_seconds:.0f}s."
            ),
        )


# ══════════════════════════════════════════════════════════════════════
# 2. Floor Time Engine
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FloorTimeEvent:
    """Tracks a single action that consumed floor time."""
    senator_id: str
    action_type: str  # evidence_submit, challenge, statement, point_of_order
    round_num: int
    timestamp: float = field(default_factory=time.time)


class FloorTimeEngine:
    """Manages floor time budgets per senator.

    Each senator has a budget of N actions (from SenatorSpec.floor_time_budget).
    Once exhausted, further submissions are rejected.
    """

    def __init__(self, senate: SenateSpec):
        # budgets: senator_id -> remaining actions
        self._budgets: dict[str, int] = {}
        # usage log
        self._events: list[FloorTimeEvent] = []
        # usage count per senator
        self._used: dict[str, int] = defaultdict(int)

        for senator in senate.senators:
            self._budgets[senator.id] = senator.floor_time_budget

    def can_speak(self, senator_id: str) -> bool:
        """Check if senator has remaining floor time."""
        return self._budgets.get(senator_id, 0) > 0

    def consume(
        self,
        senator_id: str,
        action_type: str,
        round_num: int,
    ) -> bool:
        """Consume one unit of floor time.

        Returns True if consumed, False if budget exhausted.
        """
        if not self.can_speak(senator_id):
            return False
        self._budgets[senator_id] -= 1
        self._used[senator_id] += 1
        self._events.append(FloorTimeEvent(
            senator_id=senator_id,
            action_type=action_type,
            round_num=round_num,
        ))
        return True

    def remaining(self, senator_id: str) -> int:
        """Get remaining floor time for a senator."""
        return self._budgets.get(senator_id, 0)

    def usage_summary(self) -> dict[str, dict[str, int]]:
        """Get usage summary per senator: {senator_id: {used, remaining, budget}}."""
        result: dict[str, dict[str, int]] = {}
        for sid, budget in self._budgets.items():
            used = self._used[sid]
            result[sid] = {
                "used": used,
                "remaining": self._budgets[sid],
                "budget": used + self._budgets[sid],
            }
        return result

    def make_exhaustion_record(
        self,
        senator_id: str,
        senator_name: str,
        phase: SessionPhase,
        round_num: int,
    ) -> SenateRecordEntry:
        """Generate a record entry when a senator's floor time is exhausted."""
        return SenateRecordEntry(
            entry_type=RecordEntryType.FLOOR_TIME_EVENT,
            phase=phase,
            round_num=round_num,
            senator_id=senator_id,
            senator_name=senator_name,
            content=f"Floor time exhausted for {senator_name}. No further submissions accepted.",
        )


# ══════════════════════════════════════════════════════════════════════
# 3. Filibuster Detector
# ══════════════════════════════════════════════════════════════════════

class FilibusterDetector:
    """Detects and flags redundant evidence/statements.

    Uses keyword-based Jaccard similarity against all prior submissions
    by the same senator. If a new submission is > threshold similar
    to any previous submission, it's flagged as a filibuster.
    """

    DEFAULT_THRESHOLD = 0.60

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        # senator_id -> list of keyword sets
        self._history: dict[str, list[set[str]]] = defaultdict(list)
        self._flagged_count: int = 0

    def check(self, senator_id: str, text: str) -> bool:
        """Check if text is a filibuster (redundant).

        Returns True if the text is substantially similar to prior
        submissions by this senator.
        """
        new_keywords = _keyword_set(text)
        if not new_keywords:
            return False

        for prior_keywords in self._history.get(senator_id, []):
            similarity = _jaccard_similarity(new_keywords, prior_keywords)
            if similarity >= self.threshold:
                self._flagged_count += 1
                return True

        return False

    def register(self, senator_id: str, text: str) -> None:
        """Register a text as accepted (not a filibuster)."""
        keywords = _keyword_set(text)
        if keywords:
            self._history[senator_id].append(keywords)

    @property
    def total_flagged(self) -> int:
        return self._flagged_count

    def make_filibuster_record(
        self,
        senator_id: str,
        senator_name: str,
        phase: SessionPhase,
        round_num: int,
    ) -> SenateRecordEntry:
        """Generate a record entry when a filibuster is detected."""
        return SenateRecordEntry(
            entry_type=RecordEntryType.SOCRATIC_ACTION,
            phase=phase,
            round_num=round_num,
            senator_id=senator_id,
            senator_name=senator_name,
            content=f"Filibuster flagged: {senator_name}'s submission rejected as substantively "
                    "redundant with prior contributions.",
            metadata={"action": "filibuster_rejected"},
        )


# ══════════════════════════════════════════════════════════════════════
# 4. Point of Order Handler
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PointOfOrder:
    """A procedural motion raised by a senator."""
    id: str = field(default_factory=lambda: _uid("po"))
    raiser_id: str = ""
    raiser_name: str = ""
    objection_text: str = ""
    target_senator_id: str = ""
    target_senator_name: str = ""
    ruling: str = ""        # sustained | overruled
    reason: str = ""
    round_num: int = 0
    phase: SessionPhase = SessionPhase.EVIDENCE_SUBMISSION


class PointOfOrderHandler:
    """Processes Points of Order during deliberation.

    A senator may raise a Point of Order to object to:
    - A violation of floor time rules
    - An off-topic statement
    - A logical fallacy
    - A misrepresentation of evidence

    The Socratic Engine (or EA) adjudicates.
    """

    def __init__(self):
        self._points: list[PointOfOrder] = []

    def raise_point(
        self,
        raiser_id: str,
        raiser_name: str,
        objection_text: str,
        target_senator_id: str,
        target_senator_name: str,
        round_num: int,
        phase: SessionPhase,
    ) -> PointOfOrder:
        """Raise a Point of Order."""
        point = PointOfOrder(
            raiser_id=raiser_id,
            raiser_name=raiser_name,
            objection_text=objection_text,
            target_senator_id=target_senator_id,
            target_senator_name=target_senator_name,
            round_num=round_num,
            phase=phase,
        )
        self._points.append(point)
        return point

    def rule_on_point(self, point: PointOfOrder, ruling: str, reason: str) -> None:
        """Issue ruling on a Point of Order."""
        point.ruling = ruling
        point.reason = reason

    @property
    def total_points(self) -> int:
        return len(self._points)

    @property
    def sustained_count(self) -> int:
        return sum(1 for p in self._points if p.ruling == "sustained")

    def make_point_record(self, point: PointOfOrder) -> SenateRecordEntry:
        """Generate a record entry for a Point of Order."""
        ruling_text = ""
        if point.ruling:
            ruling_text = f" Ruling: {point.ruling.upper()} — {point.reason}"
        return SenateRecordEntry(
            entry_type=RecordEntryType.POINT_OF_ORDER,
            phase=point.phase,
            round_num=point.round_num,
            senator_id=point.raiser_id,
            senator_name=point.raiser_name,
            content=(
                f"Point of Order raised by {point.raiser_name} against "
                f"{point.target_senator_name}: {point.objection_text}{ruling_text}"
            ),
            metadata={
                "target_senator": point.target_senator_id,
                "ruling": point.ruling,
            },
        )


# ══════════════════════════════════════════════════════════════════════
# 5. Quorum Engine
# ══════════════════════════════════════════════════════════════════════

class QuorumEngine:
    """Tracks participation and enforces quorum requirements.

    A quorum is met when at least ``quorum_fraction`` of seated senators
    have participated (submitted evidence, made statements, or issued
    challenges) within the session.
    """

    def __init__(
        self,
        senate: SenateSpec,
        quorum_fraction: float = 0.60,
    ):
        self._total_seated = senate.senate_size
        self._quorum_fraction = quorum_fraction
        self._participating: set[str] = set()
        self._all_senator_ids = {s.id for s in senate.senators}

    def record_participation(self, senator_id: str) -> None:
        """Record that a senator has participated."""
        if senator_id in self._all_senator_ids:
            self._participating.add(senator_id)

    @property
    def participating_count(self) -> int:
        return len(self._participating)

    @property
    def quorum_fraction_achieved(self) -> float:
        return self.participating_count / max(self._total_seated, 1)

    @property
    def is_quorum_met(self) -> bool:
        return self.quorum_fraction_achieved >= self._quorum_fraction

    @property
    def non_participating(self) -> set[str]:
        return self._all_senator_ids - self._participating

    def check_quorum_failure(self) -> bool:
        """Check if quorum has failed (below threshold and cannot recover).

        This should be checked at the end of the evidence phase.
        """
        return not self.is_quorum_met

    def make_quorum_record(self, phase: SessionPhase, round_num: int) -> SenateRecordEntry:
        """Generate a quorum status record entry."""
        status = "MET" if self.is_quorum_met else "NOT MET"
        return SenateRecordEntry(
            entry_type=RecordEntryType.QUORUM_UPDATE,
            phase=phase,
            round_num=round_num,
            content=(
                f"Quorum check: {status}. "
                f"{self.participating_count}/{self._total_seated} senators participating "
                f"({self.quorum_fraction_achieved:.0%}, "
                f"threshold: {self._quorum_fraction:.0%})."
            ),
            metadata={
                "participating": self.participating_count,
                "total": self._total_seated,
                "fraction": self.quorum_fraction_achieved,
                "met": self.is_quorum_met,
            },
        )

    def make_quorum_certificate(self) -> dict[str, Any]:
        """Generate the Quorum Certificate for the result set."""
        return {
            "quorum_met": self.is_quorum_met,
            "quorum_fraction_achieved": self.quorum_fraction_achieved,
            "quorum_threshold": self._quorum_fraction,
            "senators_total": self._total_seated,
            "senators_participating": self.participating_count,
            "non_participating_ids": list(self.non_participating),
        }
