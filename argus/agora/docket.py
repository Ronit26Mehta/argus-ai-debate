"""
Evidence Docket — formal evidence management system for AGORA.

Every piece of evidence submitted to the Senate is registered here with
full metadata, provenance, and audit trail. The Docket provides:

    1. Evidence Submission — structured intake with DEW scoring
    2. Challenge Protocol  — formal challenge / reply / EA ruling
    3. Cross-corroboration — automatic detection of supporting evidence
    4. Evidence Querying   — filter by senator, polarity, type, round
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Optional

from argus.agora.models import (
    ChallengeOutcome,
    ChallengeType,
    Challenge,
    DocketEvidenceType,
    DocketItem,
    EvidencePolarity,
    SenateRecordEntry,
    SenatorCategory,
    RecordEntryType,
    SessionPhase,
    _utcnow,
    _uid,
)

logger = logging.getLogger(__name__)

# ── keywords for cross-corroboration ──────────────────────────────────

_CORR_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "but", "or", "if",
    "that", "this", "it", "its", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "not", "no",
})


def _similarity_keywords(text: str) -> set[str]:
    return {w.lower() for w in text.split() if w.lower() not in _CORR_STOPWORDS and len(w) > 2}


def _keyword_overlap(a: str, b: str) -> float:
    """Fraction of keywords in `a` present in `b`."""
    ka = _similarity_keywords(a)
    kb = _similarity_keywords(b)
    if not ka:
        return 0.0
    return len(ka & kb) / len(ka)


# ══════════════════════════════════════════════════════════════════════
# DEW Calculator
# ══════════════════════════════════════════════════════════════════════

class DEWCalculator:
    """Dynamic Evidence Weight calculator.

    DEW(e) = f(
        source_quality,        — type of source (empirical > anecdotal)
        confidence_score,      — submitter confidence
        cross_corroboration,   — is this corroborated?
        challenge_impact,      — has it survived challenges?
        novelty,               — divergence from existing evidence
    )

    Output range: 0.0 – 1.0
    """

    # Source quality weights by evidence type
    _SOURCE_QUALITY: dict[DocketEvidenceType, float] = {
        DocketEvidenceType.EXPERIMENTAL: 0.95,
        DocketEvidenceType.QUANTITATIVE: 0.90,
        DocketEvidenceType.LEGAL: 0.85,
        DocketEvidenceType.HISTORICAL: 0.75,
        DocketEvidenceType.QUALITATIVE: 0.70,
        DocketEvidenceType.THEORETICAL: 0.65,
        DocketEvidenceType.ANECDOTAL: 0.40,
    }

    @classmethod
    def compute(
        cls,
        item: DocketItem,
        corroboration_bonus: float = 0.0,
        novelty_score: float = 0.5,
    ) -> float:
        """Compute DEW score for a docket item.

        Args:
            item: The evidence to score.
            corroboration_bonus: 0.0–0.2 bonus from cross-corroboration.
            novelty_score: 0.0–1.0 how novel this evidence is.

        Returns:
            DEW score in [0.0, 1.0].
        """
        source_q = cls._SOURCE_QUALITY.get(item.evidence_type, 0.5)
        confidence = max(0.0, min(1.0, item.confidence_score))

        # Challenge impact
        if item.is_challenged:
            if item.challenge_outcome == ChallengeOutcome.SUSTAINED:
                challenge_factor = 0.3   # heavily downweighted
            elif item.challenge_outcome == ChallengeOutcome.OVERRULED:
                challenge_factor = 1.15  # slightly upgraded
            elif item.challenge_outcome == ChallengeOutcome.NOTED:
                challenge_factor = 0.9   # minor downweight
            else:
                challenge_factor = 0.7   # pending challenge — tentative downweight
        else:
            challenge_factor = 1.0

        # Composite
        raw = (
            source_q * 0.30
            + confidence * 0.25
            + novelty_score * 0.20
            + corroboration_bonus * 0.15
            + 0.10  # base weight
        )
        raw *= challenge_factor
        return max(0.0, min(1.0, raw))


# ══════════════════════════════════════════════════════════════════════
# Challenge Handler
# ══════════════════════════════════════════════════════════════════════

class ChallengeHandler:
    """Manages the formal evidence challenge protocol.

    Protocol flow:
        1. Challenger submits Challenge(target_evidence_id, type, argument)
        2. Submitting senator has reply opportunity
        3. Epistemic Auditor rules: SUSTAINED / OVERRULED / NOTED
        4. DEW is adjusted on the target evidence
    """

    def __init__(self):
        self._challenges: list[Challenge] = []

    def issue_challenge(
        self,
        challenger_id: str,
        challenger_name: str,
        target_evidence_id: str,
        challenge_type: ChallengeType,
        argument: str,
        counter_evidence_id: Optional[str] = None,
    ) -> Challenge:
        """Issue a formal challenge against a docket item."""
        challenge = Challenge(
            challenger_id=challenger_id,
            challenger_name=challenger_name,
            target_evidence_id=target_evidence_id,
            challenge_type=challenge_type,
            challenge_argument=argument,
            counter_evidence_id=counter_evidence_id,
        )
        self._challenges.append(challenge)
        return challenge

    def submit_reply(self, challenge: Challenge, reply_text: str) -> None:
        """Submit the defending senator's reply to a challenge."""
        challenge.reply_text = reply_text
        challenge.reply_submitted = True

    def rule(
        self,
        challenge: Challenge,
        outcome: ChallengeOutcome,
        ea_reasoning: str,
    ) -> None:
        """EA rules on a challenge."""
        challenge.outcome = outcome
        challenge.ea_reasoning = ea_reasoning

    def get_challenges_for_evidence(self, evidence_id: str) -> list[Challenge]:
        """Get all challenges targeting a specific evidence item."""
        return [c for c in self._challenges if c.target_evidence_id == evidence_id]

    def get_challenges_by_challenger(self, challenger_id: str) -> list[Challenge]:
        return [c for c in self._challenges if c.challenger_id == challenger_id]

    @property
    def total_challenges(self) -> int:
        return len(self._challenges)

    @property
    def sustained_count(self) -> int:
        return sum(1 for c in self._challenges if c.outcome == ChallengeOutcome.SUSTAINED)

    def make_challenge_record(
        self,
        challenge: Challenge,
        phase: SessionPhase,
    ) -> SenateRecordEntry:
        """Generate a record entry for a challenge."""
        return SenateRecordEntry(
            entry_type=RecordEntryType.CHALLENGE_ISSUED,
            phase=phase,
            round_num=0,
            senator_id=challenge.challenger_id,
            senator_name=challenge.challenger_name,
            content=(
                f"Challenge ({challenge.challenge_type.value}) against "
                f"evidence {challenge.target_evidence_id}: "
                f"{challenge.challenge_argument}"
            ),
            metadata=challenge.to_dict(),
        )

    def make_ruling_record(
        self,
        challenge: Challenge,
        phase: SessionPhase,
    ) -> SenateRecordEntry:
        """Generate a record entry for an EA ruling."""
        return SenateRecordEntry(
            entry_type=RecordEntryType.EA_RULING,
            phase=phase,
            senator_name="Epistemic Auditor",
            content=(
                f"Ruling on challenge {challenge.id}: "
                f"{challenge.outcome.value.upper() if challenge.outcome else 'PENDING'}. "
                f"{challenge.ea_reasoning}"
            ),
            metadata={
                "challenge_id": challenge.id,
                "outcome": challenge.outcome.value if challenge.outcome else None,
            },
        )


# ══════════════════════════════════════════════════════════════════════
# Evidence Docket
# ══════════════════════════════════════════════════════════════════════

class EvidenceDocket:
    """Central evidence registry for an AGORA session.

    All evidence items are registered, scored, queried, and challenged
    through this single entry point.
    """

    CORROBORATION_THRESHOLD = 0.45

    def __init__(self):
        self._items: list[DocketItem] = []
        self._items_by_id: dict[str, DocketItem] = {}
        self._items_by_senator: dict[str, list[DocketItem]] = defaultdict(list)
        self._items_by_round: dict[int, list[DocketItem]] = defaultdict(list)
        self._challenge_handler = ChallengeHandler()
        self._dew_calculator = DEWCalculator()

    # ── Submission ────────────────────────────────────────────────────

    def submit_evidence(
        self,
        senator_id: str,
        senator_name: str,
        senator_category: SenatorCategory,
        claim_text: str,
        polarity: EvidencePolarity,
        source_reference: str = "",
        source_type: str = "general",
        confidence_score: float = 0.5,
        evidence_type: DocketEvidenceType = DocketEvidenceType.QUALITATIVE,
        relationship_to_prior: str = "",
        round_num: int = 0,
    ) -> DocketItem:
        """Submit new evidence to the docket.

        Scores DEW, detects cross-corroboration, and indexes the item.

        Returns:
            The registered DocketItem with computed DEW score.
        """
        item = DocketItem(
            senator_id=senator_id,
            senator_name=senator_name,
            senator_category=senator_category,
            claim_text=claim_text,
            polarity=polarity,
            source_reference=source_reference,
            source_type=source_type,
            confidence_score=confidence_score,
            evidence_type=evidence_type,
            relationship_to_prior=relationship_to_prior,
        )

        # Cross-corroboration detection
        corroboration_bonus = 0.0
        for existing in self._items:
            if existing.senator_id == senator_id:
                continue  # skip self-corroboration
            if existing.polarity != polarity:
                continue  # only same-polarity corroborates
            overlap = _keyword_overlap(claim_text, existing.claim_text)
            if overlap > self.CORROBORATION_THRESHOLD:
                item.corroborating_items.append(existing.id)
                existing.corroborating_items.append(item.id)
                corroboration_bonus = min(0.20, corroboration_bonus + 0.05)

        # Novelty score (inverse overlap with prior evidence)
        if self._items:
            avg_overlap = sum(
                _keyword_overlap(claim_text, e.claim_text)
                for e in self._items[-20:]
            ) / min(len(self._items), 20)
            novelty = max(0.0, 1.0 - avg_overlap)
        else:
            novelty = 1.0

        # Compute DEW
        item.dew_score = self._dew_calculator.compute(
            item,
            corroboration_bonus=corroboration_bonus,
            novelty_score=novelty,
        )

        # Index
        self._items.append(item)
        self._items_by_id[item.id] = item
        self._items_by_senator[senator_id].append(item)
        self._items_by_round[round_num].append(item)

        logger.debug(
            "Evidence submitted: %s by %s (DEW=%.2f, corr=%d)",
            item.evidence_id_display, senator_name, item.dew_score,
            len(item.corroborating_items),
        )
        return item

    # ── Challenge protocol ────────────────────────────────────────────

    def issue_challenge(
        self,
        challenger_id: str,
        challenger_name: str,
        target_evidence_id: str,
        challenge_type: ChallengeType,
        argument: str,
        counter_evidence_id: Optional[str] = None,
    ) -> Challenge:
        """Issue a formal challenge against a docket item."""
        target = self._items_by_id.get(target_evidence_id)
        if target:
            target.is_challenged = True
        return self._challenge_handler.issue_challenge(
            challenger_id, challenger_name, target_evidence_id,
            challenge_type, argument, counter_evidence_id,
        )

    def resolve_challenge(
        self,
        challenge: Challenge,
        outcome: ChallengeOutcome,
        ea_reasoning: str,
    ) -> None:
        """Resolve a challenge with an EA ruling and update DEW."""
        self._challenge_handler.rule(challenge, outcome, ea_reasoning)

        # Update target evidence
        target = self._items_by_id.get(challenge.target_evidence_id)
        if target:
            target.challenge_outcome = outcome
            # Recompute DEW with challenge impact
            target.dew_score = self._dew_calculator.compute(target)

    @property
    def challenge_handler(self) -> ChallengeHandler:
        return self._challenge_handler

    # ── Querying ──────────────────────────────────────────────────────

    def get_item(self, evidence_id: str) -> Optional[DocketItem]:
        return self._items_by_id.get(evidence_id)

    def get_by_senator(self, senator_id: str) -> list[DocketItem]:
        return list(self._items_by_senator.get(senator_id, []))

    def get_by_polarity(self, polarity: EvidencePolarity) -> list[DocketItem]:
        return [i for i in self._items if i.polarity == polarity]

    def get_by_round(self, round_num: int) -> list[DocketItem]:
        return list(self._items_by_round.get(round_num, []))

    def get_top_weighted(self, n: int = 10) -> list[DocketItem]:
        """Get top-N evidence items by DEW score."""
        return sorted(self._items, key=lambda i: i.dew_score, reverse=True)[:n]

    @property
    def total_items(self) -> int:
        return len(self._items)

    @property
    def total_challenges(self) -> int:
        return self._challenge_handler.total_challenges

    @property
    def all_items(self) -> list[DocketItem]:
        return list(self._items)

    # ── Summary ───────────────────────────────────────────────────────

    def summary_stats(self) -> dict[str, Any]:
        """Get aggregated docket statistics."""
        if not self._items:
            return {
                "total": 0, "supports": 0, "attacks": 0, "qualifies": 0,
                "avg_dew": 0.0, "challenges": 0, "challenged_pct": 0.0,
            }

        supports = sum(1 for i in self._items if i.polarity == EvidencePolarity.SUPPORTS)
        attacks = sum(1 for i in self._items if i.polarity == EvidencePolarity.ATTACKS)
        qualifies = sum(1 for i in self._items if i.polarity == EvidencePolarity.QUALIFIES)
        avg_dew = sum(i.dew_score for i in self._items) / len(self._items)
        challenged = sum(1 for i in self._items if i.is_challenged)

        return {
            "total": len(self._items),
            "supports": supports,
            "attacks": attacks,
            "qualifies": qualifies,
            "avg_dew": avg_dew,
            "challenges": self._challenge_handler.total_challenges,
            "sustained_challenges": self._challenge_handler.sustained_count,
            "challenged_pct": challenged / max(len(self._items), 1) * 100,
        }

    def make_submission_record(
        self,
        item: DocketItem,
        phase: SessionPhase,
        round_num: int,
    ) -> SenateRecordEntry:
        """Generate a record entry for an evidence submission."""
        return SenateRecordEntry(
            entry_type=RecordEntryType.EVIDENCE_SUBMISSION,
            phase=phase,
            round_num=round_num,
            senator_id=item.senator_id,
            senator_name=item.senator_name,
            content=(
                f"Evidence submitted [{item.evidence_id_display}]: {item.claim_text[:200]}"
                f" (DEW: {item.dew_score:.2f}, {item.polarity.value}, "
                f"confidence: {item.confidence_score:.2f})"
            ),
            metadata=item.to_dict(),
        )
