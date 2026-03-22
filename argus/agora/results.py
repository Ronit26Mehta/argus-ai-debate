"""
Result Architecture — assembles all nine AGORA output components.

Transforms raw session data (docket, coalitions, scorecards, etc.)
into the complete AgoraResult object with:

    1. Majority Opinion
    2. Minority Report
    3. Coalition Map
    4. Evidence Docket Summary
    5. Position Trajectory Map
    6. Senator Performance Scorecards (with ECS)
    7. What Would Change This
    8. Senate Record
    9. Quorum Certificate
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from argus.agora.models import (
    AgoraResult,
    ChallengeOutcome,
    DocketItem,
    EvidencePolarity,
    MajorityOpinion,
    MinorityReport,
    SenateRecordEntry,
    RecordEntryType,
    SenateSpec,
    SenatorScorecard,
    SessionPhase,
    StoppingTrigger,
    VerdictLabel,
    _utcnow,
)

if TYPE_CHECKING:
    from argus.agora.coalitions import CoalitionDetectionEngine
    from argus.agora.docket import EvidenceDocket
    from argus.agora.minority import MinorityReportEngine
    from argus.agora.procedures import QuorumEngine
    from argus.agora.record import SenateRecord
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── LLM prompts ───────────────────────────────────────────────────────

_MAJORITY_SYSTEM = """\
You are AGORA's Majority Opinion Generator. Given the evidence summary, \
coalition information, and aggregate posterior:

Generate the majority opinion as JSON:
{
  "verdict_label": "<Supported|Challenged|Indeterminate|Qualified>",
  "narrative": "<2-3 paragraph verdict narrative>",
  "key_evidence_reasoning": "<which evidence was most impactful>"
}
"""

_CHANGE_SYSTEM = """\
You are AGORA's Value of Information (VoI) Analyst. Given the verdict and \
evidence profile, list 3-5 specific pieces of evidence or analysis that, \
if obtained, would most likely change the current verdict.

Output ONLY a JSON array of strings:
["item1", "item2", "item3"]
"""


# ══════════════════════════════════════════════════════════════════════
# ECS Calculator
# ══════════════════════════════════════════════════════════════════════

class ECSCalculator:
    """Epistemic Contribution Score — AGORA's novel per-senator metric.

    ECS(s) = w1 * evidence_impact
           + w2 * challenge_performance
           + w3 * position_movement
           + w4 * novelty_contribution
           + w5 * cross_engagement

    where each component is normalised to [0, 1].
    """

    W_EVIDENCE = 0.30
    W_CHALLENGE = 0.20
    W_MOVEMENT = 0.20
    W_NOVELTY = 0.15
    W_ENGAGEMENT = 0.15

    @classmethod
    def compute(
        cls,
        scorecard: SenatorScorecard,
        avg_dew: float,
        total_evidence: int,
        total_senators: int,
    ) -> float:
        """Compute ECS for a single senator."""
        # Evidence impact: fraction of evidence × avg DEW
        if total_evidence > 0:
            evidence_share = scorecard.evidence_submitted / max(total_evidence, 1)
        else:
            evidence_share = 0.0
        evidence_impact = min(1.0, evidence_share * 3)  # normalise

        # Challenge performance
        ch_total = scorecard.challenges_issued + scorecard.challenges_received
        if ch_total > 0:
            challenge_perf = (
                (scorecard.challenges_sustained * 0.8
                 + scorecard.challenges_overruled * 0.2)
                / max(ch_total, 1)
            )
        else:
            challenge_perf = 0.3  # neutral if no challenges

        # Position movement (more movement = more responsive to evidence)
        if len(scorecard.position_trajectory) >= 2:
            deltas = [
                abs(scorecard.position_trajectory[i] - scorecard.position_trajectory[i - 1])
                for i in range(1, len(scorecard.position_trajectory))
            ]
            avg_movement = sum(deltas) / len(deltas)
            position_movement = min(1.0, avg_movement * 5)
        else:
            position_movement = 0.0

        # Novelty contribution (proxy: evidence / senator average)
        avg_per_senator = total_evidence / max(total_senators, 1)
        if avg_per_senator > 0:
            novelty = min(1.0, scorecard.evidence_submitted / (avg_per_senator * 1.5))
        else:
            novelty = 0.0

        # Cross-engagement (challenges + points of order)
        engagement = min(1.0, (
            scorecard.challenges_issued * 0.3
            + scorecard.challenges_received * 0.2
            + scorecard.points_of_order * 0.1
        ))

        ecs = (
            cls.W_EVIDENCE * evidence_impact
            + cls.W_CHALLENGE * challenge_perf
            + cls.W_MOVEMENT * position_movement
            + cls.W_NOVELTY * novelty
            + cls.W_ENGAGEMENT * engagement
        )
        return round(max(0.0, min(1.0, ecs)), 3)


# ══════════════════════════════════════════════════════════════════════
# Result Builder
# ══════════════════════════════════════════════════════════════════════

class AgoraResultBuilder:
    """Assembles the complete 9-component AgoraResult."""

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def build(
        self,
        proposition: str,
        senate: SenateSpec,
        docket: "EvidenceDocket",
        cde: "CoalitionDetectionEngine",
        quorum: "QuorumEngine",
        record: "SenateRecord",
        minority_engine: "MinorityReportEngine",
        position_trajectories: dict[str, list[float]],
        scorecards: list[SenatorScorecard],
        num_rounds: int,
        duration_seconds: float,
        total_tokens_used: int,
        stopping_trigger: StoppingTrigger | None,
    ) -> AgoraResult:
        """Build the complete result set."""

        # 1. Majority Opinion
        majority = self._build_majority(
            proposition, docket, position_trajectories,
            senate, cde.current_coalitions,
        )

        # 2. Compute ECS for all scorecards
        total_evidence = docket.total_items
        avg_dew = (
            sum(i.dew_score for i in docket.all_items) / max(total_evidence, 1)
            if total_evidence > 0 else 0.0
        )
        for sc in scorecards:
            sc.epistemic_contribution_score = ECSCalculator.compute(
                sc, avg_dew, total_evidence, senate.senate_size,
            )

        # 3. Minority Report
        minority = minority_engine.generate(
            proposition, majority, docket,
            cde.current_coalitions, scorecards,
            position_trajectories,
        )

        # 7. What Would Change This
        what_would_change = self._generate_what_would_change(
            proposition, majority, docket,
        )

        # 9. Quorum Certificate
        quorum_cert = quorum.make_quorum_certificate()

        # Seal the record
        record.seal()

        # Assemble result
        result = AgoraResult(
            proposition=proposition,
            senate_spec_id=senate.id,
            majority_opinion=majority,
            minority_report=minority,
            coalitions=cde.current_coalitions,
            docket_items=docket.all_items,
            position_trajectories=position_trajectories,
            scorecards=scorecards,
            what_would_change=what_would_change,
            senate_record_entries=record.entries,
            quorum_met=quorum_cert["quorum_met"],
            quorum_fraction_achieved=quorum_cert["quorum_fraction_achieved"],
            quorum_details=(
                f"{quorum_cert['senators_participating']}/{quorum_cert['senators_total']} "
                f"participating ({quorum_cert['quorum_fraction_achieved']:.0%}, "
                f"threshold: {quorum_cert['quorum_threshold']:.0%})"
            ),
            num_rounds=num_rounds,
            num_evidence=total_evidence,
            num_challenges=docket.total_challenges,
            num_senators=senate.senate_size,
            duration_seconds=duration_seconds,
            total_tokens_used=total_tokens_used,
            stopping_trigger_fired=stopping_trigger,
        )

        return result

    # ── Majority Opinion ──────────────────────────────────────────────

    def _build_majority(
        self,
        proposition: str,
        docket: "EvidenceDocket",
        trajectories: dict[str, list[float]],
        senate: SenateSpec,
        coalitions: list[Any],
    ) -> MajorityOpinion:
        """Build the Majority Opinion from evidence and positions."""
        # Compute aggregate posterior from final senator positions
        final_positions = []
        for sid, traj in trajectories.items():
            if traj:
                final_positions.append(traj[-1])
        if not final_positions:
            final_positions = [0.5]

        avg_posterior = sum(final_positions) / len(final_positions)
        std_dev = math.sqrt(
            sum((p - avg_posterior) ** 2 for p in final_positions)
            / max(len(final_positions), 1)
        )
        ci_low = max(0.0, avg_posterior - 1.96 * std_dev / math.sqrt(max(len(final_positions), 1)))
        ci_high = min(1.0, avg_posterior + 1.96 * std_dev / math.sqrt(max(len(final_positions), 1)))

        # Determine verdict label
        if avg_posterior >= 0.70:
            label = VerdictLabel.SUPPORTED
        elif avg_posterior <= 0.30:
            label = VerdictLabel.CHALLENGED
        elif std_dev > 0.20:
            label = VerdictLabel.QUALIFIED
        else:
            label = VerdictLabel.INDETERMINATE

        # Get top evidence
        top_evidence = docket.get_top_weighted(5)
        top_evidence_ids = [e.id for e in top_evidence]

        # Majority coalition
        majority_ids = []
        majority_names = []
        for c in coalitions:
            if len(c.member_ids) >= 2:
                majority_ids.extend(c.member_ids)
                majority_names.extend(c.member_names)

        # Generate narrative via LLM
        narrative = self._generate_majority_narrative(
            proposition, label, avg_posterior, top_evidence, docket,
        )

        return MajorityOpinion(
            verdict_label=label,
            posterior_probability=avg_posterior,
            confidence_interval=(ci_low, ci_high),
            narrative=narrative,
            key_supporting_evidence=top_evidence_ids,
            majority_coalition_ids=majority_ids,
            majority_coalition_names=majority_names,
        )

    def _generate_majority_narrative(
        self,
        proposition: str,
        label: VerdictLabel,
        posterior: float,
        top_evidence: list[DocketItem],
        docket: "EvidenceDocket",
    ) -> str:
        """Generate the majority opinion narrative via LLM."""
        stats = docket.summary_stats()
        evidence_text = "\n".join(
            f"- {e.claim_text[:150]} (DEW: {e.dew_score:.2f}, {e.polarity.value})"
            for e in top_evidence
        )

        prompt = (
            f"Proposition: {proposition}\n\n"
            f"Aggregate posterior: {posterior:.2%}\n"
            f"Verdict label: {label.value}\n"
            f"Evidence stats: {stats['total']} items — "
            f"{stats['supports']} support, {stats['attacks']} attack, "
            f"{stats['qualifies']} qualify\n"
            f"Avg DEW: {stats['avg_dew']:.2f}\n"
            f"Challenges: {stats['challenges']} ({stats['sustained_challenges']} sustained)\n\n"
            f"Top evidence:\n{evidence_text}\n"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=_MAJORITY_SYSTEM,
                temperature=0.4,
                max_tokens=2048,
            )
            text = response.content.strip()
            if "{" in text:
                import json
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                return data.get("narrative", text)
            return text
        except Exception as exc:
            logger.warning("Majority narrative LLM failed: %s", exc)

        # Fallback
        return (
            f"The AGORA Senate, through {stats['total']} evidence submissions and "
            f"{stats['challenges']} formal challenges, reached a verdict of "
            f"{label.value.upper()} with an aggregate posterior of {posterior:.0%}. "
            f"The verdict is based on {stats['supports']} supporting and "
            f"{stats['attacks']} challenging evidence items."
        )

    # ── What Would Change This ────────────────────────────────────────

    def _generate_what_would_change(
        self,
        proposition: str,
        majority: MajorityOpinion,
        docket: "EvidenceDocket",
    ) -> list[str]:
        """Generate VoI items via LLM."""
        stats = docket.summary_stats()
        prompt = (
            f"Proposition: {proposition}\n"
            f"Verdict: {majority.verdict_label.value} ({majority.posterior_probability:.0%})\n"
            f"Evidence profile: {stats['supports']} support, {stats['attacks']} attack\n"
            f"Challenges sustained: {stats['sustained_challenges']}\n"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=_CHANGE_SYSTEM,
                temperature=0.5,
                max_tokens=1024,
            )
            text = response.content.strip()
            if "[" in text:
                import json
                from argus.core.json_repair import extract_json_array
                items = extract_json_array(text)
                if items and isinstance(items[0], str):
                    return items[:5]
                elif items and isinstance(items[0], dict):
                    return [str(i) for i in items[:5]]
        except Exception as exc:
            logger.warning("What Would Change LLM failed: %s", exc)

        # Fallback
        return [
            "New large-scale empirical study directly addressing the proposition",
            "Cross-domain meta-analysis synthesising evidence from related fields",
            "Independent replication of the most contested evidence items",
            "Expert testimony from a domain not represented in the current Senate",
            "Historical case study with close parallels to the current proposition",
        ]
