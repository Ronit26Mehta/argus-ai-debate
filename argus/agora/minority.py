"""
Minority Report Engine — guarantees the dissenting view is always delivered.

Unlike traditional deliberation, AGORA treats the minority position as a
first-class output. The engine constructs a complete Minority Report with:

    1. The minority claim
    2. Evidence that the majority did NOT address
    3. Sustained challenges that weakened the majority
    4. "What Would Change This" — VoI (Value of Information) extension
"""

from __future__ import annotations

import logging
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
    SenatorScorecard,
    SessionPhase,
    VerdictLabel,
    _utcnow,
)

if TYPE_CHECKING:
    from argus.agora.coalitions import CoalitionDetectionEngine
    from argus.agora.docket import EvidenceDocket, ChallengeHandler
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── LLM prompts ───────────────────────────────────────────────────────

_MINORITY_SYSTEM = """\
You are AGORA's Minority Report Generator. Given the majority verdict, the \
minority senators' evidence, and sustained challenges:

1. Articulate the minority's central claim
2. Explain why the majority verdict is insufficient
3. List evidence the majority failed to address
4. Suggest "What Would Change This" — what new evidence or analysis \
could convert the minority position into the majority

Output a JSON object:
{
  "minority_claim": "...",
  "why_majority_insufficient": "...",
  "what_would_change": ["item1", "item2", "item3"],
  "narrative": "2-3 paragraph summary of the minority position"
}
"""


class MinorityReportEngine:
    """Generates the Minority Report — always a first-class AGORA output.

    If the majority is unanimous, the report documents why dissent was
    absent (which is itself epistemic information).
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def generate(
        self,
        proposition: str,
        majority_opinion: MajorityOpinion,
        docket: "EvidenceDocket",
        coalitions: list[Any],
        scorecards: list[SenatorScorecard],
        position_trajectories: dict[str, list[float]],
    ) -> MinorityReport:
        """Generate the Minority Report.

        Identifies minority senators from final positions,
        collects their evidence and challenges, and generates
        the full report via LLM.
        """
        # Identify minority senators
        minority_ids, minority_names = self._identify_minority(
            majority_opinion, position_trajectories, scorecards,
        )

        if not minority_ids:
            # Unanimous — still generate a report noting absence of dissent
            return MinorityReport(
                minority_claim="No significant dissent detected.",
                why_majority_insufficient="The assembly reached broad consensus.",
                what_would_change=[
                    "New empirical evidence from a previously unconsidered domain",
                    "A re-evaluation of methodological assumptions",
                    "Stakeholder perspectives not represented in the current Senate",
                ],
                minority_senator_ids=[],
                minority_senator_names=[],
                narrative=(
                    "No substantial minority position emerged during deliberation. "
                    "The absence of dissent itself warrants scrutiny — it may indicate "
                    "genuine consensus or may reflect insufficient adversarial pressure. "
                    "Future sessions should consider whether the Senate composition "
                    "adequately represented the full stance space."
                ),
            )

        # Collect minority evidence
        minority_evidence = []
        for sid in minority_ids:
            minority_evidence.extend(docket.get_by_senator(sid))

        # Collect sustained challenges from minority
        sustained_challenges = []
        for item in docket.all_items:
            for ch in docket.challenge_handler.get_challenges_for_evidence(item.id):
                if ch.outcome == ChallengeOutcome.SUSTAINED and ch.challenger_id in minority_ids:
                    sustained_challenges.append(ch.id)

        # Generate via LLM
        report = self._generate_report_llm(
            proposition, majority_opinion, minority_names,
            minority_evidence, sustained_challenges,
        )

        report.minority_senator_ids = minority_ids
        report.minority_senator_names = minority_names
        report.supporting_evidence_ids = [e.id for e in minority_evidence]
        report.sustained_challenges = sustained_challenges

        return report

    def _identify_minority(
        self,
        majority: MajorityOpinion,
        trajectories: dict[str, list[float]],
        scorecards: list[SenatorScorecard],
    ) -> tuple[list[str], list[str]]:
        """Identify minority senators from final positions.

        A senator is in the minority if their final position diverges
        significantly from the majority posterior.
        """
        majority_posterior = majority.posterior_probability
        minority_ids: list[str] = []
        minority_names: list[str] = []

        for sc in scorecards:
            if not sc.position_trajectory:
                continue
            final_position = sc.position_trajectory[-1]
            # Senator diverges if their position is > 0.25 away from majority
            if abs(final_position - majority_posterior) > 0.25:
                minority_ids.append(sc.senator_id)
                minority_names.append(sc.senator_name)

        return minority_ids, minority_names

    def _generate_report_llm(
        self,
        proposition: str,
        majority: MajorityOpinion,
        minority_names: list[str],
        minority_evidence: list[DocketItem],
        sustained_challenge_ids: list[str],
    ) -> MinorityReport:
        """Generate the minority report narrative via LLM."""
        evidence_summary = "\n".join(
            f"- [{e.evidence_id_display}] {e.claim_text[:150]} (DEW: {e.dew_score:.2f})"
            for e in minority_evidence[:15]
        )

        prompt = (
            f"Proposition: {proposition}\n\n"
            f"Majority verdict: {majority.verdict_label.value} "
            f"(posterior: {majority.posterior_probability:.2%})\n\n"
            f"Minority senators: {', '.join(minority_names)}\n\n"
            f"Minority evidence:\n{evidence_summary}\n\n"
            f"Sustained challenges from minority: {len(sustained_challenge_ids)}\n"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=_MINORITY_SYSTEM,
                temperature=0.4,
                max_tokens=2048,
            )
            text = response.content.strip()
            if "{" in text:
                import json
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                return MinorityReport(
                    minority_claim=data.get("minority_claim", ""),
                    why_majority_insufficient=data.get("why_majority_insufficient", ""),
                    what_would_change=data.get("what_would_change", []),
                    narrative=data.get("narrative", ""),
                )
        except Exception as exc:
            logger.warning("Minority report LLM failed: %s", exc)

        # Fallback
        return MinorityReport(
            minority_claim=(
                f"A coalition of {len(minority_names)} senators "
                "maintained positions contrary to the majority."
            ),
            why_majority_insufficient=(
                "The minority contends that key evidence was insufficiently weighted "
                "and sustained challenges were not adequately addressed."
            ),
            what_would_change=[
                "Additional empirical studies addressing the minority's concerns",
                "Independent replication of contested findings",
                "Expanded cross-domain analysis",
            ],
            narrative=(
                f"Senators {', '.join(minority_names)} dissented from the majority "
                f"verdict of {majority.verdict_label.value}. "
                f"The minority submitted {len(minority_evidence)} evidence items "
                f"and sustained {len(sustained_challenge_ids)} challenges. "
                "Their position merits further investigation."
            ),
        )

    def make_minority_record(
        self,
        report: MinorityReport,
        phase: SessionPhase,
    ) -> SenateRecordEntry:
        """Generate a Senate Record entry for the Minority Report."""
        return SenateRecordEntry(
            entry_type=RecordEntryType.MINORITY_REPORT,
            phase=phase,
            content=(
                f"Minority Report filed by {', '.join(report.minority_senator_names)}. "
                f"Claim: {report.minority_claim[:200]}"
            ),
            metadata=report.to_dict(),
        )
