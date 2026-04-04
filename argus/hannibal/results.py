"""
Campaign Result Architecture — HANNIBAL Protocol.

Assembles the nine-component Campaign Result Set:
    1. Campaign Verdict
    2. Battle Map Summary
    3. Campaign Minority Record
    4. Force Performance Scorecards
    5. Decisive Evidence Record
    6. Encirclement Report (CANNAE only)
    7. What Would Change This Verdict
    8. Armistice Record
    9. Campaign Log (Field Manual, hash-chain sealed)

Also computes the Battle Efficiency Score (BES) per force.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from argus.core.json_repair import (
    extract_json_array as _extract_json_array,
)

from argus.hannibal.models import (
    ArmisticeOption,
    BattleMap,
    CampaignLogEntry,
    CampaignLogEventType,
    CampaignMinorityRecord,
    CampaignVerdict,
    DecisiveEvidenceRecord,
    EvidenceItem,
    ForcePerformanceScorecard,
    ForceSpec,
    ForceType,
    HannibalResult,
    HannibalSessionConfig,
    SkirmishResult,
    TournamentNode,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.hannibal.tournament import TournamentTree

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# LLM Prompts
# ══════════════════════════════════════════════════════════════════════

_MINORITY_SYSTEM = """\
You are HANNIBAL's Field Marshal assembling the Campaign Minority Record — \
documenting the losing force's strongest surviving arguments.

Losing Force: {losing_force}
Their position: {position}

Evidence that was NOT decisively countered:
{surviving_evidence}

Generate a structured minority record:
1. Identify 2-3 surviving arguments that the winning force did NOT fully refute
2. Identify 1-2 sustained challenges that still have merit
3. Identify 2-3 conditions under which the losing force could prevail

Output ONLY valid JSON:
{{
  "surviving_arguments": ["...", "..."],
  "sustained_challenges": ["...", "..."],
  "conditions_to_prevail": ["...", "..."],
  "narrative": "2-3 sentence summary."
}}
"""

_WHAT_WOULD_CHANGE_SYSTEM = """\
You are HANNIBAL's Field Marshal. Given the campaign outcome, carefully analyze the debate and identify \
3-5 specific conditions, new empirical evidence, or future occurrences that, if introduced, could logically flip or alter the verdict.

Proposition: {proposition}
Current verdict: {verdict} (winner: {winner})
Campaign strength: {strength}

IMPORTANT: Provide REAL analytical conditions based on the proposition. Do NOT use placeholder text.

Output ONLY a valid JSON array of strings:
[
  "Condition 1: If a new peer-reviewed study proves...",
  "Condition 2: If the major opposing force introduces evidence of...",
  "..."
]
"""


class HannibalResultBuilder:
    """Assembles the complete Campaign Result from battle data."""

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def build(
        self,
        proposition: str,
        battle_map: BattleMap,
        verdict: CampaignVerdict,
        forces: list[ForceSpec],
        tree: "TournamentTree",
        all_evidence: list[EvidenceItem],
        campaign_log: list[CampaignLogEntry],
        force_posterior_history: dict[str, list[float]],
        duration_seconds: float = 0.0,
        armistice_fired: bool = False,
        armistice_option: ArmisticeOption | None = None,
        armistice_details: str = "",
        encirclement_report: dict[str, Any] | None = None,
    ) -> HannibalResult:
        """Assemble the full nine-component result set.

        This runs a few final LLM calls for the minority record and
        what-would-change analysis.
        """
        result = HannibalResult(
            proposition=proposition,
            battle_map_id=battle_map.id,
            verdict=verdict,
        )

        # 2. Battle Map Summary + Tournament Tree state
        result.battle_map_summary = battle_map.to_dict()
        result.battle_map_summary["tree_state"] = tree.get_bracket_state()

        # 3. Campaign Minority Record
        losing_forces = [
            f for f in forces
            if f.force_type != verdict.winning_force
        ]
        if losing_forces:
            losing_evidence = [
                e for e in all_evidence
                if e.force_type == losing_forces[0].force_type
            ]
            result.minority_record = self._build_minority_record(
                losing_forces[0], losing_evidence,
            )

        # 4. Force Performance Scorecards
        result.scorecards = self._build_scorecards(
            forces, all_evidence, tree,
        )

        # 5. Decisive Evidence Record
        result.decisive_evidence = self._build_decisive_evidence(all_evidence)

        # 6. Encirclement Report
        result.encirclement_report = encirclement_report or {}

        # 7. What Would Change This Verdict
        result.what_would_change = self._build_what_would_change(
            proposition, verdict,
        )

        # 8. Armistice Record
        result.armistice_fired = armistice_fired
        result.armistice_option = armistice_option
        result.armistice_details = armistice_details

        # 9. Campaign Log + hash-chain seal
        result.campaign_log = campaign_log
        result.log_seal_hash = self._seal_log(campaign_log)

        # Session metadata
        result.force_posterior_history = force_posterior_history
        result.num_skirmishes = tree.total_skirmishes
        result.num_engagements = len(tree.engagement_nodes)
        result.num_theatres = len(tree.theatre_nodes)
        result.total_evidence = len(all_evidence)
        result.duration_seconds = duration_seconds

        return result

    # ── Minority Record ────────────────────────────────────────────

    def _build_minority_record(
        self,
        losing_force: ForceSpec,
        losing_evidence: list[EvidenceItem],
    ) -> CampaignMinorityRecord:
        """Build the Campaign Minority Record for the losing force."""
        try:
            # Find high-quality surviving evidence (EVID-Q > 0.5)
            strong_evidence = sorted(
                [e for e in losing_evidence if e.evid_q > 0.4],
                key=lambda e: e.effective_weight,
                reverse=True,
            )[:5]

            surviving_text = "\n".join(
                f"  - {e.claim_text} (EVID-Q: {e.evid_q:.2f})"
                for e in strong_evidence
            ) or "No significant surviving evidence."

            system = _MINORITY_SYSTEM.format(
                losing_force=losing_force.force_type.display_name,
                position=losing_force.position_description[:200],
                surviving_evidence=surviving_text,
            )
            response = self.llm.generate(
                prompt="Generate the Campaign Minority Record.",
                system_prompt=system,
                temperature=0.5,
                max_tokens=1024,
            )
            text = response.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])

                return CampaignMinorityRecord(
                    losing_force=losing_force.force_type,
                    surviving_arguments=data.get("surviving_arguments", []),
                    sustained_challenges=data.get("sustained_challenges", []),
                    conditions_to_prevail=data.get("conditions_to_prevail", []),
                    narrative=data.get("narrative", ""),
                )
        except Exception as exc:
            logger.warning("Minority record LLM failed: %s", exc)

        return CampaignMinorityRecord(
            losing_force=losing_force.force_type,
            narrative=(
                f"The {losing_force.force_type.display_name} presented arguments "
                f"that were ultimately outpaced by the winning force, but several "
                f"claims remain unrefuted."
            ),
        )

    # ── Force Performance Scorecards ───────────────────────────────

    def _build_scorecards(
        self,
        forces: list[ForceSpec],
        all_evidence: list[EvidenceItem],
        tree: "TournamentTree",
    ) -> list[ForcePerformanceScorecard]:
        """Build performance scorecards for each force."""
        scorecards: list[ForcePerformanceScorecard] = []

        for force in forces:
            ft = force.force_type
            # Count wins/losses from skirmish nodes
            wins = losses = draws = 0
            for node in tree.skirmish_nodes:
                if not node.is_resolved:
                    continue
                is_participant = (
                    node.force_a_type == ft or node.force_b_type == ft
                )
                if not is_participant:
                    continue
                if node.winner_force == ft:
                    wins += 1
                elif node.confidence < 0.05:  # Near-draw
                    draws += 1
                else:
                    losses += 1

            # Count engagement wins
            eng_wins = sum(
                1 for node in tree.engagement_nodes
                if node.is_resolved and node.winner_force == ft
            )

            # Evidence stats
            force_evidence = [e for e in all_evidence if e.force_type == ft]
            evidence_count = len(force_evidence)
            avg_q = (
                sum(e.evid_q for e in force_evidence) / max(evidence_count, 1)
            )

            # Flanking stats
            flanking_evidence = [
                e for e in force_evidence if e.is_counter_evidence
            ]
            flanking_rate = (
                len(flanking_evidence) / max(evidence_count, 1)
            )

            # BES = (Skirmishes_Won × Avg_Confidence_Won) / Total_Evidence
            avg_conf_won = 0.0
            if wins > 0:
                winning_confs = [
                    n.confidence for n in tree.skirmish_nodes
                    if n.is_resolved and n.winner_force == ft
                ]
                avg_conf_won = sum(winning_confs) / max(len(winning_confs), 1)

            bes = (
                (wins * avg_conf_won) / max(evidence_count, 1)
            ) if evidence_count > 0 else 0.0

            scorecards.append(ForcePerformanceScorecard(
                force_type=ft,
                skirmishes_won=wins,
                skirmishes_lost=losses,
                skirmishes_drawn=draws,
                engagements_won=eng_wins,
                evidence_submitted=evidence_count,
                avg_evid_q=round(avg_q, 3),
                flanking_attack_success_rate=round(flanking_rate, 3),
                battle_efficiency_score=round(bes, 4),
            ))

        return scorecards

    # ── Decisive Evidence ──────────────────────────────────────────

    @staticmethod
    def _build_decisive_evidence(
        all_evidence: list[EvidenceItem],
    ) -> DecisiveEvidenceRecord:
        """Select top evidence items by effective weight."""
        sorted_ev = sorted(
            all_evidence,
            key=lambda e: e.effective_weight,
            reverse=True,
        )
        return DecisiveEvidenceRecord(items=sorted_ev[:8])

    # ── What Would Change ──────────────────────────────────────────

    def _build_what_would_change(
        self,
        proposition: str,
        verdict: CampaignVerdict,
    ) -> list[str]:
        """Identify conditions that could flip the verdict."""
        try:
            system = _WHAT_WOULD_CHANGE_SYSTEM.format(
                proposition=proposition[:200],
                verdict=verdict.verdict_label.value,
                winner=verdict.winning_force.display_name,
                strength=f"{verdict.campaign_strength_score:.0%}",
            )
            response = self.llm.generate(
                prompt="What would change this verdict?",
                system_prompt=system,
                temperature=0.5,
                max_tokens=512,
            )
            items = _extract_json_array(response.content)
            if items and isinstance(items[0], str):
                return items[:5]
            elif items and isinstance(items[0], dict):
                return [str(i) for i in items[:5]]
        except Exception as exc:
            logger.warning("What-would-change LLM failed: %s", exc)

        return [
            "New empirical evidence contradicting the winning force's key claims",
            "Methodological critique undermining the strongest evidence items",
            "A significant shift in domain expert consensus",
        ]

    # ── Log Sealing ────────────────────────────────────────────────

    @staticmethod
    def _seal_log(log_entries: list[CampaignLogEntry]) -> str:
        """Create a SHA-256 hash chain over the campaign log.

        Each entry's hash depends on the previous entry's hash,
        creating an immutable provenance trail.
        """
        chain_hash = "0" * 64  # Genesis hash
        for entry in log_entries:
            data = f"{chain_hash}:{entry.id}:{entry.event_type.value}:{entry.content}"
            chain_hash = hashlib.sha256(data.encode()).hexdigest()
        return chain_hash
