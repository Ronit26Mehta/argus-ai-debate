"""
Field Marshal — HANNIBAL's neutral arbiter.

Provides:
    1. Tie-break resolution for drawn skirmishes / engagements
    2. Armistice Protocol — when no Force achieves clear dominance
    3. Campaign verdict narrative synthesis
    4. Calibration reset — anti-escalation mechanism

The Field Marshal is the only neutral entity in the campaign.  It has
no prior commitment to any Force's position.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from argus.hannibal.models import (
    ArmisticeOption,
    CampaignVerdict,
    CampaignVerdictLabel,
    EngagementResult,
    ForceSpec,
    ForceType,
    HannibalSessionConfig,
    SkirmishResult,
    TournamentNode,
    VictoryStrength,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# LLM Prompts
# ══════════════════════════════════════════════════════════════════════

_TIE_BREAK_SYSTEM = """\
You are HANNIBAL's Field Marshal — a neutral arbiter resolving ties in \
epistemic combat.  You have no allegiance to any Force.

A skirmish on the topic "{scope}" has ended in a draw between:
  Force A ({force_a}): {evidence_a_summary}
  Force B ({force_b}): {evidence_b_summary}

Evaluate which side presented marginally stronger evidence overall.
Consider: source quality, logical coherence, relevance to scope, and
specificity of claims.

Output ONLY valid JSON:
{{
  "winner": "force_a" or "force_b",
  "confidence": 0.0,
  "reasoning": "Brief 1-2 sentence explanation."
}}
"""

_ARMISTICE_SYSTEM = """\
You are HANNIBAL's Field Marshal evaluating whether an Armistice should \
be declared.

Campaign Strength Score: {strength_score:.3f}
Victory Strength: {strength_label}
Winning Force: {winner}
Campaign Evidence Summary: {evidence_summary}

An Armistice is warranted when the campaign result is too narrow or \
contested for a confident verdict.

Options:
1. "narrow_verdict" — Issue a qualified verdict acknowledging limits
2. "redirect_agora" — The question needs deliberative governance (AGORA)
3. "redirect_aristotle" — The question needs deeper investigation (ARISTOTLE)

Output ONLY valid JSON:
{{
  "option": "narrow_verdict" or "redirect_agora" or "redirect_aristotle",
  "reasoning": "Why this option is appropriate.",
  "conditions": ["What would need to change for a stronger verdict"]
}}
"""

_VERDICT_NARRATIVE_SYSTEM = """\
You are HANNIBAL's Field Marshal writing the final campaign verdict narrative.

Proposition: {proposition}
Verdict: {verdict_label}
Winning Force: {winner} (position: {position})
Campaign Strength: {strength} ({strength_label})
Skirmishes: {num_skirmishes} | Total Evidence: {total_evidence}

Write a 4-6 sentence strategic narrative explaining:
1. What the campaign revealed about the proposition
2. How the winning Force achieved dominance
3. Where the losing Force's arguments fell short
4. The confidence level of the verdict

Write in an authoritative, analytical tone.  Do not use bullet points.
"""


class FieldMarshal:
    """Neutral arbiter for tie-breaks, armistice, and verdict narration.

    The Field Marshal makes no partisan judgements — it resolves
    procedural uncertainties and constructs the final narrative.
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    # ── Tie-Break Resolution ───────────────────────────────────────

    def break_tie(
        self,
        node: TournamentNode,
        evidence_a_summary: str,
        evidence_b_summary: str,
        force_a: ForceSpec,
        force_b: ForceSpec,
    ) -> SkirmishResult:
        """Break a skirmish tie using LLM evaluation.

        Called when the ECS margin is below the confidence threshold.
        """
        try:
            system = _TIE_BREAK_SYSTEM.format(
                scope=node.topic_scope[:200],
                force_a=force_a.force_type.display_name,
                force_b=force_b.force_type.display_name,
                evidence_a_summary=evidence_a_summary[:300],
                evidence_b_summary=evidence_b_summary[:300],
            )
            response = self.llm.generate(
                prompt="Break this skirmish tie.",
                system_prompt=system,
                temperature=0.3,
                max_tokens=512,
            )
            text = response.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])

                winner_str = data.get("winner", "force_a")
                confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                reasoning = data.get("reasoning", "")

                if winner_str == "force_b":
                    winner = force_b.force_type
                    loser = force_a.force_type
                else:
                    winner = force_a.force_type
                    loser = force_b.force_type

                return SkirmishResult(
                    skirmish_id=node.id,
                    winner_force=winner,
                    loser_force=loser,
                    confidence_score=confidence * 0.5,  # Reduced for tie-break
                    is_draw=False,
                    adjudication_summary=f"[FIELD MARSHAL TIE-BREAK] {reasoning}",
                )

        except Exception as exc:
            logger.warning("Field Marshal tie-break failed: %s", exc)

        # Default: force_a wins by coin-flip
        return SkirmishResult(
            skirmish_id=node.id,
            winner_force=force_a.force_type,
            loser_force=force_b.force_type,
            confidence_score=0.01,
            is_draw=False,
            adjudication_summary="[FIELD MARSHAL] Marginal tie-break — effectively a draw.",
        )

    # ── Engagement Tie-Break ───────────────────────────────────────

    def break_engagement_tie(
        self,
        engagement_id: str,
        force_a: ForceType,
        force_b: ForceType,
    ) -> EngagementResult:
        """Break an engagement tie — simpler than skirmish tie-break."""
        # At engagement level, defer to the force with more skirmish wins
        # Since this is called only when even, slight edge to PF by convention
        return EngagementResult(
            engagement_id=engagement_id,
            winner_force=force_a,
            margin=0.01,
        )

    # ── Armistice Protocol ─────────────────────────────────────────

    def evaluate_armistice(
        self,
        verdict: CampaignVerdict,
        config: HannibalSessionConfig,
        evidence_summary: str = "",
    ) -> tuple[bool, ArmisticeOption | None, str]:
        """Evaluate whether the Armistice Protocol should fire.

        Returns:
            (should_fire, option, details)
        """
        strength = verdict.campaign_strength_score

        # Armistice fires when victory is too contested
        if strength >= config.armistice_threshold:
            return False, None, ""

        logger.info("Field Marshal: Armistice threshold triggered (%.3f < %.3f)",
                     strength, config.armistice_threshold)

        try:
            system = _ARMISTICE_SYSTEM.format(
                strength_score=strength,
                strength_label=verdict.campaign_strength_label.value,
                winner=verdict.winning_force.display_name,
                evidence_summary=evidence_summary[:500],
            )
            response = self.llm.generate(
                prompt="Should the Armistice Protocol fire?",
                system_prompt=system,
                temperature=0.3,
                max_tokens=512,
            )
            text = response.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])

                option_str = data.get("option", "narrow_verdict")
                try:
                    option = ArmisticeOption(option_str)
                except ValueError:
                    option = ArmisticeOption.NARROW_VERDICT

                reasoning = data.get("reasoning", "")
                conditions = data.get("conditions", [])
                details = f"{reasoning}\nConditions: {'; '.join(conditions)}"

                return True, option, details

        except Exception as exc:
            logger.warning("Field Marshal armistice evaluation failed: %s", exc)

        return True, ArmisticeOption.NARROW_VERDICT, (
            "Campaign strength below threshold — issuing narrow verdict."
        )

    # ── Verdict Narration ──────────────────────────────────────────

    def narrate_verdict(
        self,
        verdict: CampaignVerdict,
        proposition: str,
        num_skirmishes: int = 0,
        total_evidence: int = 0,
    ) -> str:
        """Generate the final verdict narrative via LLM."""
        try:
            position = verdict.position_description[:200] if verdict.position_description else "N/A"
            system = _VERDICT_NARRATIVE_SYSTEM.format(
                proposition=proposition[:200],
                verdict_label=verdict.verdict_label.value,
                winner=verdict.winning_force.display_name,
                position=position,
                strength=f"{verdict.campaign_strength_score:.0%}",
                strength_label=verdict.campaign_strength_label.value,
                num_skirmishes=num_skirmishes,
                total_evidence=total_evidence,
            )
            response = self.llm.generate(
                prompt="Write the campaign verdict narrative.",
                system_prompt=system,
                temperature=0.5,
                max_tokens=512,
            )
            narrative = response.content.strip()
            if narrative and len(narrative) > 20:
                return narrative

        except Exception as exc:
            logger.warning("Field Marshal narrative failed: %s", exc)

        # Fallback narrative
        return (
            f"The HANNIBAL campaign concluded with a "
            f"{verdict.campaign_strength_label.value.lower()} verdict of "
            f"'{verdict.verdict_label.value}' in favour of the "
            f"{verdict.winning_force.display_name}. "
            f"Across {num_skirmishes} skirmishes and {total_evidence} evidence "
            f"items, the winning force demonstrated a campaign strength of "
            f"{verdict.campaign_strength_score:.0%}."
        )

    # ── Calibration Reset ──────────────────────────────────────────

    @staticmethod
    def calibration_reset(
        forces: list[ForceSpec],
    ) -> dict[str, float]:
        """Anti-escalation: reset extreme posteriors toward 0.5.

        If any force posterior has moved beyond 0.95 or below 0.05,
        apply a mild regression toward 0.5 to prevent lock-in.
        """
        adjustments: dict[str, float] = {}
        for force in forces:
            post = force.force_posterior
            if post > 0.95:
                new = 0.90
                force.force_posterior = new
                adjustments[force.force_type.value] = new
            elif post < 0.05:
                new = 0.10
                force.force_posterior = new
                adjustments[force.force_type.value] = new
        return adjustments
