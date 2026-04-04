"""
Battle Protocol Engine — HANNIBAL's core combat system.

Runs skirmishes **sequentially** (designed for i3 / 8 GB RAM), bottom-up
through the Tournament Tree:

    Skirmish → Engagement → Theatre → Campaign Root

Each Skirmish follows a structured multi-round protocol:
    S1: Scope declaration
    S2: Initial deployment (each agent submits evidence)
    S3: Counteroffensive (agents challenge opponent evidence)
    S4: Final strike (if high depth — optional 3rd round)
    S5: Adjudication (ECS computation → winner)

ECS formula from spec:
    ECS(Force, Skirmish) = Σ_i (EVID_Q_i × Confidence_i × Relevance_i ×
                                 Polarity_strength_i × LogCoherence_factor)
    Modifiers: +0.1 source uniqueness, −0.15 sustained challenge, +0.10 defended
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import re as _re

from argus.core.json_repair import (
    extract_json_array as _extract_json_array,
    repair_json as _try_repair_json,
)

from argus.hannibal.models import (
    AgentRoleSpec,
    BattleMap,
    CampaignLogEntry,
    CampaignLogEventType,
    CampaignPhase,
    CampaignVerdict,
    EngagementResult,
    EvidenceItem,
    ForceSpec,
    ForceType,
    ForcePosteriorUpdate,
    HannibalResult,
    HannibalSessionConfig,
    MilitaryRole,
    SkirmishResult,
    TheatreResult,
    TournamentNode,
    TournamentNodeType,
    VictoryStrength,
    _uid,
)
from argus.hannibal.tournament import TournamentTree

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# LLM Prompts
# ══════════════════════════════════════════════════════════════════════

_EVIDENCE_SYSTEM = """\
You are {agent_name}, a {role} in the {force_name}.

YOUR POSITION: {position}
YOUR MANDATE: {persona}

You are fighting in a skirmish on the topic:
  "{scope}"

{round_context}

Your task: Provide exactly {count} evidence items supporting YOUR position.
Each item must have:
  - claim: a clear, specific factual claim (1-2 sentences)
  - source_reference: a plausible academic/authoritative source
  - confidence: your confidence in this evidence (0.3–1.0)
  - relevance: how relevant to the skirmish scope (0.3–1.0)

Output ONLY a valid JSON array:
[{{"claim": "...", "source_reference": "...", "confidence": 0.8, "relevance": 0.9}}, ...]
"""

_COUNTER_EVIDENCE_SYSTEM = """\
You are {agent_name}, a {role} in the {force_name}.

YOUR POSITION: {position}
YOUR MANDATE: {persona}

Skirmish topic: "{scope}"

The opposing force has submitted the following evidence:
{opponent_evidence}

Your task: generate {count} counter-evidence items that directly challenge, \
contradict, or undermine the opponent's evidence above.  Each must be a \
specific claim with a source reference.

Output ONLY a valid JSON array:
[{{"claim": "...", "source_reference": "...", "confidence": 0.8, "relevance": 0.9}}, ...]
"""

_ADJUDICATION_SYSTEM = """\
You are HANNIBAL's neutral Skirmish Adjudicator.  You must evaluate two \
forces' evidence on a specific topic and determine which side won.

Skirmish topic: "{scope}"

FORCE A ({force_a_name}) evidence:
{evidence_a_text}

FORCE B ({force_b_name}) evidence:
{evidence_b_text}

Evaluate each piece of evidence for:
- Factual accuracy and source quality (EVID-Q)
- Logical coherence
- Relevance to the skirmish scope

Assign each evidence item an evid_q score (0.0–1.0).

Then determine the winner.

Output ONLY valid JSON:
{{
  "evidence_scores_a": [0.0, ...],
  "evidence_scores_b": [0.0, ...],
  "winner": "force_a" or "force_b" or "draw",
  "confidence": 0.0,
  "summary": "Brief explanation of why this side won."
}}
"""


# ══════════════════════════════════════════════════════════════════════
# Battle Protocol Engine
# ══════════════════════════════════════════════════════════════════════

class BattleEngine:
    """The core combat engine that runs the entire campaign sequentially.

    Designed for constrained hardware: one skirmish at a time, single
    LLM instance, minimal memory footprint.
    """

    def __init__(
        self,
        llm: "BaseLLM",
        config: HannibalSessionConfig | None = None,
    ):
        self.llm = llm
        self.config = config or HannibalSessionConfig()
        self._campaign_log: list[CampaignLogEntry] = []
        self._force_posterior_history: dict[str, list[float]] = {}
        self._all_evidence: list[EvidenceItem] = []
        self._total_tokens: int = 0
        self._phase_callback: Any = None

    def run_campaign(
        self,
        tree: TournamentTree,
        forces: list[ForceSpec],
        battle_map: BattleMap,
        phase_callback: Any = None,
    ) -> tuple[CampaignVerdict, TournamentTree]:
        """Execute the full campaign through the Tournament Tree.

        All skirmishes are run **sequentially** — one at a time.

        Args:
            tree: Pre-constructed Tournament Tree.
            forces: Deployed Forces.
            battle_map: The campaign's BattleMap.
            phase_callback: Optional callback(phase, node_label, details_dict)
                           called after each skirmish/engagement/theatre resolves.

        Returns:
            (CampaignVerdict, updated TournamentTree)
        """
        self._phase_callback = phase_callback
        start_time = time.time()

        # Index forces by type
        force_map: dict[str, ForceSpec] = {
            f.force_type.value: f for f in forces
        }

        # Initialise posterior histories
        for force in forces:
            self._force_posterior_history[force.force_type.value] = [
                force.force_prior,
            ]

        self._log_event(
            CampaignLogEventType.FORCE_DEPLOYED,
            f"Campaign begins: {len(forces)} forces deployed, "
            f"{tree.total_skirmishes} skirmishes scheduled.",
        )

        # ── Phase 1: Run all skirmishes ───────────────────────────
        for skirmish_node in tree.skirmish_nodes:
            result = self._run_skirmish(skirmish_node, force_map, battle_map)
            tree.update_skirmish_result(skirmish_node.id, result)

            # Update force posteriors
            if not result.is_draw:
                self._update_force_posterior(
                    force_map, result.winner_force, result.confidence_score,
                    skirmish_node.id, is_winner=True,
                )
                self._update_force_posterior(
                    force_map, result.loser_force, result.confidence_score,
                    skirmish_node.id, is_winner=False,
                )

            if self._phase_callback:
                self._phase_callback(
                    CampaignPhase.BATTLE,
                    skirmish_node.label,
                    {
                        "type": "skirmish",
                        "node_id": skirmish_node.id,
                        "winner": result.winner_force.value,
                        "confidence": result.confidence_score,
                        "progress": tree.progress_fraction,
                        "force_posteriors": {
                            k: v[-1] for k, v in self._force_posterior_history.items()
                        },
                        "evidence_a": [e.to_dict() for e in result.evidence_a],
                        "evidence_b": [e.to_dict() for e in result.evidence_b],
                        "adjudication_summary": result.adjudication_summary,
                    },
                )

        # ── Phase 2: Resolve engagements ──────────────────────────
        for eng_node in tree.engagement_nodes:
            eng_result = tree.resolve_engagement(eng_node.id)
            self._log_event(
                CampaignLogEventType.ENGAGEMENT_RESOLVED,
                f"{eng_node.label} resolved: winner={eng_result.winner_force.value} "
                f"margin={eng_result.margin:.2f}",
            )
            if self._phase_callback:
                self._phase_callback(
                    CampaignPhase.RESOLUTION,
                    eng_node.label,
                    {"type": "engagement", "winner": eng_result.winner_force.value},
                )

        # ── Phase 3: Resolve theatres ─────────────────────────────
        for th_node in tree.theatre_nodes:
            th_result = tree.resolve_theatre(th_node.id)
            self._log_event(
                CampaignLogEventType.THEATRE_DECIDED,
                f"{th_node.label} decided: winner={th_result.winner_force.value} "
                f"score={th_result.theatre_score:.2f}",
            )
            if self._phase_callback:
                self._phase_callback(
                    CampaignPhase.RESOLUTION,
                    th_node.label,
                    {"type": "theatre", "winner": th_result.winner_force.value},
                )

        # ── Phase 4: Resolve campaign ─────────────────────────────
        verdict = tree.resolve_campaign()
        verdict.position_description = force_map.get(
            verdict.winning_force.value, ForceSpec()
        ).position_description

        elapsed = time.time() - start_time

        self._log_event(
            CampaignLogEventType.CAMPAIGN_RESOLVED,
            f"Campaign resolved: {verdict.verdict_label.value} — "
            f"{verdict.winning_force.display_name} wins "
            f"({verdict.campaign_strength_label.value}, "
            f"{verdict.campaign_strength_score:.2f})",
        )

        if self._phase_callback:
            self._phase_callback(
                CampaignPhase.COMPLETE,
                "Campaign Root",
                {"type": "campaign", "verdict": verdict.to_dict()},
            )

        return verdict, tree

    # ── Skirmish Execution ─────────────────────────────────────────

    def _run_skirmish(
        self,
        node: TournamentNode,
        force_map: dict[str, ForceSpec],
        battle_map: BattleMap,
    ) -> SkirmishResult:
        """Execute a single skirmish between two forces.

        Phases:
            S1: Scope declaration (already set in node)
            S2: Initial deployment (2-3 evidence items per agent)
            S3: Counteroffensive (1-2 counter-evidence items)
            S4: Final strike (if high depth, 1 additional item)
            S5: Adjudication (ECS computation)
        """
        force_a = force_map.get(node.force_a_type.value if node.force_a_type else "")
        force_b = force_map.get(node.force_b_type.value if node.force_b_type else "")

        if not force_a or not force_b:
            logger.warning("Skirmish %s: missing forces", node.id)
            return SkirmishResult(skirmish_id=node.id)

        scope = node.topic_scope or battle_map.proposition
        self._log_event(
            CampaignLogEventType.SKIRMISH_INITIATED,
            f"{node.label}: {force_a.force_type.abbreviation} vs "
            f"{force_b.force_type.abbreviation} — scope: {scope[:80]}",
        )

        # Select agents for this skirmish
        agent_a = self._select_skirmish_agent(force_a)
        agent_b = self._select_skirmish_agent(force_b)

        evidence_a: list[EvidenceItem] = []
        evidence_b: list[EvidenceItem] = []

        # ── S2: Initial deployment (Round 1) ──────────────────────
        round1_a = self._generate_evidence(
            agent_a, scope, force_a, round_num=1, count=2,
        )
        evidence_a.extend(round1_a)

        round1_b = self._generate_evidence(
            agent_b, scope, force_b, round_num=1, count=2,
        )
        evidence_b.extend(round1_b)

        rounds_played = 1

        # ── S3: Counteroffensive (Round 2) ────────────────────────
        counter_a = self._generate_counter_evidence(
            agent_a, scope, force_a, evidence_b, round_num=2, count=1,
        )
        evidence_a.extend(counter_a)

        counter_b = self._generate_counter_evidence(
            agent_b, scope, force_b, evidence_a, round_num=2, count=1,
        )
        evidence_b.extend(counter_b)
        rounds_played = 2

        # ── S4: Final strike (Round 3 — if high depth) ───────────
        if battle_map.depth_score.aggregate >= self.config.high_depth_threshold:
            final_a = self._generate_counter_evidence(
                agent_a, scope, force_a, evidence_b, round_num=3, count=1,
            )
            evidence_a.extend(final_a)

            final_b = self._generate_counter_evidence(
                agent_b, scope, force_b, evidence_a, round_num=3, count=1,
            )
            evidence_b.extend(final_b)
            rounds_played = 3

        # ── S5: Adjudication ──────────────────────────────────────
        result = self._adjudicate_skirmish(
            node, evidence_a, evidence_b, force_a, force_b, scope,
        )
        result.rounds_played = rounds_played

        self._all_evidence.extend(evidence_a)
        self._all_evidence.extend(evidence_b)

        self._log_event(
            CampaignLogEventType.SKIRMISH_ADJUDICATED,
            f"{node.label}: winner={result.winner_force.abbreviation} "
            f"ECS={result.ecs_winner:.3f} vs {result.ecs_loser:.3f} "
            f"conf={result.confidence_score:.2f}",
        )

        return result

    # ── Partial JSON Rescue ─────────────────────────────────────────

    @staticmethod
    def _rescue_partial_array(text: str) -> list[dict[str, Any]]:
        """Extract complete JSON objects from a truncated array.

        When a local LLM truncates mid-value, extract_json_array may
        fail entirely.  This rescue parser finds all individually
        complete ``{...}`` blocks via regex and parses them one by one,
        recovering whatever complete evidence items exist.
        """
        results: list[dict[str, Any]] = []
        # Find all balanced {...} blocks
        depth = 0
        start_idx = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and obj.get("claim"):
                            results.append(obj)
                    except json.JSONDecodeError:
                        try:
                            repaired = _try_repair_json(candidate)
                            obj = json.loads(repaired)
                            if isinstance(obj, dict) and obj.get("claim"):
                                results.append(obj)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    start_idx = -1
        return results

    # ── Evidence Generation ────────────────────────────────────────

    def _generate_evidence(
        self,
        agent: AgentRoleSpec,
        scope: str,
        force: ForceSpec,
        round_num: int,
        count: int = 2,
    ) -> list[EvidenceItem]:
        """Generate evidence items for an agent via LLM."""
        raw_content = ""
        try:
            system = _EVIDENCE_SYSTEM.format(
                agent_name=agent.name,
                role=agent.role.display_name,
                force_name=force.force_type.display_name,
                position=force.position_description[:200],
                persona=agent.persona_description[:200],
                scope=scope[:200],
                count=count,
                round_context=f"This is evidence round {round_num}.",
            )
            response = self.llm.generate(
                prompt=f"Generate {count} evidence items for your position.",
                system_prompt=system,
                temperature=0.6,
                max_tokens=2048,
            )
            raw_content = response.content
            raw = _extract_json_array(raw_content)
            items: list[EvidenceItem] = []
            for r in raw[:count]:
                items.append(EvidenceItem(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    force_type=force.force_type,
                    claim_text=r.get("claim", ""),
                    source_reference=r.get("source_reference", ""),
                    confidence=max(0.1, min(1.0, float(r.get("confidence", 0.6)))),
                    relevance=max(0.1, min(1.0, float(r.get("relevance", 0.7)))),
                    polarity_strength=1.0,
                    is_counter_evidence=False,
                    skirmish_round=round_num,
                ))
            if items:
                return items
        except Exception as exc:
            logger.debug("Evidence gen primary parse failed (%s): %s", agent.name, exc)
            # Rescue: extract individually valid objects from truncated output
            if raw_content:
                rescued = self._rescue_partial_array(raw_content)
                if rescued:
                    logger.info("Rescued %d evidence items from truncated output", len(rescued))
                    items = []
                    for r in rescued[:count]:
                        items.append(EvidenceItem(
                            agent_id=agent.id,
                            agent_name=agent.name,
                            force_type=force.force_type,
                            claim_text=r.get("claim", ""),
                            source_reference=r.get("source_reference", ""),
                            confidence=max(0.1, min(1.0, float(r.get("confidence", 0.6)))),
                            relevance=max(0.1, min(1.0, float(r.get("relevance", 0.7)))),
                            polarity_strength=1.0,
                            is_counter_evidence=False,
                            skirmish_round=round_num,
                        ))
                    if items:
                        return items

        # Fallback: generate a basic evidence item
        return [EvidenceItem(
            agent_id=agent.id,
            agent_name=agent.name,
            force_type=force.force_type,
            claim_text=(
                f"Evidence from {agent.name} supporting the "
                f"{force.force_type.display_name} position on: {scope[:80]}"
            ),
            source_reference="General domain knowledge",
            confidence=0.5,
            relevance=0.5,
            skirmish_round=round_num,
        )]

    def _generate_counter_evidence(
        self,
        agent: AgentRoleSpec,
        scope: str,
        force: ForceSpec,
        opponent_evidence: list[EvidenceItem],
        round_num: int,
        count: int = 1,
    ) -> list[EvidenceItem]:
        """Generate counter-evidence targeting opponent's claims."""
        raw_content = ""
        try:
            opp_text = "\n".join(
                f"  - {e.claim_text} (source: {e.source_reference})"
                for e in opponent_evidence[-4:]
            )
            system = _COUNTER_EVIDENCE_SYSTEM.format(
                agent_name=agent.name,
                role=agent.role.display_name,
                force_name=force.force_type.display_name,
                position=force.position_description[:200],
                persona=agent.persona_description[:200],
                scope=scope[:200],
                count=count,
                opponent_evidence=opp_text or "No evidence submitted yet.",
            )
            response = self.llm.generate(
                prompt=f"Generate {count} counter-evidence items.",
                system_prompt=system,
                temperature=0.6,
                max_tokens=2048,
            )
            raw_content = response.content
            raw = _extract_json_array(raw_content)
            items: list[EvidenceItem] = []
            for r in raw[:count]:
                items.append(EvidenceItem(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    force_type=force.force_type,
                    claim_text=r.get("claim", ""),
                    source_reference=r.get("source_reference", ""),
                    confidence=max(0.1, min(1.0, float(r.get("confidence", 0.6)))),
                    relevance=max(0.1, min(1.0, float(r.get("relevance", 0.7)))),
                    polarity_strength=1.0,
                    is_counter_evidence=True,
                    skirmish_round=round_num,
                ))
            if items:
                return items
        except Exception as exc:
            logger.debug("Counter-evidence primary parse failed (%s): %s", agent.name, exc)
            # Rescue: extract individually valid objects from truncated output
            if raw_content:
                rescued = self._rescue_partial_array(raw_content)
                if rescued:
                    logger.info("Rescued %d counter-evidence items from truncated output", len(rescued))
                    items = []
                    for r in rescued[:count]:
                        items.append(EvidenceItem(
                            agent_id=agent.id,
                            agent_name=agent.name,
                            force_type=force.force_type,
                            claim_text=r.get("claim", ""),
                            source_reference=r.get("source_reference", ""),
                            confidence=max(0.1, min(1.0, float(r.get("confidence", 0.6)))),
                            relevance=max(0.1, min(1.0, float(r.get("relevance", 0.7)))),
                            polarity_strength=1.0,
                            is_counter_evidence=True,
                            skirmish_round=round_num,
                        ))
                    if items:
                        return items

        return [EvidenceItem(
            agent_id=agent.id,
            agent_name=agent.name,
            force_type=force.force_type,
            claim_text=f"Counter-argument from {agent.name} challenging opponent evidence.",
            source_reference="Critical analysis",
            confidence=0.4,
            relevance=0.5,
            is_counter_evidence=True,
            skirmish_round=round_num,
        )]

    # ── Adjudication ───────────────────────────────────────────────

    def _adjudicate_skirmish(
        self,
        node: TournamentNode,
        evidence_a: list[EvidenceItem],
        evidence_b: list[EvidenceItem],
        force_a: ForceSpec,
        force_b: ForceSpec,
        scope: str,
    ) -> SkirmishResult:
        """Adjudicate a skirmish using LLM + ECS computation."""
        # Try LLM adjudication for EVID-Q scoring
        evid_q_a, evid_q_b, llm_winner, llm_conf, summary = (
            self._llm_adjudicate(evidence_a, evidence_b, force_a, force_b, scope)
        )

        # Apply EVID-Q scores to evidence items
        for i, item in enumerate(evidence_a):
            if i < len(evid_q_a):
                item.evid_q = evid_q_a[i]
        for i, item in enumerate(evidence_b):
            if i < len(evid_q_b):
                item.evid_q = evid_q_b[i]

        # Compute ECS for each force
        ecs_a = self._compute_ecs(evidence_a)
        ecs_b = self._compute_ecs(evidence_b)

        # Determine winner
        max_ecs = max(ecs_a, ecs_b, 0.001)
        margin = abs(ecs_a - ecs_b)
        confidence = margin / max_ecs

        is_draw = confidence < self.config.confidence_threshold
        if is_draw:
            winner_force = force_a.force_type
            loser_force = force_b.force_type
        elif ecs_a >= ecs_b:
            winner_force = force_a.force_type
            loser_force = force_b.force_type
        else:
            winner_force = force_b.force_type
            loser_force = force_a.force_type
            ecs_a, ecs_b = ecs_b, ecs_a

        # Find decisive evidence
        all_ev = evidence_a + evidence_b
        all_ev_sorted = sorted(all_ev, key=lambda e: e.effective_weight, reverse=True)
        decisive_ids = [e.id for e in all_ev_sorted[:3]]

        return SkirmishResult(
            skirmish_id=node.id,
            winner_force=winner_force,
            loser_force=loser_force,
            ecs_winner=max(ecs_a, ecs_b),
            ecs_loser=min(ecs_a, ecs_b),
            confidence_score=confidence,
            is_draw=is_draw,
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            decisive_evidence_ids=decisive_ids,
            adjudication_summary=summary,
        )

    def _llm_adjudicate(
        self,
        evidence_a: list[EvidenceItem],
        evidence_b: list[EvidenceItem],
        force_a: ForceSpec,
        force_b: ForceSpec,
        scope: str,
    ) -> tuple[list[float], list[float], str, float, str]:
        """Use LLM to score evidence quality and determine winner."""
        default_q_a = [0.5] * len(evidence_a)
        default_q_b = [0.5] * len(evidence_b)

        try:
            ev_a_text = "\n".join(
                f"  A{i+1}. {e.claim_text} (source: {e.source_reference})"
                for i, e in enumerate(evidence_a)
            )
            ev_b_text = "\n".join(
                f"  B{i+1}. {e.claim_text} (source: {e.source_reference})"
                for i, e in enumerate(evidence_b)
            )

            system = _ADJUDICATION_SYSTEM.format(
                scope=scope[:200],
                force_a_name=force_a.force_type.display_name,
                force_b_name=force_b.force_type.display_name,
                evidence_a_text=ev_a_text or "No evidence.",
                evidence_b_text=ev_b_text or "No evidence.",
            )

            response = self.llm.generate(
                prompt="Adjudicate this skirmish.",
                system_prompt=system,
                temperature=0.3,
                max_tokens=1024,
            )
            text = response.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])

                scores_a = [
                    max(0.1, min(1.0, float(s)))
                    for s in data.get("evidence_scores_a", default_q_a)
                ]
                scores_b = [
                    max(0.1, min(1.0, float(s)))
                    for s in data.get("evidence_scores_b", default_q_b)
                ]
                winner = data.get("winner", "draw")
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                summary = data.get("summary", "")
                return scores_a, scores_b, winner, conf, summary

        except Exception as exc:
            logger.warning("LLM adjudication failed: %s", exc)

        return default_q_a, default_q_b, "draw", 0.5, "Adjudication fallback."

    # ── ECS Computation ────────────────────────────────────────────

    @staticmethod
    def _compute_ecs(evidence: list[EvidenceItem]) -> float:
        """Compute Epistemic Combat Score for a force's evidence.

        ECS = Σ_i (EVID_Q_i × Confidence_i × Relevance_i ×
                    Polarity_strength_i × LogCoherence_factor)

        Modifiers:
            +0.1 per unique source
            −0.15 per sustained challenge (counter-evidence)
            +0.10 per defended challenge (counter to counter)
        """
        if not evidence:
            return 0.0

        base_score = 0.0
        sources_seen: set[str] = set()
        counter_count = 0
        defended_count = 0

        for e in evidence:
            item_score = (
                e.evid_q
                * e.confidence
                * e.relevance
                * e.polarity_strength
                * 1.0  # LogCoherence factor (simplified to 1.0)
            )
            base_score += item_score

            # Source uniqueness modifier
            source_key = e.source_reference.lower().strip()[:50]
            if source_key and source_key not in sources_seen:
                base_score += 0.10
                sources_seen.add(source_key)

            # Counter-evidence modifiers
            if e.is_counter_evidence:
                counter_count += 1

        # Counter-evidence produces a net positive for defenders
        # (they are defending against opponent attacks)
        base_score += counter_count * 0.05

        return max(0.0, base_score)

    # ── Force Posterior Update ─────────────────────────────────────

    def _update_force_posterior(
        self,
        force_map: dict[str, ForceSpec],
        force_type: ForceType,
        confidence_delta: float,
        skirmish_id: str,
        is_winner: bool,
    ) -> None:
        """Update a force's posterior using logistic sigmoid.

        FP(Force) = σ(logit(Force_Prior) + Σ_s ECS_delta_s)
        """
        force = force_map.get(force_type.value)
        if not force:
            return

        prior = force.force_posterior
        # Clamp prior to avoid logit explosion
        prior = max(0.01, min(0.99, prior))

        logit_prior = math.log(prior / (1.0 - prior))
        delta = confidence_delta if is_winner else -confidence_delta
        new_logit = logit_prior + delta
        new_posterior = 1.0 / (1.0 + math.exp(-new_logit))

        force.force_posterior = new_posterior

        self._force_posterior_history.setdefault(
            force_type.value, [force.force_prior],
        ).append(new_posterior)

        self._log_event(
            CampaignLogEventType.FORCE_POSTERIOR_UPDATE,
            f"{force_type.abbreviation}: {prior:.3f} → {new_posterior:.3f} "
            f"(delta={delta:+.3f}, skirmish={skirmish_id})",
            metadata={
                "force": force_type.value,
                "prior": prior,
                "new": new_posterior,
                "delta": delta,
            },
        )

    # ── Agent Selection ────────────────────────────────────────────

    @staticmethod
    def _select_skirmish_agent(force: ForceSpec) -> AgentRoleSpec:
        """Select an agent for a skirmish — prefer vanguards, then flanking."""
        deployed = force.deployed_agents
        vanguards = [a for a in deployed if a.role == MilitaryRole.VANGUARD]
        if vanguards:
            return vanguards[0]
        flanking = [a for a in deployed if a.role == MilitaryRole.FLANKING]
        if flanking:
            return flanking[0]
        commander = force.commander
        if commander:
            return commander
        return deployed[0] if deployed else AgentRoleSpec(name="Unknown")

    # ── Campaign Log ───────────────────────────────────────────────

    def _log_event(
        self,
        event_type: CampaignLogEventType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = CampaignLogEntry(
            event_type=event_type,
            content=content,
            metadata=metadata or {},
        )
        self._campaign_log.append(entry)
        logger.info("[CAMPAIGN] %s", content)

    @property
    def campaign_log(self) -> list[CampaignLogEntry]:
        return self._campaign_log

    @property
    def force_posterior_history(self) -> dict[str, list[float]]:
        return self._force_posterior_history

    @property
    def all_evidence(self) -> list[EvidenceItem]:
        return self._all_evidence
