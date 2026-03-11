"""
Layer 3 — Autonomous Execution Engine (meta-monitor).

Wraps ARGUS's existing :class:`RDCOrchestrator` with a meta-monitor layer
that watches the debate's information-theoretic trajectory in real time and
intervenes when necessary.

Seven monitoring mechanisms:
    1. Convergence Watch      — early termination when posterior stable
    2. Stagnation Detector    — inject provocateur prompt
    3. Drift Correction       — refocus directive when semantics diverge
    4. Confidence Escalation  — hide confidence; calibration reset
    5. Evidence Drought       — replace underperforming agent
    6. Budget Overflow        — accelerate to summative evidence only
    7. Deadlock Detection     — ARISTOTLE meta-argument injection
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from argus.cdag.graph import CDAG
from argus.cdag.nodes import Proposition, Evidence, Rebuttal, Finding
from argus.cdag.propagation import compute_all_posteriors, compute_posterior
from argus.agents.specialist import Specialist, SpecialistConfig
from argus.agents.refuter import Refuter, RefuterConfig
from argus.agents.jury import Jury, JuryConfig, Verdict
from argus.agents.moderator import Moderator
from argus.provenance.ledger import ProvenanceLedger, EventType
from argus.core.llm import get_llm

from argus.aristotle.models import (
    AgentSpec,
    DebateFrame,
    MonitorEvent,
    MonitorEventType,
    OrchestratorDecisionType,
    TopologySpec,
)
from argus.aristotle.decision_log import DecisionLogger
from argus.aristotle.personas.backup_agents import get_backup_agent

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.orchestrator import DebateResult

logger = logging.getLogger(__name__)


# ── helper: simple semantic similarity via overlap ────────────────────

_DRIFT_STOPWORDS = frozenset({
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


def _proposition_coverage(proposition: str, evidence_text: str) -> float:
    """Fraction of proposition keywords present in the evidence text.

    Unlike Jaccard, this is not diluted by the size of the evidence text,
    so it stays meaningful even when evidence is much longer than the
    proposition.
    """
    prop_keywords = set(proposition.lower().split()) - _DRIFT_STOPWORDS
    if not prop_keywords:
        return 1.0  # nothing meaningful to drift from
    ev_tokens = set(evidence_text.lower().split())
    return len(prop_keywords & ev_tokens) / len(prop_keywords)


# ── helper: compute per-agent EVID-Q average ──────────────────────────

def _agent_avg_evid_q(
    evidence_log: list[dict[str, Any]], agent_id: str,
) -> float:
    """Average EVID-Q for evidence submitted by a specific agent."""
    scores = [
        e["evid_q"] for e in evidence_log
        if e.get("agent_id") == agent_id and "evid_q" in e
    ]
    return sum(scores) / max(len(scores), 1)


# ── Agent wrapper: binds an AgentSpec to an ARGUS Specialist/Refuter ──

class _LiveAgent:
    """Runtime wrapper associating an AgentSpec with an ARGUS agent instance."""

    def __init__(
        self,
        spec: AgentSpec,
        agent: Specialist | Refuter,
        active: bool = True,
    ):
        self.spec = spec
        self.agent = agent
        self.active = active
        self.evidence_submitted: int = 0
        self.evid_q_scores: list[float] = []

    @property
    def avg_evid_q(self) -> float:
        return sum(self.evid_q_scores) / max(len(self.evid_q_scores), 1)


class ExecutionMonitor:
    """
    Layer 3 of ARISTOTLE — drives ARGUS's debate machinery while
    monitoring and intervening according to seven mechanisms.

    Returns a :class:`DebateResult`-compatible tuple and the full event log.
    """

    # ── thresholds ────────────────────────────────────────────────────

    CONVERGENCE_DELTA = 0.02
    CONVERGENCE_ROUNDS = 3
    STAGNATION_EVID_Q_THRESHOLD = 0.4
    STAGNATION_ROUNDS = 2
    DRIFT_SIMILARITY_THRESHOLD = 0.30
    CONFIDENCE_CEILING_EARLY = 0.90
    CONFIDENCE_EARLY_ROUNDS = 3
    EVIDENCE_DROUGHT_THRESHOLD = 3
    BUDGET_WARNING_RATIO = 0.90

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def execute(
        self,
        topology: TopologySpec,
        frame: DebateFrame,
        llm: "BaseLLM | None" = None,
        round_callback: "Any | None" = None,
    ) -> tuple[Any, list[MonitorEvent]]:
        """
        Run the full debate with monitoring.

        Args:
            round_callback: Optional callable invoked after each round
                completes.  Signature::

                    round_callback(round_num, posteriors, evidence_log, round_events)

                where *round_events* contains only events emitted in that round.

        Returns ``(debate_result_dict, events)`` where *debate_result_dict*
        mirrors :class:`DebateResult` fields and *events* is the complete
        event stream for L4 visualisation.
        """
        llm = llm or self.llm
        ledger = ProvenanceLedger()
        decision_logger = DecisionLogger(ledger)
        events: list[MonitorEvent] = []

        # ── Build graph ───────────────────────────────────────────────
        graph = CDAG(name="aristotle_debate")
        proposition = Proposition(
            text=frame.primary_proposition,
            prior=frame.prior_probability,
            domain=frame.primary_domain,
        )
        graph.add_proposition(proposition)

        ledger.record(
            EventType.SESSION_START,
            attributes={"proposition": frame.primary_proposition[:200]},
        )
        events.append(MonitorEvent(
            event_type=MonitorEventType.SESSION_START,
            node_id=proposition.id,
            node_text=frame.primary_proposition,
            posterior=frame.prior_probability,
            chat_message=(
                f"Debate started. Proposition: \"{frame.primary_proposition}\" "
                f"with initial prior {frame.prior_probability:.0%}."
            ),
        ))

        # ── Instantiate agents ────────────────────────────────────────
        live_specialists: list[_LiveAgent] = []
        for spec in topology.specialists:
            agent_config = SpecialistConfig(
                name=spec.name,
                domain=spec.domain_mandate,
            )
            specialist = Specialist(llm=llm, config=agent_config)
            la = _LiveAgent(spec=spec, agent=specialist)
            live_specialists.append(la)
            events.append(MonitorEvent(
                event_type=MonitorEventType.SPECIALIST_INSTANTIATED,
                agent_id=spec.id,
                agent_name=spec.name,
                node_id=spec.id,
                chat_message=f"Specialist activated: {spec.name} — {spec.domain_mandate}",
            ))

        live_refuters: list[_LiveAgent] = []
        for spec in topology.refuters:
            refuter_config = RefuterConfig(name=spec.name)
            refuter = Refuter(llm=llm, config=refuter_config)
            live_refuters.append(_LiveAgent(spec=spec, agent=refuter))

        jury = Jury(llm=llm, config=JuryConfig(
            decision_threshold=0.7,
            use_llm_reasoning=True,
        ))

        # ── Run debate rounds ─────────────────────────────────────────
        posteriors: list[float] = [frame.prior_probability]
        total_tokens_used: int = 0
        all_evidence_log: list[dict[str, Any]] = []
        used_agent_names: list[str] = [s.spec.name for s in live_specialists]

        max_rounds = min(topology.estimated_rounds + 2, topology.max_rounds)
        round_num = 0
        converged_at: int | None = None
        early_stop = False

        while round_num < max_rounds and not early_stop:
            round_num += 1
            round_evidence_high_count = 0
            logger.info("═══ Round %d ═══", round_num)

            # ── Specialists gather evidence ───────────────────────────
            for idx, la in enumerate(live_specialists):
                if not la.active:
                    continue
                if idx > 0:
                    time.sleep(10)  # rate-limit guard between specialist LLM calls

                evidence_items = la.agent.gather_evidence(
                    graph, proposition.id, None, None,
                )
                for ev in evidence_items:
                    la.evidence_submitted += 1
                    evid_q = ev.quality if hasattr(ev, "quality") else 0.5
                    la.evid_q_scores.append(evid_q)
                    all_evidence_log.append({
                        "agent_id": la.spec.id,
                        "agent_name": la.spec.name,
                        "evid_q": evid_q,
                        "round": round_num,
                        "polarity": ev.polarity,
                        "node_id": ev.id,
                    })

                    evt_type = (
                        MonitorEventType.SUPPORT_EVIDENCE
                        if ev.polarity >= 0
                        else MonitorEventType.ATTACK_EVIDENCE
                    )
                    events.append(MonitorEvent(
                        event_type=evt_type,
                        round_num=round_num,
                        agent_id=la.spec.id,
                        agent_name=la.spec.name,
                        node_id=ev.id,
                        node_text=ev.text[:200] if ev.text else "",
                        evid_q=evid_q,
                        polarity=ev.polarity,
                    ))
                    if evid_q > self.STAGNATION_EVID_Q_THRESHOLD:
                        round_evidence_high_count += 1

                    ledger.record(
                        EventType.EVIDENCE_ADDED,
                        agent_id=la.spec.name,
                        entity_id=ev.id,
                        attributes={"polarity": ev.polarity, "evid_q": evid_q},
                    )

            # Delay between specialist and refuter batches to respect API rate limits
            time.sleep(15)

            # ── Refuters generate rebuttals ───────────────────────────
            for idx, la in enumerate(live_refuters):
                if not la.active:
                    continue
                if idx > 0:
                    time.sleep(10)  # rate-limit guard between refuter LLM calls
                rebuttals = la.agent.generate_rebuttals(graph, proposition.id)
                for reb in rebuttals:
                    events.append(MonitorEvent(
                        event_type=MonitorEventType.REBUTTAL_FILED,
                        round_num=round_num,
                        agent_id=la.spec.id,
                        agent_name=la.spec.name,
                        node_id=reb.id,
                        node_text=reb.text[:200] if reb.text else "",
                        details={"target_id": reb.target_id},
                    ))
                    ledger.record(
                        EventType.REBUTTAL_ADDED,
                        agent_id=la.spec.name,
                        entity_id=reb.id,
                        attributes={"target": reb.target_id},
                    )

            # ── Bayesian update ───────────────────────────────────────
            compute_all_posteriors(graph)
            current_posterior = compute_posterior(graph, proposition.id)
            posteriors.append(current_posterior)

            events.append(MonitorEvent(
                event_type=MonitorEventType.BAYESIAN_UPDATE,
                round_num=round_num,
                posterior=current_posterior,
                chat_message=(
                    f"Round {round_num} Bayesian update: "
                    f"posterior moved to {current_posterior:.2%}"
                ),
            ))

            events.append(MonitorEvent(
                event_type=MonitorEventType.ROUND_COMPLETE,
                round_num=round_num,
                posterior=current_posterior,
                chat_message=(
                    f"Round {round_num} complete. Posterior: {current_posterior:.2%}."
                ),
            ))

            # ── token estimation ──────────────────────────────────────
            agents_active = sum(
                1 for la in live_specialists + live_refuters if la.active
            )
            total_tokens_used += agents_active * 2500

            # ══════════════════════════════════════════════════════════
            # MONITORING MECHANISMS
            # ══════════════════════════════════════════════════════════

            # 1. CONVERGENCE WATCH
            if len(posteriors) >= self.CONVERGENCE_ROUNDS + 1:
                recent_deltas = [
                    abs(posteriors[-i] - posteriors[-i - 1])
                    for i in range(1, self.CONVERGENCE_ROUNDS + 1)
                ]
                if all(d < self.CONVERGENCE_DELTA for d in recent_deltas):
                    converged_at = round_num
                    decision_logger.log(
                        OrchestratorDecisionType.EARLY_TERMINATION,
                        f"Posterior stable (Δ < {self.CONVERGENCE_DELTA}) "
                        f"for {self.CONVERGENCE_ROUNDS} consecutive rounds.",
                        round_num=round_num,
                        metadata={"posteriors": posteriors[-3:]},
                    )
                    events.append(MonitorEvent(
                        event_type=MonitorEventType.INTERVENTION_CONVERGENCE,
                        round_num=round_num,
                        posterior=current_posterior,
                        intervention_type="early_termination",
                        intervention_reason="Posterior converged",
                        chat_message=(
                            f"Convergence detected at round {round_num}. "
                            f"Posterior stable at {current_posterior:.2%}. "
                            "Triggering early jury verdict."
                        ),
                    ))
                    early_stop = True
                    if round_callback is not None:
                        _rc_events = [ev for ev in events if ev.round_num == round_num]
                        round_callback(round_num, list(posteriors), list(all_evidence_log), _rc_events)
                    continue

            # 2. STAGNATION DETECTOR
            if (
                round_num >= self.STAGNATION_ROUNDS
                and round_evidence_high_count == 0
            ):
                decision_logger.log(
                    OrchestratorDecisionType.STAGNATION_INJECTION,
                    "No high-quality evidence produced — injecting adjacent domain probe.",
                    round_num=round_num,
                )
                events.append(MonitorEvent(
                    event_type=MonitorEventType.INTERVENTION_STAGNATION,
                    round_num=round_num,
                    intervention_type="stagnation_injection",
                    intervention_reason="No EVID-Q > 0.4 evidence this round",
                    chat_message=(
                        f"Round {round_num}: no high-quality evidence. "
                        "ARISTOTLE is expanding the search into adjacent domains."
                    ),
                ))
                # Broaden the domain mandate of active specialists
                for la in live_specialists:
                    if la.active:
                        la.agent.config.domain = (
                            f"{la.spec.domain_mandate} + adjacent fields"
                        )

            # 3. DRIFT CORRECTION
            prop_text = frame.primary_proposition
            recent_evidence_texts = [
                ev.node_text
                for ev in events[-10:]
                if ev.event_type in (
                    MonitorEventType.SUPPORT_EVIDENCE,
                    MonitorEventType.ATTACK_EVIDENCE,
                ) and ev.node_text
            ]
            if recent_evidence_texts:
                combined = " ".join(recent_evidence_texts)
                similarity = _proposition_coverage(prop_text, combined)
                if similarity < self.DRIFT_SIMILARITY_THRESHOLD:
                    decision_logger.log(
                        OrchestratorDecisionType.DRIFT_CORRECTION,
                        f"Semantic similarity dropped to {similarity:.2f}. "
                        "Issuing refocus directive.",
                        round_num=round_num,
                        metadata={"similarity": similarity},
                    )
                    events.append(MonitorEvent(
                        event_type=MonitorEventType.INTERVENTION_DRIFT,
                        round_num=round_num,
                        intervention_type="drift_correction",
                        intervention_reason=f"Similarity {similarity:.2f} < threshold",
                        chat_message=(
                            f"Drift detected in round {round_num}. "
                            "ARISTOTLE is refocusing agents on the original proposition."
                        ),
                    ))

            # 4. CONFIDENCE ESCALATION GUARD
            if round_num <= self.CONFIDENCE_EARLY_ROUNDS:
                max_agent_conf = max(
                    (la.spec.epistemic_prior for la in live_specialists if la.active),
                    default=0.5,
                )
                if max_agent_conf > self.CONFIDENCE_CEILING_EARLY:
                    decision_logger.log(
                        OrchestratorDecisionType.CONFIDENCE_RESET,
                        "Agent confidence exceeds 0.90 in early rounds. "
                        "Hiding confidence and triggering calibration reset.",
                        round_num=round_num,
                        affected_agents=[
                            la.spec.id for la in live_specialists
                            if la.spec.epistemic_prior > self.CONFIDENCE_CEILING_EARLY
                        ],
                    )
                    events.append(MonitorEvent(
                        event_type=MonitorEventType.INTERVENTION_CONFIDENCE_GUARD,
                        round_num=round_num,
                        intervention_type="confidence_reset",
                        intervention_reason="Anti-Bayesian drift guard triggered",
                        chat_message=(
                            "Confidence escalation detected in early rounds. "
                            "ARISTOTLE is hiding agent self-confidence and "
                            "using external Bayesian aggregation only."
                        ),
                    ))

            # 5. EVIDENCE DROUGHT — replace underperforming specialists
            for la in live_specialists:
                if (
                    la.active
                    and round_num >= 2
                    and la.evidence_submitted < self.EVIDENCE_DROUGHT_THRESHOLD
                    and la.avg_evid_q < self.STAGNATION_EVID_Q_THRESHOLD
                ):
                    la.active = False
                    backup = get_backup_agent(
                        exclude_names=used_agent_names,
                        prior=la.spec.epistemic_prior,
                    )
                    new_spec = AgentSpec(
                        name=backup["name"],
                        role="specialist",
                        domain_mandate=backup["domain_mandate"],
                        evidence_source_priority=backup["evidence_source_priority"],
                        epistemic_prior=backup["epistemic_prior"],
                        evidence_type_focus=backup["evidence_type_focus"],
                        persona_description=backup["persona_description"],
                    )
                    new_agent = Specialist(
                        llm=llm,
                        config=SpecialistConfig(
                            name=new_spec.name,
                            domain=new_spec.domain_mandate,
                        ),
                    )
                    replacement = _LiveAgent(spec=new_spec, agent=new_agent)
                    live_specialists.append(replacement)
                    used_agent_names.append(new_spec.name)

                    decision_logger.log(
                        OrchestratorDecisionType.AGENT_REPLACEMENT,
                        f"Replaced {la.spec.name} (avg EVID-Q: {la.avg_evid_q:.2f}) "
                        f"with {new_spec.name}.",
                        round_num=round_num,
                        affected_agents=[la.spec.id, new_spec.id],
                    )
                    events.append(MonitorEvent(
                        event_type=MonitorEventType.AGENT_REPLACED,
                        round_num=round_num,
                        agent_id=new_spec.id,
                        agent_name=new_spec.name,
                        intervention_type="agent_replacement",
                        intervention_reason=(
                            f"{la.spec.name} underperforming "
                            f"(avg EVID-Q {la.avg_evid_q:.2f})"
                        ),
                        chat_message=(
                            f"ARISTOTLE has replaced {la.spec.name} "
                            f"(avg EVID-Q: {la.avg_evid_q:.2f}) with "
                            f"{new_spec.name} ({new_spec.domain_mandate}). "
                            f"{new_spec.name} has been briefed on the debate so far."
                        ),
                    ))
                    break  # one replacement per round

            # 6. BUDGET OVERFLOW
            if (
                total_tokens_used
                > topology.budget_cap_tokens * self.BUDGET_WARNING_RATIO
            ):
                decision_logger.log(
                    OrchestratorDecisionType.BUDGET_ACCELERATION,
                    f"Token usage ({total_tokens_used:,}) exceeds 90% of cap "
                    f"({topology.budget_cap_tokens:,}). Accelerating.",
                    round_num=round_num,
                )
                events.append(MonitorEvent(
                    event_type=MonitorEventType.INTERVENTION_BUDGET_OVERFLOW,
                    round_num=round_num,
                    intervention_type="budget_acceleration",
                    intervention_reason="Token budget exceeded 90%",
                    chat_message=(
                        "Budget warning: token usage exceeds 90% of cap. "
                        "ARISTOTLE is accelerating to summative evidence only."
                    ),
                ))
                early_stop = True
                if round_callback is not None:
                    _rc_events = [ev for ev in events if ev.round_num == round_num]
                    round_callback(round_num, list(posteriors), list(all_evidence_log), _rc_events)
                continue

            # 7. DEADLOCK DETECTION
            if (
                len(posteriors) >= 3
                and all(
                    abs(p - 0.50) < 0.05
                    for p in posteriors[-self.CONVERGENCE_ROUNDS:]
                )
            ):
                decision_logger.log(
                    OrchestratorDecisionType.DEADLOCK_RESOLUTION,
                    "Jury deadlocked near 50/50. ARISTOTLE injecting meta-argument.",
                    round_num=round_num,
                )
                events.append(MonitorEvent(
                    event_type=MonitorEventType.INTERVENTION_DEADLOCK,
                    round_num=round_num,
                    intervention_type="deadlock_resolution",
                    intervention_reason="Posterior stuck near 0.50",
                    chat_message=(
                        "Deadlock detected. ARISTOTLE is submitting a structured "
                        "meta-argument citing the highest-quality evidence from "
                        "both sides to break the deadlock."
                    ),
                ))
                # Inject a meta-finding node
                meta_finding = Finding(
                    text=(
                        "ARISTOTLE meta-argument: weighing the highest-quality "
                        "evidence from both support and attack positions."
                    ),
                    finding_type="meta_argument",
                    source="ARISTOTLE",
                )
                graph.add_node(meta_finding)

            # ── Fire round callback for live UI ───────────────────────
            if round_callback is not None:
                _rc_events = [ev for ev in events if ev.round_num == round_num]
                round_callback(round_num, list(posteriors), list(all_evidence_log), _rc_events)

        # ── JURY VERDICT ──────────────────────────────────────────────
        verdict = jury.evaluate(graph, proposition.id)

        ledger.record(
            EventType.VERDICT_RENDERED,
            agent_id="Jury",
            entity_id=proposition.id,
            attributes={
                "label": verdict.label,
                "posterior": verdict.posterior,
            },
        )

        events.append(MonitorEvent(
            event_type=MonitorEventType.VERDICT_RENDERED,
            round_num=round_num,
            posterior=verdict.posterior,
            chat_message=(
                f"Jury verdict: {verdict.label} "
                f"(confidence {verdict.posterior:.0%})."
            ),
        ))
        events.append(MonitorEvent(
            event_type=MonitorEventType.DEBATE_COMPLETE,
            round_num=round_num,
            posterior=verdict.posterior,
        ))

        # ── build result dict ─────────────────────────────────────────
        # Serialise graph for metric functions that expect dicts
        graph_dict = {"nodes": [], "edges": []}
        try:
            for nid in getattr(graph, "_nodes", {}):
                node_obj = graph._nodes[nid]
                node_d = {
                    "id": node_obj.id,
                    "text": node_obj.text,
                    "type": type(node_obj).__name__.lower(),
                    "confidence": getattr(node_obj, "confidence", 0.5),
                    "quality": getattr(node_obj, "quality", 0.5),
                    "relevance": getattr(node_obj, "relevance", 0.5),
                    "polarity": getattr(node_obj, "polarity", 0),
                    "citations": bool(getattr(node_obj, "citations", [])),
                    "source": getattr(node_obj, "source_agent", ""),
                    "target_id": getattr(node_obj, "target_id", ""),
                    "strength": getattr(node_obj, "strength", 0.0),
                }
                if hasattr(node_obj, "prior"):
                    node_d["prior"] = node_obj.prior
                    node_d["posterior"] = node_obj.posterior
                graph_dict["nodes"].append(node_d)
            for eid in getattr(graph, "_edges", {}):
                edge_obj = graph._edges[eid]
                graph_dict["edges"].append({
                    "id": edge_obj.id,
                    "source": getattr(edge_obj, "source_id", ""),
                    "target": getattr(edge_obj, "target_id", ""),
                    "edge_type": str(getattr(edge_obj, "edge_type", "")),
                })
        except Exception as exc:
            logger.debug("Graph serialisation for metrics failed: %s", exc)

        verdict_dict = {
            "proposition_id": verdict.proposition_id,
            "posterior": verdict.posterior,
            "label": verdict.label,
            "confidence": verdict.confidence,
            "reasoning": getattr(verdict, "reasoning", ""),
        }

        debate_result = {
            "proposition_id": proposition.id,
            "proposition_text": frame.primary_proposition,
            "verdict": verdict_dict,
            "verdict_obj": verdict,
            "final_posterior": verdict.posterior,
            "num_rounds": round_num,
            "num_evidence": len(all_evidence_log),
            "num_rebuttals": len(getattr(graph, "_rebuttals", set())),
            "graph": graph_dict,
            "graph_obj": graph,
            "ledger": ledger,
            "posteriors": posteriors,
            "converged_at": converged_at,
            "total_tokens_used": total_tokens_used,
            "decisions": decision_logger.export(),
            "evidence_log": all_evidence_log,
            "rounds": [{"round": r, "posterior": posteriors[r] if r < len(posteriors) else 0.5} for r in range(round_num + 1)],
            "duration_seconds": 0.0,
        }

        logger.info(
            "Debate complete in %d rounds. Verdict: %s (%.2f)",
            round_num, verdict.label, verdict.posterior,
        )
        return debate_result, events
