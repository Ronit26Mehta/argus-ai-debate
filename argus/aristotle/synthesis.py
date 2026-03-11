"""
Layer 5 — Plain-Language Output Synthesis Engine.

Transforms the raw debate result (C-DAG graph, posteriors, evidence log,
monitor events) into a structured :class:`SynthesisResult` suitable for
display in the ARISTOTLE chat pane.

Responsibilities:
    1. Verdict formulation (posterior → VerdictLevel mapping)
    2. Reasoning narrative grounded in C-DAG node IDs
    3. Dissent log (minority jury positions)
    4. "What Could Change This Verdict" from VoI EIG
    5. Metric dashboard with plain-language interpretations

All claims in the narrative are traceable to provenance — the synthesis
receives only structured C-DAG data and ProvenanceLedger records, never
free-form agent transcripts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from argus.aristotle.models import (
    DebateFrame,
    MetricReport,
    MonitorEvent,
    MonitorEventType,
    SynthesisResult,
    TopologySpec,
    VerdictLevel,
)

logger = logging.getLogger(__name__)


# ── Verdict thresholds ────────────────────────────────────────────────
# Matches the protocol's four-level verdict scheme.
_VERDICT_THRESHOLDS: list[tuple[float, VerdictLevel]] = [
    (0.75, VerdictLevel.STRONG_SUPPORT),
    (0.55, VerdictLevel.MODERATE_SUPPORT),
    (0.35, VerdictLevel.INSUFFICIENT),
    (0.00, VerdictLevel.EVIDENCE_AGAINST),
]


def _posterior_to_verdict(posterior: float) -> VerdictLevel:
    for threshold, level in _VERDICT_THRESHOLDS:
        if posterior >= threshold:
            return level
    return VerdictLevel.EVIDENCE_AGAINST


# ── Metric interpretation templates ───────────────────────────────────

_METRIC_INTERPRETATIONS: dict[str, dict[str, str]] = {
    "ARCIS": {
        "high":   "Arguments built upon each other with strong logical consistency.",
        "medium": "Generally coherent reasoning with minor inconsistencies.",
        "low":    "Significant logical gaps or contradictions were detected.",
    },
    "EVID-Q": {
        "high":   "High-quality evidence was consistently provided.",
        "medium": "Evidence quality was mixed — some strong, some weak.",
        "low":    "Evidence quality was generally poor or unreliable.",
    },
    "DIALEC": {
        "high":   "Deep dialectical exchange with robust attack/defence cycles.",
        "medium": "Moderate argumentative depth with some back-and-forth.",
        "low":    "Shallow argumentation with minimal dialectical engagement.",
    },
    "REBUT-F": {
        "high":   "Rebuttals effectively targeted and weakened opposing evidence.",
        "medium": "Rebuttals addressed some weaknesses but missed others.",
        "low":    "Rebuttals were largely ineffective or absent.",
    },
    "CONV-S": {
        "high":   "Posterior converged quickly and stably to a clear position.",
        "medium": "Some oscillation but eventual convergence was achieved.",
        "low":    "Posterior was unstable with significant oscillation.",
    },
    "PROV-I": {
        "high":   "Complete provenance chain — all claims traceable to sources.",
        "medium": "Most claims have sources, but some gaps exist.",
        "low":    "Significant provenance gaps — many claims lack attribution.",
    },
    "CALIB-M": {
        "high":   "Confidence aligned well with actual evidential support.",
        "medium": "Some over- or under-confidence detected.",
        "low":    "Confidence poorly calibrated to the evidence base.",
    },
    "EIG-U": {
        "high":   "Evidence efficiently reduced uncertainty each round.",
        "medium": "Some redundant evidence; moderate information gain.",
        "low":    "Evidence was largely redundant or uninformative.",
    },
}


def _interpret_metric(name: str, score: float) -> str:
    """Return the appropriate plain-language interpretation band."""
    templates = _METRIC_INTERPRETATIONS.get(name, {})
    if score >= 0.7:
        return templates.get("high", f"Score {score:.2f} — high range.")
    elif score >= 0.4:
        return templates.get("medium", f"Score {score:.2f} — medium range.")
    else:
        return templates.get("low", f"Score {score:.2f} — low range.")


# ── LLM prompt for grounded narrative ────────────────────────────────

_NARRATIVE_SYSTEM = (
    "You are the ARISTOTLE synthesis engine. Your task is to write a "
    "clear, plain-English reasoning narrative summarising a structured "
    "multi-agent debate. Every claim you make MUST reference one of the "
    "provided C-DAG node IDs in square brackets, e.g. [node-abc123]. "
    "Do NOT invent information beyond what is given in the data."
)

_NARRATIVE_USER_TEMPLATE = """\
Write a 3-5 paragraph plain-English summary of this debate.

PROPOSITION: {proposition}
FINAL POSTERIOR: {posterior:.1%}
VERDICT: {verdict}

EVIDENCE SUMMARY:
{evidence_summary}

INTERVENTIONS:
{intervention_summary}

POSTERIOR TRAJECTORY: {trajectory}

RULES:
- Reference node IDs in [brackets]
- No jargon — write for a non-technical reader
- Explain WHY the evidence led to this verdict
- Mention the strongest supporting and opposing evidence
- Keep it under 500 words
"""

_DISSENT_USER_TEMPLATE = """\
Write a 1-2 paragraph minority view based on the dissenting evidence.

PROPOSITION: {proposition}
FINAL VERDICT: {verdict} ({posterior:.1%})

DISSENTING EVIDENCE (argued against the majority):
{dissent_evidence}

RULES:
- Write for a non-technical reader
- Explain why this minority view was not dismissed
- Reference node IDs in [brackets]
- Keep it under 200 words
"""

_WHAT_COULD_CHANGE_TEMPLATE = """\
Based on these highest-EIG (Expected Information Gain) nodes, write 2-4
bullet points describing what new evidence could change this verdict.
Each bullet should be a concrete, actionable research direction a
non-technical reader can understand.

PROPOSITION: {proposition}
CURRENT VERDICT: {verdict} ({posterior:.1%})

HIGH-EIG NODES:
{eig_nodes}

One bullet per line, no numbering, no sub-bullets.
"""


class SynthesisEngine:
    """
    Layer 5 — transforms raw debate output into a user-facing
    :class:`SynthesisResult`.
    """

    def __init__(self, llm=None):
        self.llm = llm

    # ── public API ────────────────────────────────────────────────────

    def synthesise(
        self,
        frame: DebateFrame,
        topology: TopologySpec,
        debate_result: dict[str, Any],
        events: list[MonitorEvent],
    ) -> SynthesisResult:
        """
        Produce the complete synthesis result.

        Args:
            frame: Layer 1 output.
            topology: Layer 2 output.
            debate_result: Raw dict from ExecutionMonitor.execute().
            events: All MonitorEvent objects from Layer 3.

        Returns:
            Fully populated SynthesisResult.
        """
        posterior = debate_result.get("final_posterior", 0.5)
        posteriors = debate_result.get("posteriors", [frame.prior_probability])
        evidence_log = debate_result.get("evidence_log", [])
        decisions = debate_result.get("decisions", [])
        graph = debate_result.get("graph")

        # 1. Verdict ───────────────────────────────────────────────────
        verdict = _posterior_to_verdict(posterior)
        convergence_round = self._find_convergence_round(posteriors)

        # 2. Metrics ───────────────────────────────────────────────────
        metrics = self._compute_metrics(debate_result, events)

        # 3. Reasoning narrative ───────────────────────────────────────
        evidence_summary = self._build_evidence_summary(evidence_log)
        intervention_summary = self._build_intervention_summary(events)
        trajectory_text = " → ".join(f"{p:.2f}" for p in posteriors)

        narrative = self._generate_narrative(
            proposition=frame.primary_proposition or frame.raw_query,
            posterior=posterior,
            verdict=verdict,
            evidence_summary=evidence_summary,
            intervention_summary=intervention_summary,
            trajectory=trajectory_text,
        )

        # 4. Dissent log ──────────────────────────────────────────────
        dissent = self._generate_dissent(
            proposition=frame.primary_proposition or frame.raw_query,
            posterior=posterior,
            verdict=verdict,
            evidence_log=evidence_log,
        )

        # 5. What Could Change This ───────────────────────────────────
        what_could_change = self._generate_what_could_change(
            proposition=frame.primary_proposition or frame.raw_query,
            posterior=posterior,
            verdict=verdict,
            graph=graph,
            evidence_log=evidence_log,
        )

        # 6. Provenance map ───────────────────────────────────────────
        provenance_map = {}
        for entry in evidence_log:
            nid = entry.get("node_id", "")
            source = entry.get("source", entry.get("agent_name", "Unknown"))
            if nid:
                provenance_map[nid] = source

        # 7. Assemble ─────────────────────────────────────────────────
        num_rebuttals = sum(
            1 for ev in events
            if ev.event_type == MonitorEventType.REBUTTAL_FILED
        )

        return SynthesisResult(
            frame_id=frame.id,
            verdict_label=verdict,
            jury_confidence=posterior,
            convergence_round=convergence_round,
            reasoning_narrative=narrative,
            dissent_log=dissent,
            what_could_change=what_could_change,
            evidence_provenance_map=provenance_map,
            metrics=metrics,
            num_rounds=debate_result.get("num_rounds", len(posteriors) - 1),
            num_evidence=len(evidence_log),
            num_rebuttals=num_rebuttals,
            duration_seconds=debate_result.get("duration_seconds", 0.0),
        )

    # ── private helpers ───────────────────────────────────────────────

    def _find_convergence_round(self, posteriors: list[float]) -> int:
        """Find the round where posterior movement drops below 0.02."""
        for i in range(2, len(posteriors)):
            if abs(posteriors[i] - posteriors[i - 1]) < 0.02:
                return i - 1  # rounds are 1-indexed (index 0 = prior)
        return max(len(posteriors) - 1, 1)

    def _compute_metrics(
        self,
        debate_result: dict[str, Any],
        events: list[MonitorEvent],
    ) -> list[MetricReport]:
        """Compute all 8 ARGUS metrics and interpret them."""
        try:
            from argus.evaluation.scoring.metrics import (
                compute_arcis,
                compute_evid_q,
                compute_dialec,
                compute_rebut_f,
                compute_conv_s,
                compute_prov_i,
                compute_calib_m,
                compute_eig_u,
                METRIC_REGISTRY,
            )

            metric_funcs = [
                ("ARCIS", compute_arcis),
                ("EVID-Q", compute_evid_q),
                ("DIALEC", compute_dialec),
                ("REBUT-F", compute_rebut_f),
                ("CONV-S", compute_conv_s),
                ("PROV-I", compute_prov_i),
                ("CALIB-M", compute_calib_m),
                ("EIG-U", compute_eig_u),
            ]

            reports = []
            for name, func in metric_funcs:
                try:
                    score = func(debate_result)
                    score = max(0.0, min(1.0, float(score)))
                except Exception:
                    score = 0.0
                    logger.debug("Metric %s computation failed, defaulting to 0", name)

                full_name = METRIC_REGISTRY.get(name, None)
                fn = full_name.full_name if full_name else name

                reports.append(MetricReport(
                    name=name,
                    full_name=fn,
                    score=score,
                    interpretation=_interpret_metric(name, score),
                ))
            return reports
        except ImportError:
            logger.warning("Evaluation scoring module not available — skipping metrics")
            return []

    def _build_evidence_summary(self, evidence_log: list[dict[str, Any]]) -> str:
        """Build a structured text summary of evidence for the LLM prompt."""
        if not evidence_log:
            return "No evidence was submitted."
        lines = []
        for entry in evidence_log:
            nid = entry.get("node_id", "?")
            agent = entry.get("agent_name", "Unknown")
            pol = "SUPPORTS" if entry.get("polarity", 1) >= 0 else "ATTACKS"
            eq = entry.get("evid_q", 0.0)
            text = entry.get("text", "")[:120]
            lines.append(f"- [{nid}] {agent}: {pol} (EVID-Q {eq:.2f}) — {text}")
        return "\n".join(lines)

    def _build_intervention_summary(self, events: list[MonitorEvent]) -> str:
        """Summarise ARISTOTLE interventions for the narrative prompt."""
        interventions = [
            ev for ev in events
            if "intervention" in ev.event_type.value or ev.event_type == MonitorEventType.AGENT_REPLACED
        ]
        if not interventions:
            return "No ARISTOTLE interventions were triggered."
        lines = []
        for ev in interventions:
            lines.append(
                f"- Round {ev.round_num}: {ev.event_type.value.replace('_', ' ').title()}"
                f" — {ev.intervention_reason or ev.chat_message[:100]}"
            )
        return "\n".join(lines)

    def _generate_narrative(
        self,
        proposition: str,
        posterior: float,
        verdict: VerdictLevel,
        evidence_summary: str,
        intervention_summary: str,
        trajectory: str,
    ) -> str:
        """Use LLM to produce a grounded reasoning narrative."""
        if not self.llm:
            return self._fallback_narrative(
                proposition, posterior, verdict, evidence_summary, trajectory,
            )

        prompt = _NARRATIVE_USER_TEMPLATE.format(
            proposition=proposition,
            posterior=posterior,
            verdict=verdict.value,
            evidence_summary=evidence_summary,
            intervention_summary=intervention_summary,
            trajectory=trajectory,
        )
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=_NARRATIVE_SYSTEM,
                temperature=0.3,
                max_tokens=16384,
            )
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
        except Exception as exc:
            logger.warning("Narrative LLM call failed: %s", exc)
            return self._fallback_narrative(
                proposition, posterior, verdict, evidence_summary, trajectory,
            )

    def _fallback_narrative(
        self,
        proposition: str,
        posterior: float,
        verdict: VerdictLevel,
        evidence_summary: str,
        trajectory: str,
    ) -> str:
        """Deterministic narrative when LLM is unavailable."""
        return (
            f"The ARGUS multi-agent debate examined the proposition: \"{proposition}\". "
            f"After structured adversarial examination, the posterior probability "
            f"reached {posterior:.1%}, resulting in a verdict of "
            f"\"{verdict.value}\".\n\n"
            f"The belief trajectory progressed as follows: {trajectory}.\n\n"
            f"Evidence summary:\n{evidence_summary}"
        )

    def _generate_dissent(
        self,
        proposition: str,
        posterior: float,
        verdict: VerdictLevel,
        evidence_log: list[dict[str, Any]],
    ) -> str:
        """Produce the minority-view dissent log."""
        # Identify dissenting evidence (counter to majority)
        majority_supports = posterior >= 0.5
        dissent_entries = [
            e for e in evidence_log
            if (e.get("polarity", 1) < 0) == majority_supports
        ]

        if not dissent_entries:
            return ""

        dissent_text = "\n".join(
            f"- [{e.get('node_id', '?')}] {e.get('agent_name', 'Unknown')}: "
            f"EVID-Q {e.get('evid_q', 0):.2f} — {e.get('text', '')[:120]}"
            for e in dissent_entries[:6]
        )

        if not self.llm:
            return (
                f"A minority of the evidence argued against the majority verdict. "
                f"Key dissenting contributions:\n{dissent_text}"
            )

        prompt = _DISSENT_USER_TEMPLATE.format(
            proposition=proposition,
            posterior=posterior,
            verdict=verdict.value,
            dissent_evidence=dissent_text,
        )
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=_NARRATIVE_SYSTEM,
                temperature=0.3,
                max_tokens=16384,
            )
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
        except Exception as exc:
            logger.warning("Dissent LLM call failed: %s", exc)
            return (
                f"A minority of the evidence argued against the majority verdict. "
                f"Key dissenting contributions:\n{dissent_text}"
            )

    def _generate_what_could_change(
        self,
        proposition: str,
        posterior: float,
        verdict: VerdictLevel,
        graph: Any | None,
        evidence_log: list[dict[str, Any]],
    ) -> list[str]:
        """
        Map highest-EIG nodes to plain-language research directions.

        If VoIPlanner is available and a graph is provided, compute real
        EIG scores. Otherwise, heuristically identify informational gaps.
        """
        eig_nodes = self._get_eig_nodes(graph, evidence_log)

        if not eig_nodes:
            return [
                "Additional high-quality evidence on this topic could shift the verdict.",
                "Cross-domain replication studies would strengthen or weaken the conclusion.",
            ]

        eig_text = "\n".join(
            f"- {node['description']} (EIG: {node['eig']:.2f})"
            for node in eig_nodes[:5]
        )

        if not self.llm:
            return [node["description"] for node in eig_nodes[:4]]

        prompt = _WHAT_COULD_CHANGE_TEMPLATE.format(
            proposition=proposition,
            posterior=posterior,
            verdict=verdict.value,
            eig_nodes=eig_text,
        )
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=_NARRATIVE_SYSTEM,
                temperature=0.4,
                max_tokens=16384,
            )
            content = response.content if hasattr(response, 'content') else str(response)
            bullets = [
                line.strip().lstrip("•-– ").strip()
                for line in content.strip().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            return bullets[:4] if bullets else [node["description"] for node in eig_nodes[:4]]
        except Exception as exc:
            logger.warning("What-could-change LLM call failed: %s", exc)
            return [node["description"] for node in eig_nodes[:4]]

    def _get_eig_nodes(
        self,
        graph: Any | None,
        evidence_log: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Attempt to compute EIG for graph nodes; fall back to heuristics.
        """
        # Try VoI planner
        if graph is not None:
            try:
                from argus.decision.eig import ActionCandidate
                from argus.decision.planner import VoIPlanner

                planner = VoIPlanner()
                candidates = []
                for prop in graph.get_all_propositions():
                    candidates.append(ActionCandidate(
                        id=prop.id,
                        name=f"Investigate: {prop.text[:60]}",
                        cost=1.0,
                        target_props=[prop.id],
                    ))
                queue = planner.plan(graph, candidates)
                results = []
                for action in queue.get_all()[:5]:
                    results.append({
                        "description": action.name,
                        "eig": action.eig,
                    })
                if results:
                    return results
            except Exception:
                logger.debug("VoI planner unavailable, falling back to heuristics")

        # Heuristic: evidence items with low EVID-Q represent areas where
        # better evidence would have the most impact.
        low_quality = sorted(evidence_log, key=lambda e: e.get("evid_q", 0))[:5]
        heuristic_nodes = []
        for entry in low_quality:
            eq = entry.get("evid_q", 0)
            heuristic_eig = max(0.01, 1.0 - eq)
            text = entry.get("text", entry.get("agent_name", "More evidence needed"))
            heuristic_nodes.append({
                "description": f"Stronger evidence addressing: {text[:80]}",
                "eig": round(heuristic_eig, 2),
            })
        return heuristic_nodes
