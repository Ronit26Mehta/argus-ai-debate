"""
Streaming Debate Engine for Argus-Viz.

Runs multi-specialist debates with per-round callbacks for live visualization.
Follows the Workflow 2 (Multi-Specialist Debate) pattern from Argus.
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Protocol
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------

class DebateCallback(Protocol):
    """Protocol for receiving real-time debate events."""

    def on_round_start(self, round_num: int, total_rounds: int) -> None: ...
    def on_specialist_evidence(self, specialist: str, evidence: list[dict]) -> None: ...
    def on_rebuttal(self, rebuttals: list[dict]) -> None: ...
    def on_round_complete(self, round_data: dict) -> None: ...
    def on_verdict(self, verdict: dict) -> None: ...


class NoOpCallback:
    """Default callback that does nothing."""
    def on_round_start(self, round_num: int, total_rounds: int) -> None: pass
    def on_specialist_evidence(self, specialist: str, evidence: list[dict]) -> None: pass
    def on_rebuttal(self, rebuttals: list[dict]) -> None: pass
    def on_round_complete(self, round_data: dict) -> None: pass
    def on_verdict(self, verdict: dict) -> None: pass


# ---------------------------------------------------------------------------
# Specialist config
# ---------------------------------------------------------------------------

@dataclass
class SpecialistDef:
    """Definition for one specialist agent."""
    name: str
    persona: str
    instruction: str


# ---------------------------------------------------------------------------
# Streaming Debate Engine
# ---------------------------------------------------------------------------

class StreamingDebateEngine:
    """
    Multi-specialist debate engine with per-round streaming callbacks.

    Unlike RDCOrchestrator.debate(), this engine gives granular control
    over each debate step, enabling live visualization updates.
    """

    def __init__(
        self,
        llm: Any,
        specialists: list[SpecialistDef],
        max_rounds: int = 3,
        refuter_enabled: bool = True,
        jury_threshold: float = 0.7,
        prior: float = 0.5,
    ):
        self.llm = llm
        self.specialists = specialists
        self.max_rounds = max_rounds
        self.refuter_enabled = refuter_enabled
        self.jury_threshold = jury_threshold
        self.prior = prior

    def run_debate(
        self,
        proposition_text: str,
        callback: Optional[Any] = None,
    ) -> dict:
        """
        Execute a full debate with per-round callbacks.

        Args:
            proposition_text: The claim to evaluate.
            callback: Object implementing DebateCallback methods.

        Returns:
            Full debate result dict with all rounds data.
        """
        from argus.cdag import CDAG, Proposition, Evidence, Rebuttal, EdgeType
        from argus.cdag.nodes import EvidenceType
        from argus.cdag.propagation import compute_posterior, compute_all_posteriors
        from argus.agents.jury import Jury, JuryConfig
        from argus.agents.refuter import Refuter
        from argus.provenance import ProvenanceLedger, EventType

        cb = callback or NoOpCallback()
        start_time = datetime.utcnow()

        # ---- Initialize graph ----
        graph = CDAG(name="argus_viz_debate")
        prop = Proposition(text=proposition_text, prior=self.prior)
        graph.add_proposition(prop)

        ledger = ProvenanceLedger()
        ledger.record(EventType.SESSION_START)

        rounds_data: list[dict] = []
        all_evidence_items: list[dict] = []
        all_rebuttal_items: list[dict] = []

        # ---- Debate rounds ----
        for round_num in range(1, self.max_rounds + 1):
            cb.on_round_start(round_num, self.max_rounds)

            posterior_before = compute_posterior(graph, prop.id)
            round_evidence: list[dict] = []
            round_rebuttals: list[dict] = []

            # --- Each specialist generates evidence ---
            for spec in self.specialists:
                evidence_items = self._gather_specialist_evidence(
                    graph, prop, spec, proposition_text, round_num,
                )
                round_evidence.extend(evidence_items)
                all_evidence_items.extend(evidence_items)

                ledger.record(
                    EventType.EVIDENCE_ADDED,
                    agent_id=spec.name,
                    attributes={"count": len(evidence_items)},
                )

                cb.on_specialist_evidence(spec.name, evidence_items)
                time.sleep(0.3)  # Rate-limit

            # --- Refuter generates rebuttals ---
            if self.refuter_enabled:
                rebuttal_items = self._generate_rebuttals(
                    graph, prop, proposition_text, round_num,
                )
                round_rebuttals.extend(rebuttal_items)
                all_rebuttal_items.extend(rebuttal_items)

                for rb in rebuttal_items:
                    ledger.record(
                        EventType.REBUTTAL_ADDED,
                        agent_id="Refuter",
                        attributes={"target": rb.get("target_text", "")[:50]},
                    )

                cb.on_rebuttal(rebuttal_items)

            # --- Update posteriors ---
            compute_all_posteriors(graph)
            posterior_after = compute_posterior(graph, prop.id)

            # --- Build round snapshot ---
            support_count = len([e for e in round_evidence if e.get("polarity", 0) > 0])
            attack_count = len([e for e in round_evidence if e.get("polarity", 0) < 0])

            round_snapshot = {
                "round": round_num,
                "posterior_before": posterior_before,
                "posterior_after": posterior_after,
                "evidence": round_evidence,
                "rebuttals": round_rebuttals,
                "support_count": support_count,
                "attack_count": attack_count,
                "total_evidence": len(round_evidence),
                "total_rebuttals": len(round_rebuttals),
                "specialist_breakdown": self._specialist_breakdown(round_evidence),
            }
            rounds_data.append(round_snapshot)

            cb.on_round_complete(round_snapshot)

        # ---- Jury verdict ----
        jury = Jury(self.llm, config=JuryConfig(
            use_llm_reasoning=True,
            decision_threshold=self.jury_threshold,
        ))
        verdict = jury.evaluate(graph, prop.id)

        ledger.record(EventType.VERDICT_RENDERED, entity_id=prop.id)
        ledger.record(EventType.SESSION_END)

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        verdict_dict = {
            "label": verdict.label,
            "posterior": verdict.posterior,
            "confidence": verdict.confidence,
            "reasoning": verdict.reasoning,
            "top_support": verdict.top_support,
            "top_attack": verdict.top_attack,
        }

        cb.on_verdict(verdict_dict)

        # ---- Build graph data for visualization ----
        graph_data = self._extract_graph_data(graph, prop.id)

        result = {
            "proposition": proposition_text,
            "prior": self.prior,
            "verdict": verdict_dict,
            "rounds": rounds_data,
            "all_evidence": all_evidence_items,
            "all_rebuttals": all_rebuttal_items,
            "graph_data": graph_data,
            "graph_summary": graph.summary(),
            "duration_seconds": duration,
            "config": {
                "max_rounds": self.max_rounds,
                "specialists": [s.name for s in self.specialists],
                "refuter_enabled": self.refuter_enabled,
                "jury_threshold": self.jury_threshold,
            },
        }

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gather_specialist_evidence(
        self,
        graph: Any,
        prop: Any,
        spec: SpecialistDef,
        proposition_text: str,
        round_num: int,
    ) -> list[dict]:
        """Prompt LLM for evidence from one specialist."""
        from argus.cdag.nodes import Evidence, EvidenceType
        from argus.cdag.edges import EdgeType

        prompt = f"""You are a {spec.name} ({spec.persona}). {spec.instruction}

PROPOSITION: {proposition_text}
ROUND: {round_num}

Provide 2 evidence points with reasoning. Return ONLY valid JSON:
{{
    "evidence": [
        {{
            "claim": "Specific factual claim with data or reasoning",
            "explanation": "Brief explanation of why this matters",
            "supports_proposition": true,
            "confidence": 0.85
        }}
    ]
}}
Only output raw JSON, no markdown fences."""

        evidence_items: list[dict] = []

        try:
            response = self.llm.generate(prompt, temperature=0.4)
            content = response.content.strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                )
            content = content.strip()

            data = json.loads(content)

            for item in data.get("evidence", [])[:2]:
                supports = item.get("supports_proposition", True)
                polarity = 1 if supports else -1
                confidence = max(0.1, min(1.0, float(item.get("confidence", 0.7))))

                evidence = Evidence(
                    text=item.get("claim", ""),
                    evidence_type=EvidenceType.EMPIRICAL,
                    polarity=polarity,
                    confidence=confidence,
                    relevance=0.8,
                    quality=0.75,
                    metadata={
                        "specialist": spec.name,
                        "persona": spec.persona,
                        "round": round_num,
                        "explanation": item.get("explanation", ""),
                    },
                )

                edge_type = EdgeType.SUPPORTS if polarity > 0 else EdgeType.ATTACKS
                graph.add_evidence(evidence, prop.id, edge_type)

                evidence_items.append({
                    "id": evidence.id,
                    "text": evidence.text,
                    "explanation": item.get("explanation", ""),
                    "polarity": polarity,
                    "confidence": confidence,
                    "specialist": spec.name,
                    "round": round_num,
                    "supports": supports,
                })

        except json.JSONDecodeError as e:
            logger.warning(f"[{spec.name}] JSON parse error: {e}")
            evidence_items.append({
                "id": f"err_{spec.name}_{round_num}",
                "text": f"[Parse error from {spec.name}]",
                "polarity": 0,
                "confidence": 0.0,
                "specialist": spec.name,
                "round": round_num,
                "error": str(e),
            })
        except Exception as e:
            logger.warning(f"[{spec.name}] Error: {e}")
            evidence_items.append({
                "id": f"err_{spec.name}_{round_num}",
                "text": f"[Error from {spec.name}: {str(e)[:100]}]",
                "polarity": 0,
                "confidence": 0.0,
                "specialist": spec.name,
                "round": round_num,
                "error": str(e),
            })

        return evidence_items

    def _generate_rebuttals(
        self,
        graph: Any,
        prop: Any,
        proposition_text: str,
        round_num: int,
    ) -> list[dict]:
        """Use the Refuter agent to generate rebuttals."""
        from argus.agents.refuter import Refuter
        rebuttal_items: list[dict] = []

        try:
            refuter = Refuter(self.llm)
            rebuttals = refuter.generate_rebuttals(graph, prop.id)

            for rb in rebuttals:
                rebuttal_items.append({
                    "id": rb.id,
                    "text": rb.text,
                    "target_id": rb.target_id,
                    "target_text": "",
                    "strength": getattr(rb, "strength", 0.5),
                    "rebuttal_type": getattr(rb, "rebuttal_type", "general"),
                    "round": round_num,
                })

        except Exception as e:
            logger.warning(f"[Refuter] Error in round {round_num}: {e}")
            rebuttal_items.append({
                "id": f"err_refuter_{round_num}",
                "text": f"[Refuter error: {str(e)[:100]}]",
                "strength": 0.0,
                "round": round_num,
                "error": str(e),
            })

        return rebuttal_items

    def _specialist_breakdown(self, evidence: list[dict]) -> dict:
        """Count evidence per specialist with polarity split."""
        breakdown: dict[str, dict] = {}
        for e in evidence:
            name = e.get("specialist", "Unknown")
            if name not in breakdown:
                breakdown[name] = {"total": 0, "support": 0, "attack": 0}
            breakdown[name]["total"] += 1
            if e.get("polarity", 0) > 0:
                breakdown[name]["support"] += 1
            elif e.get("polarity", 0) < 0:
                breakdown[name]["attack"] += 1
        return breakdown

    def _extract_graph_data(self, graph: Any, prop_id: str) -> dict:
        """Extract CDAG nodes and edges for visualization."""
        nodes: list[dict] = []
        edges: list[dict] = []

        # Add all nodes
        for node_id, node in graph._nodes.items():
            node_type = type(node).__name__
            node_data = {
                "id": node_id,
                "type": node_type,
                "text": node.text[:200] if hasattr(node, "text") else "",
                "confidence": getattr(node, "confidence", 0.0),
                "status": getattr(node, "status", "active"),
            }

            if node_type == "Evidence":
                node_data["polarity"] = getattr(node, "polarity", 0)
                node_data["specialist"] = node.metadata.get("specialist", "") if hasattr(node, "metadata") and isinstance(node.metadata, dict) else ""
            elif node_type == "Proposition":
                node_data["prior"] = getattr(node, "prior", 0.5)
                node_data["posterior"] = getattr(node, "posterior", 0.5)
            elif node_type == "Rebuttal":
                node_data["strength"] = getattr(node, "strength", 0.5)
                node_data["target_id"] = getattr(node, "target_id", "")

            nodes.append(node_data)

        # Add all edges
        for edge_id, edge in graph._edges.items():
            edges.append({
                "id": edge_id,
                "source": edge.source_id,
                "target": edge.target_id,
                "edge_type": edge.edge_type.value if hasattr(edge.edge_type, 'value') else str(edge.edge_type),
                "weight": getattr(edge, "weight", 1.0),
            })

        return {"nodes": nodes, "edges": edges}
