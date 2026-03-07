"""
Streaming CRUX Debate Engine for CRUX-Viz.

Uses the REAL CRUXOrchestrator from argus.crux — all 7 CRUX primitives
run internally by the orchestrator:
  1. Epistemic Agent Cards (EAC)
  2. Claim Bundles (CB)
  3. Dialectical Routing (DR)
  4. Belief Reconciliation Protocol (BRP)
  5. Credibility Ledger (CL)
  6. Epistemic Dead Reckoning (EDR)
  7. Challenger Auction (CA)

The CRUXDebateResult contains a real CRUXSession object, which is
passed as _crux_session_obj so plot_crux_debate_flow() renders the
real debate flow (not synthetic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------

class NoOpCRUXCallback:
    """Default no-op callback."""
    def on_phase_change(self, phase: str) -> None: pass
    def on_round_start(self, round_num: int, total: int) -> None: pass
    def on_specialist_evidence(self, name: str, items: list) -> None: pass
    def on_rebuttal(self, items: list) -> None: pass
    def on_claim_bundle(self, cb: dict) -> None: pass
    def on_auction(self, auction: dict) -> None: pass
    def on_brp(self, brp: dict) -> None: pass
    def on_round_complete(self, data: dict) -> None: pass
    def on_verdict(self, verdict: dict) -> None: pass


# ---------------------------------------------------------------------------
# Specialist config (mirrors argus_viz)
# ---------------------------------------------------------------------------

@dataclass
class SpecialistDef:
    """Definition for one specialist agent."""
    name: str
    persona: str
    instruction: str


# ---------------------------------------------------------------------------
# Streaming CRUX Engine — uses real CRUXOrchestrator
# ---------------------------------------------------------------------------

class StreamingCRUXEngine:
    """
    crux-viz debate engine backed by the real CRUXOrchestrator.

    Runs a full CRUX-protocol debate using CRUXOrchestrator.debate()
    which internally executes all 7 CRUX primitives and returns a
    CRUXDebateResult carrying a real CRUXSession object.
    """

    def __init__(
        self,
        llm: Any,
        specialists: list[SpecialistDef],
        max_rounds: int = 3,
        refuter_enabled: bool = True,
        jury_threshold: float = 0.7,
        prior: float = 0.5,
        contradiction_threshold: float = 0.20,
        enable_edr: bool = True,
        auction_timeout_seconds: int = 30,
    ):
        self.llm = llm
        self.specialists = specialists
        self.max_rounds = max_rounds
        self.refuter_enabled = refuter_enabled
        self.jury_threshold = jury_threshold
        self.prior = prior
        self.contradiction_threshold = contradiction_threshold
        self.enable_edr = enable_edr
        self.auction_timeout_seconds = auction_timeout_seconds

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_debate(
        self,
        proposition_text: str,
        callback: Optional[Any] = None,
    ) -> dict:
        """
        Run a full CRUX debate using CRUXOrchestrator.

        Returns a result dict that includes a real CRUXSession under the
        key '_crux_session_obj', which lets crux_viz use the real
        plot_crux_debate_flow() instead of the synthetic fallback.
        """
        from argus.crux import CRUXOrchestrator, CRUXConfig
        from argus.crux.ledger import CredibilityLedger
        from argus.orchestrator import RDCOrchestrator

        cb = callback or NoOpCRUXCallback()
        start_time = datetime.utcnow()

        cb.on_phase_change("initializing")

        # ---- Build config ----
        crux_config = CRUXConfig(
            contradiction_threshold=self.contradiction_threshold,
            auction_timeout_seconds=self.auction_timeout_seconds,
            enable_edr=self.enable_edr,
        )

        # ---- Build base RDCOrchestrator ----
        # CRUXOrchestrator.debate() calls base.debate() internally,
        # which runs the ARGUS debate (evidence gathering, refuter, jury)
        cb.on_phase_change("building_orchestrator")

        # RDCOrchestrator.__init__(llm?, ledger?, max_rounds) — verified from source.
        # refuter_enabled and jury_threshold are NOT constructor params.
        base_orchestrator = RDCOrchestrator(
            llm=self.llm,
            max_rounds=self.max_rounds,
        )

        # Shared credibility ledger
        crux_ledger = CredibilityLedger(config=crux_config)

        # ---- Build REAL CRUXOrchestrator ----
        # This initialises all 7 CRUX primitives internally:
        #   EAC registry, CB factory, dialectical router, BRP, ledger, EDR, CA
        crux = CRUXOrchestrator(
            base=base_orchestrator,
            llm=self.llm,
            ledger=crux_ledger,
            config=crux_config,
        )

        # ---- Register streaming callbacks on the orchestrator ----
        crux._on_claim_bundle.append(
            lambda c: cb.on_claim_bundle(self._cb_to_dict(c))
        )
        crux._on_auction_complete.append(
            lambda a: cb.on_auction(self._auction_to_dict(a))
        )
        crux._on_brp_triggered.append(
            lambda b: cb.on_brp(self._brp_to_dict(b))
        )

        cb.on_phase_change("debate")

        # ---- Run the REAL CRUX debate ----
        # This calls all 7 CRUX phases internally and returns a
        # CRUXDebateResult with a real CRUXSession.
        crux_result = crux.debate(
            proposition_text=proposition_text,
            prior=self.prior,
            domain="general",
        )

        # ---- Extract the real CRUXSession ----
        session = crux_result.session  # real CRUXSession dataclass

        cb.on_phase_change("verdict")

        verdict = crux_result.verdict
        verdict_dict = {
            "label": verdict.label,
            "posterior": crux_result.final_posterior,
            "confidence": getattr(verdict, "confidence", 0.0),
            "reasoning": getattr(verdict, "reasoning", ""),
            "top_support": getattr(verdict, "top_support", None),
            "top_attack": getattr(verdict, "top_attack", None),
        }
        cb.on_verdict(verdict_dict)

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # ---- Build serialisable claim bundles list ----
        claim_bundles_dicts = [
            self._cb_to_dict(cb_obj)
            for cb_obj in (session.claim_bundles if session else [])
        ]

        # ---- Build serialisable auctions list ----
        auctions_dicts = [
            self._auction_to_dict(a)
            for a in (session.auctions if session else [])
        ]

        # ---- Build serialisable BRP sessions list ----
        brp_dicts = [
            self._brp_to_dict(b)
            for b in (session.brp_sessions if session else [])
        ]

        # ---- Build rounds data from base result ----
        base_result = crux_result.base_result
        rounds_data = self._extract_rounds(base_result, claim_bundles_dicts)

        # ---- Build credibility snapshot from real ledger ----
        credibility_snapshot = self._build_credibility_snapshot(crux_ledger)

        # ---- EDR checkpoints from real session ----
        edr_checkpoints = []
        if session:
            for ckpt in session.checkpoints:
                edr_checkpoints.append({
                    "checkpoint_id": ckpt.checkpoint_id,
                    "checkpoint_hash": ckpt.checkpoint_hash,
                    "timestamp": ckpt.timestamp.isoformat(),
                    "posterior": crux_result.final_posterior,
                    "num_bundles": len(session.claim_bundles),
                })

        # ---- CRUX stats from real session stats object ----
        stats = session.stats if session else None
        crux_stats = {
            "num_claim_bundles": stats.num_claim_bundles if stats else len(claim_bundles_dicts),
            "num_challenges": stats.num_challenges if stats else 0,
            "num_brp_sessions": stats.num_brp_sessions if stats else len(brp_dicts),
            "num_auctions": stats.num_auctions if stats else len(auctions_dicts),
            "num_unchallenged": stats.num_unchallenged if stats else 0,
            "avg_dfs": stats.avg_dfs if stats else 0.0,
            "total_credibility_updates": len(credibility_snapshot),
        }

        return {
            "proposition": proposition_text,
            "prior": self.prior,
            "verdict": verdict_dict,
            "rounds": rounds_data,
            "all_evidence": self._extract_all_evidence(base_result),
            "all_rebuttals": self._extract_all_rebuttals(base_result),
            "duration_seconds": duration,
            # CRUX-specific (serialisable dicts for chart functions)
            "claim_bundles": claim_bundles_dicts,
            "auctions": auctions_dicts,
            "brp_sessions": brp_dicts,
            "crux_session_stats": crux_stats,
            "credibility_ledger": credibility_snapshot,
            "edr_checkpoints": edr_checkpoints,
            "dfs_scores": {},
            # Real CRUXSession — used by plot_crux_debate_flow() for the
            # real (non-synthetic) CRUX Debate Flow chart
            "_crux_session_obj": session,
            "config": {
                "max_rounds": self.max_rounds,
                "specialists": [s.name for s in self.specialists],
                "refuter_enabled": self.refuter_enabled,
                "jury_threshold": self.jury_threshold,
                "contradiction_threshold": self.contradiction_threshold,
                "enable_edr": self.enable_edr,
            },
        }

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _cb_to_dict(self, cb: Any) -> dict:
        """Serialise a ClaimBundle object to a plain dict."""
        polarity = getattr(cb, "polarity", 0)
        polarity_val = polarity.value if hasattr(polarity, "value") else int(polarity)

        issued_at = getattr(cb, "issued_at", None)
        issued_at_str = (
            issued_at.isoformat() if hasattr(issued_at, "isoformat") else str(issued_at)
        ) if issued_at else datetime.utcnow().isoformat()

        return {
            "cb_id": getattr(cb, "cb_id", ""),
            "claim_text": getattr(cb, "claim_text", ""),
            "polarity": polarity_val,
            "prior": getattr(cb, "prior", 0.5),
            "posterior": getattr(cb, "posterior", 0.5),
            "issuer_agent": getattr(cb, "issuer_agent", ""),
            "issuer_credibility": getattr(cb, "issuer_credibility", 0.8),
            "challenge_open": getattr(cb, "challenge_open", False),
            "issued_at": issued_at_str,
        }

    def _auction_to_dict(self, a: Any) -> dict:
        """Serialise an AuctionResult to a plain dict."""
        return {
            "auction_id": getattr(a, "auction_id", ""),
            "cb_id": getattr(a, "cb_id", ""),
            "winner": getattr(a, "winner", None),
            "num_bids": getattr(a, "num_bids", 0),
            "unchallenged": getattr(a, "unchallenged", True),
            "winning_dfs": (
                a.winning_bid.estimated_dfs
                if getattr(a, "winning_bid", None) else 0.0
            ),
        }

    def _brp_to_dict(self, b: Any) -> dict:
        """Serialise a BRPSession to a plain dict."""
        resolution = getattr(b, "resolution", None)
        res_dict: dict = {}
        if resolution:
            res_dict = {
                "resolution_id": getattr(resolution, "resolution_id", ""),
                "reconciled_posterior": getattr(resolution, "reconciled_posterior", 0.5),
                "contributor_agents": getattr(resolution, "contributor_agents", []),
                "contributor_weights": getattr(resolution, "contributor_weights", []),
                "original_cb_ids": getattr(resolution, "original_cb_ids", []),
                "strategy": getattr(resolution, "strategy", ""),
            }

        contradiction = getattr(b, "contradiction", None)
        delta = 0.0
        if contradiction:
            delta = getattr(contradiction, "divergence", 0.0)

        return {
            "brp_id": getattr(b, "session_id", ""),
            "trigger": "contradiction",
            "contradiction_delta": delta,
            "resolution": res_dict,
        }

    def _extract_rounds(self, base_result: Any, all_cbs: list[dict]) -> list[dict]:
        """
        Build rounds_data from the base DebateResult.
        If not available, synthesise one round from claim bundles.
        """
        if base_result is None:
            return self._synthetic_rounds(all_cbs)

        try:
            rounds = getattr(base_result, "rounds", None)
            if rounds:
                return [
                    {
                        "round": i + 1,
                        "posterior_before": r.get("posterior_before", 0.5),
                        "posterior_after": r.get("posterior_after", 0.5),
                        "evidence": r.get("evidence", []),
                        "rebuttals": r.get("rebuttals", []),
                        "support_count": r.get("support_count", 0),
                        "attack_count": r.get("attack_count", 0),
                        "total_evidence": r.get("total_evidence", 0),
                        "total_rebuttals": r.get("total_rebuttals", 0),
                        "claim_bundles_issued": r.get("claim_bundles_issued", 0),
                    }
                    for i, r in enumerate(rounds)
                ]
        except Exception:
            pass

        return self._synthetic_rounds(all_cbs)

    def _synthetic_rounds(self, all_cbs: list[dict]) -> list[dict]:
        """Synthesise one round from claim bundle dicts when base result has no rounds."""
        support = sum(1 for c in all_cbs if str(c.get("polarity", 0)) in ("1", "1"))
        attack = sum(1 for c in all_cbs if str(c.get("polarity", 0)) in ("-1",))
        posteriors = [c.get("posterior", 0.5) for c in all_cbs]
        avg_post = sum(posteriors) / len(posteriors) if posteriors else 0.5

        return [{
            "round": 1,
            "posterior_before": 0.5,
            "posterior_after": avg_post,
            "evidence": [],
            "rebuttals": [],
            "support_count": support,
            "attack_count": attack,
            "total_evidence": len(all_cbs),
            "total_rebuttals": 0,
            "claim_bundles_issued": len(all_cbs),
        }]

    def _extract_all_evidence(self, base_result: Any) -> list[dict]:
        """Extract flat evidence list from base DebateResult."""
        if base_result is None:
            return []
        try:
            evidence = getattr(base_result, "all_evidence", None)
            if evidence:
                return list(evidence)
            # Try rounds
            rounds = getattr(base_result, "rounds", [])
            items = []
            for r in rounds:
                items.extend(r.get("evidence", []))
            return items
        except Exception:
            return []

    def _extract_all_rebuttals(self, base_result: Any) -> list[dict]:
        """Extract flat rebuttals list from base DebateResult."""
        if base_result is None:
            return []
        try:
            rebuttals = getattr(base_result, "all_rebuttals", None)
            if rebuttals:
                return list(rebuttals)
            rounds = getattr(base_result, "rounds", [])
            items = []
            for r in rounds:
                items.extend(r.get("rebuttals", []))
            return items
        except Exception:
            return []

    def _build_credibility_snapshot(self, ledger: Any) -> list[dict]:
        """
        Extract live credibility state from the real CredibilityLedger.
        Falls back to initial calibration values if ledger is empty.
        """
        _defaults = [
            {"agent_id": "argus-specialist-001", "credibility": 0.85,
             "brier_score": 0.15, "num_updates": 0, "is_suspended": False},
            {"agent_id": "argus-refuter-001",    "credibility": 0.88,
             "brier_score": 0.12, "num_updates": 0, "is_suspended": False},
            {"agent_id": "argus-jury-001",        "credibility": 0.92,
             "brier_score": 0.10, "num_updates": 0, "is_suspended": False},
        ]
        try:
            agent_ids = ledger.get_all_agent_ids()
            snapshot = []
            for agent_id in agent_ids:
                state = ledger.get_agent_state(agent_id)
                if state:
                    snapshot.append({
                        "agent_id": agent_id,
                        "credibility": round(state.credibility_rating, 4),
                        "brier_score": round(state.average_brier, 4),
                        "num_updates": state.sample_size,
                        "is_suspended": state.is_suspended,
                    })
            # Merge: add defaults for any agent not in live ledger
            live_ids = {s["agent_id"] for s in snapshot}
            for d in _defaults:
                if d["agent_id"] not in live_ids:
                    snapshot.append(d)
            return snapshot
        except Exception:
            return _defaults
