"""
HANNIBAL — Hierarchical Adversarial Network for Nested Intelligence
Battles and Logic.

Façade class providing a clean entry point for the HANNIBAL protocol.
Mirror of :class:`argus.agora.AGORA` and :class:`argus.aristotle.ARISTOTLE`.

Usage (programmatic)::

    from argus.hannibal import HANNIBAL
    campaign = HANNIBAL()
    result = campaign.run("Nuclear fusion will be commercially viable by 2040")

Usage (CLI)::

    argus hannibal run

Usage (Streamlit)::

    streamlit run argus/hannibal/hannibal_app.py
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from argus.hannibal.models import (
    BattleMap,
    CampaignPhase,
    CampaignVerdict,
    ForceSpec,
    HannibalResult,
    HannibalSessionConfig,
)
from argus.hannibal.pda import PropositionDepthAnalyzer
from argus.hannibal.tournament import TournamentTree
from argus.hannibal.forces import ForceDeploymentEngine
from argus.hannibal.battle import BattleEngine
from argus.hannibal.cannae import CANNAEEngine
from argus.hannibal.field_marshal import FieldMarshal
from argus.hannibal.results import HannibalResultBuilder

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

__all__ = [
    "HANNIBAL",
    "BattleMap",
    "CampaignVerdict",
    "ForceSpec",
    "HannibalResult",
    "HannibalSessionConfig",
    "PropositionDepthAnalyzer",
    "TournamentTree",
    "ForceDeploymentEngine",
    "BattleEngine",
    "CANNAEEngine",
    "FieldMarshal",
    "HannibalResultBuilder",
]


class HANNIBAL:
    """Façade for the HANNIBAL Battlefield Epistemic Protocol.

    Orchestrates the full pipeline:
        1. PDA (Proposition Depth Analyzer) → BattleMap
        2. Force Deployment Engine → list[ForceSpec]
        3. Tournament Tree construction
        4. Battle Protocol Engine (sequential skirmishes)
        5. CANNAE Engine (if multipolar)
        6. Field Marshal (armistice check + verdict narration)
        7. Result Builder → HannibalResult

    Designed for sequential execution on constrained hardware
    (i3 processor, 8 GB RAM).
    """

    def __init__(
        self,
        llm: "BaseLLM | None" = None,
        config: HannibalSessionConfig | None = None,
    ):
        """Initialise HANNIBAL.

        Args:
            llm: LLM instance. If None, uses ``get_llm()`` with env defaults.
            config: Session configuration overrides.
        """
        if llm is None:
            from argus.core.llm import get_llm
            llm = get_llm()
        self.llm = llm
        self.config = config or HannibalSessionConfig()

        # Engines (lazy-initialised, share single LLM instance)
        self._pda = PropositionDepthAnalyzer(llm)
        self._forces = ForceDeploymentEngine(llm)
        self._field_marshal = FieldMarshal(llm)
        self._cannae = CANNAEEngine(llm)
        self._result_builder = HannibalResultBuilder(llm)

    def run(
        self,
        proposition: str,
        phase_callback: Any = None,
    ) -> HannibalResult:
        """Execute a complete HANNIBAL campaign.

        Args:
            proposition: The proposition to battle over.
            phase_callback: Optional ``fn(phase, label, details_dict)``
                           called after each major event (for UI updates).

        Returns:
            Complete :class:`HannibalResult` with all nine components.
        """
        start_time = time.time()

        def _cb(phase, label, details):
            if phase_callback:
                phase_callback(phase, label, details)

        # ── Phase 1: Proposition Analysis ─────────────────────────
        _cb(CampaignPhase.ANALYSIS, "PDA", {"status": "starting"})
        logger.info("HANNIBAL: Analysing proposition: %s", proposition[:100])

        battle_map = self._pda.analyze(proposition, self.config)
        _cb(CampaignPhase.ANALYSIS, "PDA", {
            "status": "complete",
            "battle_map": battle_map.to_dict(),
        })

        # ── Phase 2: Force Deployment ─────────────────────────────
        _cb(CampaignPhase.DEPLOYMENT, "Forces", {"status": "starting"})
        logger.info("HANNIBAL: Deploying forces…")

        forces = self._forces.deploy(battle_map, self.config)
        _cb(CampaignPhase.DEPLOYMENT, "Forces", {
            "status": "complete",
            "num_forces": len(forces),
            "total_agents": sum(f.force_size for f in forces),
            "force_types": [f.force_type.value for f in forces],
            "forces_data": [f.to_dict() for f in forces]
        })

        # ── Phase 3: Tournament Tree Construction ─────────────────
        tree = TournamentTree(config=self.config)
        tree.build_tree(battle_map)

        # Expose initial tree state for UI
        self._last_tree_state = tree.get_bracket_state()

        _cb(CampaignPhase.BATTLE, "Tournament Tree", {
            "status": "constructed",
            "total_skirmishes": tree.total_skirmishes,
        })

        # ── Phase 4: Battle ───────────────────────────────────────
        logger.info("HANNIBAL: Battle phase — %d skirmishes (sequential)",
                     tree.total_skirmishes)

        # Wrap user callback to inject live tree state after each skirmish
        def _battle_callback(phase, label, details):
            # Update live tree state snapshot after each event
            self._last_tree_state = tree.get_bracket_state()
            if phase_callback:
                phase_callback(phase, label, details)

        battle_engine = BattleEngine(llm=self.llm, config=self.config)
        verdict, tree = battle_engine.run_campaign(
            tree=tree,
            forces=forces,
            battle_map=battle_map,
            phase_callback=_battle_callback,
        )

        # Final tree state after battle complete
        self._last_tree_state = tree.get_bracket_state()

        # ── Phase 5: CANNAE Resolution (if multipolar) ────────────
        encirclement_report: dict[str, Any] = {}
        if battle_map.cannae_activated:
            logger.info("HANNIBAL: CANNAE Engine activated (multipolar)")
            _cb(CampaignPhase.RESOLUTION, "CANNAE", {"status": "starting"})

            theatre_results = []
            for th_node in tree.theatre_nodes:
                if th_node.is_resolved and th_node.winner_force:
                    from argus.hannibal.models import TheatreResult
                    theatre_results.append(TheatreResult(
                        theatre_id=th_node.id,
                        theatre_name=th_node.label,
                        winner_force=th_node.winner_force,
                        theatre_score=th_node.confidence,
                    ))

            if theatre_results:
                cannae_verdict, encirclement_report = self._cannae.resolve(
                    theatre_results, forces,
                )
                # CANNAE verdict overrides standard for multipolar
                verdict = cannae_verdict

        # ── Phase 6: Armistice Check ──────────────────────────────
        armistice_fired = False
        armistice_option = None
        armistice_details = ""

        fired, option, details = self._field_marshal.evaluate_armistice(
            verdict, self.config,
        )
        if fired:
            armistice_fired = True
            armistice_option = option
            armistice_details = details
            logger.info("HANNIBAL: Armistice Protocol fired: %s", option)

        # ── Phase 7: Verdict Narration ────────────────────────────
        verdict.narrative = self._field_marshal.narrate_verdict(
            verdict=verdict,
            proposition=proposition,
            num_skirmishes=tree.total_skirmishes,
            total_evidence=len(battle_engine.all_evidence),
        )

        # ── Phase 8: Assemble Result ──────────────────────────────
        elapsed = time.time() - start_time

        result = self._result_builder.build(
            proposition=proposition,
            battle_map=battle_map,
            verdict=verdict,
            forces=forces,
            tree=tree,
            all_evidence=battle_engine.all_evidence,
            campaign_log=battle_engine.campaign_log,
            force_posterior_history=battle_engine.force_posterior_history,
            duration_seconds=elapsed,
            armistice_fired=armistice_fired,
            armistice_option=armistice_option,
            armistice_details=armistice_details,
            encirclement_report=encirclement_report,
        )

        _cb(CampaignPhase.COMPLETE, "Campaign", {
            "status": "complete",
            "verdict": verdict.to_dict(),
            "duration": elapsed,
        })

        logger.info(
            "HANNIBAL: Campaign complete — %s (%s, %.0f%%) in %.0fs",
            verdict.verdict_label.value,
            verdict.winning_force.display_name,
            verdict.campaign_strength_score * 100,
            elapsed,
        )
        return result

    @staticmethod
    def launch_chat() -> None:
        """Launch the War Room Streamlit interface."""
        import subprocess
        import sys
        from pathlib import Path

        app_path = Path(__file__).parent / "hannibal_app.py"
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless", "true",
        ])
