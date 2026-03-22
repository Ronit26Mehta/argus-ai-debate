"""Production-oriented ARGUS Sandbox orchestrator for all blueprint modules."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

from argus.chronos import ChronosConfig, ChronosOrchestrator
from argus.fractal import FRACTALConfig, FRACTALOrchestrator
from argus.mneme import MNEMEConfig, MNEMEPlugin
from argus.mirror import MIRRORConfig, MIRROROrchestrator
from argus.orchestrator import RDCOrchestrator
from argus.phalanx import PHALANXConfig, PHALANXOrchestrator
from argus.pulse import PULSEConfig, PULSEOrchestrator
from argus.seed import SEEDConfig, SEEDOrchestrator
from argus.verichain import EpistemicPrecedentInjector, VERICHAINRegistry

from argus.sandbox.lifecycle import LifecycleDAG
from argus.sandbox.storage import JsonFolderStore, SandboxRunPaths
from argus_viz.debate_engine import SpecialistDef, StreamingDebateEngine


@dataclass
class SandboxConfig:
    storage_dir: str = "./argus_sandbox_runs"
    max_rounds: int = 4
    population_size: int = 120
    parallel_workers: int = 4


class ArgusSandboxRunner:
    """Coordinates all ARGUS evolution modules with incremental persisted outputs."""

    def __init__(self, config: SandboxConfig | None = None, llm: Any | None = None):
        self.config = config or SandboxConfig()
        self.store = JsonFolderStore(self.config.storage_dir)

        self.base = RDCOrchestrator(llm=llm, max_rounds=self.config.max_rounds)
        self.seed = SEEDOrchestrator(
            config=SEEDConfig(max_claims=30, top_claims=5, min_debatability=0.25)
        )
        self.chronos = ChronosOrchestrator(
            base=self.base,
            config=ChronosConfig(
                temporal_resolution="month",
                lookback_years=3.0,
                drift_min_magnitude=0.03,
                drift_window_size=3,
                enable_drift_detection=True,
            ),
        )
        self.phalanx = PHALANXOrchestrator(
            base=self.base,
            config=PHALANXConfig(
                population_size=self.config.population_size,
                parallel_workers=self.config.parallel_workers,
            ),
        )
        self.mneme = MNEMEPlugin(config=MNEMEConfig(backend="memory", max_entries=8000))
        self.fractal = FRACTALOrchestrator(
            base=self.base,
            config=FRACTALConfig(
                max_depth=3,
                max_children=4,
                parallel_workers=self.config.parallel_workers,
            ),
        )
        self.mirror = MIRROROrchestrator(base=self.base, config=MIRRORConfig())
        self.pulse = PULSEOrchestrator(
            base=self.base,
            config=PULSEConfig(export_format="json", output_dir=self.config.storage_dir),
        )
        self.verichain = VERICHAINRegistry(backend="memory")
        self.injector = EpistemicPrecedentInjector(max_precedents=5)

    @staticmethod
    def _default_specialists(domain: str) -> list[SpecialistDef]:
        return [
            SpecialistDef(
                name="Evidence Analyst",
                persona=f"{domain} empirical",
                instruction="Find strongest supporting evidence with concise factual claims.",
            ),
            SpecialistDef(
                name="Critical Analyst",
                persona=f"{domain} skeptic",
                instruction="Find strongest opposing evidence and methodological weaknesses.",
            ),
            SpecialistDef(
                name="Systems Analyst",
                persona=f"{domain} systems",
                instruction="Assess tradeoffs, uncertainty, and long-term system impacts.",
            ),
        ]

    @staticmethod
    def _clamp(value: float, lo: float = 0.01, hi: float = 0.99) -> float:
        return max(lo, min(hi, value))

    def _event(
        self,
        run: SandboxRunPaths,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "message": message,
            "payload": payload or {},
        }
        self.store.append_event(run, event)
        return event

    def _stage_write(self, run: SandboxRunPaths, stage: str, data: dict[str, Any]) -> None:
        self.store.write_stage(run, stage, data)

    def run_iter(
        self,
        proposition: str,
        prior: float = 0.5,
        source_text: str | None = None,
        domain: str = "general",
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        """Yield incremental events while executing a complete sandbox run."""
        lifecycle = LifecycleDAG()
        run = self.store.create_run(proposition)
        started_at = datetime.now(timezone.utc)

        try:
            yield self._event(run, "system", f"run_started:{run.run_id}", {"run_path": str(run.root)})

            # SEED
            lifecycle.start("seed")
            yield self._event(run, "seed", "extracting_claims")
            seed_input = source_text or proposition
            bundle = self.seed.from_text(seed_input, source_title="sandbox_input")
            claim = bundle.top_claim
            debate_claim = claim.text if claim else proposition
            skeleton = claim.cdag_skeleton if claim else None

            seed_data = {
                "bundle": bundle.to_dict(),
                "selected_claim": debate_claim,
                "has_skeleton": bool(skeleton),
            }
            self._stage_write(run, "seed", seed_data)
            lifecycle.complete("seed", {"claims": bundle.num_claims})
            yield self._event(run, "seed", "completed", {"claims": bundle.num_claims})

            # VERICHAIN injection planning
            lifecycle.start("verichain_injection")
            precedents = [(node, 1.0) for node in self.verichain.search(debate_claim, top_k=5)]
            injection = self.injector.plan_injection(precedents, debate_claim)
            adjusted_prior = self._clamp(prior + injection.prior_adjustment)
            inject_data = {
                "precedent_count": injection.num_precedents,
                "prior_adjustment": injection.prior_adjustment,
                "adjusted_prior": adjusted_prior,
                "evidence_texts": injection.evidence_texts,
            }
            self._stage_write(run, "verichain_injection", inject_data)
            lifecycle.complete("verichain_injection", {"precedents": injection.num_precedents})
            yield self._event(run, "verichain_injection", "completed", inject_data)

            # Debate flow (ARISTOTLE-like round progression)
            lifecycle.start("debate_flow")
            yield self._event(run, "debate_flow", "starting_round_based_debate")
            debate_engine = StreamingDebateEngine(
                llm=self.base.llm,
                specialists=self._default_specialists(domain),
                max_rounds=self.config.max_rounds,
                refuter_enabled=True,
                jury_threshold=0.7,
                prior=adjusted_prior,
            )
            debate_flow_result = debate_engine.run_debate(debate_claim)

            rounds = debate_flow_result.get("rounds", [])
            for snapshot in rounds:
                yield self._event(
                    run,
                    "debate_flow",
                    "round_completed",
                    {
                        "round": snapshot.get("round"),
                        "posterior_after": snapshot.get("posterior_after"),
                        "support_count": snapshot.get("support_count", 0),
                        "attack_count": snapshot.get("attack_count", 0),
                        "rebuttals": len(snapshot.get("rebuttals", [])),
                    },
                )

            debate_flow_data = {
                "proposition": debate_flow_result.get("proposition", debate_claim),
                "prior": debate_flow_result.get("prior", adjusted_prior),
                "verdict": debate_flow_result.get("verdict", {}),
                "rounds": rounds,
                "graph_data": debate_flow_result.get("graph_data", {"nodes": [], "edges": []}),
                "duration_seconds": debate_flow_result.get("duration_seconds", 0.0),
                "config": debate_flow_result.get("config", {}),
            }
            self._stage_write(run, "debate_flow", debate_flow_data)
            lifecycle.complete("debate_flow", {"rounds": len(rounds)})
            yield self._event(
                run,
                "debate_flow",
                "completed",
                {
                    "rounds": len(rounds),
                    "final_posterior": debate_flow_data.get("verdict", {}).get("posterior", adjusted_prior),
                    "label": debate_flow_data.get("verdict", {}).get("label", "undecided"),
                },
            )

            # CHRONOS
            lifecycle.start("chronos")
            yield self._event(run, "chronos", "running_temporal_debate")
            evidence_timestamps: dict[str, str] = {}
            if skeleton:
                now = datetime.now(timezone.utc)
                all_frags = skeleton.supporting + skeleton.opposing + skeleton.neutral
                for idx, frag in enumerate(all_frags):
                    evidence_timestamps[frag.fragment_id] = (
                        now - timedelta(days=idx * 17)
                    ).replace(tzinfo=None).isoformat()

            chronos_result = self.chronos.debate(
                proposition=debate_claim,
                prior=adjusted_prior,
                evidence_timestamps=evidence_timestamps,
            )
            latest = (
                chronos_result.temporal_posterior.latest.posterior
                if chronos_result.temporal_posterior and chronos_result.temporal_posterior.latest
                else adjusted_prior
            )
            chronos_data = {
                "latest_posterior": latest,
                "trend": (
                    chronos_result.temporal_posterior.trend_direction
                    if chronos_result.temporal_posterior
                    else "unknown"
                ),
                "result": chronos_result.to_dict(),
            }
            self._stage_write(run, "chronos", chronos_data)
            lifecycle.complete("chronos", {"posterior": latest})
            yield self._event(run, "chronos", "completed", {"latest_posterior": latest})

            # PHALANX
            lifecycle.start("phalanx")
            yield self._event(run, "phalanx", "running_population_debate")
            evidence_items: list[dict[str, Any]] = []
            if skeleton:
                for frag in skeleton.supporting + skeleton.opposing:
                    evidence_items.append(
                        {
                            "confidence": frag.confidence,
                            "relevance": 0.75,
                            "polarity": frag.polarity if frag.polarity != 0 else 1,
                            "recency": 0.65,
                            "prestige": 0.7,
                            "alignment": 0.55,
                        }
                    )

            phalanx_result = self.phalanx.debate(
                proposition=debate_claim,
                prior=adjusted_prior,
                evidence_items=evidence_items or None,
            )
            population_mean = (
                phalanx_result.population_posterior.mean
                if phalanx_result.population_posterior
                else adjusted_prior
            )
            phalanx_data = {
                "consensus_type": phalanx_result.consensus_type,
                "population_mean": population_mean,
                "beliefs": (
                    phalanx_result.population_posterior.beliefs
                    if phalanx_result.population_posterior
                    else []
                ),
                "result": phalanx_result.to_dict(),
            }
            self._stage_write(run, "phalanx", phalanx_data)
            lifecycle.complete("phalanx", {"population_mean": population_mean})
            yield self._event(run, "phalanx", "completed", phalanx_data)

            # MNEME
            lifecycle.start("mneme")
            agents = ["moderator", "specialist", "refuter", "jury"]
            relevant = self.mneme.on_debate_start(debate_claim, agents=agents)
            confidence = float((latest + population_mean) / 2.0)
            verdict_label = "supported" if confidence >= 0.5 else "rejected"
            self.mneme.on_debate_end(
                proposition=debate_claim,
                verdict=verdict_label,
                domain=domain,
                confidence=confidence,
                agents=agents,
            )
            for agent in agents:
                self.mneme.record_outcome(
                    agent_id=agent,
                    domain=domain,
                    correct=True,
                    predicted=confidence,
                )
            mneme_data = {
                "retrieved_memory": {k: len(v) for k, v in relevant.items()},
                "summary": self.mneme.summary(),
            }
            self._stage_write(run, "mneme", mneme_data)
            lifecycle.complete("mneme", {"agents": len(agents)})
            yield self._event(run, "mneme", "completed", mneme_data)

            # FRACTAL
            lifecycle.start("fractal")
            fractal_result = self.fractal.debate(debate_claim)
            fractal_data = fractal_result.to_dict()
            self._stage_write(run, "fractal", fractal_data)
            lifecycle.complete("fractal", {"num_leaves": fractal_result.num_leaves})
            yield self._event(
                run,
                "fractal",
                "completed",
                {
                    "root_posterior": fractal_result.root_posterior,
                    "num_leaves": fractal_result.num_leaves,
                },
            )

            # MIRROR
            lifecycle.start("mirror")
            mirror_result = self.mirror.debate(debate_claim, prior=confidence)
            mirror_data = mirror_result.to_dict()
            self._stage_write(run, "mirror", mirror_data)
            lifecycle.complete("mirror", {"num_consequences": mirror_result.num_consequences})
            yield self._event(
                run,
                "mirror",
                "completed",
                {"num_consequences": mirror_result.num_consequences},
            )

            # PULSE
            lifecycle.start("pulse")
            pulse_result = self.pulse.debate(debate_claim, prior=confidence)
            pulse_data = pulse_result.to_dict()
            self._stage_write(run, "pulse", pulse_data)
            lifecycle.complete("pulse", {"has_report": bool(pulse_result.report)})
            yield self._event(run, "pulse", "completed", {"has_report": bool(pulse_result.report)})

            # VERICHAIN commit
            lifecycle.start("verichain_commit")
            node = self.verichain.register_verdict(
                proposition=debate_claim,
                verdict=verdict_label,
                posterior=confidence,
                domain=domain,
                debate_id=run.run_id,
            )
            verichain_data = {
                "node_id": node.node_id,
                "chain_length": self.verichain.chain_length,
                "posterior": confidence,
            }
            self._stage_write(run, "verichain_commit", verichain_data)
            lifecycle.complete("verichain_commit", {"chain_length": self.verichain.chain_length})
            yield self._event(run, "verichain_commit", "completed", verichain_data)

            completed_at = datetime.now(timezone.utc)
            summary = {
                "run_id": run.run_id,
                "paths": {
                    "root": str(run.root),
                    "events": str(run.events_file),
                },
                "input": {
                    "proposition": proposition,
                    "selected_claim": debate_claim,
                    "prior": prior,
                    "adjusted_prior": adjusted_prior,
                    "domain": domain,
                },
                "outputs": {
                    "verdict": verdict_label,
                    "posterior": confidence,
                    "chronos_latest": latest,
                    "population_mean": population_mean,
                    "fractal_root": fractal_result.root_posterior,
                    "mirror_consequences": mirror_result.num_consequences,
                    "verichain_chain_length": self.verichain.chain_length,
                },
                "lifecycle": lifecycle.to_dict(),
                "timing": {
                    "started_at": started_at.isoformat(),
                    "completed_at": completed_at.isoformat(),
                    "duration_seconds": (completed_at - started_at).total_seconds(),
                },
            }
            self.store.write_summary(run, summary)
            yield self._event(run, "system", "run_completed", {"summary_path": str(run.summary_file)})
            return summary

        except Exception as exc:
            err = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            self._stage_write(run, "failure", err)
            yield self._event(run, "system", "run_failed", err)
            raise
