"""
fsociety VAPT Orchestrator.

Wraps ARGUS RDCOrchestrator with VAPT-specific 6-round debate lifecycle.
Irving is the meta-orchestrator; this module manages the session lifecycle.
"""

from __future__ import annotations

import json
import logging
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from argus.orchestrator import RDCOrchestrator
from argus.cdag.graph import CDAG

from fsociety.config import FsocietyConfig, get_config
from fsociety.vkg import VulnerabilityKnowledgeGraph
from fsociety.models import (
    SeverityLevel,
    FindingStatus,
    ScanTarget,
    ScanSession,
)
from fsociety.ingestion.code_chunker import CodeChunker
from fsociety.ingestion.dependency_scanner import DependencyScanner
from fsociety.ingestion.git_analyzer import GitAnalyzer

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class VAPTOrchestrator:
    """
    fsociety VAPT Orchestrator — manages the complete VAPT session lifecycle.

    Lifecycle:
        1. Target Ingestion  — code chunking, dep scanning, git analysis
        2. Index & Embed     — FAISS + BM25 hybrid index
        3. Recon Round       — Elliot maps attack surface
        4. Debate Rounds     — multi-agent adversarial debate (up to 6 rounds)
        5. Verdict           — Whiterose renders P0-P3 verdicts
        6. Output            — Leon + Cisco produce reports
    """

    def __init__(self, config: Optional[FsocietyConfig] = None):
        self.config = config or get_config()
        self._llm: Optional["BaseLLM"] = None
        self._vkg: Optional[VulnerabilityKnowledgeGraph] = None
        self._session: Optional[ScanSession] = None
        self._agents: dict[str, Any] = {}
        self._code_chunks: list[Any] = []
        self._dependencies: list[Any] = []
        self._git_insights: list[Any] = []
        self._cloned_dir: Optional[str] = None  # temp dir for cloned repos

    # ── Properties ────────────────────────────────────────────────

    @property
    def llm(self) -> "BaseLLM":
        """Lazy-init the LLM using config."""
        if self._llm is None:
            self._llm = self.config.get_llm()
        return self._llm

    @property
    def vkg(self) -> VulnerabilityKnowledgeGraph:
        """Lazy-init the VKG."""
        if self._vkg is None:
            self._vkg = VulnerabilityKnowledgeGraph(name="fsociety-vkg")
        return self._vkg

    @property
    def session(self) -> ScanSession:
        if self._session is None:
            self._session = ScanSession(session_id=self.config.generate_session_id())
        return self._session

    # ── Agent initialization ──────────────────────────────────────

    def _init_agents(self) -> None:
        """Initialize agents based on config."""
        from fsociety.agents import (
            ElliotAgent, MrRobotAgent, DarleneAgent, WhiteroseAgent, IrvingAgent,
            RomeroAgent, MobleyAgent, TrentonAgent, TyrellAgent, AngelaAgent, DomAgent,
            LeonAgent, CiscoAgent,
        )

        agent_map = {
            "elliot": ElliotAgent,
            "mrrobot": MrRobotAgent,
            "darlene": DarleneAgent,
            "whiterose": WhiteroseAgent,
            "irving": IrvingAgent,
            "romero": RomeroAgent,
            "mobley": MobleyAgent,
            "trenton": TrentonAgent,
            "tyrell": TyrellAgent,
            "angela": AngelaAgent,
            "dom": DomAgent,
            "leon": LeonAgent,
            "cisco": CiscoAgent,
        }

        for name in self.config.scan.agents:
            name_lower = name.lower()
            if name_lower in agent_map:
                self._agents[name_lower] = agent_map[name_lower](llm=self.llm)
                logger.info(f"Initialized agent: {name}")

        # Always include Irving (orchestrator) and output agents
        for always_on in ("irving", "leon", "cisco"):
            if always_on not in self._agents and always_on in agent_map:
                self._agents[always_on] = agent_map[always_on](llm=self.llm)

    # ── Phase 1: Ingestion ────────────────────────────────────────

    def _clone_repo(self, repo_url: str) -> str:
        """Clone a git repo to a temp directory and return the path."""
        import git as gitmodule

        tmp_dir = tempfile.mkdtemp(prefix="fsociety_")
        logger.info(f"Cloning {repo_url} → {tmp_dir}")
        gitmodule.Repo.clone_from(repo_url, tmp_dir, depth=1)
        self._cloned_dir = tmp_dir
        logger.info(f"Clone complete: {tmp_dir}")
        return tmp_dir

    def ingest(self, target: ScanTarget) -> dict[str, Any]:
        """Phase 1: Ingest the target — code chunking, deps, git."""
        self.session.target = target
        self.session.started_at = datetime.now()

        # Auto-clone if repo_url provided but no local path
        if target.repo_url and not target.path:
            cloned_path = self._clone_repo(target.repo_url)
            target.path = cloned_path
            logger.info(f"Repo cloned to {cloned_path}")

        results: dict[str, Any] = {
            "code_chunks": 0,
            "dependencies": 0,
            "git_insights": 0,
        }

        # Code chunking
        if target.path:
            logger.info(f"Chunking codebase at {target.path}")
            chunker = CodeChunker()
            self._code_chunks = chunker.chunk_directory(target.path)
            results["code_chunks"] = len(self._code_chunks)
            logger.info(f"Produced {len(self._code_chunks)} code chunks")

        # Dependency scanning
        if target.path:
            scanner = DependencyScanner()
            self._dependencies = scanner.scan_directory(target.path)
            results["dependencies"] = len(self._dependencies)

        # Git analysis
        if target.path:
            analyzer = GitAnalyzer()
            self._git_insights = analyzer.analyze(target.path)
            results["git_insights"] = len(self._git_insights)

        logger.info(f"Ingestion complete: {results}")
        return results

    # ── Phase 2: Debate ───────────────────────────────────────────

    def debate(
        self,
        max_rounds: Optional[int] = None,
        round_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Phase 2: Run the multi-agent VAPT debate.

        Each round:
          1. Active agents analyze target context + previous findings
          2. Propositions and evidence flow into the VKG
          3. Posterior probabilities update
          4. Convergence check (posterior stability or P0 escalation)
        """
        if not self._agents:
            self._init_agents()

        max_rounds = max_rounds or self.config.scan.max_debate_rounds
        all_findings: list[dict[str, Any]] = []
        posteriors: list[float] = [0.5]  # Starting prior

        for round_num in range(1, max_rounds + 1):
            logger.info(f"=== Debate Round {round_num}/{max_rounds} ===")
            round_findings: list[dict[str, Any]] = []

            # Build context from ingested data + previous findings
            context = self._build_debate_context(round_num, all_findings)

            # Each active agent analyzes
            for agent_name, agent in self._agents.items():
                if agent_name in ("irving", "leon", "cisco"):
                    continue  # These are special-purpose, not debaters

                try:
                    response = agent.act(self.vkg, context)
                    findings = agent.parse_findings(response.content)
                    for f in findings:
                        f["agent"] = agent_name
                        f["round"] = round_num
                        round_findings.append(f)
                except Exception as e:
                    logger.warning(f"Agent {agent_name} failed in round {round_num}: {e}")

            # Add findings to VKG
            for finding in round_findings:
                sev = self._parse_severity(finding.get("severity", "P2"))
                self.vkg.add_vulnerability(
                    text=finding.get("text", "Unknown finding"),
                    severity=sev,
                    agent_source=finding.get("agent", "unknown"),
                )

            all_findings.extend(round_findings)

            # Update posterior
            if round_findings:
                # Simple confidence update based on findings density
                delta = len(round_findings) * 0.05
                new_posterior = min(0.99, posteriors[-1] + delta)
                posteriors.append(new_posterior)
            else:
                posteriors.append(posteriors[-1])

            # Callback for live UI updates
            if round_callback:
                round_callback(round_num, posteriors, all_findings, [])

            # Convergence check
            if posteriors[-1] >= self.config.scan.posterior_threshold:
                logger.info(f"Posterior {posteriors[-1]:.2f} exceeded threshold — stopping early")
                break

            # Stability check (posterior hasn't changed in 2 rounds)
            if round_num >= 3 and abs(posteriors[-1] - posteriors[-2]) < 0.01:
                logger.info("Posterior stable — stopping debate")
                break

        self.session.total_findings = len(all_findings)
        self.session.debate_rounds = len(posteriors) - 1
        self.session.agents_used = list(self._agents.keys())

        # Count severities
        for f in all_findings:
            sev = f.get("severity", "P2")
            if sev in self.session.severity_counts:
                self.session.severity_counts[sev] += 1

        return {
            "findings": all_findings,
            "posteriors": posteriors,
            "vkg_summary": self.vkg.vkg_summary(),
            "rounds_completed": len(posteriors) - 1,
            "session": self.session.model_dump(),
        }

    # ── Phase 3: Report ───────────────────────────────────────────

    def generate_report(self, debate_result: dict[str, Any]) -> Path:
        """Phase 3: Generate the output report + visualizations."""
        from fsociety.outputs.report_builder import ReportBuilder
        from fsociety.outputs.directory_manager import OutputDirectoryManager
        from fsociety.outputs.visualizations import generate_all_plots

        target_name = self.session.target.name
        dir_mgr = OutputDirectoryManager(self.config.output_dir)
        session_dir = dir_mgr.create_session_tree(target_name, self.session.session_id)

        builder = ReportBuilder()
        report_path = builder.generate(
            debate_result=debate_result,
            session=self.session,
            output_dir=session_dir,
        )

        # Generate visualizations
        try:
            plots = generate_all_plots(debate_result, session_dir)
            logger.info(f"Generated {len(plots)} visualizations")
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

        self.session.completed_at = datetime.now()
        logger.info(f"Report generated: {report_path}")
        return report_path

    # ── Full pipeline ─────────────────────────────────────────────

    def scan(
        self,
        path: Optional[str] = None,
        repo_url: Optional[str] = None,
        url: Optional[str] = None,
        round_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Run the complete fsociety pipeline: ingest → debate → report."""
        target = ScanTarget(path=path, repo_url=repo_url, url=url)
        logger.info(f"Starting fsociety scan: {target.name} (mode={target.mode})")

        # Phase 1
        ingestion_stats = self.ingest(target)

        # Phase 2
        debate_result = self.debate(round_callback=round_callback)

        # Phase 3
        try:
            report_path = self.generate_report(debate_result)
            debate_result["report_path"] = str(report_path)
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            debate_result["report_path"] = None

        debate_result["ingestion"] = ingestion_stats
        return debate_result

    # ── Helpers ────────────────────────────────────────────────────

    def _build_debate_context(self, round_num: int, previous_findings: list[dict]) -> dict[str, Any]:
        """Build context dict for agents in a debate round."""
        context: dict[str, Any] = {"round": round_num}

        # Code sample (first N chunks)
        if self._code_chunks:
            sample_chunks = self._code_chunks[:10]
            context["code_chunks"] = "\n---\n".join(
                f"[{c.filename}:{c.line_start}-{c.line_end}]\n{c.text[:500]}"
                for c in sample_chunks
            )

        # Dependencies
        if self._dependencies:
            context["dependencies"] = "\n".join(
                f"  {d.ecosystem}: {d.name} {d.version}" for d in self._dependencies[:30]
            )

        # Git insights
        if self._git_insights:
            context["git_insights"] = "\n".join(
                f"  [{i.severity}] {i.description}" for i in self._git_insights[:10]
            )

        # Previous findings
        if previous_findings:
            context["previous_findings"] = json.dumps(previous_findings[-20:], indent=2)

        return context

    def _parse_severity(self, sev: str) -> SeverityLevel:
        """Parse severity string to enum."""
        sev_map = {
            "P0": SeverityLevel.CRITICAL,
            "P1": SeverityLevel.HIGH,
            "P2": SeverityLevel.MEDIUM,
            "P3": SeverityLevel.LOW,
            "CRITICAL": SeverityLevel.CRITICAL,
            "HIGH": SeverityLevel.HIGH,
            "MEDIUM": SeverityLevel.MEDIUM,
            "LOW": SeverityLevel.LOW,
            "INFO": SeverityLevel.INFO,
        }
        return sev_map.get(sev.upper(), SeverityLevel.MEDIUM)
