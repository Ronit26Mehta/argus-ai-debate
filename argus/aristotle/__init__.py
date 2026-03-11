"""
ARISTOTLE — Autonomous Reasoning Intelligence for Structured
Topic-Orchestrated Logical Engagement.

A five-layer meta-orchestration module inside argus-debate-ai that transforms
a single natural-language question into a fully autonomous, visualised,
auditable multi-agent debate.

Layers:
    L1  framing.py      — Intent Parsing & Framing Engine
    L2  topology.py     — Dynamic Topology Builder
    L3  monitor.py      — Autonomous Execution Engine (meta-monitor)
    L4  interface.py    — Single-Pane Split Streamlit UI
        dag_viz.py      — Full Lifecycle DAG (Plotly)
        panels.py       — Belief Trajectory, Evidence Heatmap, Sankey, etc.
    L5  synthesis.py    — Plain-Language Output Synthesis

Usage (programmatic):
    >>> from argus.aristotle import ARISTOTLE
    >>> agent = ARISTOTLE(llm=get_llm("openai", model="gpt-4o"))
    >>> result = agent.run("Is social media causing the mental health crisis?")

Usage (Streamlit):
    $ argus aristotle run
"""

__version__ = "4.5.0"

from argus.aristotle.models import (
    DebateFrame,
    DebateType,
    TopologySpec,
    AgentSpec,
    JurySpec,
    JuryArchitecture,
    RefuterIntensity,
    MonitorEvent,
    MonitorEventType,
    SynthesisResult,
    VerdictLevel,
)
from argus.aristotle.framing import FramingEngine
from argus.aristotle.topology import TopologyBuilder
from argus.aristotle.monitor import ExecutionMonitor
from argus.aristotle.synthesis import SynthesisEngine
from argus.aristotle.decision_log import DecisionLogger, OrchestratorDecisionType


class ARISTOTLE:
    """
    Top-level façade for the five-layer ARISTOTLE protocol.

    Accepts a natural-language query and drives ARGUS end-to-end:
    framing → topology → execution → synthesis.
    """

    def __init__(self, llm=None, fast_llm=None, config: dict | None = None):
        """
        Args:
            llm: Primary LLM for debate agents (BaseLLM instance).
            fast_llm: Lightweight LLM for framing (defaults to *llm*).
            config: Optional overrides (max_rounds, budget_tokens, …).
        """
        from argus.core.llm import get_llm

        self.llm = llm or get_llm()
        self.fast_llm = fast_llm or self.llm
        self.config = config or {}

        self.framing_engine = FramingEngine(llm=self.fast_llm)
        self.topology_builder = TopologyBuilder(llm=self.fast_llm)
        self.monitor = ExecutionMonitor(llm=self.llm)
        self.synthesis_engine = SynthesisEngine(llm=self.fast_llm)

    # ── Programmatic interface (no Streamlit) ──────────────────────────

    def run(self, query: str) -> SynthesisResult:
        """
        Full autonomous pipeline for a single query.

        Returns a :class:`SynthesisResult` containing verdict, narrative,
        dissent log, *What Could Change This*, and metric dashboard.
        """
        # L1 — framing
        frame = self.framing_engine.frame(query)

        # L2 — topology
        topology = self.topology_builder.build(frame)

        # L3 — execution
        debate_result, events = self.monitor.execute(
            topology=topology,
            frame=frame,
            llm=self.llm,
        )

        # L5 — synthesis
        result = self.synthesis_engine.synthesise(
            frame=frame,
            topology=topology,
            debate_result=debate_result,
            events=events,
        )
        return result

    def frame(self, query: str) -> DebateFrame:
        """Run Layer 1 only and return the DebateFrame."""
        return self.framing_engine.frame(query)

    def build_topology(self, frame: DebateFrame) -> TopologySpec:
        """Run Layer 2 only and return the TopologySpec."""
        return self.topology_builder.build(frame)


__all__ = [
    "ARISTOTLE",
    "__version__",
    "DebateFrame",
    "DebateType",
    "TopologySpec",
    "AgentSpec",
    "JurySpec",
    "JuryArchitecture",
    "RefuterIntensity",
    "MonitorEvent",
    "MonitorEventType",
    "SynthesisResult",
    "VerdictLevel",
    "FramingEngine",
    "TopologyBuilder",
    "ExecutionMonitor",
    "SynthesisEngine",
    "DecisionLogger",
    "OrchestratorDecisionType",
    "launch_chat",
]


def launch_chat() -> None:
    """Entry point for the ``aristotle-chat`` console script.

    Launches the ARISTOTLE Streamlit chat interface.
    """
    import os
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "interface.py"
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=false",
            "--theme.base=dark",
            "--theme.primaryColor=#00d4ff",
            "--theme.backgroundColor=#0B141A",
            "--theme.secondaryBackgroundColor=#1F2C34",
            "--theme.textColor=#E9EDEF",
            "--browser.gatherUsageStats=false",
        ],
    )
