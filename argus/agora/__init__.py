"""
AGORA — Autonomous Governed Open Reasoning Assembly
═══════════════════════════════════════════════════

Third ARGUS sub-module implementing a procedurally governed,
dynamically composed, real-time deliberative body for evidence-based
multi-agent debate.

Architecture
────────────
    Layer 1 — Senate Generation  (senate_gen.py)
    Layer 2 — Procedural Rules   (procedures.py)
    Layer 3 — Evidence Docket    (docket.py)
    Layer 4 — Socratic Engine    (socratic.py)
    Layer 5 — Coalition Detection (coalitions.py)
    Layer 6 — Minority Report    (minority.py)
    Layer 7 — Senate Record      (record.py)
    Layer 8 — Result Assembly    (results.py)

Usage (programmatic)
────────────────────
    >>> from argus.agora import AGORA
    >>> agora = AGORA()
    >>> result = agora.run("Is nuclear fusion commercially viable by 2040?")
    >>> print(result.majority_opinion.verdict_label)
    'Qualified'
    >>> print(result.minority_report.narrative)
    ...

Launch sandbox
──────────────
    $ streamlit run argus/agora/agora_app.py
    $ agora-chat                                    # via console script
    $ python -c "from argus.agora import launch_chat; launch_chat()"

Version
───────
"""

from __future__ import annotations

__version__ = "5.5.0"
__all__ = [
    "AGORA",
    "launch_chat",
    # Models
    "AgoraResult",
    "AgoraSessionConfig",
    "SenateSpec",
    "SenatorSpec",
    "SenatorCategory",
    "SessionPhase",
    "StoppingTrigger",
    "DocketItem",
    "Challenge",
    "CoalitionInfo",
    "MinorityReport",
    "MajorityOpinion",
    "SenatorScorecard",
    "VerdictLabel",
    # Engines
    "SenateGenerator",
    "SocraticEngine",
    "EvidenceDocket",
    "CoalitionDetectionEngine",
    "MinorityReportEngine",
    "SenateRecord",
    "AgoraResultBuilder",
]

import logging
from typing import TYPE_CHECKING, Optional

from argus.agora.models import (
    AgoraResult,
    AgoraSessionConfig,
    Challenge,
    CoalitionInfo,
    DocketItem,
    MajorityOpinion,
    MinorityReport,
    SenateSpec,
    SenatorCategory,
    SenatorScorecard,
    SenatorSpec,
    SessionPhase,
    StoppingTrigger,
    VerdictLabel,
)
from argus.agora.senate_gen import SenateGenerator
from argus.agora.socratic import SocraticEngine
from argus.agora.docket import EvidenceDocket
from argus.agora.coalitions import CoalitionDetectionEngine
from argus.agora.minority import MinorityReportEngine
from argus.agora.record import SenateRecord
from argus.agora.results import AgoraResultBuilder

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class AGORA:
    """Façade for the Autonomous Governed Open Reasoning Assembly.

    Mirrors the ARISTOTLE interface: construct once, call ``run()``
    to execute a full deliberation, or use individual engines.

    Example::

        >>> from argus.agora import AGORA
        >>> agora = AGORA()
        >>> result = agora.run("Should AI systems require licensing?")
        >>> print(result.chat_card())
    """

    def __init__(
        self,
        llm: "BaseLLM | None" = None,
        config: AgoraSessionConfig | None = None,
    ):
        """Initialise AGORA.

        Args:
            llm: LLM provider instance. If None, uses the ARGUS default.
            config: Session configuration. If None, uses defaults
                    (unbounded time, 5 rounds per phase, 7-25 senators).
        """
        if llm is None:
            from argus.core.llm import get_llm
            llm = get_llm()

        self.llm = llm
        self.config = config or AgoraSessionConfig()

        # Engines
        self.senate_generator = SenateGenerator(llm=self.llm)
        self.socratic_engine = SocraticEngine(llm=self.llm, config=self.config)

        logger.info(
            "AGORA v%s initialised (provider=%s, model=%s)",
            __version__, llm.provider_name, llm.model,
        )

    def run(
        self,
        proposition: str,
        config: AgoraSessionConfig | None = None,
        round_callback=None,
    ) -> AgoraResult:
        """Run a complete AGORA deliberation session.

        Pipeline:
            1. Senate Generation → SenateSpec
            2. Socratic Engine   → AgoraResult (drives all 5 phases)

        Args:
            proposition: The proposition to deliberate.
            config: Override session configuration.
            round_callback: Optional callback for live UI updates.

        Returns:
            Complete AgoraResult with all 9 output components.
        """
        config = config or self.config
        logger.info("AGORA session starting: %s", proposition[:100])

        # Layer 1: Generate Senate
        senate = self.senate_generator.generate(proposition, config)
        logger.info("Senate generated: %d senators", senate.senate_size)

        # Layers 2–8: Run session
        result = self.socratic_engine.run_session(
            senate=senate,
            proposition=proposition,
            config=config,
            round_callback=round_callback,
        )

        logger.info(
            "AGORA complete: %s (%s)",
            result.majority_opinion.verdict_label.value,
            f"{result.majority_opinion.posterior_probability:.0%}",
        )
        return result

    def generate_senate(
        self,
        proposition: str,
        config: AgoraSessionConfig | None = None,
    ) -> SenateSpec:
        """Run only the Senate Generation Engine (Layer 1).

        Useful for previewing the Senate composition before launching
        a full session.
        """
        return self.senate_generator.generate(proposition, config or self.config)


def launch_chat() -> None:
    """Launch the AGORA Streamlit sandbox.

    Entry point for the ``agora-chat`` console script.
    """
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "agora_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
    )
