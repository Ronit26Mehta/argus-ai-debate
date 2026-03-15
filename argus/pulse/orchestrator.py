"""
PULSE Orchestrator — wraps base orchestrator with full instrumentation.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass

from argus.pulse.dashboard import PULSEDashboard, PULSEConfig, DashboardReport

logger = logging.getLogger(__name__)


@dataclass
class PULSEResult:
    """Debate result with operational metrics."""
    base_result: Any = None
    report: Optional[DashboardReport] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.report:
            result["pulse"] = self.report.to_dict()
        return result


class PULSEOrchestrator:
    """
    Instrumented orchestrator with PULSE monitoring.

    Wraps the base orchestrator and instruments each debate stage
    with lightweight probes. Generates operational reports.

    Example:
        >>> pulse = PULSEOrchestrator(base=rdc)
        >>> result = pulse.debate("proposition")
        >>> print(result.report.recommendations)
    """

    def __init__(
        self,
        base: Optional[Any] = None,
        config: Optional[PULSEConfig] = None,
    ):
        self.base = base
        self.dashboard = PULSEDashboard(config or PULSEConfig())
        self.dashboard.start()

    def debate(self, proposition: str, prior: float = 0.5, **kwargs: Any) -> PULSEResult:
        """Run instrumented debate."""
        base_result = None

        with self.dashboard.probe.track("full_debate"):
            if self.base:
                try:
                    with self.dashboard.probe.track("base_debate"):
                        base_result = self.base.debate(
                            proposition, prior=prior, **kwargs,
                        )
                except Exception as e:
                    self.dashboard.record_failure(str(e))
                    logger.warning(f"Instrumented debate failed: {e}")

        report = self.dashboard.generate_report()

        return PULSEResult(base_result=base_result, report=report)

    def export_report(self, output_path: Optional[str] = None) -> str:
        return self.dashboard.export_report(output_path=output_path)
