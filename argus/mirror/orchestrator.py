"""
MIRROR Orchestrator — consequence inference after debate.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass

from argus.mirror.inference_agent import ConsequenceInferenceAgent
from argus.mirror.graph import ConsequenceGraph
from argus.mirror.counterfactual import CounterfactualChallenger, CounterfactualReport

logger = logging.getLogger(__name__)


@dataclass
class MIRRORConfig:
    max_consequences_per_agent: int = 8
    min_sensitivity: float = 0.1
    enable_counterfactual: bool = True


@dataclass
class MIRRORResult:
    base_result: Any = None
    consequence_graph: Optional[ConsequenceGraph] = None
    counterfactual_report: Optional[CounterfactualReport] = None
    num_consequences: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"num_consequences": self.num_consequences}
        if self.consequence_graph:
            result["graph"] = self.consequence_graph.to_dict()
        if self.counterfactual_report:
            result["counterfactual"] = self.counterfactual_report.to_dict()
        return result


class MIRROROrchestrator:
    """Consequence Inference Orchestrator."""

    def __init__(
        self,
        base: Optional[Any] = None,
        config: Optional[MIRRORConfig] = None,
    ):
        self.base = base
        self.config = config or MIRRORConfig()
        self._opp_agent = ConsequenceInferenceAgent(
            role="opportunity", max_consequences=self.config.max_consequences_per_agent,
        )
        self._risk_agent = ConsequenceInferenceAgent(
            role="risk", max_consequences=self.config.max_consequences_per_agent,
        )
        self._challenger = CounterfactualChallenger(
            min_sensitivity=self.config.min_sensitivity,
        )

    def debate(self, proposition: str, prior: float = 0.5, **kwargs: Any) -> MIRRORResult:
        """Run debate + consequence inference."""
        # Run base debate
        base_result = None
        verdict_posterior = prior
        verdict_str = "undetermined"

        if self.base:
            try:
                base_result = self.base.debate(proposition, prior=prior, **kwargs)
                verdict_posterior = getattr(base_result, 'posterior', prior)
                verdict_str = str(getattr(base_result, 'verdict', 'undetermined'))
            except Exception as e:
                logger.warning(f"Base debate failed: {e}")

        # Infer consequences
        opp = self._opp_agent.infer(verdict_str, proposition, verdict_posterior)
        risk = self._risk_agent.infer(verdict_str, proposition, verdict_posterior)

        # Build graph
        graph = ConsequenceGraph(
            root_verdict=verdict_str,
            root_posterior=verdict_posterior,
            proposition=proposition,
        )
        graph.add_consequences(opp)
        graph.add_consequences(risk)
        graph.compute_marginals()

        # Counterfactual analysis
        cf_report = None
        if self.config.enable_counterfactual:
            cf_report = self._challenger.analyse(graph)

        return MIRRORResult(
            base_result=base_result,
            consequence_graph=graph,
            counterfactual_report=cf_report,
            num_consequences=graph.num_nodes,
        )
