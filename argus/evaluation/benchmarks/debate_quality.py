"""
Debate Quality Benchmark.

Evaluates overall quality of AI debate including verdict accuracy,
evidence coverage, and argument structure.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from argus.evaluation.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    SampleResult,
)
from argus.evaluation.scoring.metrics import compute_all_scores

logger = logging.getLogger(__name__)


class DebateQualityBenchmark(Benchmark):
    """Benchmark for evaluating debate quality.
    
    Measures:
        - Verdict accuracy against ground truth
        - Evidence coverage completeness
        - Argument structure coherence
        - Overall debate effectiveness
    
    Example:
        >>> from argus.evaluation import DebateQualityBenchmark, load_dataset
        >>> benchmark = DebateQualityBenchmark()
        >>> results = benchmark.run(load_dataset("factual_claims"), llm)
    """
    
    def __init__(self):
        super().__init__(
            name="debate_quality",
            description="Evaluates overall debate quality and verdict accuracy",
        )
    
    def evaluate_sample(
        self,
        sample: dict[str, Any],
        llm: Any,
        config: BenchmarkConfig,
    ) -> SampleResult:
        """Evaluate a single proposition using ARGUS debate.
        
        Args:
            sample: Dataset sample with proposition and ground truth
            llm: LLM instance for debate
            config: Benchmark configuration
            
        Returns:
            SampleResult with verdict and scores
        """
        from argus.orchestrator import RDCOrchestrator
        from argus.cdag.graph import CDAG
        from argus.cdag.nodes import Proposition
        from argus.agents import Specialist, Refuter, Jury
        
        proposition_text = sample.get("proposition", "")
        expected_verdict = sample.get("ground_truth", "undecided").lower()
        sample_id = str(sample.get("id", "unknown"))
        domain = sample.get("domain", "general")
        
        try:
            # Create orchestrator with configured rounds
            orchestrator = RDCOrchestrator(
                llm=llm,
                max_rounds=config.num_rounds,
            )
            
            # Run debate
            result = orchestrator.debate(
                proposition_text=proposition_text,
                prior=0.5,
                domain=domain,
            )
            
            # Extract verdict
            predicted_verdict = result.verdict.label.lower()
            posterior = result.verdict.posterior
            
            # Check correctness
            correct = self._check_verdict_match(predicted_verdict, expected_verdict)
            
            # Compute all ARGUS scores
            scores = compute_all_scores(result.to_dict())
            
            return SampleResult(
                sample_id=sample_id,
                proposition=proposition_text,
                expected=expected_verdict,
                predicted=predicted_verdict,
                posterior=posterior,
                correct=correct,
                duration_seconds=result.duration_seconds,
                scores=scores,
                metadata={
                    "num_rounds": result.num_rounds,
                    "num_evidence": result.num_evidence,
                    "num_rebuttals": result.num_rebuttals,
                    "domain": domain,
                },
            )
            
        except Exception as e:
            logger.error(f"Debate failed for sample {sample_id}: {e}")
            return SampleResult(
                sample_id=sample_id,
                proposition=proposition_text,
                expected=expected_verdict,
                predicted="error",
                posterior=0.5,
                correct=False,
                duration_seconds=0.0,
                scores={},
                metadata={"error": str(e)},
            )
    
    def _check_verdict_match(self, predicted: str, expected: str) -> bool:
        """Check if predicted verdict matches expected.
        
        Handles various verdict formats and synonyms.
        
        Args:
            predicted: Predicted verdict
            expected: Expected verdict
            
        Returns:
            True if verdicts match
        """
        # Normalize verdicts
        support_terms = {"supported", "support", "true", "yes", "confirmed", "valid"}
        reject_terms = {"rejected", "reject", "false", "no", "refuted", "invalid"}
        undecided_terms = {"undecided", "uncertain", "unknown", "inconclusive"}
        
        def normalize(verdict: str) -> str:
            verdict = verdict.lower().strip()
            if verdict in support_terms:
                return "supported"
            elif verdict in reject_terms:
                return "rejected"
            elif verdict in undecided_terms:
                return "undecided"
            return verdict
        
        return normalize(predicted) == normalize(expected)
