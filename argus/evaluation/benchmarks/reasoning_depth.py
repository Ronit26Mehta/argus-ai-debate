"""
Reasoning Depth Benchmark.

Evaluates the sophistication and depth of reasoning in ARGUS debates.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from argus.evaluation.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    SampleResult,
)
from argus.evaluation.scoring.metrics import compute_all_scores

logger = logging.getLogger(__name__)


class ReasoningDepthBenchmark(Benchmark):
    """Benchmark for evaluating reasoning depth and sophistication.
    
    Measures:
        - Multi-hop reasoning detection
        - Logical chain length analysis
        - Assumption identification rate
        - Inference complexity
    
    Example:
        >>> benchmark = ReasoningDepthBenchmark()
        >>> results = benchmark.run(dataset, llm)
    """
    
    def __init__(self):
        super().__init__(
            name="reasoning_depth",
            description="Evaluates reasoning sophistication and logical depth",
        )
    
    def evaluate_sample(
        self,
        sample: dict[str, Any],
        llm: Any,
        config: BenchmarkConfig,
    ) -> SampleResult:
        """Evaluate reasoning depth for a single proposition.
        
        Args:
            sample: Dataset sample
            llm: LLM instance
            config: Benchmark configuration
            
        Returns:
            SampleResult with reasoning metrics
        """
        from argus.orchestrator import RDCOrchestrator
        
        proposition_text = sample.get("proposition", "")
        expected_verdict = sample.get("ground_truth", "undecided").lower()
        sample_id = str(sample.get("id", "unknown"))
        expected_depth = sample.get("expected_rounds", 3)
        
        try:
            orchestrator = RDCOrchestrator(
                llm=llm,
                max_rounds=config.num_rounds,
            )
            
            result = orchestrator.debate(
                proposition_text=proposition_text,
                prior=0.5,
            )
            
            predicted_verdict = result.verdict.label.lower()
            posterior = result.verdict.posterior
            
            # Analyze reasoning depth
            reasoning_analysis = self._analyze_reasoning(result)
            
            # Compute base scores
            scores = compute_all_scores(result.to_dict())
            
            # Add reasoning-specific scores
            scores.update({
                "reasoning_chain_length": reasoning_analysis["chain_length"],
                "multi_hop_count": reasoning_analysis["multi_hop_count"],
                "assumption_count": reasoning_analysis["assumptions_identified"],
                "inference_complexity": reasoning_analysis["complexity_score"],
            })
            
            # Correctness based on verdict match
            correct = predicted_verdict.lower() == expected_verdict.lower()
            
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
                    "reasoning_analysis": reasoning_analysis,
                    "expected_depth": expected_depth,
                    "actual_rounds": result.num_rounds,
                },
            )
            
        except Exception as e:
            logger.error(f"Reasoning analysis failed for sample {sample_id}: {e}")
            return SampleResult(
                sample_id=sample_id,
                proposition=proposition_text,
                expected=expected_verdict,
                predicted="error",
                posterior=0.5,
                correct=False,
                duration_seconds=0.0,
                metadata={"error": str(e)},
            )
    
    def _analyze_reasoning(self, result: Any) -> dict[str, Any]:
        """Analyze reasoning depth from debate result.
        
        Args:
            result: Debate result object
            
        Returns:
            Dictionary with reasoning metrics
        """
        analysis = {
            "chain_length": 0,
            "multi_hop_count": 0,
            "assumptions_identified": 0,
            "complexity_score": 0.0,
        }
        
        try:
            # Analyze graph structure for reasoning chains
            if result.graph:
                graph = result.graph
                
                # Count evidence nodes
                evidence_nodes = [n for n in graph.nodes.values() 
                                 if hasattr(n, 'evidence_type')]
                
                # Count rebuttal depth (multi-hop)
                rebuttal_nodes = [n for n in graph.nodes.values()
                                 if hasattr(n, 'rebuttal_type')]
                
                analysis["chain_length"] = len(evidence_nodes) + len(rebuttal_nodes)
                analysis["multi_hop_count"] = len(rebuttal_nodes)
                
                # Estimate assumptions from evidence
                analysis["assumptions_identified"] = sum(
                    1 for n in evidence_nodes 
                    if hasattr(n, 'text') and 'assum' in n.text.lower()
                )
                
                # Complexity score based on structure
                if analysis["chain_length"] > 0:
                    analysis["complexity_score"] = min(1.0, (
                        0.3 * min(len(evidence_nodes), 10) / 10 +
                        0.4 * min(len(rebuttal_nodes), 5) / 5 +
                        0.3 * min(result.num_rounds, 5) / 5
                    ))
                    
        except Exception as e:
            logger.warning(f"Reasoning analysis error: {e}")
        
        return analysis
