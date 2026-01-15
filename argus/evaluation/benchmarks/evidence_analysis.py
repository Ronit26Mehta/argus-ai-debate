"""
Evidence Analysis Benchmark.

Evaluates the quality and handling of evidence in ARGUS debates.
"""

from __future__ import annotations

import logging
from typing import Any

from argus.evaluation.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    SampleResult,
)
from argus.evaluation.scoring.metrics import compute_all_scores

logger = logging.getLogger(__name__)


class EvidenceAnalysisBenchmark(Benchmark):
    """Benchmark for evaluating evidence handling quality.
    
    Measures:
        - Source quality assessment
        - Citation accuracy
        - Evidence relevance scoring
        - Support/attack balance
    
    Example:
        >>> benchmark = EvidenceAnalysisBenchmark()
        >>> results = benchmark.run(dataset, llm)
    """
    
    def __init__(self):
        super().__init__(
            name="evidence_analysis",
            description="Evaluates evidence quality and citation handling",
        )
    
    def evaluate_sample(
        self,
        sample: dict[str, Any],
        llm: Any,
        config: BenchmarkConfig,
    ) -> SampleResult:
        """Evaluate evidence handling for a single proposition.
        
        Args:
            sample: Dataset sample
            llm: LLM instance
            config: Benchmark configuration
            
        Returns:
            SampleResult with evidence metrics
        """
        from argus.orchestrator import RDCOrchestrator
        
        proposition_text = sample.get("proposition", "")
        expected_verdict = sample.get("ground_truth", "undecided").lower()
        sample_id = str(sample.get("id", "unknown"))
        evidence_hints = sample.get("evidence_hints", "")
        
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
            
            # Analyze evidence quality
            evidence_analysis = self._analyze_evidence(result)
            
            # Compute base scores
            scores = compute_all_scores(result.to_dict())
            
            # Add evidence-specific scores
            scores.update({
                "evidence_coverage": evidence_analysis["coverage"],
                "source_diversity": evidence_analysis["source_diversity"],
                "polarity_balance": evidence_analysis["polarity_balance"],
                "avg_confidence": evidence_analysis["avg_confidence"],
                "avg_relevance": evidence_analysis["avg_relevance"],
            })
            
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
                    "evidence_analysis": evidence_analysis,
                    "num_evidence": result.num_evidence,
                    "num_rebuttals": result.num_rebuttals,
                },
            )
            
        except Exception as e:
            logger.error(f"Evidence analysis failed for sample {sample_id}: {e}")
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
    
    def _analyze_evidence(self, result: Any) -> dict[str, Any]:
        """Analyze evidence quality from debate result.
        
        Args:
            result: Debate result object
            
        Returns:
            Dictionary with evidence metrics
        """
        analysis = {
            "coverage": 0.0,
            "source_diversity": 0.0,
            "polarity_balance": 0.0,
            "avg_confidence": 0.0,
            "avg_relevance": 0.0,
            "support_count": 0,
            "attack_count": 0,
        }
        
        try:
            if result.graph:
                graph = result.graph
                
                evidence_nodes = [n for n in graph.nodes.values()
                                 if hasattr(n, 'confidence')]
                
                if evidence_nodes:
                    # Confidence average
                    confidences = [getattr(n, 'confidence', 0.5) 
                                  for n in evidence_nodes]
                    analysis["avg_confidence"] = sum(confidences) / len(confidences)
                    
                    # Relevance average
                    relevances = [getattr(n, 'relevance', 0.5)
                                 for n in evidence_nodes]
                    analysis["avg_relevance"] = sum(relevances) / len(relevances)
                    
                    # Polarity balance
                    polarities = [getattr(n, 'polarity', 0) 
                                 for n in evidence_nodes]
                    support_count = sum(1 for p in polarities if p > 0)
                    attack_count = sum(1 for p in polarities if p < 0)
                    
                    analysis["support_count"] = support_count
                    analysis["attack_count"] = attack_count
                    
                    total = support_count + attack_count
                    if total > 0:
                        # Balance is 1.0 when equal, lower when unbalanced
                        min_count = min(support_count, attack_count)
                        analysis["polarity_balance"] = 2 * min_count / total
                    
                    # Coverage score (evidence count / expected)
                    expected_evidence = 6  # Baseline expectation
                    analysis["coverage"] = min(1.0, len(evidence_nodes) / expected_evidence)
                    
                    # Source diversity (unique specialists)
                    sources = set(
                        getattr(n, 'source', 'unknown') 
                        for n in evidence_nodes
                    )
                    analysis["source_diversity"] = min(1.0, len(sources) / 3)
                    
        except Exception as e:
            logger.warning(f"Evidence analysis error: {e}")
        
        return analysis
