"""
Results Aggregator for ARGUS Evaluation.

Collects and analyzes results from multiple benchmark runs.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from argus.evaluation.benchmarks.base import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs.
    
    Attributes:
        metric_name: Name of the metric
        mean: Mean value
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        count: Number of samples
    """
    metric_name: str
    mean: float
    std: float
    min: float
    max: float
    count: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }


class ResultsAggregator:
    """Aggregates results from multiple benchmark runs.
    
    Example:
        >>> aggregator = ResultsAggregator()
        >>> aggregator.add_results(results)
        >>> summary = aggregator.summarize()
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.results: list[BenchmarkResult] = []
    
    def add_result(self, result: BenchmarkResult):
        """Add a single result.
        
        Args:
            result: BenchmarkResult to add
        """
        self.results.append(result)
    
    def add_results(self, results: list[BenchmarkResult]):
        """Add multiple results.
        
        Args:
            results: List of BenchmarkResult objects
        """
        self.results.extend(results)
    
    def load_from_directory(self, directory: Path):
        """Load results from saved JSON files.
        
        Args:
            directory: Directory containing result JSON files
        """
        directory = Path(directory)
        for json_file in directory.glob("*.json"):
            if json_file.name == "summary.json":
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    # Convert to BenchmarkResult (simplified)
                    result = BenchmarkResult(
                        benchmark_name=data.get("benchmark_name", "unknown"),
                        dataset_name=data.get("dataset_name", "unknown"),
                        accuracy=data.get("accuracy", 0.0),
                        aggregate_scores=data.get("aggregate_scores", {}),
                    )
                    self.results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
    
    def summarize(self) -> dict[str, Any]:
        """Generate summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Group by benchmark
        by_benchmark: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            if r.benchmark_name not in by_benchmark:
                by_benchmark[r.benchmark_name] = []
            by_benchmark[r.benchmark_name].append(r)
        
        summary = {
            "total_runs": len(self.results),
            "benchmarks": {},
            "overall_accuracy": 0.0,
            "metrics": {},
        }
        
        all_accuracies = []
        all_metrics: dict[str, list[float]] = {}
        
        for bench_name, bench_results in by_benchmark.items():
            accuracies = [r.accuracy for r in bench_results]
            all_accuracies.extend(accuracies)
            
            summary["benchmarks"][bench_name] = {
                "runs": len(bench_results),
                "mean_accuracy": statistics.mean(accuracies),
                "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            }
            
            # Collect metrics
            for r in bench_results:
                for metric, value in r.aggregate_scores.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Overall accuracy
        summary["overall_accuracy"] = statistics.mean(all_accuracies) if all_accuracies else 0.0
        
        # Aggregate metrics
        for metric, values in all_metrics.items():
            summary["metrics"][metric] = AggregatedMetrics(
                metric_name=metric,
                mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0,
                min=min(values),
                max=max(values),
                count=len(values),
            ).to_dict()
        
        return summary
    
    def compare_runs(
        self,
        baseline_name: str,
        comparison_name: str,
    ) -> dict[str, Any]:
        """Compare two benchmark runs.
        
        Args:
            baseline_name: Name of baseline benchmark
            comparison_name: Name of comparison benchmark
            
        Returns:
            Comparison statistics
        """
        baseline = [r for r in self.results if r.benchmark_name == baseline_name]
        comparison = [r for r in self.results if r.benchmark_name == comparison_name]
        
        if not baseline or not comparison:
            return {"error": "Benchmarks not found"}
        
        baseline_acc = statistics.mean([r.accuracy for r in baseline])
        comparison_acc = statistics.mean([r.accuracy for r in comparison])
        
        return {
            "baseline": baseline_name,
            "comparison": comparison_name,
            "baseline_accuracy": baseline_acc,
            "comparison_accuracy": comparison_acc,
            "accuracy_change": comparison_acc - baseline_acc,
            "improvement": comparison_acc > baseline_acc,
        }


def aggregate_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Convenience function to aggregate results.
    
    Args:
        results: List of BenchmarkResult objects
        
    Returns:
        Summary statistics dictionary
    """
    aggregator = ResultsAggregator()
    aggregator.add_results(results)
    return aggregator.summarize()
