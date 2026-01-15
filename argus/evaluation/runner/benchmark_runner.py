"""
Benchmark Runner for ARGUS Evaluation.

Executes benchmarks and manages result collection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from argus.evaluation.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStatus,
)
from argus.evaluation.datasets.loader import load_dataset, list_datasets

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for benchmark runner.
    
    Attributes:
        benchmarks: List of benchmark names to run
        datasets: List of dataset names to use
        output_dir: Directory for results
        max_samples_per_dataset: Limit samples per dataset
        num_rounds: Number of debate rounds
        num_agents: Number of agents to use
        save_results: Save results to disk
        dry_run: Run without actual LLM calls
    """
    benchmarks: list[str] = field(default_factory=lambda: ["debate_quality"])
    datasets: list[str] = field(default_factory=lambda: ["factual_claims"])
    output_dir: Path = field(default_factory=lambda: Path("./evaluation_results"))
    max_samples_per_dataset: int = 10
    num_rounds: int = 1
    num_agents: int = 3
    save_results: bool = True
    dry_run: bool = False
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


class BenchmarkRunner:
    """Runs benchmarks and collects results.
    
    Example:
        >>> from argus.evaluation import BenchmarkRunner
        >>> from argus.core.llm import GeminiLLM
        >>> 
        >>> runner = BenchmarkRunner()
        >>> llm = GeminiLLM(api_key=\"...\")
        >>> results = runner.run(llm)
        >>> print(results)
    """
    
    def __init__(self, config: Optional[RunConfig] = None):
        """Initialize runner.
        
        Args:
            config: Runner configuration
        """
        self.config = config or RunConfig()
        self.results: list[BenchmarkResult] = []
        
        # Initialize output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        llm: Any,
        config: Optional[RunConfig] = None,
    ) -> list[BenchmarkResult]:
        """Run all configured benchmarks.
        
        Args:
            llm: LLM instance for debates
            config: Override configuration
            
        Returns:
            List of BenchmarkResult objects
        """
        config = config or self.config
        self.results = []
        
        # Get benchmark instances
        benchmarks = self._get_benchmarks(config.benchmarks)
        
        logger.info(
            f"Starting benchmark run: {len(benchmarks)} benchmarks, "
            f"{len(config.datasets)} datasets"
        )
        
        for dataset_name in config.datasets:
            try:
                # Load dataset
                dataset = load_dataset(dataset_name, config.max_samples_per_dataset)
                
                for benchmark in benchmarks:
                    logger.info(f"Running {benchmark.name} on {dataset_name}")
                    
                    # Create benchmark config
                    bench_config = BenchmarkConfig(
                        name=f"{benchmark.name}_{dataset_name}",
                        max_samples=config.max_samples_per_dataset,
                        output_dir=config.output_dir,
                        num_rounds=config.num_rounds,
                        num_agents=config.num_agents,
                    )
                    
                    if config.dry_run:
                        result = self._dry_run(benchmark, dataset_name, bench_config)
                    else:
                        result = benchmark.run(dataset, llm, bench_config)
                    
                    self.results.append(result)
                    
                    # Save intermediate results
                    if config.save_results:
                        self._save_result(result)
                        
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
        
        # Save summary
        if config.save_results:
            self._save_summary()
        
        return self.results
    
    def _get_benchmarks(self, names: list[str]) -> list[Benchmark]:
        """Get benchmark instances by name.
        
        Args:
            names: List of benchmark names
            
        Returns:
            List of Benchmark instances
        """
        from argus.evaluation.benchmarks import (
            DebateQualityBenchmark,
            ReasoningDepthBenchmark,
            EvidenceAnalysisBenchmark,
        )
        
        benchmark_map = {
            "debate_quality": DebateQualityBenchmark,
            "reasoning_depth": ReasoningDepthBenchmark,
            "evidence_analysis": EvidenceAnalysisBenchmark,
        }
        
        benchmarks = []
        for name in names:
            if name in benchmark_map:
                benchmarks.append(benchmark_map[name]())
            else:
                logger.warning(f"Unknown benchmark: {name}")
        
        return benchmarks
    
    def _dry_run(
        self,
        benchmark: Benchmark,
        dataset_name: str,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """Perform dry run without actual LLM calls.
        
        Args:
            benchmark: Benchmark instance
            dataset_name: Name of dataset
            config: Benchmark config
            
        Returns:
            Simulated BenchmarkResult
        """
        import random
        
        return BenchmarkResult(
            benchmark_name=benchmark.name,
            dataset_name=dataset_name,
            status=BenchmarkStatus.COMPLETED,
            num_samples=config.max_samples,
            num_correct=int(config.max_samples * 0.7),
            accuracy=0.7,
            avg_posterior=0.65,
            avg_duration=1.5,
            aggregate_scores={
                "ARCIS": random.uniform(0.6, 0.8),
                "EVID-Q": random.uniform(0.5, 0.7),
                "DIALEC": random.uniform(0.4, 0.6),
                "REBUT-F": random.uniform(0.5, 0.7),
                "CONV-S": random.uniform(0.6, 0.8),
                "PROV-I": random.uniform(0.5, 0.7),
                "CALIB-M": random.uniform(0.5, 0.7),
                "EIG-U": random.uniform(0.4, 0.6),
            },
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            config=config,
        )
    
    def _save_result(self, result: BenchmarkResult):
        """Save individual result to disk.
        
        Args:
            result: BenchmarkResult to save
        """
        filename = f"{result.benchmark_name}_{result.dataset_name}.json"
        path = self.config.output_dir / filename
        
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved result to {path}")
    
    def _save_summary(self):
        """Save summary of all results."""
        summary = {
            "run_time": datetime.utcnow().isoformat(),
            "num_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }
        
        path = self.config.output_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved summary to {path}")


def main():
    """CLI entry point for benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARGUS Benchmark Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run without LLM calls")
    parser.add_argument("--datasets", nargs="+", default=["factual_claims"])
    parser.add_argument("--benchmarks", nargs="+", default=["debate_quality"])
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./evaluation_results")
    
    args = parser.parse_args()
    
    config = RunConfig(
        benchmarks=args.benchmarks,
        datasets=args.datasets,
        max_samples_per_dataset=args.max_samples,
        num_rounds=args.num_rounds,
        num_agents=args.num_agents,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )
    
    runner = BenchmarkRunner(config)
    
    if args.dry_run:
        print("Running in dry-run mode...")
        results = runner.run(llm=None)
    else:
        from argus.core.llm import GeminiLLM
        import os
        
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("Error: GEMINI_API_KEY not set")
            return
        
        llm = GeminiLLM(api_key=api_key)
        results = runner.run(llm)
    
    print(f"\nCompleted {len(results)} benchmark runs")
    for r in results:
        print(f"  {r.benchmark_name} on {r.dataset_name}: accuracy={r.accuracy:.3f}")


if __name__ == "__main__":
    main()
