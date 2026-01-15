"""
Base Benchmark Classes for ARGUS Evaluation.

Provides abstract base class and result containers for all benchmarks.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.
    
    Attributes:
        name: Benchmark name
        max_samples: Maximum samples to evaluate (-1 for all)
        timeout_seconds: Timeout per sample
        save_intermediate: Save results after each sample
        output_dir: Directory for results
        num_rounds: Number of debate rounds
        num_agents: Number of agents to use
    """
    name: str = "default"
    max_samples: int = -1
    timeout_seconds: float = 300.0
    save_intermediate: bool = True
    output_dir: Path = field(default_factory=lambda: Path("./evaluation_results"))
    num_rounds: int = 1
    num_agents: int = 3
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SampleResult:
    """Result for a single benchmark sample.
    
    Attributes:
        sample_id: Unique sample identifier
        proposition: The evaluated proposition
        expected: Expected verdict
        predicted: Predicted verdict
        posterior: Posterior probability
        correct: Whether prediction matched expected
        duration_seconds: Time taken
        scores: Dictionary of metric scores
        metadata: Additional information
    """
    sample_id: str
    proposition: str
    expected: str
    predicted: str
    posterior: float
    correct: bool
    duration_seconds: float
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated result of a benchmark run.
    
    Attributes:
        benchmark_name: Name of the benchmark
        dataset_name: Name of the dataset used
        status: Overall status
        num_samples: Total samples evaluated
        num_correct: Correct predictions
        accuracy: Overall accuracy
        avg_posterior: Average posterior probability
        avg_duration: Average time per sample
        aggregate_scores: Aggregated metric scores
        sample_results: Individual sample results
        started_at: Start timestamp
        completed_at: Completion timestamp
        config: Benchmark configuration used
        errors: List of errors encountered
    """
    benchmark_name: str
    dataset_name: str
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    num_samples: int = 0
    num_correct: int = 0
    accuracy: float = 0.0
    avg_posterior: float = 0.0
    avg_duration: float = 0.0
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    sample_results: list[SampleResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: Optional[BenchmarkConfig] = None
    errors: list[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Total benchmark duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "dataset_name": self.dataset_name,
            "status": self.status.value,
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "avg_posterior": self.avg_posterior,
            "avg_duration": self.avg_duration,
            "aggregate_scores": self.aggregate_scores,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
        }


class Benchmark(ABC):
    """Abstract base class for benchmarks.
    
    All benchmarks must implement the `evaluate_sample` method.
    The `run` method handles iteration, timing, and result aggregation.
    
    Example:
        >>> class MyBenchmark(Benchmark):
        ...     def evaluate_sample(self, sample, llm, config):
        ...         # Run debate and return result
        ...         return SampleResult(...)
        ...
        >>> benchmark = MyBenchmark("my_benchmark")
        >>> results = benchmark.run(dataset, llm)
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize benchmark.
        
        Args:
            name: Benchmark name
            description: Human-readable description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def evaluate_sample(
        self,
        sample: dict[str, Any],
        llm: Any,
        config: BenchmarkConfig,
    ) -> SampleResult:
        """Evaluate a single sample.
        
        Args:
            sample: Sample data from dataset
            llm: LLM instance for debate
            config: Benchmark configuration
            
        Returns:
            SampleResult with evaluation outcome
        """
        pass
    
    def run(
        self,
        dataset: "pd.DataFrame",
        llm: Any,
        config: Optional[BenchmarkConfig] = None,
    ) -> BenchmarkResult:
        """Run benchmark on dataset.
        
        Args:
            dataset: DataFrame with samples
            llm: LLM instance
            config: Optional configuration
            
        Returns:
            BenchmarkResult with aggregated metrics
        """
        config = config or BenchmarkConfig()
        
        result = BenchmarkResult(
            benchmark_name=self.name,
            dataset_name=getattr(dataset, "name", "unknown"),
            status=BenchmarkStatus.RUNNING,
            started_at=datetime.utcnow(),
            config=config,
        )
        
        # Limit samples if configured
        samples = dataset.to_dict("records")
        if config.max_samples > 0:
            samples = samples[:config.max_samples]
        
        logger.info(f"Running benchmark '{self.name}' on {len(samples)} samples")
        
        total_posterior = 0.0
        total_duration = 0.0
        
        for i, sample in enumerate(samples):
            try:
                start_time = time.time()
                sample_result = self.evaluate_sample(sample, llm, config)
                sample_result.duration_seconds = time.time() - start_time
                
                result.sample_results.append(sample_result)
                result.num_samples += 1
                
                if sample_result.correct:
                    result.num_correct += 1
                
                total_posterior += sample_result.posterior
                total_duration += sample_result.duration_seconds
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(samples)} samples")
                    
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.get('id', i)}: {e}")
                result.errors.append(f"Sample {sample.get('id', i)}: {str(e)}")
        
        # Compute aggregates
        if result.num_samples > 0:
            result.accuracy = result.num_correct / result.num_samples
            result.avg_posterior = total_posterior / result.num_samples
            result.avg_duration = total_duration / result.num_samples
        
        # Aggregate scores from samples
        result.aggregate_scores = self._aggregate_scores(result.sample_results)
        
        result.status = BenchmarkStatus.COMPLETED
        result.completed_at = datetime.utcnow()
        
        logger.info(
            f"Benchmark '{self.name}' completed: "
            f"accuracy={result.accuracy:.3f}, "
            f"samples={result.num_samples}"
        )
        
        return result
    
    def _aggregate_scores(
        self,
        sample_results: list[SampleResult],
    ) -> dict[str, float]:
        """Aggregate scores across samples.
        
        Args:
            sample_results: List of sample results
            
        Returns:
            Dictionary of aggregated scores
        """
        if not sample_results:
            return {}
        
        # Collect all score keys
        score_keys = set()
        for sr in sample_results:
            score_keys.update(sr.scores.keys())
        
        # Compute means
        aggregated = {}
        for key in score_keys:
            values = [sr.scores.get(key, 0.0) for sr in sample_results]
            aggregated[key] = sum(values) / len(values)
        
        return aggregated
