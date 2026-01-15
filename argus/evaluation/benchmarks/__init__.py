"""
Benchmark Framework for ARGUS Evaluation.

Provides base classes and utilities for defining and running benchmarks.
"""

from argus.evaluation.benchmarks.base import (
    Benchmark,
    BenchmarkResult,
    BenchmarkConfig,
)

from argus.evaluation.benchmarks.debate_quality import DebateQualityBenchmark
from argus.evaluation.benchmarks.reasoning_depth import ReasoningDepthBenchmark
from argus.evaluation.benchmarks.evidence_analysis import EvidenceAnalysisBenchmark

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    "DebateQualityBenchmark",
    "ReasoningDepthBenchmark",
    "EvidenceAnalysisBenchmark",
]
