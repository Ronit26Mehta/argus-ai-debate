"""
ARGUS Evaluation Framework.

Comprehensive benchmarking, scoring, and testing infrastructure for the
ARGUS AI Debate system.

Components:
    - benchmarks: Benchmark definitions for debate quality assessment
    - datasets: Test datasets (10 domains, 1000+ points each)
    - scoring: Novel metrics unique to ARGUS (ARCIS, EVID-Q, DIALEC, etc.)
    - runner: Test execution and result aggregation

Example:
    >>> from argus.evaluation import BenchmarkRunner, load_dataset
    >>> runner = BenchmarkRunner()
    >>> dataset = load_dataset("factual_claims")
    >>> results = runner.run(dataset, benchmarks=["debate_quality"])
    >>> print(results.score_card())
"""

from argus.evaluation.scoring.metrics import (
    compute_arcis,
    compute_evid_q,
    compute_dialec,
    compute_rebut_f,
    compute_conv_s,
    compute_prov_i,
    compute_calib_m,
    compute_eig_u,
    compute_all_scores,
    ScoreCard,
)

from argus.evaluation.benchmarks.base import (
    Benchmark,
    BenchmarkResult,
    BenchmarkConfig,
)

from argus.evaluation.runner.benchmark_runner import BenchmarkRunner
from argus.evaluation.datasets.loader import load_dataset, list_datasets

__all__ = [
    # Scoring
    "compute_arcis",
    "compute_evid_q",
    "compute_dialec",
    "compute_rebut_f",
    "compute_conv_s",
    "compute_prov_i",
    "compute_calib_m",
    "compute_eig_u",
    "compute_all_scores",
    "ScoreCard",
    # Benchmarks
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    # Runner
    "BenchmarkRunner",
    # Datasets
    "load_dataset",
    "list_datasets",
]
