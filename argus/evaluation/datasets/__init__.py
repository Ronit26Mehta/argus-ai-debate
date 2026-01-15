"""
Dataset Management for ARGUS Evaluation.

Provides utilities for loading, validating, and managing test datasets,
including both custom ARGUS datasets and global benchmark datasets from Hugging Face.

Custom Datasets:
    - 10 domain-specific datasets with 1050+ samples each
    
Global Benchmarks (via Hugging Face):
    - FEVER, SNLI, MultiNLI, TruthfulQA, BoolQ, ARC, HellaSwag, WinoGrande
"""

from argus.evaluation.datasets.loader import (
    load_dataset,
    list_datasets,
    validate_dataset,
    get_dataset_path,
    DatasetInfo,
)

from argus.evaluation.datasets.global_benchmarks import (
    # Main loader class
    HuggingFaceDatasetLoader,
    HFBenchmarkInfo,
    HF_BENCHMARKS,
    # Convenience functions
    list_global_benchmarks,
    load_global_benchmark,
    get_benchmark_info,
    download_all_benchmarks,
    # Legacy aliases
    GlobalBenchmarkLoader,
    BenchmarkInfo,
    GLOBAL_BENCHMARKS,
)

__all__ = [
    # Custom ARGUS datasets
    "load_dataset",
    "list_datasets",
    "validate_dataset",
    "get_dataset_path",
    "DatasetInfo",
    # Hugging Face global benchmarks
    "HuggingFaceDatasetLoader",
    "HFBenchmarkInfo",
    "HF_BENCHMARKS",
    "list_global_benchmarks",
    "load_global_benchmark",
    "get_benchmark_info",
    "download_all_benchmarks",
    # Legacy compatibility
    "GlobalBenchmarkLoader",
    "BenchmarkInfo",
    "GLOBAL_BENCHMARKS",
]
