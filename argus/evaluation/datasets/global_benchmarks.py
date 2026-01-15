"""
Global Benchmark Dataset Loader using Hugging Face Datasets Library.

Provides utilities to load and use standard NLP/AI benchmark datasets
from Hugging Face Hub for evaluating the ARGUS debate system.

Supported Benchmarks:
- FEVER (Fact Extraction and VERification)
- SNLI (Stanford Natural Language Inference)
- MultiNLI (Multi-Genre NLI)
- TruthfulQA (Truthfulness in QA)
- HellaSwag (Commonsense Reasoning)
- BoolQ (Boolean Questions)
- ARC (AI2 Reasoning Challenge)
- WinoGrande (Winograd Schema)

Requirements:
    pip install datasets

Usage:
    from argus.evaluation.datasets import load_global_benchmark
    df = load_global_benchmark("fever", max_samples=1000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Callable
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Definitions with Hugging Face Dataset IDs
# =============================================================================

@dataclass
class HFBenchmarkInfo:
    """Information about a Hugging Face benchmark dataset."""
    name: str
    hf_dataset_id: str
    hf_config: Optional[str]  # Configuration/subset name
    hf_split: str  # Default split to use
    description: str
    citation: str
    task_type: str
    text_field: str  # Field containing main text
    label_field: str  # Field containing label
    context_field: Optional[str]  # Optional context field
    label_mapping: dict[str, str]  # Maps HF labels to ARGUS labels
    source: str = ""
    license: str = ""


# Hugging Face dataset configurations
HF_BENCHMARKS = {
    "fever": HFBenchmarkInfo(
        name="FEVER",
        hf_dataset_id="fever",
        hf_config="v1.0",
        hf_split="paper_test",
        description="Fact Extraction and VERification - classify claims as SUPPORTED, REFUTED, or NOT ENOUGH INFO",
        citation="Thorne et al., 2018",
        task_type="classification",
        text_field="claim",
        label_field="label",
        context_field="evidence",
        label_mapping={
            "SUPPORTS": "supported",
            "REFUTES": "rejected",
            "NOT ENOUGH INFO": "undecided",
        },
        source="Wikipedia",
        license="CC BY-SA 3.0",
    ),
    "snli": HFBenchmarkInfo(
        name="SNLI",
        hf_dataset_id="snli",
        hf_config=None,
        hf_split="test",
        description="Stanford Natural Language Inference - classify premise-hypothesis pairs",
        citation="Bowman et al., 2015",
        task_type="nli",
        text_field="hypothesis",
        label_field="label",
        context_field="premise",
        label_mapping={
            "0": "supported",    # entailment
            "1": "undecided",    # neutral
            "2": "rejected",     # contradiction
        },
        source="Image captions",
        license="CC BY-SA 4.0",
    ),
    "multinli": HFBenchmarkInfo(
        name="MultiNLI",
        hf_dataset_id="multi_nli",
        hf_config=None,
        hf_split="validation_matched",
        description="Multi-Genre Natural Language Inference - 10 genres",
        citation="Williams et al., 2018",
        task_type="nli",
        text_field="hypothesis",
        label_field="label",
        context_field="premise",
        label_mapping={
            "0": "supported",
            "1": "undecided",
            "2": "rejected",
        },
        source="Multiple genres",
        license="OANC",
    ),
    "truthfulqa": HFBenchmarkInfo(
        name="TruthfulQA",
        hf_dataset_id="truthful_qa",
        hf_config="generation",
        hf_split="validation",
        description="Benchmark for measuring truthfulness in language model responses",
        citation="Lin et al., 2022",
        task_type="qa",
        text_field="question",
        label_field="best_answer",
        context_field="correct_answers",
        label_mapping={},  # Custom handling
        source="Human-written questions",
        license="Apache 2.0",
    ),
    "boolq": HFBenchmarkInfo(
        name="BoolQ",
        hf_dataset_id="google/boolq",
        hf_config=None,
        hf_split="validation",
        description="Boolean yes/no questions from natural queries",
        citation="Clark et al., 2019",
        task_type="qa",
        text_field="question",
        label_field="answer",
        context_field="passage",
        label_mapping={
            "True": "supported",
            "False": "rejected",
        },
        source="Google queries",
        license="CC BY-SA 3.0",
    ),
    "arc": HFBenchmarkInfo(
        name="ARC",
        hf_dataset_id="allenai/ai2_arc",
        hf_config="ARC-Challenge",
        hf_split="test",
        description="AI2 Reasoning Challenge - grade-school science questions",
        citation="Clark et al., 2018",
        task_type="qa",
        text_field="question",
        label_field="answerKey",
        context_field="choices",
        label_mapping={},  # Multiple choice
        source="Science exams",
        license="CC BY-SA 4.0",
    ),
    "hellaswag": HFBenchmarkInfo(
        name="HellaSwag",
        hf_dataset_id="Rowan/hellaswag",
        hf_config=None,
        hf_split="validation",
        description="Commonsense reasoning about everyday events",
        citation="Zellers et al., 2019",
        task_type="qa",
        text_field="ctx",
        label_field="label",
        context_field="endings",
        label_mapping={},  # Multiple choice
        source="WikiHow + ActivityNet",
        license="MIT",
    ),
    "winogrande": HFBenchmarkInfo(
        name="WinoGrande",
        hf_dataset_id="allenai/winogrande",
        hf_config="winogrande_xl",
        hf_split="validation",
        description="Commonsense reasoning with Winograd-style problems",
        citation="Sakaguchi et al., 2020",
        task_type="classification",
        text_field="sentence",
        label_field="answer",
        context_field=None,
        label_mapping={
            "1": "option1",
            "2": "option2",
        },
        source="Crowdsourced",
        license="CC BY 4.0",
    ),
}


# =============================================================================
# Hugging Face Dataset Loader
# =============================================================================

class HuggingFaceDatasetLoader:
    """
    Loader for global benchmark datasets using Hugging Face datasets library.
    
    Provides direct access to popular NLP benchmarks for evaluation.
    
    Example:
        >>> loader = HuggingFaceDatasetLoader()
        >>> df = loader.load("fever", max_samples=1000)
        >>> print(df.head())
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "datasets"
        self._datasets_available = self._check_datasets_available()
    
    def _check_datasets_available(self) -> bool:
        """Check if huggingface datasets library is available."""
        try:
            import datasets
            return True
        except ImportError:
            logger.warning(
                "Hugging Face datasets library not available. "
                "Install with: pip install datasets"
            )
            return False
    
    def list_benchmarks(self) -> list[str]:
        """List available benchmarks."""
        return list(HF_BENCHMARKS.keys())
    
    def get_benchmark_info(self, name: str) -> Optional[HFBenchmarkInfo]:
        """Get information about a benchmark.
        
        Args:
            name: Benchmark name
            
        Returns:
            HFBenchmarkInfo or None if not found
        """
        return HF_BENCHMARKS.get(name.lower())
    
    def load(
        self,
        name: str,
        max_samples: int = 1000,
        split: Optional[str] = None,
        streaming: bool = False,
    ) -> "pd.DataFrame":
        """
        Load a benchmark dataset from Hugging Face.
        
        Args:
            name: Benchmark name (fever, snli, multinli, etc.)
            max_samples: Maximum samples to load (-1 for all)
            split: Override default split
            streaming: Use streaming mode for large datasets
            
        Returns:
            DataFrame in ARGUS evaluation format
            
        Raises:
            ValueError: If benchmark not found
            ImportError: If datasets library not installed
        """
        if not self._datasets_available:
            raise ImportError(
                "Hugging Face datasets library required. "
                "Install with: pip install datasets"
            )
        
        from datasets import load_dataset
        import pandas as pd
        
        benchmark = HF_BENCHMARKS.get(name.lower())
        if not benchmark:
            available = ", ".join(self.list_benchmarks())
            raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
        
        logger.info(f"Loading {benchmark.name} from Hugging Face Hub...")
        
        # Load from Hugging Face
        try:
            split_name = split or benchmark.hf_split
            
            if benchmark.hf_config:
                hf_dataset = load_dataset(
                    benchmark.hf_dataset_id,
                    benchmark.hf_config,
                    split=split_name,
                    streaming=streaming,
                    trust_remote_code=True,
                )
            else:
                hf_dataset = load_dataset(
                    benchmark.hf_dataset_id,
                    split=split_name,
                    streaming=streaming,
                    trust_remote_code=True,
                )
            
            # Convert to list if streaming
            if streaming:
                samples = []
                for i, sample in enumerate(hf_dataset):
                    if max_samples > 0 and i >= max_samples:
                        break
                    samples.append(sample)
                hf_dataset = samples
            else:
                # Limit samples
                if max_samples > 0 and len(hf_dataset) > max_samples:
                    hf_dataset = hf_dataset.select(range(max_samples))
            
        except Exception as e:
            logger.error(f"Failed to load {name} from Hugging Face: {e}")
            raise
        
        # Convert to ARGUS format
        df = self._convert_to_argus_format(hf_dataset, benchmark)
        
        logger.info(f"Loaded {len(df)} samples from {benchmark.name}")
        return df
    
    def _convert_to_argus_format(
        self,
        hf_dataset: Any,
        benchmark: HFBenchmarkInfo,
    ) -> "pd.DataFrame":
        """Convert Hugging Face dataset to ARGUS evaluation format.
        
        Args:
            hf_dataset: Hugging Face dataset
            benchmark: Benchmark info
            
        Returns:
            DataFrame in ARGUS format
        """
        import pandas as pd
        
        rows = []
        
        # Handle different dataset formats
        if hasattr(hf_dataset, '__iter__'):
            items = list(hf_dataset)
        else:
            items = hf_dataset
        
        for i, sample in enumerate(items):
            # Convert sample to dict if needed
            if hasattr(sample, 'keys'):
                sample_dict = dict(sample)
            else:
                sample_dict = sample
            
            # Extract fields
            text = self._get_field(sample_dict, benchmark.text_field, f"Sample {i}")
            label = self._get_field(sample_dict, benchmark.label_field, "undecided")
            context = ""
            if benchmark.context_field:
                ctx = self._get_field(sample_dict, benchmark.context_field, "")
                if isinstance(ctx, list):
                    context = " | ".join(str(c) for c in ctx[:3])
                else:
                    context = str(ctx)[:500]
            
            # Map label to ARGUS format
            if benchmark.label_mapping:
                ground_truth = benchmark.label_mapping.get(str(label), "undecided")
            else:
                # Default mapping for boolean/binary
                if isinstance(label, bool):
                    ground_truth = "supported" if label else "rejected"
                elif str(label).lower() in ("true", "yes", "1"):
                    ground_truth = "supported"
                elif str(label).lower() in ("false", "no", "0"):
                    ground_truth = "rejected"
                else:
                    ground_truth = "undecided"
            
            # Determine difficulty based on text length
            text_len = len(text) if text else 0
            if text_len < 50:
                difficulty = "easy"
            elif text_len < 150:
                difficulty = "medium"
            elif text_len < 300:
                difficulty = "hard"
            else:
                difficulty = "expert"
            
            rows.append({
                "id": f"{benchmark.name.lower()}_{i+1:06d}",
                "proposition": text[:1000],  # Limit length
                "domain": benchmark.task_type,
                "ground_truth": ground_truth,
                "confidence": 0.8,
                "difficulty": difficulty,
                "evidence_hints": context[:500] if context else "",
                "expected_rounds": 2,
                "source_benchmark": benchmark.name,
                "hf_dataset": benchmark.hf_dataset_id,
            })
        
        return pd.DataFrame(rows)
    
    def _get_field(
        self,
        sample: dict,
        field_name: str,
        default: Any = "",
    ) -> Any:
        """Safely get field from sample."""
        if not field_name:
            return default
        
        # Handle nested fields
        if "." in field_name:
            parts = field_name.split(".")
            value = sample
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, default)
                else:
                    return default
            return value
        
        return sample.get(field_name, default)
    
    def to_csv(
        self,
        name: str,
        output_path: Path,
        max_samples: int = 1000,
    ) -> Path:
        """Load benchmark and save as CSV.
        
        Args:
            name: Benchmark name
            output_path: Output file path
            max_samples: Maximum samples
            
        Returns:
            Path to saved file
        """
        df = self.load(name, max_samples)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} samples to {output_path}")
        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def list_global_benchmarks() -> list[str]:
    """List all available global benchmarks.
    
    Returns:
        List of benchmark names
    """
    return list(HF_BENCHMARKS.keys())


def load_global_benchmark(
    name: str,
    max_samples: int = 1000,
    split: Optional[str] = None,
) -> "pd.DataFrame":
    """Load a global benchmark from Hugging Face as DataFrame.
    
    This function loads benchmark datasets directly from Hugging Face Hub
    and converts them to ARGUS evaluation format.
    
    Args:
        name: Benchmark name. Available options:
            - "fever": Fact verification (185K samples)
            - "snli": Natural language inference (570K samples)
            - "multinli": Multi-genre NLI (433K samples)
            - "truthfulqa": Truthfulness evaluation (817 samples)
            - "boolq": Boolean questions (16K samples)
            - "arc": Science questions (8K samples)
            - "hellaswag": Commonsense reasoning (70K samples)
            - "winogrande": Winograd schema (44K samples)
        max_samples: Maximum samples to load (-1 for all)
        split: Override default data split
        
    Returns:
        DataFrame with columns:
            - id: Unique identifier
            - proposition: Text to evaluate
            - domain: Task type
            - ground_truth: Expected verdict (supported/rejected/undecided)
            - confidence: Confidence level
            - difficulty: Estimated difficulty
            - evidence_hints: Context/evidence
            - source_benchmark: Source benchmark name
            - hf_dataset: Hugging Face dataset ID
            
    Raises:
        ImportError: If datasets library not installed
        ValueError: If benchmark name not found
        
    Example:
        >>> from argus.evaluation.datasets import load_global_benchmark
        >>> 
        >>> # Load FEVER benchmark
        >>> fever_df = load_global_benchmark("fever", max_samples=1000)
        >>> print(fever_df.head())
        >>> 
        >>> # Load SNLI for NLI evaluation
        >>> snli_df = load_global_benchmark("snli", max_samples=500)
        >>> 
        >>> # Load TruthfulQA for truthfulness testing
        >>> tqa_df = load_global_benchmark("truthfulqa")
    """
    loader = HuggingFaceDatasetLoader()
    return loader.load(name, max_samples, split)


def get_benchmark_info(name: str) -> Optional[HFBenchmarkInfo]:
    """Get detailed information about a benchmark.
    
    Args:
        name: Benchmark name
        
    Returns:
        HFBenchmarkInfo with dataset details, or None if not found
    """
    return HF_BENCHMARKS.get(name.lower())


def download_all_benchmarks(
    output_dir: Path,
    max_samples_per_benchmark: int = 1000,
) -> dict[str, Path]:
    """Download all benchmarks as CSV files.
    
    Args:
        output_dir: Directory to save CSV files
        max_samples_per_benchmark: Max samples per benchmark
        
    Returns:
        Dictionary mapping benchmark names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = HuggingFaceDatasetLoader()
    paths = {}
    
    for name in list_global_benchmarks():
        try:
            output_path = output_dir / f"{name}_benchmark.csv"
            loader.to_csv(name, output_path, max_samples_per_benchmark)
            paths[name] = output_path
            logger.info(f"Downloaded {name} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
    
    return paths


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Keep these for backward compatibility with existing code
GlobalBenchmarkLoader = HuggingFaceDatasetLoader
BenchmarkInfo = HFBenchmarkInfo
GLOBAL_BENCHMARKS = HF_BENCHMARKS
BenchmarkSample = None  # Deprecated, use DataFrame rows instead
