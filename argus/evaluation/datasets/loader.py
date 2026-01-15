"""
Dataset Loader for ARGUS Evaluation.

Provides utilities for loading and validating test datasets.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Dataset directory relative to this file
DATASETS_DIR = Path(__file__).parent


@dataclass
class DatasetInfo:
    """Information about a dataset.
    
    Attributes:
        name: Dataset name
        path: Path to CSV file
        description: Dataset description
        domain: Primary domain
        num_samples: Number of samples
        difficulty_distribution: Count by difficulty level
    """
    name: str
    path: Path
    description: str
    domain: str
    num_samples: int = 0
    difficulty_distribution: dict[str, int] = None
    
    def __post_init__(self):
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {}


# Dataset registry with descriptions
DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "factual_claims": DatasetInfo(
        name="factual_claims",
        path=DATASETS_DIR / "factual_claims.csv",
        description="General knowledge factual verification",
        domain="general",
    ),
    "scientific_hypotheses": DatasetInfo(
        name="scientific_hypotheses",
        path=DATASETS_DIR / "scientific_hypotheses.csv",
        description="Research and scientific claims evaluation",
        domain="science",
    ),
    "financial_analysis": DatasetInfo(
        name="financial_analysis",
        path=DATASETS_DIR / "financial_analysis.csv",
        description="Stock market and financial claims",
        domain="finance",
    ),
    "medical_efficacy": DatasetInfo(
        name="medical_efficacy",
        path=DATASETS_DIR / "medical_efficacy.csv",
        description="Treatment and drug efficacy claims",
        domain="medical",
    ),
    "legal_reasoning": DatasetInfo(
        name="legal_reasoning",
        path=DATASETS_DIR / "legal_reasoning.csv",
        description="Legal case and precedent analysis",
        domain="legal",
    ),
    "technical_comparison": DatasetInfo(
        name="technical_comparison",
        path=DATASETS_DIR / "technical_comparison.csv",
        description="Technology product and system claims",
        domain="technology",
    ),
    "policy_impact": DatasetInfo(
        name="policy_impact",
        path=DATASETS_DIR / "policy_impact.csv",
        description="Socio-economic policy analysis",
        domain="policy",
    ),
    "historical_interpretation": DatasetInfo(
        name="historical_interpretation",
        path=DATASETS_DIR / "historical_interpretation.csv",
        description="Historical event analysis",
        domain="history",
    ),
    "environmental_risk": DatasetInfo(
        name="environmental_risk",
        path=DATASETS_DIR / "environmental_risk.csv",
        description="Climate and environmental claims",
        domain="environment",
    ),
    "adversarial_edge_cases": DatasetInfo(
        name="adversarial_edge_cases",
        path=DATASETS_DIR / "adversarial_edge_cases.csv",
        description="Stress testing and edge cases",
        domain="adversarial",
    ),
}


def list_datasets() -> list[str]:
    """List all available dataset names.
    
    Returns:
        List of dataset names
    """
    return list(DATASET_REGISTRY.keys())


def get_dataset_path(name: str) -> Path:
    """Get the file path for a dataset.
    
    Args:
        name: Dataset name
        
    Returns:
        Path to dataset CSV file
        
    Raises:
        ValueError: If dataset not found
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(list_datasets())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    return DATASET_REGISTRY[name].path


def load_dataset(
    name: str,
    max_samples: int = -1,
) -> "pd.DataFrame":
    """Load a dataset by name.
    
    Args:
        name: Dataset name
        max_samples: Maximum samples to load (-1 for all)
        
    Returns:
        DataFrame with dataset contents
        
    Raises:
        ValueError: If dataset not found
        FileNotFoundError: If CSV file missing
    """
    import pandas as pd
    
    path = get_dataset_path(name)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    df = pd.read_csv(path)
    df.name = name  # Attach name for reference
    
    if max_samples > 0:
        df = df.head(max_samples)
    
    logger.info(f"Loaded dataset '{name}' with {len(df)} samples")
    return df


def validate_dataset(name: str) -> tuple[bool, list[str]]:
    """Validate a dataset for completeness and correctness.
    
    Args:
        name: Dataset name
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    try:
        path = get_dataset_path(name)
    except ValueError as e:
        return False, [str(e)]
    
    if not path.exists():
        return False, [f"File not found: {path}"]
    
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as e:
        return False, [f"Failed to load CSV: {e}"]
    
    # Check minimum samples
    if len(df) < 1000:
        issues.append(f"Insufficient samples: {len(df)} < 1000 required")
    
    # Check required columns
    required_cols = {"id", "proposition", "ground_truth", "difficulty"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for empty propositions
    if "proposition" in df.columns:
        empty_props = df["proposition"].isna().sum()
        if empty_props > 0:
            issues.append(f"Empty propositions: {empty_props}")
    
    # Check ground truth values
    if "ground_truth" in df.columns:
        valid_verdicts = {"supported", "rejected", "undecided"}
        invalid_verdicts = set(df["ground_truth"].str.lower().unique()) - valid_verdicts
        if invalid_verdicts:
            issues.append(f"Invalid ground truth values: {invalid_verdicts}")
    
    # Check difficulty distribution
    if "difficulty" in df.columns:
        valid_difficulties = {"easy", "medium", "hard", "expert"}
        invalid_diffs = set(df["difficulty"].str.lower().unique()) - valid_difficulties
        if invalid_diffs:
            issues.append(f"Invalid difficulty values: {invalid_diffs}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_all() -> dict[str, tuple[bool, list[str]]]:
    """Validate all registered datasets.
    
    Returns:
        Dictionary mapping dataset names to (is_valid, issues) tuples
    """
    results = {}
    for name in list_datasets():
        results[name] = validate_dataset(name)
    return results
