"""
Standard/Global Scoring Metrics for ARGUS Evaluation.

Industry-standard metrics commonly used to evaluate AI reasoning, 
NLP systems, and debate/argumentation systems.

These complement the novel ARGUS-specific metrics (ARCIS, EVID-Q, etc.)
with widely-recognized evaluation measures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Standard Classification Metrics
# =============================================================================

def compute_accuracy(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted labels
        ground_truths: True labels
        
    Returns:
        Accuracy score (0-1)
    """
    if not predictions or not ground_truths:
        return 0.0
    
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p.lower() == g.lower())
    return correct / len(predictions)


def compute_precision_recall_f1(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    positive_label: str = "supported",
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        predictions: Predicted labels
        ground_truths: True labels
        positive_label: Label considered as positive class
        
    Returns:
        Dictionary with precision, recall, f1, support
    """
    if not predictions or not ground_truths:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    
    tp = sum(1 for p, g in zip(predictions, ground_truths) 
             if p.lower() == positive_label.lower() and g.lower() == positive_label.lower())
    fp = sum(1 for p, g in zip(predictions, ground_truths) 
             if p.lower() == positive_label.lower() and g.lower() != positive_label.lower())
    fn = sum(1 for p, g in zip(predictions, ground_truths) 
             if p.lower() != positive_label.lower() and g.lower() == positive_label.lower())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    support = sum(1 for g in ground_truths if g.lower() == positive_label.lower())
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def compute_macro_f1(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> float:
    """
    Compute macro-averaged F1 score across all classes.
    
    Args:
        predictions: Predicted labels
        ground_truths: True labels
        labels: Optional list of labels to consider
        
    Returns:
        Macro F1 score
    """
    if labels is None:
        labels = list(set(ground_truths))
    
    f1_scores = []
    for label in labels:
        metrics = compute_precision_recall_f1(predictions, ground_truths, label)
        f1_scores.append(metrics["f1"])
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# =============================================================================
# Calibration Metrics
# =============================================================================

def compute_brier_score(
    confidences: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    """
    Compute Brier Score for probability calibration.
    
    Lower is better. Range: 0 (perfect) to 1 (worst).
    
    Args:
        confidences: Predicted probabilities
        outcomes: Binary outcomes (0 or 1)
        
    Returns:
        Brier score
    """
    if not confidences or not outcomes:
        return 1.0
    
    if len(confidences) != len(outcomes):
        raise ValueError("Confidences and outcomes must have same length")
    
    squared_errors = [(c - o) ** 2 for c, o in zip(confidences, outcomes)]
    return sum(squared_errors) / len(squared_errors)


def compute_ece(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    num_bins: int = 10,
) -> tuple[float, dict[str, Any]]:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures how well confidence aligns with accuracy.
    Lower is better.
    
    Args:
        confidences: Predicted probabilities
        outcomes: Binary outcomes (0 or 1)
        num_bins: Number of calibration bins
        
    Returns:
        Tuple of (ECE value, bin details)
    """
    if not confidences or not outcomes:
        return 1.0, {}
    
    confidences = np.array(confidences)
    outcomes = np.array(outcomes)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_data = []
    
    for i in range(num_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_confidences = confidences[bin_mask]
        bin_outcomes = outcomes[bin_mask]
        
        if len(bin_confidences) > 0:
            bin_accuracy = bin_outcomes.mean()
            bin_confidence = bin_confidences.mean()
            bin_count = len(bin_confidences)
            bin_data.append({
                "bin": i,
                "accuracy": float(bin_accuracy),
                "confidence": float(bin_confidence),
                "count": int(bin_count),
                "error": abs(bin_accuracy - bin_confidence),
            })
    
    # Weighted average by bin count
    total_samples = len(confidences)
    ece = sum(b["error"] * b["count"] / total_samples for b in bin_data)
    
    return float(ece), {"bins": bin_data, "num_bins": num_bins}


def compute_mce(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    num_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    The maximum gap between confidence and accuracy across bins.
    
    Args:
        confidences: Predicted probabilities
        outcomes: Binary outcomes
        num_bins: Number of calibration bins
        
    Returns:
        MCE value
    """
    _, data = compute_ece(confidences, outcomes, num_bins)
    bins = data.get("bins", [])
    
    if not bins:
        return 1.0
    
    return max(b["error"] for b in bins)


# =============================================================================
# Information-Theoretic Metrics
# =============================================================================

def compute_cross_entropy(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    epsilon: float = 1e-15,
) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        confidences: Predicted probabilities
        outcomes: Binary outcomes
        epsilon: Small value to avoid log(0)
        
    Returns:
        Cross-entropy loss
    """
    if not confidences or not outcomes:
        return float('inf')
    
    ce = 0.0
    for c, o in zip(confidences, outcomes):
        c = max(epsilon, min(1 - epsilon, c))
        ce -= o * math.log(c) + (1 - o) * math.log(1 - c)
    
    return ce / len(confidences)


def compute_log_loss(
    confidences: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    """
    Compute log loss (same as cross-entropy for binary classification).
    
    Args:
        confidences: Predicted probabilities
        outcomes: Binary outcomes
        
    Returns:
        Log loss
    """
    return compute_cross_entropy(confidences, outcomes)


# =============================================================================
# Argumentation/Debate-Specific Standard Metrics
# =============================================================================

def compute_argument_coverage(
    evidence_count: int,
    expected_evidence: int = 5,
) -> float:
    """
    Compute argument coverage score.
    
    Measures how much of the expected evidence space was explored.
    
    Args:
        evidence_count: Actual evidence collected
        expected_evidence: Expected/target evidence count
        
    Returns:
        Coverage ratio (capped at 1.0)
    """
    if expected_evidence <= 0:
        return 0.0
    return min(1.0, evidence_count / expected_evidence)


def compute_dialectical_balance(
    support_count: int,
    attack_count: int,
) -> float:
    """
    Compute dialectical balance score.
    
    Measures balance between supporting and attacking arguments.
    1.0 = perfectly balanced, 0.0 = completely one-sided.
    
    Args:
        support_count: Number of supporting arguments
        attack_count: Number of attacking arguments
        
    Returns:
        Balance score (0-1)
    """
    total = support_count + attack_count
    if total == 0:
        return 0.5  # Neutral when no arguments
    
    min_count = min(support_count, attack_count)
    max_count = max(support_count, attack_count)
    
    return 2 * min_count / total


def compute_verdict_confidence_correlation(
    posteriors: Sequence[float],
    correctness: Sequence[bool],
) -> float:
    """
    Compute correlation between posterior confidence and correctness.
    
    Higher is better - confident predictions should be correct.
    
    Args:
        posteriors: Posterior probabilities
        correctness: Whether each prediction was correct
        
    Returns:
        Pearson correlation coefficient
    """
    if len(posteriors) < 2:
        return 0.0
    
    posteriors = np.array(posteriors)
    correctness = np.array(correctness, dtype=float)
    
    # Convert posteriors to "confidence" (distance from 0.5)
    confidences = np.abs(posteriors - 0.5) * 2
    
    # Pearson correlation
    if np.std(confidences) == 0 or np.std(correctness) == 0:
        return 0.0
    
    correlation = np.corrcoef(confidences, correctness)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


# =============================================================================
# Aggregate Standard Metrics
# =============================================================================

@dataclass
class StandardMetricsResult:
    """Results from standard metrics computation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    macro_f1: float = 0.0
    brier_score: float = 0.0
    ece: float = 0.0
    mce: float = 0.0
    log_loss: float = 0.0
    argument_coverage: float = 0.0
    dialectical_balance: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_f1": self.macro_f1,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "mce": self.mce,
            "log_loss": self.log_loss,
            "argument_coverage": self.argument_coverage,
            "dialectical_balance": self.dialectical_balance,
        }


def compute_all_standard_metrics(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    confidences: Optional[Sequence[float]] = None,
    outcomes: Optional[Sequence[int]] = None,
    evidence_count: int = 0,
    support_count: int = 0,
    attack_count: int = 0,
) -> StandardMetricsResult:
    """
    Compute all standard metrics in one call.
    
    Args:
        predictions: Predicted verdict labels
        ground_truths: True verdict labels
        confidences: Posterior probabilities (optional)
        outcomes: Binary correctness outcomes (optional)
        evidence_count: Total evidence collected
        support_count: Supporting evidence count
        attack_count: Attacking evidence count
        
    Returns:
        StandardMetricsResult with all metrics
    """
    result = StandardMetricsResult()
    
    # Classification metrics
    result.accuracy = compute_accuracy(predictions, ground_truths)
    
    prf = compute_precision_recall_f1(predictions, ground_truths, "supported")
    result.precision = prf["precision"]
    result.recall = prf["recall"]
    result.f1 = prf["f1"]
    
    result.macro_f1 = compute_macro_f1(
        predictions, ground_truths,
        labels=["supported", "rejected", "undecided"]
    )
    
    # Calibration metrics (if confidences provided)
    if confidences and outcomes:
        result.brier_score = compute_brier_score(confidences, outcomes)
        result.ece, _ = compute_ece(confidences, outcomes)
        result.mce = compute_mce(confidences, outcomes)
        result.log_loss = compute_log_loss(confidences, outcomes)
    
    # Argumentation metrics
    result.argument_coverage = compute_argument_coverage(evidence_count)
    result.dialectical_balance = compute_dialectical_balance(support_count, attack_count)
    
    return result


# =============================================================================
# Metric Descriptions for Documentation
# =============================================================================

STANDARD_METRIC_DESCRIPTIONS = {
    "accuracy": {
        "name": "Accuracy",
        "description": "Proportion of correct predictions",
        "range": (0.0, 1.0),
        "higher_is_better": True,
        "standard": "Machine Learning",
    },
    "precision": {
        "name": "Precision",
        "description": "True positives / (True positives + False positives)",
        "range": (0.0, 1.0),
        "higher_is_better": True,
        "standard": "Machine Learning",
    },
    "recall": {
        "name": "Recall",
        "description": "True positives / (True positives + False negatives)",
        "range": (0.0, 1.0),
        "higher_is_better": True,
        "standard": "Machine Learning",
    },
    "f1": {
        "name": "F1 Score",
        "description": "Harmonic mean of precision and recall",
        "range": (0.0, 1.0),
        "higher_is_better": True,
        "standard": "Machine Learning",
    },
    "macro_f1": {
        "name": "Macro F1",
        "description": "Average F1 across all classes",
        "range": (0.0, 1.0),
        "higher_is_better": True,
        "standard": "Machine Learning",
    },
    "brier_score": {
        "name": "Brier Score",
        "description": "Mean squared error of probability estimates",
        "range": (0.0, 1.0),
        "higher_is_better": False,
        "standard": "Probability Calibration",
    },
    "ece": {
        "name": "Expected Calibration Error",
        "description": "Weighted average of confidence-accuracy gap across bins",
        "range": (0.0, 1.0),
        "higher_is_better": False,
        "standard": "Probability Calibration",
    },
    "mce": {
        "name": "Maximum Calibration Error",
        "description": "Maximum confidence-accuracy gap across bins",
        "range": (0.0, 1.0),
        "higher_is_better": False,
        "standard": "Probability Calibration",
    },
    "log_loss": {
        "name": "Log Loss",
        "description": "Cross-entropy loss for probability predictions",
        "range": (0.0, float("inf")),
        "higher_is_better": False,
        "standard": "Information Theory",
    },
    "argument_coverage": {
        "name": "Argument Coverage",
        "description": "Ratio of collected evidence to expected evidence",
        "range": (0.0, 1.0),
        "higher_is_better": True,
        "standard": "Argumentation Theory",
    },
    "dialectical_balance": {
        "name": "Dialectical Balance",
        "description": "Balance between supporting and attacking arguments",
        "range": (0.0, 1.0),
        "higher_is_better": True,
        "standard": "Argumentation Theory",
    },
}
