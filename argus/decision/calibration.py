"""
Calibration Metrics for ARGUS.

Implements calibration assessment and temperature scaling:
    - Brier Score
    - Expected Calibration Error (ECE)
    - Temperature Scaling
    - Reliability Diagrams
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """
    Calibration metrics for a set of predictions.
    
    Attributes:
        brier_score: Brier score (lower is better)
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        nll: Negative Log Likelihood
        num_samples: Number of samples
        reliability_data: Data for reliability diagram
    """
    brier_score: float = 0.0
    ece: float = 0.0
    mce: float = 0.0
    nll: float = 0.0
    num_samples: int = 0
    reliability_data: dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Brier: {self.brier_score:.4f}, "
            f"ECE: {self.ece:.4f}, "
            f"MCE: {self.mce:.4f}, "
            f"NLL: {self.nll:.4f}"
        )


def compute_brier_score(
    confidences: list[float],
    outcomes: list[int],
) -> float:
    """
    Compute Brier score.
    
    Brier score = (1/N) Σ (p_i - o_i)²
    
    Lower is better. Perfect = 0, worst = 1.
    
    Args:
        confidences: Predicted probabilities [0, 1]
        outcomes: Actual binary outcomes (0 or 1)
        
    Returns:
        Brier score
    """
    if len(confidences) != len(outcomes):
        raise ValueError("Lengths must match")
    
    if len(confidences) == 0:
        return 0.0
    
    brier = 0.0
    for p, o in zip(confidences, outcomes):
        brier += (p - o) ** 2
    
    return brier / len(confidences)


def compute_ece(
    confidences: list[float],
    outcomes: list[int],
    num_bins: int = 10,
) -> tuple[float, dict[str, Any]]:
    """
    Compute Expected Calibration Error.
    
    ECE = Σ (|B_m|/N) × |acc(B_m) - conf(B_m)|
    
    For each bin, measures gap between average confidence and accuracy.
    
    Args:
        confidences: Predicted probabilities
        outcomes: Actual binary outcomes
        num_bins: Number of bins
        
    Returns:
        (ECE value, reliability data)
    """
    if len(confidences) != len(outcomes):
        raise ValueError("Lengths must match")
    
    n = len(confidences)
    if n == 0:
        return 0.0, {}
    
    confidences = np.array(confidences)
    outcomes = np.array(outcomes)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])
    
    ece = 0.0
    mce = 0.0
    reliability_data = {
        "bins": [],
        "accuracies": [],
        "confidences": [],
        "counts": [],
    }
    
    for i in range(num_bins):
        mask = bin_indices == i
        bin_size = mask.sum()
        
        if bin_size > 0:
            bin_accuracy = outcomes[mask].mean()
            bin_confidence = confidences[mask].mean()
            
            gap = abs(bin_accuracy - bin_confidence)
            ece += (bin_size / n) * gap
            mce = max(mce, gap)
            
            reliability_data["bins"].append(i)
            reliability_data["accuracies"].append(float(bin_accuracy))
            reliability_data["confidences"].append(float(bin_confidence))
            reliability_data["counts"].append(int(bin_size))
    
    reliability_data["ece"] = ece
    reliability_data["mce"] = mce
    
    return ece, reliability_data


def compute_nll(
    confidences: list[float],
    outcomes: list[int],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Negative Log Likelihood.
    
    NLL = -(1/N) Σ [o_i log(p_i) + (1-o_i) log(1-p_i)]
    
    Args:
        confidences: Predicted probabilities
        outcomes: Actual binary outcomes
        epsilon: Small value to avoid log(0)
        
    Returns:
        NLL value
    """
    if len(confidences) != len(outcomes):
        raise ValueError("Lengths must match")
    
    if len(confidences) == 0:
        return 0.0
    
    nll = 0.0
    for p, o in zip(confidences, outcomes):
        p = max(epsilon, min(1 - epsilon, p))
        if o == 1:
            nll -= math.log(p)
        else:
            nll -= math.log(1 - p)
    
    return nll / len(confidences)


def temperature_scaling(
    logits: list[float],
    labels: list[int],
    initial_temp: float = 1.0,
) -> tuple[float, list[float]]:
    """
    Find optimal temperature for calibration.
    
    Temperature scaling: p = sigmoid(logit / T)
    Find T that minimizes NLL.
    
    Args:
        logits: Pre-softmax logits
        labels: True binary labels
        initial_temp: Initial temperature guess
        
    Returns:
        (optimal_temperature, calibrated_probabilities)
    """
    logits_arr = np.array(logits)
    labels_arr = np.array(labels)
    
    def nll_objective(temp: float) -> float:
        """NLL as function of temperature."""
        temp = max(0.01, temp)  # Avoid division by zero
        scaled_logits = logits_arr / temp
        probs = 1 / (1 + np.exp(-scaled_logits))
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        nll = -np.mean(
            labels_arr * np.log(probs) + (1 - labels_arr) * np.log(1 - probs)
        )
        return nll
    
    # Optimize temperature
    result = optimize.minimize_scalar(
        nll_objective,
        bounds=(0.1, 10.0),
        method="bounded",
    )
    
    optimal_temp = result.x
    
    # Compute calibrated probabilities
    calibrated_logits = logits_arr / optimal_temp
    calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
    
    return optimal_temp, calibrated_probs.tolist()


def apply_temperature(
    confidence: float,
    temperature: float,
) -> float:
    """
    Apply temperature scaling to a single confidence.
    
    Args:
        confidence: Original confidence [0, 1]
        temperature: Temperature value (>1 = softer, <1 = sharper)
        
    Returns:
        Calibrated confidence
    """
    # Convert to logit
    epsilon = 1e-10
    p = max(epsilon, min(1 - epsilon, confidence))
    logit = math.log(p / (1 - p))
    
    # Scale
    scaled_logit = logit / temperature
    
    # Back to probability
    return 1 / (1 + math.exp(-scaled_logit))


class ReliabilityDiagram:
    """
    Reliability diagram for visualizing calibration.
    
    Compares predicted confidence to actual accuracy
    across confidence bins.
    """
    
    def __init__(
        self,
        num_bins: int = 10,
    ):
        """
        Initialize reliability diagram.
        
        Args:
            num_bins: Number of bins
        """
        self.num_bins = num_bins
        self.confidences: list[float] = []
        self.outcomes: list[int] = []
    
    def add_sample(
        self,
        confidence: float,
        outcome: int,
    ) -> None:
        """
        Add a prediction-outcome pair.
        
        Args:
            confidence: Predicted probability
            outcome: Actual outcome (0 or 1)
        """
        self.confidences.append(confidence)
        self.outcomes.append(outcome)
    
    def add_batch(
        self,
        confidences: list[float],
        outcomes: list[int],
    ) -> None:
        """
        Add multiple prediction-outcome pairs.
        
        Args:
            confidences: List of predicted probabilities
            outcomes: List of actual outcomes
        """
        self.confidences.extend(confidences)
        self.outcomes.extend(outcomes)
    
    def compute(self) -> CalibrationMetrics:
        """
        Compute calibration metrics.
        
        Returns:
            CalibrationMetrics with all values
        """
        if len(self.confidences) == 0:
            return CalibrationMetrics()
        
        brier = compute_brier_score(self.confidences, self.outcomes)
        ece, reliability_data = compute_ece(
            self.confidences, self.outcomes, self.num_bins
        )
        nll = compute_nll(self.confidences, self.outcomes)
        
        return CalibrationMetrics(
            brier_score=brier,
            ece=ece,
            mce=reliability_data.get("mce", 0.0),
            nll=nll,
            num_samples=len(self.confidences),
            reliability_data=reliability_data,
        )
    
    def get_diagram_data(self) -> dict[str, Any]:
        """
        Get data for plotting reliability diagram.
        
        Returns:
            Dict with bin_centers, accuracies, counts
        """
        _, reliability_data = compute_ece(
            self.confidences, self.outcomes, self.num_bins
        )
        
        bin_centers = np.linspace(
            0.5 / self.num_bins,
            1 - 0.5 / self.num_bins,
            self.num_bins,
        ).tolist()
        
        return {
            "bin_centers": bin_centers,
            "accuracies": reliability_data.get("accuracies", []),
            "confidences": reliability_data.get("confidences", []),
            "counts": reliability_data.get("counts", []),
            "perfect_calibration": bin_centers,
        }
    
    def reset(self) -> None:
        """Clear all samples."""
        self.confidences.clear()
        self.outcomes.clear()


def calibrate_model_outputs(
    train_confidences: list[float],
    train_outcomes: list[int],
    test_confidences: list[float],
) -> tuple[float, list[float], CalibrationMetrics]:
    """
    Calibrate model outputs using temperature scaling.
    
    Uses training data to find optimal temperature,
    then applies to test confidences.
    
    Args:
        train_confidences: Training predictions
        train_outcomes: Training outcomes
        test_confidences: Test predictions to calibrate
        
    Returns:
        (temperature, calibrated_test_confidences, metrics)
    """
    # Convert to logits
    epsilon = 1e-10
    train_logits = [
        math.log(max(epsilon, min(1 - epsilon, p)) / 
                 max(epsilon, 1 - max(epsilon, min(1 - epsilon, p))))
        for p in train_confidences
    ]
    
    # Find optimal temperature
    optimal_temp, _ = temperature_scaling(train_logits, train_outcomes)
    
    # Calibrate test confidences
    calibrated = [
        apply_temperature(conf, optimal_temp)
        for conf in test_confidences
    ]
    
    # Compute metrics on training data
    diagram = ReliabilityDiagram()
    calibrated_train = [
        apply_temperature(conf, optimal_temp)
        for conf in train_confidences
    ]
    diagram.add_batch(calibrated_train, train_outcomes)
    metrics = diagram.compute()
    
    logger.info(
        f"Calibration complete: T={optimal_temp:.3f}, "
        f"ECE={metrics.ece:.4f}, Brier={metrics.brier_score:.4f}"
    )
    
    return optimal_temp, calibrated, metrics
