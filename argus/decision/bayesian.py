"""
Bayesian Updating for ARGUS.

Implements log-odds based posterior updates for proposition beliefs.
Evidence contributions are aggregated using signed likelihood ratios.

The update equation (Eq. 4.5):
    logit P(θ|x) = logit P(θ) + log(P(x|θ) / P(x|¬θ))
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from argus.cdag.graph import CDAG
    from argus.cdag.nodes import Proposition

logger = logging.getLogger(__name__)


def log_odds(p: float) -> float:
    """
    Convert probability to log-odds.
    
    logit(p) = log(p / (1-p))
    
    Args:
        p: Probability in (0, 1)
        
    Returns:
        Log-odds value
    """
    # Clip to avoid infinities
    p = max(1e-10, min(1 - 1e-10, p))
    return math.log(p / (1 - p))


def from_log_odds(lo: float) -> float:
    """
    Convert log-odds to probability.
    
    p = 1 / (1 + exp(-lo))
    
    Args:
        lo: Log-odds value
        
    Returns:
        Probability in (0, 1)
    """
    # Clip to avoid overflow
    lo = max(-500, min(500, lo))
    return 1.0 / (1.0 + math.exp(-lo))


def compute_likelihood_ratio(
    evidence_confidence: float,
    sensitivity: float = 0.8,
    specificity: float = 0.8,
) -> float:
    """
    Compute likelihood ratio for evidence.
    
    LR = P(evidence | hypothesis true) / P(evidence | hypothesis false)
    
    For binary evidence with known sensitivity/specificity:
        LR_positive = sensitivity / (1 - specificity)
        LR_negative = (1 - sensitivity) / specificity
    
    Args:
        evidence_confidence: Confidence that evidence is positive
        sensitivity: True positive rate
        specificity: True negative rate
        
    Returns:
        Log-likelihood ratio
    """
    # Compute positive and negative LRs
    lr_pos = sensitivity / max(0.01, 1 - specificity)
    lr_neg = (1 - sensitivity) / max(0.01, specificity)
    
    # Weighted combination based on confidence
    # High confidence -> more like positive LR
    # Low confidence -> more like negative LR
    
    log_lr_pos = math.log(max(0.01, lr_pos))
    log_lr_neg = math.log(max(0.01, lr_neg))
    
    # Interpolate based on confidence
    log_lr = evidence_confidence * log_lr_pos + (1 - evidence_confidence) * log_lr_neg
    
    return log_lr


@dataclass
class UpdateResult:
    """Result of a Bayesian update."""
    prior: float
    posterior: float
    prior_log_odds: float
    posterior_log_odds: float
    total_contribution: float
    num_evidence: int
    support_contribution: float
    attack_contribution: float


class BayesianUpdater:
    """
    Bayesian updater for proposition beliefs.
    
    Aggregates evidence contributions in log-odds space
    and computes posterior probabilities.
    
    Example:
        >>> updater = BayesianUpdater()
        >>> posterior = updater.update(
        ...     prior=0.5,
        ...     evidence_contributions=[0.3, 0.2, -0.1],
        ... )
        >>> print(f"Posterior: {posterior:.3f}")
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        regularization: float = 0.0,
    ):
        """
        Initialize Bayesian updater.
        
        Args:
            temperature: Temperature for calibration scaling
            regularization: Regularization toward prior (0=none, 1=full)
        """
        self.temperature = temperature
        self.regularization = regularization
    
    def update(
        self,
        prior: float,
        evidence_contributions: list[float],
        weights: Optional[list[float]] = None,
    ) -> float:
        """
        Update belief given evidence contributions.
        
        Args:
            prior: Prior probability
            evidence_contributions: List of LLR contributions (positive=support)
            weights: Optional weights for each contribution
            
        Returns:
            Posterior probability
        """
        # Convert prior to log-odds
        prior_lo = log_odds(prior)
        
        # Apply temperature scaling
        scaled_contributions = [
            c / self.temperature for c in evidence_contributions
        ]
        
        # Apply weights if provided
        if weights:
            if len(weights) != len(scaled_contributions):
                raise ValueError("Weights must match contributions length")
            scaled_contributions = [
                c * w for c, w in zip(scaled_contributions, weights)
            ]
        
        # Aggregate contributions
        total_contribution = sum(scaled_contributions)
        
        # Apply regularization toward prior
        if self.regularization > 0:
            total_contribution *= (1 - self.regularization)
        
        # Compute posterior
        posterior_lo = prior_lo + total_contribution
        posterior = from_log_odds(posterior_lo)
        
        return posterior
    
    def update_from_graph(
        self,
        graph: "CDAG",
        prop_id: str,
    ) -> UpdateResult:
        """
        Update proposition belief from C-DAG evidence.
        
        Args:
            graph: The C-DAG graph
            prop_id: Proposition ID
            
        Returns:
            UpdateResult with detailed breakdown
        """
        from argus.cdag.edges import EdgePolarity
        
        prop = graph.get_proposition(prop_id)
        if not prop:
            raise ValueError(f"Proposition {prop_id} not found")
        
        prior = prop.prior
        prior_lo = log_odds(prior)
        
        # Collect contributions from incoming edges
        incoming = graph.get_incoming_edges(prop_id)
        
        support_total = 0.0
        attack_total = 0.0
        
        for edge in incoming:
            source = graph.get_node(edge.source_id)
            if not source:
                continue
            
            # Get evidence confidence
            conf = getattr(source, "confidence", 0.5)
            
            # Compute LLR
            llr = compute_likelihood_ratio(conf)
            
            # Weight by edge weight
            weighted_llr = edge.effective_weight * llr
            
            if edge.polarity == EdgePolarity.POSITIVE:
                support_total += weighted_llr
            elif edge.polarity == EdgePolarity.NEGATIVE:
                attack_total += weighted_llr
        
        # Temperature scaling
        support_total /= self.temperature
        attack_total /= self.temperature
        
        # Net contribution (support - attack because attacks are negative)
        total_contribution = support_total - attack_total
        
        # Regularization
        if self.regularization > 0:
            total_contribution *= (1 - self.regularization)
        
        # Compute posterior
        posterior_lo = prior_lo + total_contribution
        posterior = from_log_odds(posterior_lo)
        
        # Update the proposition
        prop.update_posterior(posterior)
        
        return UpdateResult(
            prior=prior,
            posterior=posterior,
            prior_log_odds=prior_lo,
            posterior_log_odds=posterior_lo,
            total_contribution=total_contribution,
            num_evidence=len(incoming),
            support_contribution=support_total,
            attack_contribution=attack_total,
        )


def update_posterior(
    prior: float,
    support_scores: list[tuple[float, float]],  # (confidence, weight)
    attack_scores: list[tuple[float, float]],   # (confidence, weight)
    temperature: float = 1.0,
) -> float:
    """
    Convenience function to update posterior given support/attack scores.
    
    Args:
        prior: Prior probability
        support_scores: List of (confidence, weight) for support evidence
        attack_scores: List of (confidence, weight) for attack evidence
        temperature: Calibration temperature
        
    Returns:
        Posterior probability
    """
    prior_lo = log_odds(prior)
    
    # Process support
    support_contribution = 0.0
    for conf, weight in support_scores:
        llr = compute_likelihood_ratio(conf)
        support_contribution += weight * llr
    
    # Process attack
    attack_contribution = 0.0
    for conf, weight in attack_scores:
        llr = compute_likelihood_ratio(conf)
        attack_contribution += weight * llr
    
    # Net contribution
    total = (support_contribution - attack_contribution) / temperature
    
    # Posterior
    posterior_lo = prior_lo + total
    posterior = from_log_odds(posterior_lo)
    
    return posterior


def batch_update(
    priors: list[float],
    contributions: list[list[float]],
    temperature: float = 1.0,
) -> list[float]:
    """
    Batch update multiple propositions.
    
    Args:
        priors: List of prior probabilities
        contributions: List of contribution lists per proposition
        temperature: Calibration temperature
        
    Returns:
        List of posterior probabilities
    """
    posteriors = []
    updater = BayesianUpdater(temperature=temperature)
    
    for prior, contribs in zip(priors, contributions):
        posterior = updater.update(prior, contribs)
        posteriors.append(posterior)
    
    return posteriors


def sensitivity_analysis(
    prior: float,
    base_contributions: list[float],
    target_index: int,
    delta_range: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 20,
) -> list[tuple[float, float]]:
    """
    Analyze posterior sensitivity to a single evidence contribution.
    
    Args:
        prior: Prior probability
        base_contributions: Base contribution values
        target_index: Index of contribution to vary
        delta_range: Range of deltas to apply
        num_points: Number of points to evaluate
        
    Returns:
        List of (contribution_value, posterior) pairs
    """
    updater = BayesianUpdater()
    results = []
    
    deltas = np.linspace(delta_range[0], delta_range[1], num_points)
    
    for delta in deltas:
        modified = base_contributions.copy()
        modified[target_index] += delta
        
        posterior = updater.update(prior, modified)
        results.append((modified[target_index], posterior))
    
    return results
