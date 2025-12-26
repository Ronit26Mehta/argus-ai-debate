"""
Expected Information Gain (EIG) Estimation.

Implements Monte Carlo EIG estimation for experiment planning.
EIG measures how much we expect to learn from an action.

The EIG formula (Eq. 4.6, 4.24):
    EIG(a) = H(θ) - E_y[H(θ|y, a)]
           = E_y[D_KL(P(θ|y,a) || P(θ))]
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from argus.cdag.graph import CDAG

logger = logging.getLogger(__name__)


@dataclass
class ActionCandidate:
    """
    A candidate action/experiment for EIG evaluation.
    
    Attributes:
        id: Action identifier
        name: Human-readable name
        cost: Cost of the action
        description: Action description
        expected_outcomes: Possible outcomes with probabilities
        target_props: Propositions this action would affect
        metadata: Additional action data
    """
    id: str
    name: str
    cost: float = 1.0
    description: str = ""
    expected_outcomes: list[dict[str, Any]] = field(default_factory=list)
    target_props: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Computed fields
    eig: float = 0.0
    utility: float = 0.0
    rank: int = 0


def entropy(p: float) -> float:
    """
    Compute binary entropy.
    
    H(p) = -p log(p) - (1-p) log(1-p)
    
    Args:
        p: Probability
        
    Returns:
        Binary entropy in nats
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def kl_divergence(p: float, q: float) -> float:
    """
    Compute KL divergence for binary distributions.
    
    D_KL(P || Q) = p log(p/q) + (1-p) log((1-p)/(1-q))
    
    Args:
        p: True probability
        q: Approximate probability
        
    Returns:
        KL divergence
    """
    eps = 1e-10
    p = max(eps, min(1 - eps, p))
    q = max(eps, min(1 - eps, q))
    
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


class EIGEstimator:
    """
    Monte Carlo EIG estimator.
    
    Estimates expected information gain through sampling possible
    outcomes and computing expected posterior entropies.
    
    Example:
        >>> estimator = EIGEstimator(num_samples=100)
        >>> eig = estimator.estimate(
        ...     prior=0.5,
        ...     outcome_model=lambda theta: np.random.binomial(1, theta),
        ...     update_model=lambda prior, outcome: 0.8 if outcome else 0.2,
        ... )
        >>> print(f"EIG: {eig:.4f}")
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_theta_samples: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize EIG estimator.
        
        Args:
            num_samples: Number of outcome samples
            num_theta_samples: Number of theta samples for inner expectation
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_theta_samples = num_theta_samples
        self.rng = np.random.default_rng(seed)
    
    def estimate(
        self,
        prior: float,
        outcome_model: Callable[[float], Any],
        update_model: Callable[[float, Any], float],
    ) -> float:
        """
        Estimate EIG given outcome and update models.
        
        Args:
            prior: Prior probability
            outcome_model: Function mapping theta to observed outcome
            update_model: Function mapping (prior, outcome) to posterior
            
        Returns:
            Estimated EIG
        """
        # Current entropy
        prior_entropy = entropy(prior)
        
        # Sample outcomes and compute expected posterior entropy
        posterior_entropies = []
        
        for _ in range(self.num_samples):
            # Sample theta from prior (Beta approximation)
            theta = self.rng.beta(prior * 10 + 1, (1 - prior) * 10 + 1)
            
            # Generate outcome
            outcome = outcome_model(theta)
            
            # Update to posterior
            posterior = update_model(prior, outcome)
            
            # Compute posterior entropy
            post_entropy = entropy(posterior)
            posterior_entropies.append(post_entropy)
        
        # Expected posterior entropy
        expected_post_entropy = np.mean(posterior_entropies)
        
        # EIG = prior entropy - expected posterior entropy
        eig = prior_entropy - expected_post_entropy
        
        return max(0.0, eig)
    
    def estimate_discrete(
        self,
        prior: float,
        outcomes: list[Any],
        outcome_probs: list[float],
        update_probs: list[float],
    ) -> float:
        """
        Estimate EIG for discrete outcomes.
        
        Args:
            prior: Prior probability
            outcomes: List of possible outcomes
            outcome_probs: Probability of each outcome
            update_probs: Posterior given each outcome
            
        Returns:
            EIG estimate
        """
        if len(outcomes) != len(outcome_probs) or len(outcomes) != len(update_probs):
            raise ValueError("Lists must have same length")
        
        # Normalize outcome probabilities
        total_prob = sum(outcome_probs)
        outcome_probs = [p / total_prob for p in outcome_probs]
        
        # Prior entropy
        prior_entropy = entropy(prior)
        
        # Expected posterior entropy
        expected_post_entropy = 0.0
        for prob, posterior in zip(outcome_probs, update_probs):
            expected_post_entropy += prob * entropy(posterior)
        
        eig = prior_entropy - expected_post_entropy
        return max(0.0, eig)
    
    def estimate_for_action(
        self,
        graph: "CDAG",
        action: ActionCandidate,
        prior_getter: Callable[[str], float],
        update_simulator: Callable[[str, dict], float],
    ) -> float:
        """
        Estimate EIG for a specific action on a C-DAG.
        
        Args:
            graph: The C-DAG graph
            action: Action candidate
            prior_getter: Function to get prior for a proposition
            update_simulator: Function to simulate update given action outcome
            
        Returns:
            Estimated EIG
        """
        total_eig = 0.0
        
        for prop_id in action.target_props:
            prior = prior_getter(prop_id)
            
            # Use expected outcomes if specified
            if action.expected_outcomes:
                outcomes = []
                outcome_probs = []
                update_probs = []
                
                for outcome_spec in action.expected_outcomes:
                    outcomes.append(outcome_spec.get("outcome"))
                    outcome_probs.append(outcome_spec.get("probability", 1.0))
                    update_probs.append(
                        update_simulator(prop_id, outcome_spec)
                    )
                
                eig = self.estimate_discrete(
                    prior, outcomes, outcome_probs, update_probs
                )
            else:
                # Default: binary outcome model
                def outcome_model(theta: float) -> int:
                    return int(self.rng.random() < theta)
                
                def update_model(p: float, outcome: int) -> float:
                    # Simple Bayesian update
                    if outcome:
                        return (p * 0.9) / (p * 0.9 + (1 - p) * 0.1)
                    else:
                        return (p * 0.1) / (p * 0.1 + (1 - p) * 0.9)
                
                eig = self.estimate(prior, outcome_model, update_model)
            
            total_eig += eig
        
        return total_eig


def estimate_eig(
    prior: float,
    sensitivity: float = 0.8,
    specificity: float = 0.8,
) -> float:
    """
    Quick EIG estimate for a binary test.
    
    Computes expected information gain for a test with
    known sensitivity and specificity.
    
    Args:
        prior: Prior probability
        sensitivity: True positive rate
        specificity: True negative rate
        
    Returns:
        Estimated EIG
    """
    # Probability of positive result
    p_pos = prior * sensitivity + (1 - prior) * (1 - specificity)
    p_neg = 1 - p_pos
    
    # Posteriors given results (Bayes)
    if p_pos > 0:
        post_pos = (prior * sensitivity) / p_pos
    else:
        post_pos = prior
    
    if p_neg > 0:
        post_neg = (prior * (1 - sensitivity)) / p_neg
    else:
        post_neg = prior
    
    # Prior entropy
    h_prior = entropy(prior)
    
    # Expected posterior entropy
    h_expected = p_pos * entropy(post_pos) + p_neg * entropy(post_neg)
    
    # EIG
    eig = h_prior - h_expected
    return max(0.0, eig)


def rank_actions_by_eig(
    actions: list[ActionCandidate],
    priors: dict[str, float],
    sensitivity: float = 0.8,
    specificity: float = 0.8,
    cost_weight: float = 0.1,
) -> list[ActionCandidate]:
    """
    Rank actions by EIG-adjusted utility.
    
    Utility = EIG - cost_weight × cost
    
    Args:
        actions: List of action candidates
        priors: Dict mapping proposition IDs to priors
        sensitivity: Test sensitivity
        specificity: Test specificity
        cost_weight: Weight for cost penalty
        
    Returns:
        Actions sorted by utility (highest first)
    """
    for action in actions:
        # Sum EIG over target propositions
        total_eig = 0.0
        for prop_id in action.target_props:
            prior = priors.get(prop_id, 0.5)
            eig = estimate_eig(prior, sensitivity, specificity)
            total_eig += eig
        
        action.eig = total_eig
        action.utility = total_eig - cost_weight * action.cost
    
    # Sort by utility
    sorted_actions = sorted(actions, key=lambda a: a.utility, reverse=True)
    
    # Assign ranks
    for i, action in enumerate(sorted_actions):
        action.rank = i + 1
    
    return sorted_actions
