"""
Signed Influence Propagation for C-DAG.

Implements the signed message passing algorithm for computing
proposition scores based on evidence support/attack weights.

The propagation formula:
    s^{t+1}(p) = σ(α Σ_{supports} w_e s^t(e) - β Σ_{attacks} w_e s^t(e))

Where:
    - σ is a squashing function (tanh or sigmoid)
    - α, β are positive scaling coefficients
    - w_e is the effective edge weight
    - s^t(e) is the evidence score at iteration t
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    from argus.cdag.graph import CDAG
    from argus.cdag.nodes import Proposition

logger = logging.getLogger(__name__)


@dataclass
class PropagationConfig:
    """
    Configuration for influence propagation.
    
    Attributes:
        alpha: Scaling factor for support edges
        beta: Scaling factor for attack edges
        max_iterations: Maximum propagation iterations
        convergence_threshold: Convergence threshold (max score change)
        squashing_function: Function to squash scores ('tanh', 'sigmoid', 'clip')
        temperature: Temperature for calibration
        use_log_odds: Whether to use log-odds space
    """
    alpha: float = 1.0
    beta: float = 1.0
    max_iterations: int = 10
    convergence_threshold: float = 0.001
    squashing_function: str = "tanh"
    temperature: float = 1.0
    use_log_odds: bool = True


def sigmoid(x: float) -> float:
    """
    Sigmoid squashing function.
    
    Args:
        x: Input value
        
    Returns:
        Value in (0, 1)
    """
    # Clip to avoid overflow
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def squash(
    x: float,
    method: str = "tanh",
    temperature: float = 1.0,
) -> float:
    """
    Apply squashing function to a value.
    
    Args:
        x: Input value
        method: Squashing method ('tanh', 'sigmoid', 'clip')
        temperature: Temperature scaling
        
    Returns:
        Squashed value
    """
    scaled = x / temperature
    
    if method == "tanh":
        return (math.tanh(scaled) + 1) / 2  # Map to [0, 1]
    elif method == "sigmoid":
        return sigmoid(scaled)
    elif method == "clip":
        return max(0.0, min(1.0, (scaled + 1) / 2))
    else:
        return sigmoid(scaled)


def log_odds(p: float) -> float:
    """
    Convert probability to log-odds.
    
    Args:
        p: Probability in (0, 1)
        
    Returns:
        Log-odds value
    """
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def from_log_odds(lo: float) -> float:
    """
    Convert log-odds to probability.
    
    Args:
        lo: Log-odds value
        
    Returns:
        Probability in (0, 1)
    """
    return sigmoid(lo)


def compute_log_likelihood_ratio(
    confidence: float,
    temperature: float = 1.0,
) -> float:
    """
    Compute log-likelihood ratio from confidence.
    
    From Eq. (4.17):
        ℓ_e = σ(1/T × logit(p̂_e))
        s_e = log(ℓ_e / (1 - ℓ_e))
    
    Args:
        confidence: Calibrated confidence (0-1)
        temperature: Calibration temperature
        
    Returns:
        Log-likelihood ratio
    """
    # Convert to logit
    lo = log_odds(confidence)
    
    # Temperature scale
    scaled_lo = lo / temperature
    
    # Convert back through sigmoid
    l_e = sigmoid(scaled_lo)
    
    # Compute LLR
    llr = log_odds(l_e)
    
    return llr


def propagate_influence(
    graph: "CDAG",
    config: Optional[PropagationConfig] = None,
) -> dict[str, float]:
    """
    Propagate influence through the C-DAG.
    
    Implements signed message passing from evidence to propositions.
    Updates node scores based on weighted support/attack edges.
    
    Args:
        graph: The C-DAG graph
        config: Propagation configuration
        
    Returns:
        Dictionary mapping node IDs to final scores
    """
    from argus.cdag.edges import EdgePolarity
    
    config = config or PropagationConfig()
    
    # Get all node IDs
    node_ids = list(graph._nodes.keys())
    
    # Initialize scores with confidence values
    scores: dict[str, float] = {}
    for node_id in node_ids:
        node = graph._nodes[node_id]
        scores[node_id] = getattr(node, "confidence", 0.5)
    
    logger.debug(f"Starting propagation with {len(node_ids)} nodes")
    
    # Iterate until convergence
    for iteration in range(config.max_iterations):
        new_scores: dict[str, float] = {}
        max_change = 0.0
        
        for node_id in node_ids:
            node = graph._nodes[node_id]
            
            # Get incoming edges
            incoming = graph.get_incoming_edges(node_id)
            
            if not incoming:
                # No incoming edges: use base confidence
                new_scores[node_id] = scores[node_id]
                continue
            
            # Compute support and attack contributions
            support_sum = 0.0
            attack_sum = 0.0
            
            for edge in incoming:
                source_score = scores.get(edge.source_id, 0.5)
                w_eff = edge.effective_weight
                
                if edge.polarity == EdgePolarity.POSITIVE:
                    # Support contribution
                    if config.use_log_odds:
                        llr = compute_log_likelihood_ratio(
                            source_score,
                            config.temperature,
                        )
                        support_sum += w_eff * llr
                    else:
                        support_sum += w_eff * source_score
                        
                elif edge.polarity == EdgePolarity.NEGATIVE:
                    # Attack contribution
                    if config.use_log_odds:
                        llr = compute_log_likelihood_ratio(
                            source_score,
                            config.temperature,
                        )
                        attack_sum += w_eff * llr
                    else:
                        attack_sum += w_eff * source_score
            
            # Compute net influence
            net = config.alpha * support_sum - config.beta * attack_sum
            
            if config.use_log_odds:
                # In log-odds space: add to base log-odds
                base_lo = log_odds(getattr(node, "prior", 0.5))
                new_lo = base_lo + net
                new_score = from_log_odds(new_lo)
            else:
                # Simple squashing
                new_score = squash(
                    net,
                    method=config.squashing_function,
                    temperature=config.temperature,
                )
            
            new_scores[node_id] = new_score
            
            # Track change
            change = abs(new_score - scores[node_id])
            max_change = max(max_change, change)
        
        # Update scores
        scores = new_scores
        
        logger.debug(f"Iteration {iteration + 1}: max_change={max_change:.6f}")
        
        # Check convergence
        if max_change < config.convergence_threshold:
            logger.debug(f"Converged after {iteration + 1} iterations")
            break
    
    # Update node scores in graph
    for node_id, score in scores.items():
        node = graph._nodes[node_id]
        node._score = score
    
    return scores


def compute_posterior(
    graph: "CDAG",
    prop_id: str,
    config: Optional[PropagationConfig] = None,
) -> float:
    """
    Compute posterior probability for a proposition.
    
    Uses Bayesian updating with log-odds aggregation:
        logit P(θ|E) = logit P(θ) + Σ_e ω_e z_e
    
    Where:
        - ω_e is the effective edge weight
        - z_e is the signed log-likelihood ratio
    
    Args:
        graph: The C-DAG graph
        prop_id: Proposition ID
        config: Propagation configuration
        
    Returns:
        Posterior probability
    """
    from argus.cdag.nodes import Proposition
    from argus.cdag.edges import EdgePolarity
    
    config = config or PropagationConfig()
    
    prop = graph.get_proposition(prop_id)
    if not prop:
        raise ValueError(f"Proposition {prop_id} not found")
    
    # Start with prior in log-odds
    prior_lo = log_odds(prop.prior)
    
    # Get incoming evidence edges
    incoming = graph.get_incoming_edges(prop_id)
    
    # Aggregate signed LLRs
    total_contribution = 0.0
    
    for edge in incoming:
        source = graph.get_node(edge.source_id)
        if not source:
            continue
        
        # Get source confidence
        source_conf = getattr(source, "confidence", 0.5)
        
        # Compute LLR
        llr = compute_log_likelihood_ratio(source_conf, config.temperature)
        
        # Apply sign based on polarity
        if edge.polarity == EdgePolarity.POSITIVE:
            sign = 1.0
        elif edge.polarity == EdgePolarity.NEGATIVE:
            sign = -1.0
        else:
            sign = 0.0
        
        # Weight by effective edge weight
        z_e = sign * llr
        contribution = edge.effective_weight * z_e
        total_contribution += contribution
    
    # Compute posterior in log-odds
    posterior_lo = prior_lo + total_contribution
    
    # Convert to probability
    posterior = from_log_odds(posterior_lo)
    
    # Update proposition
    prop.update_posterior(posterior)
    
    logger.debug(
        f"Proposition {prop_id}: prior={prop.prior:.3f} -> "
        f"posterior={posterior:.3f} (contribution={total_contribution:.3f})"
    )
    
    return posterior


def compute_all_posteriors(
    graph: "CDAG",
    config: Optional[PropagationConfig] = None,
) -> dict[str, float]:
    """
    Compute posteriors for all propositions.
    
    Args:
        graph: The C-DAG graph
        config: Propagation configuration
        
    Returns:
        Dictionary mapping proposition IDs to posteriors
    """
    posteriors: dict[str, float] = {}
    
    for prop in graph.get_all_propositions():
        posterior = compute_posterior(graph, prop.id, config)
        posteriors[prop.id] = posterior
    
    return posteriors


def compute_disagreement_index(
    graph: "CDAG",
    prop_id: str,
) -> float:
    """
    Compute disagreement index for a proposition.
    
    Measures the balance between support and attack.
    High disagreement = strong evidence on both sides.
    
    Args:
        graph: The C-DAG graph
        prop_id: Proposition ID
        
    Returns:
        Disagreement index (0 = no disagreement, 1 = max)
    """
    support = graph.compute_support_score(prop_id)
    attack = graph.compute_attack_score(prop_id)
    
    total = support + attack
    if total == 0:
        return 0.0
    
    # Use product of normalized scores as disagreement
    # Max when support == attack
    disagreement = 4 * (support / total) * (attack / total)
    
    return disagreement


def get_counter_evidence(
    graph: "CDAG",
    prop_id: str,
    threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Get evidence that could flip the verdict.
    
    Identifies minimal evidence changes that would change the posterior.
    
    Args:
        graph: The C-DAG graph
        prop_id: Proposition ID
        threshold: Posterior threshold for flipping
        
    Returns:
        List of counter-evidence specifications
    """
    from argus.cdag.edges import EdgePolarity
    
    prop = graph.get_proposition(prop_id)
    if not prop:
        return []
    
    current_posterior = prop.posterior
    
    # Determine current verdict
    is_supported = current_posterior > 0.5
    
    # Get current evidence
    incoming = graph.get_incoming_edges(prop_id)
    
    counter_specs = []
    
    # Find evidence that, if removed/weakened, would flip
    for edge in incoming:
        source = graph.get_node(edge.source_id)
        if not source:
            continue
        
        # Calculate contribution
        source_conf = getattr(source, "confidence", 0.5)
        llr = compute_log_likelihood_ratio(source_conf)
        contribution = edge.effective_weight * llr
        
        if edge.polarity == EdgePolarity.POSITIVE:
            contribution = contribution
        else:
            contribution = -contribution
        
        # Check if removing this would flip
        prior_lo = log_odds(prop.prior)
        total_lo = log_odds(current_posterior)
        remaining_lo = total_lo - contribution
        remaining_prob = from_log_odds(remaining_lo)
        
        would_flip = (remaining_prob > 0.5) != is_supported
        
        if would_flip or abs(contribution) > threshold:
            counter_specs.append({
                "edge_id": edge.id,
                "node_id": source.id,
                "node_text": getattr(source, "text", "")[:100],
                "polarity": edge.polarity.value,
                "contribution": contribution,
                "would_flip": would_flip,
            })
    
    # Sort by absolute contribution
    counter_specs.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    
    return counter_specs
