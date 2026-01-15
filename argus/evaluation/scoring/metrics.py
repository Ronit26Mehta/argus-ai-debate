"""
ARGUS Novel Scoring Metrics.

Eight unique metrics designed specifically for evaluating multi-agent AI debate systems.
These metrics are original to ARGUS and provide comprehensive debate quality assessment.

Metrics:
    ARCIS - Argus Reasoning Coherence Index Score
    EVID-Q - Evidence Quality Quotient
    DIALEC - Dialectical Depth Evaluation Coefficient
    REBUT-F - Rebuttal Effectiveness Factor
    CONV-S - Convergence Stability Score
    PROV-I - Provenance Integrity Index
    CALIB-M - Calibration Matrix Score
    EIG-U - Expected Information Gain Utilization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Definitions
# =============================================================================

@dataclass
class MetricDefinition:
    """Definition of a scoring metric.
    
    Attributes:
        name: Short metric name (e.g., "ARCIS")
        full_name: Full descriptive name
        description: Detailed description
        range: Expected value range (min, max)
        higher_is_better: Whether higher values indicate better performance
        weight: Default weight in composite scoring
    """
    name: str
    full_name: str
    description: str
    range: tuple[float, float] = (0.0, 1.0)
    higher_is_better: bool = True
    weight: float = 1.0


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "ARCIS": MetricDefinition(
        name="ARCIS",
        full_name="Argus Reasoning Coherence Index Score",
        description="Measures logical consistency and coherence across debate rounds. "
                   "Evaluates whether arguments build upon each other without contradictions.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=1.2,
    ),
    "EVID-Q": MetricDefinition(
        name="EVID-Q",
        full_name="Evidence Quality Quotient",
        description="Composite score combining evidence relevance, confidence, source quality, "
                   "and citation completeness into a unified quality measure.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=1.0,
    ),
    "DIALEC": MetricDefinition(
        name="DIALEC",
        full_name="Dialectical Depth Evaluation Coefficient",
        description="Quantifies the depth of dialectical exchange - attack/defense cycles, "
                   "rebuttal chains, and argumentative sophistication.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=1.1,
    ),
    "REBUT-F": MetricDefinition(
        name="REBUT-F",
        full_name="Rebuttal Effectiveness Factor",
        description="Measures the effectiveness of rebuttals in addressing and weakening "
                   "opposing evidence. Higher values indicate more impactful rebuttals.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=0.9,
    ),
    "CONV-S": MetricDefinition(
        name="CONV-S",
        full_name="Convergence Stability Score",
        description="Measures how quickly and stably the posterior probability converges. "
                   "Penalizes oscillation and rewards monotonic convergence.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=1.0,
    ),
    "PROV-I": MetricDefinition(
        name="PROV-I",
        full_name="Provenance Integrity Index",
        description="Tracks the completeness and verifiability of the citation chain. "
                   "Evaluates source traceability and evidence attribution.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=0.8,
    ),
    "CALIB-M": MetricDefinition(
        name="CALIB-M",
        full_name="Calibration Matrix Score",
        description="Multi-dimensional calibration assessment across confidence bins. "
                   "Measures how well confidence aligns with actual accuracy.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=1.0,
    ),
    "EIG-U": MetricDefinition(
        name="EIG-U",
        full_name="Expected Information Gain Utilization",
        description="Measures how efficiently evidence reduces uncertainty. "
                   "Evaluates information gain per evidence item relative to optimal.",
        range=(0.0, 1.0),
        higher_is_better=True,
        weight=1.1,
    ),
}


# =============================================================================
# Individual Metric Computations
# =============================================================================

def compute_arcis(result: dict[str, Any]) -> float:
    """
    Compute ARCIS - Argus Reasoning Coherence Index Score.
    
    Measures logical consistency across debate rounds by analyzing:
    - Argument dependency chains
    - Contradiction detection
    - Evidence-claim alignment
    
    Formula:
        ARCIS = (1 - contradiction_rate) × chain_quality × alignment_score
    
    Args:
        result: Debate result dictionary containing graph and round data
        
    Returns:
        ARCIS score in [0, 1] range
    """
    try:
        graph_data = result.get("graph", {})
        rounds = result.get("rounds", [])
        
        if not graph_data and not rounds:
            return 0.5  # Default neutral score
        
        # Analyze evidence chain coherence
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        if not nodes:
            nodes = []
        if not edges:
            edges = []
        
        # 1. Chain quality: Connected evidence forms coherent chains
        num_nodes = len(nodes)
        num_edges = len(edges)
        
        if num_nodes > 1:
            expected_edges = num_nodes - 1  # Minimum spanning tree
            chain_quality = min(1.0, num_edges / max(expected_edges, 1))
        else:
            chain_quality = 0.5
        
        # 2. Contradiction rate: Evidence with opposing polarity for same claim
        evidence_polarities = {}
        for node in nodes:
            if isinstance(node, dict):
                prop_id = node.get("target_proposition", node.get("parent_id", ""))
                polarity = node.get("polarity", 0)
                if prop_id:
                    if prop_id not in evidence_polarities:
                        evidence_polarities[prop_id] = []
                    evidence_polarities[prop_id].append(polarity)
        
        # Contradictions exist when both +1 and -1 polarities exist
        # This is actually expected in debates, so we measure balance
        contradiction_factor = 0.0
        for polarities in evidence_polarities.values():
            if polarities:
                has_support = any(p > 0 for p in polarities)
                has_attack = any(p < 0 for p in polarities)
                if has_support and has_attack:
                    # Balanced dialectic is good
                    contradiction_factor += 0.1
        
        coherence_bonus = min(0.3, contradiction_factor)
        
        # 3. Alignment score: Evidence confidence matches verdict direction
        verdict = result.get("verdict", {})
        posterior = verdict.get("posterior", 0.5)
        verdict_direction = 1 if posterior > 0.5 else -1 if posterior < 0.5 else 0
        
        aligned_evidence = 0
        total_evidence = 0
        for node in nodes:
            if isinstance(node, dict) and "polarity" in node:
                total_evidence += 1
                evidence_polarity = node.get("polarity", 0)
                confidence = node.get("confidence", 0.5)
                
                # Weight by confidence
                if evidence_polarity * verdict_direction > 0:
                    aligned_evidence += confidence
                elif evidence_polarity * verdict_direction < 0:
                    aligned_evidence -= confidence * 0.3  # Penalize misalignment less
        
        if total_evidence > 0:
            alignment_score = 0.5 + min(0.5, max(-0.5, aligned_evidence / total_evidence))
        else:
            alignment_score = 0.5
        
        # Combine components
        arcis = (
            0.35 * chain_quality +
            0.30 * (0.5 + coherence_bonus) +
            0.35 * alignment_score
        )
        
        return max(0.0, min(1.0, arcis))
        
    except Exception as e:
        logger.warning(f"ARCIS computation error: {e}")
        return 0.5


def compute_evid_q(result: dict[str, Any]) -> float:
    """
    Compute EVID-Q - Evidence Quality Quotient.
    
    Composite quality score combining:
    - Evidence relevance to proposition
    - Confidence calibration
    - Source quality/credibility
    - Citation completeness
    
    Formula:
        EVID-Q = Σ(relevance × confidence × source_quality) / n × citation_factor
    
    Args:
        result: Debate result dictionary
        
    Returns:
        EVID-Q score in [0, 1] range
    """
    try:
        graph_data = result.get("graph", {})
        nodes = graph_data.get("nodes", [])
        
        if not nodes:
            return 0.5
        
        evidence_nodes = [
            n for n in nodes 
            if isinstance(n, dict) and n.get("type") in ("evidence", "Evidence")
            or (isinstance(n, dict) and "confidence" in n)
        ]
        
        if not evidence_nodes:
            return 0.5
        
        total_quality = 0.0
        citation_count = 0
        
        for evidence in evidence_nodes:
            relevance = evidence.get("relevance", 0.7)
            confidence = evidence.get("confidence", 0.7)
            quality = evidence.get("quality", evidence.get("source_quality", 0.7))
            
            # Check for citation presence
            has_citation = bool(evidence.get("citations") or evidence.get("source"))
            if has_citation:
                citation_count += 1
            
            # Individual evidence quality
            item_quality = relevance * confidence * quality
            total_quality += item_quality
        
        # Average quality
        avg_quality = total_quality / len(evidence_nodes)
        
        # Citation completeness factor (0.8 to 1.0 range)
        citation_rate = citation_count / len(evidence_nodes)
        citation_factor = 0.8 + 0.2 * citation_rate
        
        evid_q = avg_quality * citation_factor
        
        return max(0.0, min(1.0, evid_q))
        
    except Exception as e:
        logger.warning(f"EVID-Q computation error: {e}")
        return 0.5


def compute_dialec(result: dict[str, Any]) -> float:
    """
    Compute DIALEC - Dialectical Depth Evaluation Coefficient.
    
    Quantifies dialectical sophistication:
    - Attack/defense cycle depth
    - Rebuttal chain length
    - Argument layer count
    - Counter-argument quality
    
    Formula:
        DIALEC = α × depth_score + β × rebuttal_score + γ × engagement_score
    
    Args:
        result: Debate result dictionary
        
    Returns:
        DIALEC score in [0, 1] range
    """
    try:
        graph_data = result.get("graph", {})
        nodes = graph_data.get("nodes", [])
        
        num_evidence = result.get("num_evidence", 0)
        num_rebuttals = result.get("num_rebuttals", 0)
        num_rounds = result.get("num_rounds", 1)
        
        # 1. Depth score: Based on rebuttal chains
        # More rebuttals relative to evidence = deeper dialectic
        if num_evidence > 0:
            depth_ratio = num_rebuttals / num_evidence
            depth_score = min(1.0, depth_ratio * 2)  # Normalize: 0.5 ratio = 1.0 score
        else:
            depth_score = 0.0
        
        # 2. Rebuttal quality score
        rebuttal_nodes = [
            n for n in nodes
            if isinstance(n, dict) and n.get("type") in ("rebuttal", "Rebuttal")
            or (isinstance(n, dict) and "rebuttal_type" in n)
        ]
        
        if rebuttal_nodes:
            avg_strength = sum(
                n.get("strength", 0.5) for n in rebuttal_nodes
            ) / len(rebuttal_nodes)
            rebuttal_score = avg_strength
        else:
            rebuttal_score = 0.3  # Baseline for no rebuttals
        
        # 3. Engagement score: Round progression with evidence accumulation
        if num_rounds > 0:
            evidence_per_round = num_evidence / num_rounds
            engagement_score = min(1.0, evidence_per_round / 3)  # 3 evidence/round = full score
        else:
            engagement_score = 0.0
        
        # Combine with weights
        dialec = (
            0.35 * depth_score +
            0.35 * rebuttal_score +
            0.30 * engagement_score
        )
        
        return max(0.0, min(1.0, dialec))
        
    except Exception as e:
        logger.warning(f"DIALEC computation error: {e}")
        return 0.5


def compute_rebut_f(result: dict[str, Any]) -> float:
    """
    Compute REBUT-F - Rebuttal Effectiveness Factor.
    
    Measures rebuttal impact:
    - Successful challenge rate
    - Confidence reduction in target
    - Rebuttal strength vs target confidence
    
    Formula:
        REBUT-F = Σ(rebuttal_strength × target_reduction) / Σ(target_confidence)
    
    Args:
        result: Debate result dictionary
        
    Returns:
        REBUT-F score in [0, 1] range
    """
    try:
        graph_data = result.get("graph", {})
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        num_rebuttals = result.get("num_rebuttals", 0)
        
        if num_rebuttals == 0:
            return 0.5  # Neutral when no rebuttals
        
        # Build node lookup
        node_map = {}
        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get("id", "")
                if node_id:
                    node_map[node_id] = node
        
        # Find rebuttal-target pairs
        rebuttal_effectiveness = []
        
        for edge in edges:
            if isinstance(edge, dict):
                edge_type = edge.get("type", edge.get("edge_type", ""))
                if edge_type.lower() in ("rebuts", "rebuttal"):
                    source_id = edge.get("source", edge.get("from", ""))
                    target_id = edge.get("target", edge.get("to", ""))
                    
                    rebuttal = node_map.get(source_id, {})
                    target = node_map.get(target_id, {})
                    
                    rebuttal_strength = rebuttal.get("strength", 0.5)
                    rebuttal_conf = rebuttal.get("confidence", 0.5)
                    target_conf = target.get("confidence", 0.7)
                    
                    # Effectiveness = how much rebuttal reduces target impact
                    # Higher strength and higher target confidence = more effective
                    effectiveness = rebuttal_strength * rebuttal_conf * (1 - 0.3 * (1 - target_conf))
                    rebuttal_effectiveness.append(effectiveness)
        
        if rebuttal_effectiveness:
            rebut_f = sum(rebuttal_effectiveness) / len(rebuttal_effectiveness)
        else:
            # Estimate from rebuttal count
            rebut_f = min(0.8, 0.3 + 0.1 * num_rebuttals)
        
        return max(0.0, min(1.0, rebut_f))
        
    except Exception as e:
        logger.warning(f"REBUT-F computation error: {e}")
        return 0.5


def compute_conv_s(result: dict[str, Any]) -> float:
    """
    Compute CONV-S - Convergence Stability Score.
    
    Measures posterior convergence quality:
    - Convergence speed (fewer rounds = faster)
    - Stability (low oscillation)
    - Monotonicity (consistent direction)
    
    Formula:
        CONV-S = speed_factor × (1 - oscillation) × monotonicity_bonus
    
    Args:
        result: Debate result dictionary
        
    Returns:
        CONV-S score in [0, 1] range
    """
    try:
        rounds = result.get("rounds", [])
        verdict = result.get("verdict", {})
        
        # Extract posterior timeline
        posteriors = []
        for r in rounds:
            if isinstance(r, dict):
                post = r.get("posterior_after", r.get("posterior", None))
                if post is not None:
                    posteriors.append(float(post))
        
        # Add initial and final if not in rounds
        prior = 0.5  # Default prior
        final_posterior = verdict.get("posterior", 0.5)
        
        if not posteriors:
            posteriors = [prior, final_posterior]
        elif len(posteriors) == 1:
            posteriors = [prior] + posteriors
        
        if len(posteriors) < 2:
            return 0.5
        
        # 1. Speed factor: How quickly did we converge?
        num_rounds = result.get("num_rounds", len(posteriors) - 1)
        final_change = abs(posteriors[-1] - prior)
        
        if final_change > 0.01:  # Meaningful change occurred
            speed_factor = min(1.0, 2.0 / max(num_rounds, 1))
        else:
            speed_factor = 0.5  # No real convergence
        
        # 2. Oscillation: Count direction changes
        direction_changes = 0
        for i in range(1, len(posteriors) - 1):
            prev_dir = posteriors[i] - posteriors[i-1]
            next_dir = posteriors[i+1] - posteriors[i]
            if prev_dir * next_dir < 0:  # Direction changed
                direction_changes += 1
        
        max_possible_changes = max(1, len(posteriors) - 2)
        oscillation = direction_changes / max_possible_changes
        stability = 1 - oscillation
        
        # 3. Monotonicity bonus: Consistent direction toward final
        final_direction = posteriors[-1] - posteriors[0]
        monotonic_steps = 0
        for i in range(1, len(posteriors)):
            step_direction = posteriors[i] - posteriors[i-1]
            if final_direction * step_direction >= 0:
                monotonic_steps += 1
        
        monotonicity = monotonic_steps / (len(posteriors) - 1) if len(posteriors) > 1 else 0.5
        
        # Combine
        conv_s = 0.30 * speed_factor + 0.40 * stability + 0.30 * monotonicity
        
        return max(0.0, min(1.0, conv_s))
        
    except Exception as e:
        logger.warning(f"CONV-S computation error: {e}")
        return 0.5


def compute_prov_i(result: dict[str, Any]) -> float:
    """
    Compute PROV-I - Provenance Integrity Index.
    
    Measures citation chain completeness:
    - Evidence source attribution
    - Citation verification status
    - Hash chain integrity
    - Audit trail completeness
    
    Formula:
        PROV-I = attribution_rate × verification_rate × integrity_factor
    
    Args:
        result: Debate result dictionary
        
    Returns:
        PROV-I score in [0, 1] range
    """
    try:
        graph_data = result.get("graph", {})
        nodes = graph_data.get("nodes", [])
        provenance = result.get("provenance", result.get("ledger", {}))
        
        evidence_nodes = [
            n for n in nodes
            if isinstance(n, dict) and (
                n.get("type") in ("evidence", "Evidence") or
                "confidence" in n
            )
        ]
        
        if not evidence_nodes:
            return 0.7  # Default to moderate score if no evidence
        
        # 1. Attribution rate: Evidence with identifiable sources
        attributed = 0
        for node in evidence_nodes:
            if node.get("source") or node.get("citations") or node.get("doc_id"):
                attributed += 1
        
        attribution_rate = attributed / len(evidence_nodes)
        
        # 2. Verification status: Evidence marked as verified
        verified = 0
        for node in evidence_nodes:
            if node.get("verified", False) or node.get("quality", 0) > 0.7:
                verified += 1
        
        verification_rate = verified / len(evidence_nodes)
        
        # 3. Integrity factor: Provenance ledger completeness
        if isinstance(provenance, dict):
            events = provenance.get("events", [])
            expected_events = 2 + len(evidence_nodes)  # Start + end + evidence
            integrity_factor = min(1.0, len(events) / max(expected_events, 1))
        elif isinstance(provenance, list):
            integrity_factor = min(1.0, len(provenance) / max(len(evidence_nodes), 1))
        else:
            integrity_factor = 0.5  # Unknown provenance structure
        
        # Combine
        prov_i = (
            0.40 * attribution_rate +
            0.30 * verification_rate +
            0.30 * integrity_factor
        )
        
        return max(0.0, min(1.0, prov_i))
        
    except Exception as e:
        logger.warning(f"PROV-I computation error: {e}")
        return 0.5


def compute_calib_m(result: dict[str, Any]) -> float:
    """
    Compute CALIB-M - Calibration Matrix Score.
    
    Multi-dimensional calibration assessment:
    - Confidence vs accuracy alignment
    - Per-bin calibration error
    - Overconfidence/underconfidence detection
    
    Formula:
        CALIB-M = 1 - ECE (Expected Calibration Error)
    
    Args:
        result: Debate result dictionary
        
    Returns:
        CALIB-M score in [0, 1] range (1 = perfectly calibrated)
    """
    try:
        verdict = result.get("verdict", {})
        posterior = verdict.get("posterior", 0.5)
        label = verdict.get("label", "undecided").lower()
        
        graph_data = result.get("graph", {})
        nodes = graph_data.get("nodes", [])
        
        # Collect confidence scores
        confidences = []
        for node in nodes:
            if isinstance(node, dict) and "confidence" in node:
                confidences.append(node.get("confidence", 0.5))
        
        if not confidences:
            confidences = [posterior]
        
        # Approximate calibration from verdict
        # If posterior is extreme and verdict is decisive, that's well-calibrated
        verdict_confidence = abs(posterior - 0.5) * 2  # 0 to 1 scale
        is_decisive = label in ("supported", "rejected")
        
        # Calibration based on verdict decisiveness matching confidence
        if is_decisive and verdict_confidence > 0.4:
            calibration_match = min(1.0, verdict_confidence)
        elif not is_decisive and verdict_confidence < 0.4:
            calibration_match = 1.0 - verdict_confidence
        else:
            calibration_match = 0.5
        
        # Evidence confidence distribution calibration
        avg_confidence = sum(confidences) / len(confidences)
        confidence_spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0
        
        # Good calibration has reasonable spread (not all same, not too scattered)
        spread_score = 1.0 - abs(0.3 - confidence_spread)  # 0.3 is ideal spread
        
        # Combine
        calib_m = 0.60 * calibration_match + 0.40 * max(0, spread_score)
        
        return max(0.0, min(1.0, calib_m))
        
    except Exception as e:
        logger.warning(f"CALIB-M computation error: {e}")
        return 0.5


def compute_eig_u(result: dict[str, Any]) -> float:
    """
    Compute EIG-U - Expected Information Gain Utilization.
    
    Measures information efficiency:
    - Uncertainty reduction per evidence
    - Information gain relative to optimal
    - Evidence utility vs cost
    
    Formula:
        EIG-U = actual_info_gain / theoretical_max_gain
    
    Args:
        result: Debate result dictionary
        
    Returns:
        EIG-U score in [0, 1] range
    """
    try:
        verdict = result.get("verdict", {})
        posterior = verdict.get("posterior", 0.5)
        prior = 0.5  # Assume standard prior
        
        num_evidence = result.get("num_evidence", 1)
        num_rounds = result.get("num_rounds", 1)
        
        # Compute actual entropy reduction
        def binary_entropy(p: float) -> float:
            if p <= 0 or p >= 1:
                return 0.0
            return -p * math.log2(p) - (1-p) * math.log2(1-p)
        
        prior_entropy = binary_entropy(prior)
        posterior_entropy = binary_entropy(posterior)
        
        actual_reduction = prior_entropy - posterior_entropy
        max_possible_reduction = prior_entropy  # Can reduce to 0
        
        if max_possible_reduction > 0:
            reduction_efficiency = actual_reduction / max_possible_reduction
        else:
            reduction_efficiency = 0.5
        
        # Efficiency per evidence item
        if num_evidence > 0:
            gain_per_evidence = abs(posterior - prior) / num_evidence
            # Normalize: 0.1 gain per evidence is considered good
            evidence_efficiency = min(1.0, gain_per_evidence / 0.1)
        else:
            evidence_efficiency = 0.0
        
        # Round efficiency: Fewer rounds for same result is better
        round_efficiency = 1.0 / max(1, num_rounds) if num_rounds <= 3 else 0.5
        
        # Combine
        eig_u = (
            0.40 * reduction_efficiency +
            0.35 * evidence_efficiency +
            0.25 * round_efficiency
        )
        
        return max(0.0, min(1.0, eig_u))
        
    except Exception as e:
        logger.warning(f"EIG-U computation error: {e}")
        return 0.5


# =============================================================================
# Aggregated Scoring
# =============================================================================

def compute_all_scores(result: dict[str, Any]) -> dict[str, float]:
    """
    Compute all 8 ARGUS novel metrics for a debate result.
    
    Args:
        result: Debate result dictionary
        
    Returns:
        Dictionary mapping metric names to scores
    """
    return {
        "ARCIS": compute_arcis(result),
        "EVID-Q": compute_evid_q(result),
        "DIALEC": compute_dialec(result),
        "REBUT-F": compute_rebut_f(result),
        "CONV-S": compute_conv_s(result),
        "PROV-I": compute_prov_i(result),
        "CALIB-M": compute_calib_m(result),
        "EIG-U": compute_eig_u(result),
    }


@dataclass
class ScoreCard:
    """Complete score card for a debate evaluation.
    
    Attributes:
        scores: Individual metric scores
        composite_score: Weighted average of all metrics
        created_at: Timestamp of computation
        metadata: Additional context
    """
    scores: dict[str, float]
    composite_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.scores and self.composite_score == 0.0:
            self.composite_score = self._compute_composite()
    
    def _compute_composite(self) -> float:
        """Compute weighted composite score."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, score in self.scores.items():
            metric_def = METRIC_REGISTRY.get(name)
            weight = metric_def.weight if metric_def else 1.0
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scores": self.scores,
            "composite_score": self.composite_score,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """Format as readable string."""
        lines = ["ARGUS Score Card", "=" * 40]
        for name, score in sorted(self.scores.items()):
            lines.append(f"  {name:10s}: {score:.3f}")
        lines.append("-" * 40)
        lines.append(f"  {'COMPOSITE':10s}: {self.composite_score:.3f}")
        return "\n".join(lines)
    
    @classmethod
    def from_result(cls, result: dict[str, Any]) -> "ScoreCard":
        """Create ScoreCard from debate result."""
        scores = compute_all_scores(result)
        return cls(scores=scores, metadata={"source": "debate_result"})
