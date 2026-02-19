"""
Dialectical Routing (DR) for CRUX Protocol.

Unlike standard capability routing (A2A), Dialectical Routing intentionally
routes Claim Bundles toward agents best positioned to CHALLENGE them —
maximizing epistemic pressure and reducing echo-chamber failure modes.

Key Features:
    - Dialectical Fitness Score (DFS) computation
    - Adversarial potential assessment
    - Domain matching with hierarchical support
    - Recency-aware routing to avoid over-engagement
"""

from __future__ import annotations

import re
import math
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

from argus.crux.models import CRUXConfig, Polarity

if TYPE_CHECKING:
    from argus.crux.agent_card import EpistemicAgentCard, EACRegistry
    from argus.crux.claim_bundle import ClaimBundle

logger = logging.getLogger(__name__)


@dataclass
class DialecticalFitnessScore:
    """
    Dialectical Fitness Score (DFS) for an agent-claim pair.
    
    DFS measures how well-positioned an agent is to challenge
    a specific Claim Bundle.
    
    Attributes:
        agent_id: Agent being scored
        cb_id: Claim Bundle being evaluated
        total_score: Combined DFS score
        domain_match: Domain match component (w1)
        adversarial_potential: Adversarial potential component (w2)
        credibility_factor: Credibility component (w3)
        recency_factor: Recency component (w4)
        computed_at: When score was computed
    """
    agent_id: str
    cb_id: str
    total_score: float = 0.0
    domain_match: float = 0.0
    adversarial_potential: float = 0.0
    credibility_factor: float = 0.0
    recency_factor: float = 0.0
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "cb_id": self.cb_id,
            "total_score": self.total_score,
            "components": {
                "domain_match": self.domain_match,
                "adversarial_potential": self.adversarial_potential,
                "credibility_factor": self.credibility_factor,
                "recency_factor": self.recency_factor,
            },
            "computed_at": self.computed_at.isoformat(),
        }


class DomainMatcher:
    """
    Matches agent belief domains against claim text.
    
    Supports hierarchical domain matching and semantic similarity.
    """
    
    # Common domain hierarchies
    DOMAIN_HIERARCHIES = {
        "medicine": ["clinical", "oncology", "cardiology", "neurology", "pharmacology"],
        "clinical": ["rct", "clinical-trials", "diagnosis", "treatment"],
        "oncology": ["cancer", "tumor", "chemotherapy", "radiation"],
        "finance": ["trading", "banking", "investment", "risk"],
        "technology": ["software", "hardware", "ai", "ml", "data"],
        "science": ["physics", "chemistry", "biology", "research"],
    }
    
    # Keyword patterns for domain detection
    DOMAIN_KEYWORDS = {
        "oncology": r"cancer|tumor|oncolog|chemo|radiation|malignant",
        "clinical-trials": r"rct|trial|placebo|randomized|blind|cohort",
        "cardiology": r"heart|cardiac|arrhythmia|coronary|cardiovascular",
        "pharmacology": r"drug|medication|dose|pharma|therapeutic|efficacy",
        "finance": r"market|stock|trading|investment|portfolio|risk",
        "technology": r"software|algorithm|system|data|compute|digital",
        "methodology": r"method|approach|technique|procedure|protocol",
    }
    
    @classmethod
    def extract_domains_from_text(cls, text: str) -> list[str]:
        """
        Extract domain indicators from claim text.
        
        Args:
            text: Claim text
            
        Returns:
            List of detected domains
        """
        text_lower = text.lower()
        detected = []
        
        for domain, pattern in cls.DOMAIN_KEYWORDS.items():
            if re.search(pattern, text_lower):
                detected.append(domain)
        
        return detected
    
    @classmethod
    def compute_domain_match(
        cls,
        agent_domains: list[str],
        claim_domains: list[str],
    ) -> float:
        """
        Compute domain match score.
        
        Args:
            agent_domains: Agent's belief domains
            claim_domains: Domains extracted from claim
            
        Returns:
            Match score in [0, 1]
        """
        if not claim_domains:
            return 0.5  # Neutral if no domains detected
        
        if not agent_domains:
            return 0.3  # Low match if agent has no domains
        
        if "all" in agent_domains:
            return 1.0
        
        # Direct matches
        direct_matches = set(d.lower() for d in agent_domains) & set(d.lower() for d in claim_domains)
        
        # Hierarchical matches
        hierarchical_matches = 0
        for agent_domain in agent_domains:
            agent_lower = agent_domain.lower()
            
            # Check if agent domain is parent of claim domain
            if agent_lower in cls.DOMAIN_HIERARCHIES:
                children = cls.DOMAIN_HIERARCHIES[agent_lower]
                for claim_domain in claim_domains:
                    if claim_domain.lower() in children:
                        hierarchical_matches += 0.7
            
            # Check if agent domain is child of claim domain
            for parent, children in cls.DOMAIN_HIERARCHIES.items():
                if agent_lower in children:
                    if parent in [d.lower() for d in claim_domains]:
                        hierarchical_matches += 0.5
        
        total_matches = len(direct_matches) + hierarchical_matches
        max_possible = len(claim_domains)
        
        return min(1.0, total_matches / max_possible)


class AdversarialAssessor:
    """
    Assesses adversarial potential of an agent against a claim.
    
    Rewards agents whose prior beliefs differ from the claim's
    posterior — they are intrinsically motivated to push back.
    """
    
    @staticmethod
    def compute_adversarial_potential(
        agent_prior: Optional[float],
        claim_posterior: float,
        claim_polarity: Polarity,
    ) -> float:
        """
        Compute adversarial potential score.
        
        An agent with a different belief is more likely to
        provide a rigorous challenge.
        
        Args:
            agent_prior: Agent's prior belief (if known)
            claim_posterior: Claim's posterior probability
            claim_polarity: Claim's polarity
            
        Returns:
            Adversarial potential score in [0, 1]
        """
        if agent_prior is None:
            # Unknown prior - assume moderate adversarial potential
            return 0.5
        
        # Compute belief divergence
        divergence = abs(agent_prior - claim_posterior)
        
        # Amplify if polarity suggests opposition
        if claim_polarity == Polarity.SUPPORTS and agent_prior < 0.5:
            # Agent is skeptical, claim is supportive
            divergence *= 1.2
        elif claim_polarity == Polarity.ATTACKS and agent_prior > 0.5:
            # Agent is optimistic, claim is attacking
            divergence *= 1.2
        
        # Map to [0, 1] with sigmoid-like curve
        # Maximum potential at divergence = 0.5 (opposite beliefs)
        potential = 2 * divergence  # Scale to [0, 1] range
        
        return min(1.0, potential)


class RecencyTracker:
    """
    Tracks recent agent engagements to avoid over-engagement.
    
    Agents that have recently engaged with a proposition receive
    a recency penalty to encourage diverse participation.
    """
    
    def __init__(self, decay_hours: float = 1.0):
        """
        Initialize tracker.
        
        Args:
            decay_hours: Hours for engagement to decay
        """
        self.decay_hours = decay_hours
        self._engagements: dict[str, dict[str, datetime]] = defaultdict(dict)
    
    def record_engagement(
        self,
        agent_id: str,
        proposition_id: str,
    ) -> None:
        """
        Record an agent engagement.
        
        Args:
            agent_id: Engaging agent
            proposition_id: Proposition engaged with
        """
        self._engagements[agent_id][proposition_id] = datetime.utcnow()
    
    def compute_recency_penalty(
        self,
        agent_id: str,
        proposition_id: str,
    ) -> float:
        """
        Compute recency penalty for agent-proposition pair.
        
        Args:
            agent_id: Agent ID
            proposition_id: Proposition ID
            
        Returns:
            Penalty in [0, 1], higher = more recent engagement
        """
        if agent_id not in self._engagements:
            return 0.0  # No penalty
        
        if proposition_id not in self._engagements[agent_id]:
            return 0.0
        
        last_engagement = self._engagements[agent_id][proposition_id]
        elapsed = datetime.utcnow() - last_engagement
        hours = elapsed.total_seconds() / 3600
        
        # Exponential decay
        penalty = math.exp(-hours / self.decay_hours)
        
        return penalty
    
    def cleanup_old(self, max_age_hours: float = 24.0) -> int:
        """
        Remove old engagement records.
        
        Args:
            max_age_hours: Maximum age to keep
            
        Returns:
            Number of records removed
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed = 0
        
        for agent_id in list(self._engagements.keys()):
            for prop_id in list(self._engagements[agent_id].keys()):
                if self._engagements[agent_id][prop_id] < cutoff:
                    del self._engagements[agent_id][prop_id]
                    removed += 1
            
            # Clean up empty agent entries
            if not self._engagements[agent_id]:
                del self._engagements[agent_id]
        
        return removed


def compute_dfs(
    agent: "EpistemicAgentCard",
    claim_bundle: "ClaimBundle",
    config: Optional[CRUXConfig] = None,
    agent_prior: Optional[float] = None,
    recency_penalty: float = 0.0,
) -> DialecticalFitnessScore:
    """
    Compute Dialectical Fitness Score for an agent-claim pair.
    
    DFS = w1 * domain_match + w2 * adversarial_potential + 
          w3 * credibility + w4 * (1 - recency_penalty)
    
    Args:
        agent: Agent's EAC
        claim_bundle: Claim Bundle to evaluate
        config: CRUX configuration (for weights)
        agent_prior: Agent's prior belief on proposition
        recency_penalty: Recency penalty from tracker
        
    Returns:
        DialecticalFitnessScore instance
    """
    config = config or CRUXConfig()
    
    # Extract domains from claim
    claim_domains = DomainMatcher.extract_domains_from_text(claim_bundle.claim_text)
    
    # Compute components
    domain_match = DomainMatcher.compute_domain_match(
        agent.belief_domains,
        claim_domains,
    )
    
    adversarial = AdversarialAssessor.compute_adversarial_potential(
        agent_prior,
        claim_bundle.posterior,
        claim_bundle.polarity,
    )
    
    credibility = agent.calibration.credibility_rating
    recency = 1.0 - recency_penalty
    
    # Compute weighted total
    total = (
        config.dfs_domain_weight * domain_match +
        config.dfs_adversarial_weight * adversarial +
        config.dfs_credibility_weight * credibility +
        config.dfs_recency_weight * recency
    )
    
    return DialecticalFitnessScore(
        agent_id=agent.agent_id,
        cb_id=claim_bundle.cb_id,
        total_score=total,
        domain_match=domain_match,
        adversarial_potential=adversarial,
        credibility_factor=credibility,
        recency_factor=recency,
    )


class DialecticalRouter:
    """
    Routes Claim Bundles to agents using Dialectical Fitness Scores.
    
    Key constraint: An agent that issued the Claim Bundle is excluded
    from its own routing table — self-challenge is not permitted.
    
    Example:
        >>> router = DialecticalRouter(registry, config)
        >>> rankings = router.rank_challengers(claim_bundle)
        >>> best_challenger = rankings[0].agent_id
    """
    
    def __init__(
        self,
        registry: "EACRegistry",
        config: Optional[CRUXConfig] = None,
    ):
        """
        Initialize router.
        
        Args:
            registry: EAC registry for agent lookup
            config: CRUX configuration
        """
        self.registry = registry
        self.config = config or CRUXConfig()
        self.recency_tracker = RecencyTracker()
        
        # Cache of agent priors per proposition
        self._agent_priors: dict[str, dict[str, float]] = defaultdict(dict)
    
    def set_agent_prior(
        self,
        agent_id: str,
        proposition_id: str,
        prior: float,
    ) -> None:
        """
        Record an agent's prior belief on a proposition.
        
        Args:
            agent_id: Agent ID
            proposition_id: Proposition ID
            prior: Prior probability belief
        """
        self._agent_priors[agent_id][proposition_id] = prior
    
    def get_agent_prior(
        self,
        agent_id: str,
        proposition_id: str,
    ) -> Optional[float]:
        """
        Get agent's prior belief on a proposition.
        
        Returns:
            Prior probability or None if unknown
        """
        return self._agent_priors.get(agent_id, {}).get(proposition_id)
    
    def rank_challengers(
        self,
        claim_bundle: "ClaimBundle",
        top_k: Optional[int] = None,
    ) -> list[DialecticalFitnessScore]:
        """
        Rank potential challengers by DFS.
        
        Args:
            claim_bundle: Claim Bundle to find challengers for
            top_k: Return only top K challengers
            
        Returns:
            List of DFS scores, sorted descending
        """
        # Get eligible challengers (exclude issuer)
        challengers = self.registry.get_challengers(
            exclude_agent=claim_bundle.issuer_agent,
        )
        
        if not challengers:
            logger.warning(f"No challengers available for {claim_bundle.cb_id}")
            return []
        
        # Compute DFS for each
        scores = []
        for agent in challengers:
            agent_prior = self.get_agent_prior(
                agent.agent_id,
                claim_bundle.proposition_id,
            )
            recency_penalty = self.recency_tracker.compute_recency_penalty(
                agent.agent_id,
                claim_bundle.proposition_id,
            )
            
            dfs = compute_dfs(
                agent=agent,
                claim_bundle=claim_bundle,
                config=self.config,
                agent_prior=agent_prior,
                recency_penalty=recency_penalty,
            )
            scores.append(dfs)
        
        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        
        if top_k:
            return scores[:top_k]
        
        return scores
    
    def get_best_challenger(
        self,
        claim_bundle: "ClaimBundle",
    ) -> Optional[DialecticalFitnessScore]:
        """
        Get the best challenger for a Claim Bundle.
        
        Args:
            claim_bundle: Claim Bundle to challenge
            
        Returns:
            Best DFS score or None if no challengers
        """
        rankings = self.rank_challengers(claim_bundle, top_k=1)
        return rankings[0] if rankings else None
    
    def record_challenge(
        self,
        agent_id: str,
        proposition_id: str,
    ) -> None:
        """
        Record that an agent has challenged a proposition.
        
        Updates recency tracking.
        """
        self.recency_tracker.record_engagement(agent_id, proposition_id)
    
    def get_routing_table(
        self,
        claim_bundle: "ClaimBundle",
    ) -> dict[str, DialecticalFitnessScore]:
        """
        Get full routing table for a Claim Bundle.
        
        Args:
            claim_bundle: Claim Bundle
            
        Returns:
            Dictionary mapping agent_id to DFS
        """
        rankings = self.rank_challengers(claim_bundle)
        return {s.agent_id: s for s in rankings}
    
    def visualize_routing(
        self,
        claim_bundle: "ClaimBundle",
    ) -> dict[str, Any]:
        """
        Get visualization data for routing decisions.
        
        Returns data suitable for plotting.
        """
        rankings = self.rank_challengers(claim_bundle)
        
        return {
            "cb_id": claim_bundle.cb_id,
            "claim_text": claim_bundle.claim_text[:100],
            "rankings": [
                {
                    "rank": i + 1,
                    "agent_id": s.agent_id,
                    "total_score": s.total_score,
                    "components": {
                        "domain": s.domain_match,
                        "adversarial": s.adversarial_potential,
                        "credibility": s.credibility_factor,
                        "recency": s.recency_factor,
                    },
                }
                for i, s in enumerate(rankings)
            ],
        }
