"""
CRUX Core Data Models.

Defines the fundamental data structures for the CRUX protocol:
    - BetaDistribution: Parametric uncertainty representation
    - ConfidenceDistribution: Wrapper for various distribution types
    - ChallengerBid: Bid for challenging a claim
    - BRPResolution: Result of belief reconciliation
    - EDRDelta: Delta bundle for reconnection sync
    - CRUXConfig: Protocol configuration
"""

from __future__ import annotations

import math
import hashlib
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator, computed_field


def generate_crux_id(prefix: str) -> str:
    """Generate a unique CRUX identifier with prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


def compute_crux_hash(content: str, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash for CRUX entities."""
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))
    return f"{algorithm}:{hasher.hexdigest()[:16]}"


class CRUXVersion(str, Enum):
    """CRUX protocol versions."""
    V1_0 = "1.0"


class Polarity(int, Enum):
    """Claim polarity relative to proposition."""
    SUPPORTS = 1      # Supports the proposition
    ATTACKS = -1      # Attacks the proposition
    NEUTRAL = 0       # Neither supports nor attacks


class DistributionType(str, Enum):
    """Types of confidence distributions."""
    BETA = "beta"
    NORMAL = "normal"
    POINT = "point"


class BRPState(str, Enum):
    """States of the Belief Reconciliation Protocol."""
    IDLE = "idle"
    TRIGGERED = "triggered"
    MINI_DEBATE = "mini_debate"
    BAYESIAN_MERGE = "bayesian_merge"
    PROVENANCE_FORK = "provenance_fork"
    RESOLVED = "resolved"


class AuctionState(str, Enum):
    """States of the Challenger Auction."""
    OPEN = "open"
    BIDDING = "bidding"
    EVALUATING = "evaluating"
    CLOSED = "closed"
    UNCHALLENGED = "unchallenged"


class BetaDistribution(BaseModel):
    """
    Beta distribution for uncertainty representation.
    
    The Beta distribution is conjugate prior for binary outcomes,
    making it ideal for representing calibrated beliefs in CRUX.
    
    Attributes:
        alpha: Alpha parameter (successes + prior)
        beta: Beta parameter (failures + prior)
        
    Example:
        >>> dist = BetaDistribution(alpha=7.3, beta=2.7)
        >>> print(f"Mean: {dist.mean:.3f}, Std: {dist.std:.3f}")
        Mean: 0.730, Std: 0.134
    """
    
    model_config = {"frozen": True}
    
    alpha: float = Field(
        ge=0.001,
        description="Alpha parameter (shape1)",
    )
    
    beta: float = Field(
        ge=0.001,
        description="Beta parameter (shape2)",
    )
    
    @computed_field
    @property
    def mean(self) -> float:
        """Compute distribution mean."""
        return self.alpha / (self.alpha + self.beta)
    
    @computed_field
    @property
    def mode(self) -> float:
        """Compute distribution mode (for alpha, beta > 1)."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return self.mean
    
    @computed_field
    @property
    def variance(self) -> float:
        """Compute distribution variance."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab ** 2 * (ab + 1))
    
    @computed_field
    @property
    def std(self) -> float:
        """Compute distribution standard deviation."""
        return math.sqrt(self.variance)
    
    @computed_field
    @property
    def concentration(self) -> float:
        """Total concentration (alpha + beta)."""
        return self.alpha + self.beta
    
    def pdf(self, x: float) -> float:
        """
        Probability density function at x.
        
        Args:
            x: Value in [0, 1]
            
        Returns:
            PDF value at x
        """
        if x <= 0 or x >= 1:
            return 0.0
        
        from scipy import special
        
        log_beta = special.betaln(self.alpha, self.beta)
        log_pdf = (
            (self.alpha - 1) * math.log(x) +
            (self.beta - 1) * math.log(1 - x) -
            log_beta
        )
        return math.exp(log_pdf)
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function at x.
        
        Args:
            x: Value in [0, 1]
            
        Returns:
            CDF value at x
        """
        from scipy import special
        return special.betainc(self.alpha, self.beta, x)
    
    def sample(self, n: int = 1) -> list[float]:
        """
        Draw samples from the distribution.
        
        Args:
            n: Number of samples
            
        Returns:
            List of samples
        """
        import numpy as np
        return list(np.random.beta(self.alpha, self.beta, n))
    
    def quantile(self, q: float) -> float:
        """
        Compute quantile (inverse CDF).
        
        Args:
            q: Quantile in [0, 1]
            
        Returns:
            Value at quantile q
        """
        from scipy import special
        return special.betaincinv(self.alpha, self.beta, q)
    
    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """
        Compute credible interval.
        
        Args:
            level: Credible level (e.g., 0.95 for 95%)
            
        Returns:
            (lower, upper) bounds
        """
        tail = (1 - level) / 2
        return self.quantile(tail), self.quantile(1 - tail)
    
    @classmethod
    def from_mean_concentration(
        cls,
        mean: float,
        concentration: float,
    ) -> "BetaDistribution":
        """
        Create from mean and concentration.
        
        Args:
            mean: Desired mean in (0, 1)
            concentration: Total concentration (alpha + beta)
            
        Returns:
            BetaDistribution instance
        """
        alpha = mean * concentration
        beta = (1 - mean) * concentration
        return cls(alpha=alpha, beta=beta)
    
    @classmethod
    def from_mean_std(
        cls,
        mean: float,
        std: float,
    ) -> "BetaDistribution":
        """
        Create from mean and standard deviation.
        
        Args:
            mean: Desired mean in (0, 1)
            std: Desired standard deviation
            
        Returns:
            BetaDistribution instance
        """
        # Derive concentration from mean and variance
        variance = std ** 2
        if variance >= mean * (1 - mean):
            # Invalid: use minimum concentration
            concentration = 2.0
        else:
            concentration = mean * (1 - mean) / variance - 1
        
        concentration = max(2.0, concentration)
        return cls.from_mean_concentration(mean, concentration)
    
    @classmethod
    def uniform(cls) -> "BetaDistribution":
        """Create uniform (non-informative) distribution."""
        return cls(alpha=1.0, beta=1.0)
    
    @classmethod
    def jeffreys(cls) -> "BetaDistribution":
        """Create Jeffreys prior (minimally informative)."""
        return cls(alpha=0.5, beta=0.5)
    
    def update(
        self,
        successes: int = 0,
        failures: int = 0,
    ) -> "BetaDistribution":
        """
        Bayesian update with new observations.
        
        Args:
            successes: Number of successes
            failures: Number of failures
            
        Returns:
            Updated BetaDistribution
        """
        return BetaDistribution(
            alpha=self.alpha + successes,
            beta=self.beta + failures,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "beta",
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "std": self.std,
        }


class ConfidenceDistribution(BaseModel):
    """
    Wrapper for confidence distributions of various types.
    
    Supports Beta, Normal, and Point distributions.
    
    Attributes:
        distribution_type: Type of distribution
        beta: Beta distribution parameters (if type is beta)
        mean: Mean value
        std: Standard deviation
    """
    
    model_config = {"frozen": True}
    
    distribution_type: DistributionType = Field(
        default=DistributionType.BETA,
        description="Type of underlying distribution",
    )
    
    beta: Optional[BetaDistribution] = Field(
        default=None,
        description="Beta distribution parameters",
    )
    
    mean: float = Field(
        ge=0.0,
        le=1.0,
        description="Distribution mean",
    )
    
    std: float = Field(
        ge=0.0,
        default=0.1,
        description="Standard deviation",
    )
    
    @classmethod
    def from_beta(
        cls,
        alpha: float,
        beta: float,
    ) -> "ConfidenceDistribution":
        """Create from Beta parameters."""
        beta_dist = BetaDistribution(alpha=alpha, beta=beta)
        return cls(
            distribution_type=DistributionType.BETA,
            beta=beta_dist,
            mean=beta_dist.mean,
            std=beta_dist.std,
        )
    
    @classmethod
    def from_point(cls, value: float) -> "ConfidenceDistribution":
        """Create point estimate (no uncertainty)."""
        return cls(
            distribution_type=DistributionType.POINT,
            mean=value,
            std=0.0,
        )
    
    @classmethod
    def from_mean_std(
        cls,
        mean: float,
        std: float,
    ) -> "ConfidenceDistribution":
        """Create from mean and standard deviation."""
        beta_dist = BetaDistribution.from_mean_std(mean, std)
        return cls(
            distribution_type=DistributionType.BETA,
            beta=beta_dist,
            mean=beta_dist.mean,
            std=beta_dist.std,
        )
    
    def sample(self, n: int = 1) -> list[float]:
        """Draw samples from distribution."""
        if self.distribution_type == DistributionType.POINT:
            return [self.mean] * n
        elif self.beta:
            return self.beta.sample(n)
        else:
            import numpy as np
            samples = np.random.normal(self.mean, self.std, n)
            return list(np.clip(samples, 0, 1))
    
    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Compute credible interval."""
        if self.distribution_type == DistributionType.POINT:
            return (self.mean, self.mean)
        elif self.beta:
            return self.beta.credible_interval(level)
        else:
            from scipy import stats
            z = stats.norm.ppf((1 + level) / 2)
            lower = max(0, self.mean - z * self.std)
            upper = min(1, self.mean + z * self.std)
            return (lower, upper)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.distribution_type.value,
            "mean": self.mean,
            "std": self.std,
        }
        if self.beta:
            result["alpha"] = self.beta.alpha
            result["beta"] = self.beta.beta
        return result


@dataclass
class ChallengerBid:
    """
    Bid submitted by an agent to challenge a Claim Bundle.
    
    Attributes:
        bid_id: Unique bid identifier
        bidder_agent: Agent submitting the bid
        cb_id: Claim Bundle being challenged
        estimated_dfs: Estimated Dialectical Fitness Score
        domain_confidence: Confidence in domain expertise
        adversarial_strategy: Strategy for challenging
        bid_expires: Expiration timestamp
        submitted_at: Submission timestamp
    """
    bid_id: str = field(default_factory=lambda: generate_crux_id("bid"))
    bidder_agent: str = ""
    cb_id: str = ""
    estimated_dfs: float = 0.0
    domain_confidence: float = 0.5
    adversarial_strategy: str = "general_critique"
    bid_expires: Optional[datetime] = None
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.bid_expires is None:
            self.bid_expires = self.submitted_at + timedelta(minutes=5)
    
    @property
    def is_expired(self) -> bool:
        """Check if bid has expired."""
        return datetime.utcnow() > self.bid_expires
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bid_id": self.bid_id,
            "bidder_agent": self.bidder_agent,
            "cb_id": self.cb_id,
            "estimated_dfs": self.estimated_dfs,
            "domain_confidence": self.domain_confidence,
            "adversarial_strategy": self.adversarial_strategy,
            "bid_expires": self.bid_expires.isoformat() if self.bid_expires else None,
            "submitted_at": self.submitted_at.isoformat(),
        }


@dataclass
class BRPResolution:
    """
    Result of Belief Reconciliation Protocol.
    
    Attributes:
        resolution_id: Unique resolution identifier
        original_cb_ids: IDs of original conflicting Claim Bundles
        reconciled_posterior: Merged posterior probability
        reconciled_distribution: Merged confidence distribution
        contributor_agents: Agents that contributed to merge
        contributor_weights: Credibility weights used
        lineage_hash: New lineage hash after merge
        ancestor_hashes: Original lineage hashes preserved
        rounds_completed: Number of BRP rounds
        resolution_timestamp: When resolution was completed
    """
    resolution_id: str = field(default_factory=lambda: generate_crux_id("brp"))
    original_cb_ids: list[str] = field(default_factory=list)
    reconciled_posterior: float = 0.5
    reconciled_distribution: Optional[ConfidenceDistribution] = None
    contributor_agents: list[str] = field(default_factory=list)
    contributor_weights: list[float] = field(default_factory=list)
    lineage_hash: str = ""
    ancestor_hashes: list[str] = field(default_factory=list)
    rounds_completed: int = 0
    resolution_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resolution_id": self.resolution_id,
            "original_cb_ids": self.original_cb_ids,
            "reconciled_posterior": self.reconciled_posterior,
            "reconciled_distribution": (
                self.reconciled_distribution.to_dict()
                if self.reconciled_distribution else None
            ),
            "contributor_agents": self.contributor_agents,
            "contributor_weights": self.contributor_weights,
            "lineage_hash": self.lineage_hash,
            "ancestor_hashes": self.ancestor_hashes,
            "rounds_completed": self.rounds_completed,
            "resolution_timestamp": self.resolution_timestamp.isoformat(),
        }


@dataclass
class EDRDelta:
    """
    Delta bundle for Epistemic Dead Reckoning synchronization.
    
    Contains all changes since a checkpoint for agent reconnection.
    
    Attributes:
        delta_id: Unique delta identifier  
        checkpoint_hash: Hash of checkpoint being synced from
        claim_bundles: New Claim Bundles since checkpoint
        resolutions: BRP resolutions that occurred
        credibility_updates: Credibility changes since checkpoint
        current_head_hash: Current ledger head hash
        timestamp: When delta was computed
    """
    delta_id: str = field(default_factory=lambda: generate_crux_id("edr"))
    checkpoint_hash: str = ""
    claim_bundles: list[Any] = field(default_factory=list)
    resolutions: list[BRPResolution] = field(default_factory=list)
    credibility_updates: list[Any] = field(default_factory=list)
    current_head_hash: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def num_changes(self) -> int:
        """Total number of changes in delta."""
        return (
            len(self.claim_bundles) +
            len(self.resolutions) +
            len(self.credibility_updates)
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "delta_id": self.delta_id,
            "checkpoint_hash": self.checkpoint_hash,
            "num_claim_bundles": len(self.claim_bundles),
            "num_resolutions": len(self.resolutions),
            "num_credibility_updates": len(self.credibility_updates),
            "current_head_hash": self.current_head_hash,
            "timestamp": self.timestamp.isoformat(),
        }


class CRUXConfig(BaseModel):
    """
    Configuration for CRUX protocol.
    
    Attributes:
        crux_version: Protocol version
        contradiction_threshold: Threshold for triggering BRP (θ)
        credibility_recency_weight: Weight for credibility decay (λ)
        auction_timeout_seconds: Timeout for Challenger Auction
        min_credibility_floor: Minimum credibility before suspension
        brp_max_rounds: Maximum BRP mini-debate rounds
        dfs_weights: Weights for DFS calculation components
        enable_edr: Enable Epistemic Dead Reckoning
        signature_required: Require Ed25519 signatures
    """
    
    crux_version: CRUXVersion = Field(
        default=CRUXVersion.V1_0,
        description="CRUX protocol version",
    )
    
    contradiction_threshold: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="Threshold for BRP trigger (θ)",
    )
    
    credibility_recency_weight: float = Field(
        default=0.15,
        ge=0.01,
        le=0.50,
        description="Credibility EMA weight (λ)",
    )
    
    auction_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Challenger Auction timeout",
    )
    
    min_credibility_floor: float = Field(
        default=0.30,
        ge=0.1,
        le=0.5,
        description="Minimum credibility before suspension",
    )
    
    brp_max_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum BRP mini-debate rounds",
    )
    
    dfs_domain_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="DFS weight for domain match (w1)",
    )
    
    dfs_adversarial_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="DFS weight for adversarial potential (w2)",
    )
    
    dfs_credibility_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="DFS weight for credibility (w3)",
    )
    
    dfs_recency_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="DFS weight for recency (w4)",
    )
    
    enable_edr: bool = Field(
        default=True,
        description="Enable Epistemic Dead Reckoning",
    )
    
    signature_required: bool = Field(
        default=False,
        description="Require Ed25519 signatures on Claim Bundles",
    )
    
    ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Default TTL for Claim Bundles",
    )
    
    def validate_dfs_weights(self) -> bool:
        """Validate that DFS weights sum to approximately 1.0."""
        total = (
            self.dfs_domain_weight +
            self.dfs_adversarial_weight +
            self.dfs_credibility_weight +
            self.dfs_recency_weight
        )
        return abs(total - 1.0) < 0.01
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "crux_version": self.crux_version.value,
            "contradiction_threshold": self.contradiction_threshold,
            "credibility_recency_weight": self.credibility_recency_weight,
            "auction_timeout_seconds": self.auction_timeout_seconds,
            "min_credibility_floor": self.min_credibility_floor,
            "brp_max_rounds": self.brp_max_rounds,
            "dfs_weights": {
                "domain": self.dfs_domain_weight,
                "adversarial": self.dfs_adversarial_weight,
                "credibility": self.dfs_credibility_weight,
                "recency": self.dfs_recency_weight,
            },
            "enable_edr": self.enable_edr,
            "signature_required": self.signature_required,
            "ttl_seconds": self.ttl_seconds,
        }
