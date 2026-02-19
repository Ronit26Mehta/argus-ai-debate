"""
Claim Bundle (CB) for CRUX Protocol.

The Claim Bundle is the atomic unit of CRUX exchange. It is not a task,
not a message, and not a tool call — it is a belief, fully specified
with its uncertainty distribution and argument ancestry.

Key Features:
    - Full Beta distribution for uncertainty
    - Argument lineage tracking via hash chains
    - Challenge status and deadlines
    - Issuer credibility attachment
    - Ed25519 signature support
"""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, computed_field

from argus.crux.models import (
    generate_crux_id,
    compute_crux_hash,
    Polarity,
    ConfidenceDistribution,
    BetaDistribution,
)

if TYPE_CHECKING:
    from argus.cdag.nodes import Evidence, Proposition
    from argus.crux.agent_card import EpistemicAgentCard

logger = logging.getLogger(__name__)


class ClaimBundle(BaseModel):
    """
    Claim Bundle (CB) - CRUX Primitive 2.
    
    The atomic unit of epistemic exchange in CRUX. Each Claim Bundle
    carries a belief with its full uncertainty distribution and
    argument ancestry.
    
    Attributes:
        cb_id: Unique Claim Bundle identifier (SHA-256 based)
        schema: Schema identifier for versioning
        claim_text: The textual claim being made
        proposition_id: Related proposition ID
        polarity: Support (+1), Attack (-1), or Neutral (0)
        prior: Prior probability before evidence
        posterior: Computed posterior probability
        confidence_distribution: Full parametric uncertainty
        evidence_refs: List of evidence IDs supporting this claim
        argument_lineage_hash: Hash of argument ancestry
        issuer_agent: Agent that issued this bundle
        issuer_credibility: Credibility rating of issuer
        challenge_open: Whether claim is open for challenge
        challenge_deadline_utc: Deadline for challenges
        issued_at: When the bundle was issued
        ttl_seconds: Time-to-live for the bundle
        signature: Ed25519 signature (optional)
        contributors: List of contributing agents (for merged bundles)
        metadata: Additional claim-specific data
        
    Example:
        >>> cb = ClaimBundle(
        ...     claim_text="Treatment X reduces symptoms by more than 20%",
        ...     proposition_id="prop_abc123",
        ...     polarity=Polarity.SUPPORTS,
        ...     prior=0.50,
        ...     posterior=0.73,
        ...     confidence_distribution=ConfidenceDistribution.from_beta(7.3, 2.7),
        ...     issuer_agent="specialist-oncology-001",
        ...     challenge_open=True,
        ... )
    """
    
    model_config = {"frozen": False}  # Allow challenge status updates
    
    cb_id: str = Field(
        default="",
        description="Unique Claim Bundle identifier",
    )
    
    schema: str = Field(
        default="crux/claim-bundle/1.0",
        description="Schema identifier",
    )
    
    claim_text: str = Field(
        min_length=1,
        description="The claim being made",
    )
    
    proposition_id: str = Field(
        default="",
        description="Related proposition ID",
    )
    
    polarity: Polarity = Field(
        default=Polarity.NEUTRAL,
        description="Claim polarity",
    )
    
    prior: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Prior probability",
    )
    
    posterior: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Posterior probability",
    )
    
    confidence_distribution: ConfidenceDistribution = Field(
        default_factory=lambda: ConfidenceDistribution.from_beta(1.0, 1.0),
        description="Full uncertainty distribution",
    )
    
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Supporting evidence IDs",
    )
    
    argument_lineage_hash: str = Field(
        default="",
        description="Hash of argument ancestry",
    )
    
    issuer_agent: str = Field(
        default="",
        description="Agent that issued this bundle",
    )
    
    issuer_credibility: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Credibility of issuer",
    )
    
    challenge_open: bool = Field(
        default=True,
        description="Whether open for challenge",
    )
    
    challenge_deadline_utc: Optional[datetime] = Field(
        default=None,
        description="Challenge deadline",
    )
    
    issued_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Issue timestamp",
    )
    
    ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Time-to-live",
    )
    
    signature: Optional[str] = Field(
        default=None,
        description="Ed25519 signature",
    )
    
    contributors: list[str] = Field(
        default_factory=list,
        description="Contributing agents (for merged bundles)",
    )
    
    nonce: str = Field(
        default_factory=lambda: generate_crux_id("nonce"),
        description="Replay prevention nonce",
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    def __init__(self, **data):
        """Initialize and compute CB ID if not provided."""
        super().__init__(**data)
        
        # Compute CB ID from content hash if not provided
        if not self.cb_id:
            object.__setattr__(self, "cb_id", self._compute_cb_id())
        
        # Compute lineage hash if not provided
        if not self.argument_lineage_hash:
            object.__setattr__(
                self, 
                "argument_lineage_hash", 
                self._compute_lineage_hash()
            )
        
        # Set challenge deadline if not provided
        if self.challenge_open and not self.challenge_deadline_utc:
            deadline = datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
            object.__setattr__(self, "challenge_deadline_utc", deadline)
    
    def _compute_cb_id(self) -> str:
        """Compute CB ID from content hash."""
        content = json.dumps({
            "claim_text": self.claim_text,
            "proposition_id": self.proposition_id,
            "polarity": self.polarity.value,
            "posterior": self.posterior,
            "issuer_agent": self.issuer_agent,
            "nonce": self.nonce,
        }, sort_keys=True)
        hash_val = hashlib.sha256(content.encode()).hexdigest()
        return f"cb_{hash_val[:16]}"
    
    def _compute_lineage_hash(self) -> str:
        """Compute argument lineage hash."""
        content = json.dumps({
            "evidence_refs": sorted(self.evidence_refs),
            "issuer_agent": self.issuer_agent,
            "issued_at": self.issued_at.isoformat(),
        }, sort_keys=True)
        return compute_crux_hash(content)
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if bundle has expired."""
        expiry = self.issued_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry
    
    @computed_field
    @property
    def challenge_expired(self) -> bool:
        """Check if challenge window has closed."""
        if not self.challenge_deadline_utc:
            return True
        return datetime.utcnow() > self.challenge_deadline_utc
    
    @computed_field
    @property
    def is_supporting(self) -> bool:
        """Check if bundle supports proposition."""
        return self.polarity == Polarity.SUPPORTS
    
    @computed_field
    @property
    def is_attacking(self) -> bool:
        """Check if bundle attacks proposition."""
        return self.polarity == Polarity.ATTACKS
    
    @computed_field
    @property
    def effective_weight(self) -> float:
        """
        Compute effective weight for BRP merges.
        
        Combines posterior uncertainty with issuer credibility.
        """
        # Lower std = higher confidence = higher weight
        confidence_factor = max(0.1, 1 - self.confidence_distribution.std)
        return self.issuer_credibility * confidence_factor
    
    @computed_field
    @property
    def is_merged(self) -> bool:
        """Check if this is a merged/reconciled bundle."""
        return len(self.contributors) > 1
    
    def close_challenge(self) -> None:
        """Close the bundle for challenges."""
        object.__setattr__(self, "challenge_open", False)
    
    def mark_unchallenged(self) -> None:
        """Mark as unchallenged (no challengers responded)."""
        self.close_challenge()
        self.metadata["unchallenged"] = True
        self.metadata["unchallenged_at"] = datetime.utcnow().isoformat()
    
    def conflicts_with(
        self,
        other: "ClaimBundle",
        threshold: float = 0.20,
    ) -> bool:
        """
        Check if this bundle conflicts with another.
        
        Conflict occurs when:
        1. Same proposition AND
        2. Opposite polarity OR posterior divergence > threshold
        
        Args:
            other: Another ClaimBundle
            threshold: Posterior divergence threshold (θ)
            
        Returns:
            True if bundles conflict
        """
        if self.proposition_id != other.proposition_id:
            return False
        
        # Check polarity conflict
        if self.polarity != other.polarity and self.polarity != Polarity.NEUTRAL:
            return True
        
        # Check posterior divergence
        divergence = abs(self.posterior - other.posterior)
        return divergence > threshold
    
    def verify_signature(self, public_key: str) -> bool:
        """
        Verify Ed25519 signature.
        
        Args:
            public_key: Ed25519 public key
            
        Returns:
            True if signature is valid
        """
        if not self.signature:
            return False
        
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ed25519
            
            # Parse public key
            pub_bytes = bytes.fromhex(public_key.replace("ed25519:", ""))
            pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            
            # Get content to verify
            content = self._get_signable_content()
            sig_bytes = bytes.fromhex(self.signature.replace("ed25519:", ""))
            
            pub_key.verify(sig_bytes, content.encode())
            return True
            
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False
    
    def _get_signable_content(self) -> str:
        """Get content for signing/verification."""
        return json.dumps({
            "cb_id": self.cb_id,
            "claim_text": self.claim_text,
            "proposition_id": self.proposition_id,
            "polarity": self.polarity.value,
            "posterior": self.posterior,
            "issuer_agent": self.issuer_agent,
            "nonce": self.nonce,
        }, sort_keys=True)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (CRUX wire format)."""
        return {
            "cb_id": self.cb_id,
            "schema": self.schema,
            "claim_text": self.claim_text,
            "proposition_id": self.proposition_id,
            "polarity": self.polarity.value,
            "prior": self.prior,
            "posterior": self.posterior,
            "confidence_distribution": self.confidence_distribution.to_dict(),
            "evidence_refs": self.evidence_refs,
            "argument_lineage_hash": self.argument_lineage_hash,
            "issuer_agent": self.issuer_agent,
            "issuer_credibility": self.issuer_credibility,
            "challenge_open": self.challenge_open,
            "challenge_deadline_utc": (
                self.challenge_deadline_utc.isoformat()
                if self.challenge_deadline_utc else None
            ),
            "issued_at": self.issued_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "signature": self.signature,
            "contributors": self.contributors,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimBundle":
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ClaimBundle instance
        """
        # Parse nested objects
        conf_dist = ConfidenceDistribution(**data.get("confidence_distribution", {}))
        
        # Parse timestamps
        issued_at = datetime.fromisoformat(data["issued_at"])
        challenge_deadline = None
        if data.get("challenge_deadline_utc"):
            challenge_deadline = datetime.fromisoformat(data["challenge_deadline_utc"])
        
        return cls(
            cb_id=data["cb_id"],
            schema=data.get("schema", "crux/claim-bundle/1.0"),
            claim_text=data["claim_text"],
            proposition_id=data["proposition_id"],
            polarity=Polarity(data["polarity"]),
            prior=data["prior"],
            posterior=data["posterior"],
            confidence_distribution=conf_dist,
            evidence_refs=data.get("evidence_refs", []),
            argument_lineage_hash=data.get("argument_lineage_hash", ""),
            issuer_agent=data["issuer_agent"],
            issuer_credibility=data.get("issuer_credibility", 0.5),
            challenge_open=data.get("challenge_open", True),
            challenge_deadline_utc=challenge_deadline,
            issued_at=issued_at,
            ttl_seconds=data.get("ttl_seconds", 3600),
            signature=data.get("signature"),
            contributors=data.get("contributors", []),
        )


class ClaimBundleFactory:
    """
    Factory for creating Claim Bundles from ARGUS entities.
    
    Provides convenient methods to convert ARGUS Evidence and
    Proposition nodes to CRUX Claim Bundles.
    """
    
    @staticmethod
    def from_evidence(
        evidence: "Evidence",
        proposition_id: str,
        agent_card: "EpistemicAgentCard",
        prior: float = 0.5,
    ) -> ClaimBundle:
        """
        Create Claim Bundle from ARGUS Evidence node.
        
        Args:
            evidence: ARGUS Evidence node
            proposition_id: Related proposition ID
            agent_card: EAC of issuing agent
            prior: Prior probability
            
        Returns:
            ClaimBundle instance
        """
        # Determine polarity
        if hasattr(evidence, 'polarity'):
            polarity = (
                Polarity.SUPPORTS if evidence.polarity > 0
                else Polarity.ATTACKS if evidence.polarity < 0
                else Polarity.NEUTRAL
            )
        else:
            polarity = Polarity.NEUTRAL
        
        # Create confidence distribution from evidence confidence
        confidence = getattr(evidence, 'confidence', 0.5)
        
        # Estimate alpha/beta from confidence
        # Higher confidence = more concentrated around mean
        concentration = 10 + 20 * abs(confidence - 0.5)
        conf_dist = ConfidenceDistribution.from_beta(
            alpha=confidence * concentration,
            beta=(1 - confidence) * concentration,
        )
        
        return ClaimBundle(
            claim_text=evidence.text,
            proposition_id=proposition_id,
            polarity=polarity,
            prior=prior,
            posterior=confidence,
            confidence_distribution=conf_dist,
            evidence_refs=[evidence.id],
            issuer_agent=agent_card.agent_id,
            issuer_credibility=agent_card.calibration.credibility_rating,
            challenge_open=True,
            metadata={
                "evidence_type": getattr(evidence, 'evidence_type', 'unknown'),
                "relevance": getattr(evidence, 'relevance', 1.0),
                "quality": getattr(evidence, 'quality', 1.0),
            },
        )
    
    @staticmethod
    def from_proposition_posterior(
        proposition: "Proposition",
        agent_card: "EpistemicAgentCard",
        evidence_refs: Optional[list[str]] = None,
    ) -> ClaimBundle:
        """
        Create Claim Bundle from Proposition posterior.
        
        Args:
            proposition: ARGUS Proposition node
            agent_card: EAC of issuing agent
            evidence_refs: Optional list of evidence IDs
            
        Returns:
            ClaimBundle instance
        """
        posterior = getattr(proposition, 'posterior', proposition.prior)
        
        # Determine polarity from posterior relative to prior
        if posterior > 0.5 + 0.1:
            polarity = Polarity.SUPPORTS
        elif posterior < 0.5 - 0.1:
            polarity = Polarity.ATTACKS
        else:
            polarity = Polarity.NEUTRAL
        
        # Create distribution
        concentration = 20.0  # Moderate confidence
        conf_dist = ConfidenceDistribution.from_beta(
            alpha=posterior * concentration,
            beta=(1 - posterior) * concentration,
        )
        
        return ClaimBundle(
            claim_text=proposition.text,
            proposition_id=proposition.id,
            polarity=polarity,
            prior=proposition.prior,
            posterior=posterior,
            confidence_distribution=conf_dist,
            evidence_refs=evidence_refs or [],
            issuer_agent=agent_card.agent_id,
            issuer_credibility=agent_card.calibration.credibility_rating,
            challenge_open=True,
        )


def merge_claim_bundles(
    bundles: list[ClaimBundle],
    credibility_weights: Optional[list[float]] = None,
) -> ClaimBundle:
    """
    Merge multiple Claim Bundles into a reconciled bundle.
    
    Uses credibility-weighted Bayesian fusion to merge posteriors
    and moment-matching to combine distributions.
    
    Args:
        bundles: List of Claim Bundles to merge
        credibility_weights: Optional weights (uses issuer_credibility if None)
        
    Returns:
        Merged ClaimBundle
    """
    if not bundles:
        raise ValueError("No bundles to merge")
    
    if len(bundles) == 1:
        return bundles[0]
    
    # Use credibility weights if not provided
    if credibility_weights is None:
        credibility_weights = [b.issuer_credibility for b in bundles]
    
    # Normalize weights
    total_weight = sum(credibility_weights)
    weights = [w / total_weight for w in credibility_weights]
    
    # Weighted mean posterior
    merged_posterior = sum(
        w * b.posterior for w, b in zip(weights, bundles)
    )
    
    # Weighted mean prior
    merged_prior = sum(
        w * b.prior for w, b in zip(weights, bundles)
    )
    
    # Merge distributions via moment matching
    # Weighted mean
    merged_mean = sum(
        w * b.confidence_distribution.mean 
        for w, b in zip(weights, bundles)
    )
    
    # Weighted variance (including between-component variance)
    within_var = sum(
        w * (b.confidence_distribution.std ** 2)
        for w, b in zip(weights, bundles)
    )
    between_var = sum(
        w * ((b.confidence_distribution.mean - merged_mean) ** 2)
        for w, b in zip(weights, bundles)
    )
    merged_std = (within_var + between_var) ** 0.5
    
    # Create merged distribution
    merged_dist = ConfidenceDistribution.from_mean_std(merged_mean, merged_std)
    
    # Collect evidence refs and contributors
    all_evidence = []
    all_contributors = []
    ancestor_hashes = []
    
    for b in bundles:
        all_evidence.extend(b.evidence_refs)
        if b.issuer_agent not in all_contributors:
            all_contributors.append(b.issuer_agent)
        ancestor_hashes.append(b.argument_lineage_hash)
    
    # Determine consensus polarity
    polarity_votes = {}
    for w, b in zip(weights, bundles):
        pol = b.polarity
        polarity_votes[pol] = polarity_votes.get(pol, 0) + w
    
    consensus_polarity = max(polarity_votes.keys(), key=lambda p: polarity_votes[p])
    
    # Average credibility of contributors
    merged_credibility = sum(credibility_weights) / len(credibility_weights)
    
    return ClaimBundle(
        claim_text=bundles[0].claim_text,  # Use first bundle's text
        proposition_id=bundles[0].proposition_id,
        polarity=consensus_polarity,
        prior=merged_prior,
        posterior=merged_posterior,
        confidence_distribution=merged_dist,
        evidence_refs=list(set(all_evidence)),
        issuer_agent=f"brp-merge-{generate_crux_id('')[:8]}",
        issuer_credibility=merged_credibility,
        challenge_open=False,  # Merged bundles are final
        contributors=all_contributors,
        metadata={
            "merged_from": [b.cb_id for b in bundles],
            "ancestor_hashes": ancestor_hashes,
            "merge_weights": credibility_weights,
        },
    )
