"""
Epistemic Agent Card (EAC) for CRUX Protocol.

The Epistemic Agent Card extends A2A's Agent Card concept with
belief-domain declarations and calibration metadata. Every agent
participating in a CRUX network publishes an EAC.

Key Features:
    - Belief domain declarations
    - Calibration metrics (Brier score, ECE, credibility rating)
    - Capability declarations (emit claims, challenge, render verdicts)
    - CRUX endpoint information
    - Ledger public key for signature verification
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, computed_field

from argus.crux.models import generate_crux_id, compute_crux_hash

if TYPE_CHECKING:
    from argus.agents.base import BaseAgent, AgentConfig
    from argus.decision.calibration import CalibrationMetrics

logger = logging.getLogger(__name__)


class AgentCalibration(BaseModel):
    """
    Calibration metrics for an agent.
    
    These metrics are derived from the agent's history in the
    Credibility Ledger and updated after every debate session.
    
    Attributes:
        brier_score: Average Brier score (lower is better)
        ece: Expected Calibration Error
        credibility_rating: Overall credibility (0-1)
        sample_size: Number of predictions evaluated
        last_updated: When metrics were last computed
    """
    
    brier_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Average Brier score",
    )
    
    ece: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Expected Calibration Error",
    )
    
    credibility_rating: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Overall credibility rating",
    )
    
    sample_size: int = Field(
        default=0,
        ge=0,
        description="Number of evaluated predictions",
    )
    
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp",
    )
    
    @computed_field
    @property
    def is_well_calibrated(self) -> bool:
        """Check if agent is well-calibrated (ECE < 0.10)."""
        return self.ece < 0.10
    
    @computed_field
    @property
    def is_credible(self) -> bool:
        """Check if agent has sufficient credibility (> 0.50)."""
        return self.credibility_rating > 0.50
    
    @computed_field
    @property
    def effective_weight(self) -> float:
        """
        Compute effective weight for BRP merges.
        
        Combines credibility with calibration quality.
        """
        calibration_factor = max(0.1, 1 - self.ece)
        return self.credibility_rating * calibration_factor
    
    @classmethod
    def neutral(cls) -> "AgentCalibration":
        """Create neutral calibration for new agents."""
        return cls(
            brier_score=0.25,
            ece=0.10,
            credibility_rating=0.50,
            sample_size=0,
            last_updated=datetime.utcnow(),
        )
    
    @classmethod
    def from_metrics(
        cls,
        metrics: "CalibrationMetrics",
        current_credibility: float = 0.50,
    ) -> "AgentCalibration":
        """
        Create from ARGUS CalibrationMetrics.
        
        Args:
            metrics: ARGUS calibration metrics
            current_credibility: Current credibility rating
            
        Returns:
            AgentCalibration instance
        """
        return cls(
            brier_score=metrics.brier_score,
            ece=metrics.ece,
            credibility_rating=current_credibility,
            sample_size=metrics.num_samples,
            last_updated=datetime.utcnow(),
        )
    
    def update(
        self,
        new_brier: float,
        recency_weight: float = 0.15,
    ) -> "AgentCalibration":
        """
        Update calibration with new observation.
        
        Uses exponential moving average for recency weighting.
        
        Args:
            new_brier: Brier contribution from new prediction
            recency_weight: Weight for new observation (λ)
            
        Returns:
            Updated AgentCalibration
        """
        # Update Brier score
        new_brier_avg = (
            (1 - recency_weight) * self.brier_score +
            recency_weight * new_brier
        )
        
        # Update credibility: higher Brier = lower credibility
        new_credibility = (
            (1 - recency_weight) * self.credibility_rating +
            recency_weight * (1 - new_brier)
        )
        
        return AgentCalibration(
            brier_score=new_brier_avg,
            ece=self.ece,  # ECE needs full re-computation
            credibility_rating=max(0.0, min(1.0, new_credibility)),
            sample_size=self.sample_size + 1,
            last_updated=datetime.utcnow(),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "brier_score": self.brier_score,
            "ece": self.ece,
            "credibility_rating": self.credibility_rating,
            "sample_size": self.sample_size,
            "last_updated": (
                self.last_updated.isoformat() 
                if self.last_updated else None
            ),
        }


class AgentCapabilities(BaseModel):
    """
    Capabilities declared by an agent.
    
    Determines what actions the agent can perform in CRUX exchanges.
    
    Attributes:
        emit_claims: Can emit Claim Bundles
        challenge_claims: Can challenge/rebut claims
        render_verdicts: Can render final verdicts
        max_rounds: Maximum debate rounds supported
        supported_domains: List of supported belief domains
    """
    
    emit_claims: bool = Field(
        default=True,
        description="Can emit Claim Bundles",
    )
    
    challenge_claims: bool = Field(
        default=True,
        description="Can challenge claims",
    )
    
    render_verdicts: bool = Field(
        default=False,
        description="Can render final verdicts",
    )
    
    max_rounds: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum debate rounds",
    )
    
    supported_domains: list[str] = Field(
        default_factory=list,
        description="Domains agent can handle",
    )
    
    @classmethod
    def specialist(cls, domains: list[str]) -> "AgentCapabilities":
        """Create capabilities for a Specialist agent."""
        return cls(
            emit_claims=True,
            challenge_claims=False,
            render_verdicts=False,
            max_rounds=5,
            supported_domains=domains,
        )
    
    @classmethod
    def refuter(cls, domains: list[str]) -> "AgentCapabilities":
        """Create capabilities for a Refuter agent."""
        return cls(
            emit_claims=True,
            challenge_claims=True,
            render_verdicts=False,
            max_rounds=5,
            supported_domains=domains,
        )
    
    @classmethod
    def jury(cls) -> "AgentCapabilities":
        """Create capabilities for a Jury agent."""
        return cls(
            emit_claims=False,
            challenge_claims=False,
            render_verdicts=True,
            max_rounds=1,
            supported_domains=["all"],
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emit_claims": self.emit_claims,
            "challenge_claims": self.challenge_claims,
            "render_verdicts": self.render_verdicts,
            "max_rounds": self.max_rounds,
            "supported_domains": self.supported_domains,
        }


class EpistemicAgentCard(BaseModel):
    """
    Epistemic Agent Card (EAC) - CRUX Primitive 1.
    
    Every agent that participates in a CRUX network publishes an
    Epistemic Agent Card — a JSON document that extends A2A's Agent
    Card with belief-domain declarations and calibration metadata.
    
    Attributes:
        crux_version: CRUX protocol version
        agent_id: Unique agent identifier
        agent_type: Type of agent (Specialist, Refuter, Jury, etc.)
        agent_name: Human-readable name
        belief_domains: Domains of expertise
        calibration: Calibration metrics
        capabilities: Agent capabilities
        crux_endpoint: CRUX protocol endpoint
        ledger_pubkey: Public key for signature verification
        created_at: Card creation timestamp
        expires_at: Card expiration timestamp
        metadata: Additional metadata
        
    Example:
        >>> card = EpistemicAgentCard(
        ...     agent_id="specialist-oncology-001",
        ...     agent_type="Specialist",
        ...     belief_domains=["oncology", "clinical-trials"],
        ...     calibration=AgentCalibration(brier_score=0.112, credibility_rating=0.91),
        ... )
        >>> card_json = card.to_json()
    """
    
    crux_version: str = Field(
        default="1.0",
        description="CRUX protocol version",
    )
    
    agent_id: str = Field(
        default_factory=lambda: generate_crux_id("agent"),
        description="Unique agent identifier",
    )
    
    agent_type: str = Field(
        default="Specialist",
        description="Type of agent",
    )
    
    agent_name: str = Field(
        default="",
        description="Human-readable agent name",
    )
    
    belief_domains: list[str] = Field(
        default_factory=list,
        description="Domains of expertise",
    )
    
    calibration: AgentCalibration = Field(
        default_factory=AgentCalibration.neutral,
        description="Calibration metrics",
    )
    
    capabilities: AgentCapabilities = Field(
        default_factory=AgentCapabilities,
        description="Agent capabilities",
    )
    
    crux_endpoint: Optional[str] = Field(
        default=None,
        description="CRUX protocol endpoint",
    )
    
    ledger_pubkey: Optional[str] = Field(
        default=None,
        description="Ed25519 public key for signatures",
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Card creation timestamp",
    )
    
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Card expiration timestamp",
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if card has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @computed_field
    @property
    def card_hash(self) -> str:
        """Compute hash of card for verification."""
        content = json.dumps({
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "belief_domains": self.belief_domains,
            "calibration": self.calibration.to_dict(),
            "capabilities": self.capabilities.to_dict(),
        }, sort_keys=True)
        return compute_crux_hash(content)
    
    @classmethod
    def from_argus_agent(
        cls,
        agent: "BaseAgent",
        belief_domains: Optional[list[str]] = None,
        crux_endpoint: Optional[str] = None,
    ) -> "EpistemicAgentCard":
        """
        Create EAC from existing ARGUS agent.
        
        Args:
            agent: ARGUS BaseAgent instance
            belief_domains: Optional domain override
            crux_endpoint: CRUX endpoint URL
            
        Returns:
            EpistemicAgentCard instance
        """
        from argus.agents.base import AgentRole
        
        # Determine agent type
        agent_type = agent.role.value.capitalize()
        
        # Determine capabilities based on role
        if agent.role == AgentRole.SPECIALIST:
            capabilities = AgentCapabilities.specialist(belief_domains or [])
        elif agent.role == AgentRole.REFUTER:
            capabilities = AgentCapabilities.refuter(belief_domains or [])
        elif agent.role == AgentRole.JURY:
            capabilities = AgentCapabilities.jury()
        else:
            capabilities = AgentCapabilities()
        
        # Get domain from specialist config if available
        if belief_domains is None:
            belief_domains = []
            if hasattr(agent.config, 'domain'):
                belief_domains = [agent.config.domain]
        
        return cls(
            agent_id=f"argus-{agent_type.lower()}-{generate_crux_id('')[:8]}",
            agent_type=agent_type,
            agent_name=agent.name,
            belief_domains=belief_domains,
            calibration=AgentCalibration.neutral(),
            capabilities=capabilities,
            crux_endpoint=crux_endpoint,
        )
    
    def can_handle_domain(self, domain: str) -> bool:
        """
        Check if agent can handle a specific domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if agent can handle domain
        """
        if "all" in self.belief_domains:
            return True
        
        domain_lower = domain.lower()
        for d in self.belief_domains:
            if d.lower() == domain_lower:
                return True
            # Partial match for hierarchical domains
            if domain_lower.startswith(d.lower()) or d.lower().startswith(domain_lower):
                return True
        
        return False
    
    def domain_match_score(self, domains: list[str]) -> float:
        """
        Compute domain match score against a list of domains.
        
        Args:
            domains: List of domains to match against
            
        Returns:
            Match score in [0, 1]
        """
        if not domains:
            return 0.5  # Neutral
        
        if "all" in self.belief_domains:
            return 1.0
        
        matches = 0
        for domain in domains:
            if self.can_handle_domain(domain):
                matches += 1
        
        return matches / len(domains)
    
    def refresh_calibration(
        self,
        new_calibration: AgentCalibration,
    ) -> "EpistemicAgentCard":
        """
        Create new card with updated calibration.
        
        Args:
            new_calibration: Updated calibration metrics
            
        Returns:
            New EpistemicAgentCard with updated calibration
        """
        return EpistemicAgentCard(
            crux_version=self.crux_version,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            agent_name=self.agent_name,
            belief_domains=self.belief_domains,
            calibration=new_calibration,
            capabilities=self.capabilities,
            crux_endpoint=self.crux_endpoint,
            ledger_pubkey=self.ledger_pubkey,
            created_at=datetime.utcnow(),
            expires_at=self.expires_at,
            metadata=self.metadata,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (compatible with A2A Agent Card format)."""
        return {
            "crux_version": self.crux_version,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "belief_domains": self.belief_domains,
            "calibration": self.calibration.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "crux_endpoint": self.crux_endpoint,
            "ledger_pubkey": self.ledger_pubkey,
            "created_at": self.created_at.isoformat(),
            "expires_at": (
                self.expires_at.isoformat() 
                if self.expires_at else None
            ),
            "card_hash": self.card_hash,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "EpistemicAgentCard":
        """
        Parse from JSON string.
        
        Args:
            json_str: JSON representation of card
            
        Returns:
            EpistemicAgentCard instance
        """
        data = json.loads(json_str)
        
        # Parse nested objects
        calibration = AgentCalibration(**data.get("calibration", {}))
        capabilities = AgentCapabilities(**data.get("capabilities", {}))
        
        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"])
        expires_at = (
            datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at") else None
        )
        
        return cls(
            crux_version=data.get("crux_version", "1.0"),
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            agent_name=data.get("agent_name", ""),
            belief_domains=data.get("belief_domains", []),
            calibration=calibration,
            capabilities=capabilities,
            crux_endpoint=data.get("crux_endpoint"),
            ledger_pubkey=data.get("ledger_pubkey"),
            created_at=created_at,
            expires_at=expires_at,
            metadata=data.get("metadata", {}),
        )


class EACRegistry:
    """
    Registry for Epistemic Agent Cards.
    
    Maintains a collection of EACs for agents in the CRUX network.
    Supports lookup by ID, type, and domain.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._cards: dict[str, EpistemicAgentCard] = {}
        self._by_type: dict[str, set[str]] = {}
        self._by_domain: dict[str, set[str]] = {}
    
    def register(self, card: EpistemicAgentCard) -> None:
        """
        Register an agent card.
        
        Args:
            card: EpistemicAgentCard to register
        """
        self._cards[card.agent_id] = card
        
        # Index by type
        if card.agent_type not in self._by_type:
            self._by_type[card.agent_type] = set()
        self._by_type[card.agent_type].add(card.agent_id)
        
        # Index by domain
        for domain in card.belief_domains:
            if domain not in self._by_domain:
                self._by_domain[domain] = set()
            self._by_domain[domain].add(card.agent_id)
        
        logger.debug(f"Registered EAC: {card.agent_id} ({card.agent_type})")
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent card.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if found and removed
        """
        if agent_id not in self._cards:
            return False
        
        card = self._cards[agent_id]
        
        # Remove from indexes
        if card.agent_type in self._by_type:
            self._by_type[card.agent_type].discard(agent_id)
        
        for domain in card.belief_domains:
            if domain in self._by_domain:
                self._by_domain[domain].discard(agent_id)
        
        del self._cards[agent_id]
        logger.debug(f"Unregistered EAC: {agent_id}")
        return True
    
    def get(self, agent_id: str) -> Optional[EpistemicAgentCard]:
        """Get card by agent ID."""
        return self._cards.get(agent_id)
    
    def get_card(self, agent_id: str) -> Optional[EpistemicAgentCard]:
        """Get card by agent ID (alias for get)."""
        return self.get(agent_id)
    
    def get_by_type(self, agent_type: str) -> list[EpistemicAgentCard]:
        """Get all cards of a specific type."""
        agent_ids = self._by_type.get(agent_type, set())
        return [self._cards[aid] for aid in agent_ids if aid in self._cards]
    
    def get_by_domain(self, domain: str) -> list[EpistemicAgentCard]:
        """Get all cards handling a specific domain."""
        agent_ids = self._by_domain.get(domain, set())
        return [self._cards[aid] for aid in agent_ids if aid in self._cards]
    
    def get_challengers(
        self,
        exclude_agent: str,
        domain: Optional[str] = None,
    ) -> list[EpistemicAgentCard]:
        """
        Get agents capable of challenging claims.
        
        Args:
            exclude_agent: Agent ID to exclude (issuer)
            domain: Optional domain filter
            
        Returns:
            List of challenger-capable cards
        """
        challengers = []
        
        for card in self._cards.values():
            if card.agent_id == exclude_agent:
                continue
            if not card.capabilities.challenge_claims:
                continue
            if card.is_expired:
                continue
            if domain and not card.can_handle_domain(domain):
                continue
            
            challengers.append(card)
        
        return challengers
    
    def get_all(self) -> list[EpistemicAgentCard]:
        """Get all registered cards."""
        return list(self._cards.values())
    
    def __len__(self) -> int:
        """Number of registered cards."""
        return len(self._cards)
    
    def __contains__(self, agent_id: str) -> bool:
        """Check if agent is registered."""
        return agent_id in self._cards
