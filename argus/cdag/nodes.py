"""
C-DAG Node Types.

Defines the node types for the Conceptual Debate Graph:
    - Proposition: Claims to be evaluated
    - Evidence: Supporting or attacking evidence
    - Rebuttal: Counter-arguments
    - Finding: Observations and facts
    - Assumption: Prior beliefs
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, field_validator, computed_field


def generate_node_id(prefix: str) -> str:
    """Generate a unique node identifier with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class NodeStatus(str, Enum):
    """Status of a node in the debate."""
    PENDING = "pending"           # Awaiting evaluation
    ACTIVE = "active"             # Under active consideration
    SUPPORTED = "supported"       # Has more support than attack
    CONTESTED = "contested"       # Has significant attack
    REJECTED = "rejected"         # Rejected by evidence
    ENDORSED = "endorsed"         # Promoted/accepted
    ARCHIVED = "archived"         # No longer active


class NodeBase(BaseModel):
    """
    Base class for all C-DAG nodes.
    
    Provides common attributes and methods used by all node types.
    
    Attributes:
        id: Unique node identifier
        text: Node content text
        confidence: Initial/calibrated confidence (0-1)
        weight: Importance weight for scoring
        status: Current status in debate
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional node-specific data
    """
    
    model_config = {"frozen": False}  # Allow status updates
    
    id: str = Field(
        description="Unique node identifier",
    )
    
    text: str = Field(
        min_length=1,
        description="Node content text",
    )
    
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)",
    )
    
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Importance weight",
    )
    
    status: NodeStatus = Field(
        default=NodeStatus.PENDING,
        description="Current status",
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp",
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    # Computed score from influence propagation
    _score: float = 0.5
    
    @property
    def score(self) -> float:
        """Get the computed influence score."""
        return self._score
    
    @score.setter
    def score(self, value: float) -> None:
        """Set the influence score."""
        self._score = max(0.0, min(1.0, value))
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is bounded."""
        return max(0.0, min(1.0, v))
    
    def update_status(self, new_status: NodeStatus) -> None:
        """
        Update node status with timestamp.
        
        Args:
            new_status: New status to set
        """
        object.__setattr__(self, "status", new_status)
        object.__setattr__(self, "updated_at", datetime.utcnow())
    
    def __hash__(self) -> int:
        """Hash by node ID."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on node ID."""
        if not isinstance(other, NodeBase):
            return False
        return self.id == other.id


class Proposition(NodeBase):
    """
    A proposition (hypothesis) to be evaluated.
    
    Propositions are the central claims in the debate graph.
    Evidence nodes connect to propositions via support/attack edges.
    
    Attributes:
        prior: Prior probability (0-1)
        posterior: Computed posterior probability
        domain: Domain/topic area
        source_agent: Agent that proposed this
        
    Example:
        >>> prop = Proposition(
        ...     text="Drug X reduces biomarker Y by >20%",
        ...     prior=0.5,
        ...     domain="clinical",
        ... )
    """
    
    id: str = Field(
        default_factory=lambda: generate_node_id("prop"),
        description="Proposition ID",
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
    
    domain: Optional[str] = Field(
        default=None,
        description="Domain/topic area",
    )
    
    source_agent: Optional[str] = Field(
        default=None,
        description="Agent that proposed this",
    )
    
    @computed_field
    @property
    def log_odds_prior(self) -> float:
        """Prior in log-odds space."""
        import math
        p = max(0.001, min(0.999, self.prior))
        return math.log(p / (1 - p))
    
    @computed_field
    @property
    def log_odds_posterior(self) -> float:
        """Posterior in log-odds space."""
        import math
        p = max(0.001, min(0.999, self.posterior))
        return math.log(p / (1 - p))
    
    def update_posterior(self, new_posterior: float) -> None:
        """
        Update the posterior probability.
        
        Args:
            new_posterior: New posterior value (0-1)
        """
        bounded = max(0.0, min(1.0, new_posterior))
        object.__setattr__(self, "posterior", bounded)
        object.__setattr__(self, "updated_at", datetime.utcnow())
        
        # Update status based on posterior
        if bounded >= 0.8:
            self.update_status(NodeStatus.ENDORSED)
        elif bounded >= 0.6:
            self.update_status(NodeStatus.SUPPORTED)
        elif bounded <= 0.2:
            self.update_status(NodeStatus.REJECTED)
        elif bounded <= 0.4:
            self.update_status(NodeStatus.CONTESTED)
        else:
            self.update_status(NodeStatus.ACTIVE)


class EvidenceType(str, Enum):
    """Type of evidence."""
    EMPIRICAL = "empirical"       # Data from experiments/observations
    LITERATURE = "literature"     # Published research
    EXPERT = "expert"             # Expert opinion
    LOGICAL = "logical"           # Logical argument
    COMPUTATIONAL = "computational"  # Computational/simulation results


class Evidence(NodeBase):
    """
    Evidence node supporting or attacking a proposition.
    
    Evidence nodes link to propositions via typed edges.
    Each evidence node carries citations and quality metrics.
    
    Attributes:
        evidence_type: Type of evidence
        polarity: Support (+1), Attack (-1), or Neutral (0)
        citations: Source citations
        relevance: Relevance to target proposition
        quality: Source quality score
        
    Example:
        >>> evidence = Evidence(
        ...     text="RCT showed 25% reduction (p<0.01)",
        ...     evidence_type=EvidenceType.EMPIRICAL,
        ...     polarity=1,
        ...     confidence=0.85,
        ... )
    """
    
    id: str = Field(
        default_factory=lambda: generate_node_id("evid"),
        description="Evidence ID",
    )
    
    evidence_type: EvidenceType = Field(
        default=EvidenceType.LITERATURE,
        description="Type of evidence",
    )
    
    polarity: int = Field(
        default=0,
        ge=-1,
        le=1,
        description="Support (+1), Attack (-1), Neutral (0)",
    )
    
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source citations",
    )
    
    relevance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance to target",
    )
    
    quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Source quality",
    )
    
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Source chunk IDs for provenance",
    )
    
    @computed_field
    @property
    def effective_weight(self) -> float:
        """
        Compute effective weight for scoring.
        
        weight = confidence × relevance × quality × base_weight
        """
        return self.confidence * self.relevance * self.quality * self.weight
    
    @computed_field
    @property
    def is_supporting(self) -> bool:
        """Check if this is supporting evidence."""
        return self.polarity > 0
    
    @computed_field
    @property
    def is_attacking(self) -> bool:
        """Check if this is attacking evidence."""
        return self.polarity < 0
    
    def add_citation(
        self,
        doc_id: str,
        chunk_id: str,
        quote: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        """
        Add a citation to this evidence.
        
        Args:
            doc_id: Source document ID
            chunk_id: Source chunk ID
            quote: Direct quote (optional)
            url: Source URL (optional)
        """
        citation = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "quote": quote,
            "url": url,
            "added_at": datetime.utcnow().isoformat(),
        }
        self.citations.append(citation)
        self.chunk_ids.append(chunk_id)


class Rebuttal(NodeBase):
    """
    A rebuttal or counter-argument.
    
    Rebuttals attack other evidence or rebuttals,
    creating nested argumentation structure.
    
    Attributes:
        target_id: ID of the node being rebutted
        rebuttal_type: Type of rebuttal argument
        strength: Strength of the rebuttal
    """
    
    id: str = Field(
        default_factory=lambda: generate_node_id("rebt"),
        description="Rebuttal ID",
    )
    
    target_id: str = Field(
        description="ID of rebutted node",
    )
    
    rebuttal_type: str = Field(
        default="general",
        description="Type of rebuttal (methodological, logical, empirical)",
    )
    
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of rebuttal",
    )


class Finding(NodeBase):
    """
    A factual finding or observation.
    
    Findings are objective observations that can
    be used as evidence inputs.
    
    Attributes:
        finding_type: Type of finding
        source: Where the finding came from
        verified: Whether the finding has been verified
    """
    
    id: str = Field(
        default_factory=lambda: generate_node_id("find"),
        description="Finding ID",
    )
    
    finding_type: str = Field(
        default="observation",
        description="Type of finding",
    )
    
    source: Optional[str] = Field(
        default=None,
        description="Source of finding",
    )
    
    verified: bool = Field(
        default=False,
        description="Whether verified",
    )


class Assumption(NodeBase):
    """
    An assumption or prior belief.
    
    Assumptions are explicit premises that underlie
    the argument structure.
    
    Attributes:
        assumption_type: Type of assumption
        justification: Justification for the assumption
        can_be_tested: Whether assumption can be tested
    """
    
    id: str = Field(
        default_factory=lambda: generate_node_id("asmp"),
        description="Assumption ID",
    )
    
    assumption_type: str = Field(
        default="general",
        description="Type of assumption",
    )
    
    justification: Optional[str] = Field(
        default=None,
        description="Justification for assumption",
    )
    
    can_be_tested: bool = Field(
        default=True,
        description="Whether testable",
    )
