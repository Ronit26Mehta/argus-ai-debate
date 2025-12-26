"""
C-DAG Edge Types.

Defines the edge types for the Conceptual Debate Graph:
    - supports: Evidence supports a proposition
    - attacks: Evidence attacks a proposition
    - refines: Refinement or clarification
    - suggests_test: Suggests an experiment/test
    - causes: Causal relationship
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, computed_field


def generate_edge_id() -> str:
    """Generate a unique edge identifier."""
    return f"edge_{uuid.uuid4().hex[:12]}"


class EdgeType(str, Enum):
    """Type of edge in the C-DAG."""
    SUPPORTS = "supports"           # Evidence supports proposition
    ATTACKS = "attacks"             # Evidence attacks proposition
    REFINES = "refines"             # Refinement/clarification
    SUGGESTS_TEST = "suggests_test" # Suggests an experiment
    CAUSES = "causes"               # Causal relationship
    REBUTS = "rebuts"               # Rebuttal attacks evidence
    UNDERMINES = "undermines"       # Undermines the connection
    DERIVED_FROM = "derived_from"   # Derived from another node


class EdgePolarity(str, Enum):
    """Polarity of an edge (positive/negative influence)."""
    POSITIVE = "positive"   # Positive influence (+1)
    NEGATIVE = "negative"   # Negative influence (-1)
    NEUTRAL = "neutral"     # No polarity (0)


# Mapping from edge type to default polarity
EDGE_TYPE_POLARITY = {
    EdgeType.SUPPORTS: EdgePolarity.POSITIVE,
    EdgeType.ATTACKS: EdgePolarity.NEGATIVE,
    EdgeType.REFINES: EdgePolarity.NEUTRAL,
    EdgeType.SUGGESTS_TEST: EdgePolarity.NEUTRAL,
    EdgeType.CAUSES: EdgePolarity.POSITIVE,
    EdgeType.REBUTS: EdgePolarity.NEGATIVE,
    EdgeType.UNDERMINES: EdgePolarity.NEGATIVE,
    EdgeType.DERIVED_FROM: EdgePolarity.NEUTRAL,
}


class Edge(BaseModel):
    """
    An edge connecting two nodes in the C-DAG.
    
    Edges carry typed relations with weights that influence
    the propagation of scores through the graph.
    
    Attributes:
        id: Unique edge identifier
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Type of relation
        polarity: Positive, negative, or neutral
        weight: Edge weight (0-1)
        confidence: Confidence in the relation
        relevance: Relevance of source to target
        quality: Quality/reliability score
        created_at: Creation timestamp
        metadata: Additional edge data
        
    Example:
        >>> edge = Edge(
        ...     source_id="evid_abc123",
        ...     target_id="prop_xyz789",
        ...     edge_type=EdgeType.SUPPORTS,
        ...     weight=0.8,
        ...     confidence=0.9,
        ... )
    """
    
    model_config = {"frozen": False}
    
    id: str = Field(
        default_factory=generate_edge_id,
        description="Unique edge identifier",
    )
    
    source_id: str = Field(
        description="Source node ID",
    )
    
    target_id: str = Field(
        description="Target node ID",
    )
    
    edge_type: EdgeType = Field(
        description="Type of relation",
    )
    
    polarity: EdgePolarity = Field(
        default=EdgePolarity.NEUTRAL,
        description="Edge polarity",
    )
    
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Edge weight (0-1)",
    )
    
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in relation",
    )
    
    relevance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance score",
    )
    
    quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Quality/reliability score",
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
    
    def model_post_init(self, __context: Any) -> None:
        """Set default polarity based on edge type."""
        if self.polarity == EdgePolarity.NEUTRAL:
            default_polarity = EDGE_TYPE_POLARITY.get(
                self.edge_type, EdgePolarity.NEUTRAL
            )
            object.__setattr__(self, "polarity", default_polarity)
    
    @computed_field
    @property
    def effective_weight(self) -> float:
        """
        Compute effective weight for propagation.
        
        w_eff = weight × confidence × relevance × quality
        """
        return self.weight * self.confidence * self.relevance * self.quality
    
    @computed_field
    @property
    def signed_weight(self) -> float:
        """
        Compute signed weight for influence propagation.
        
        Returns positive for support, negative for attack.
        """
        sign = 1.0 if self.polarity == EdgePolarity.POSITIVE else (
            -1.0 if self.polarity == EdgePolarity.NEGATIVE else 0.0
        )
        return sign * self.effective_weight
    
    @computed_field
    @property
    def is_supporting(self) -> bool:
        """Check if this is a supporting edge."""
        return self.polarity == EdgePolarity.POSITIVE
    
    @computed_field
    @property
    def is_attacking(self) -> bool:
        """Check if this is an attacking edge."""
        return self.polarity == EdgePolarity.NEGATIVE
    
    def update_weight(
        self,
        weight: Optional[float] = None,
        confidence: Optional[float] = None,
        relevance: Optional[float] = None,
        quality: Optional[float] = None,
    ) -> None:
        """
        Update edge weights.
        
        Args:
            weight: New weight value
            confidence: New confidence value
            relevance: New relevance value
            quality: New quality value
        """
        if weight is not None:
            object.__setattr__(self, "weight", max(0.0, min(1.0, weight)))
        if confidence is not None:
            object.__setattr__(self, "confidence", max(0.0, min(1.0, confidence)))
        if relevance is not None:
            object.__setattr__(self, "relevance", max(0.0, min(1.0, relevance)))
        if quality is not None:
            object.__setattr__(self, "quality", max(0.0, min(1.0, quality)))
        object.__setattr__(self, "updated_at", datetime.utcnow())
    
    def __hash__(self) -> int:
        """Hash by edge ID."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on edge ID."""
        if not isinstance(other, Edge):
            return False
        return self.id == other.id
    
    def to_tuple(self) -> tuple[str, str]:
        """Return (source, target) tuple for graph operations."""
        return (self.source_id, self.target_id)
    
    def to_prov_dict(self) -> dict[str, Any]:
        """
        Convert edge to PROV-O compatible dictionary.
        
        Returns:
            Dictionary with PROV-O relation attributes
        """
        return {
            "prov:type": f"argus:{self.edge_type.value}",
            "prov:id": self.id,
            "argus:source": self.source_id,
            "argus:target": self.target_id,
            "argus:polarity": self.polarity.value,
            "argus:weight": self.weight,
            "argus:effective_weight": self.effective_weight,
            "prov:generatedAtTime": self.created_at.isoformat(),
        }


def create_support_edge(
    source_id: str,
    target_id: str,
    confidence: float = 1.0,
    relevance: float = 1.0,
    **kwargs: Any,
) -> Edge:
    """
    Create a support edge.
    
    Args:
        source_id: Source node ID (evidence)
        target_id: Target node ID (proposition)
        confidence: Confidence in the support
        relevance: Relevance to the target
        **kwargs: Additional edge attributes
        
    Returns:
        Edge with SUPPORTS type and positive polarity
    """
    return Edge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.SUPPORTS,
        polarity=EdgePolarity.POSITIVE,
        confidence=confidence,
        relevance=relevance,
        **kwargs,
    )


def create_attack_edge(
    source_id: str,
    target_id: str,
    confidence: float = 1.0,
    relevance: float = 1.0,
    **kwargs: Any,
) -> Edge:
    """
    Create an attack edge.
    
    Args:
        source_id: Source node ID (evidence)
        target_id: Target node ID (proposition)
        confidence: Confidence in the attack
        relevance: Relevance to the target
        **kwargs: Additional edge attributes
        
    Returns:
        Edge with ATTACKS type and negative polarity
    """
    return Edge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.ATTACKS,
        polarity=EdgePolarity.NEGATIVE,
        confidence=confidence,
        relevance=relevance,
        **kwargs,
    )


def create_rebuttal_edge(
    source_id: str,
    target_id: str,
    confidence: float = 1.0,
    **kwargs: Any,
) -> Edge:
    """
    Create a rebuttal edge.
    
    Args:
        source_id: Rebuttal node ID
        target_id: Target evidence/rebuttal ID
        confidence: Confidence in the rebuttal
        **kwargs: Additional edge attributes
        
    Returns:
        Edge with REBUTS type and negative polarity
    """
    return Edge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.REBUTS,
        polarity=EdgePolarity.NEGATIVE,
        confidence=confidence,
        **kwargs,
    )
