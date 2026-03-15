"""
TruthNode — versioned, hash-linked, signed verdict record.
"""

from __future__ import annotations

import uuid
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NodeVersion:
    """Versioned snapshot of a TruthNode."""
    version: int = 1
    posterior: float = 0.5
    verdict: str = "undetermined"
    evidence_count: int = 0
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    hash: str = ""

    def compute_hash(self, prev_hash: str = "") -> str:
        payload = f"{self.version}:{self.posterior}:{self.verdict}:{self.updated_at}:{prev_hash}"
        self.hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.hash


@dataclass
class TruthNode:
    """
    A versioned, hash-linked verdict record in the VERICHAIN.

    Each node represents a debated proposition with its full version
    history. Hash-chain integrity ensures tamper detection.

    Attributes:
        node_id: Unique node ID
        proposition: Debated proposition text
        domain: Domain classification
        current_posterior: Latest posterior probability
        current_verdict: Latest verdict
        created_at: Creation timestamp
        versions: Version history
        prev_hash: Previous hash in the chain
        signature: Optional cryptographic signature
        debate_id: ID of the originating debate
        citation_count: How many debates have cited this node
        embedding: Vector embedding for semantic search
    """
    node_id: str = field(default_factory=lambda: f"truth_{uuid.uuid4().hex[:12]}")
    proposition: str = ""
    domain: str = "general"
    current_posterior: float = 0.5
    current_verdict: str = "undetermined"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    versions: list[NodeVersion] = field(default_factory=list)
    prev_hash: str = ""
    signature: str = ""
    debate_id: str = ""
    citation_count: int = 0
    embedding: Optional[list[float]] = None

    @property
    def current_hash(self) -> str:
        if self.versions:
            return self.versions[-1].hash
        return ""

    @property
    def version_count(self) -> int:
        return len(self.versions)

    @property
    def authority_score(self) -> float:
        """Authority = f(citation_count, version_count, posterior_confidence)."""
        confidence = abs(self.current_posterior - 0.5) * 2
        citation_factor = min(1.0, self.citation_count / 10.0)
        version_factor = min(1.0, self.version_count / 5.0)
        return 0.4 * confidence + 0.4 * citation_factor + 0.2 * version_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "proposition": self.proposition[:200],
            "domain": self.domain,
            "current_posterior": round(self.current_posterior, 4),
            "verdict": self.current_verdict,
            "version_count": self.version_count,
            "authority_score": round(self.authority_score, 4),
            "citation_count": self.citation_count,
            "current_hash": self.current_hash,
        }


class TruthNodeBuilder:
    """
    Builder for constructing TruthNode objects.

    Example:
        >>> node = (TruthNodeBuilder()
        ...     .proposition("Drug X is effective")
        ...     .verdict("supported", 0.78)
        ...     .domain("clinical")
        ...     .build())
    """

    def __init__(self):
        self._node = TruthNode()

    def proposition(self, text: str) -> "TruthNodeBuilder":
        self._node.proposition = text
        return self

    def verdict(self, verdict: str, posterior: float) -> "TruthNodeBuilder":
        self._node.current_verdict = verdict
        self._node.current_posterior = posterior
        version = NodeVersion(
            version=len(self._node.versions) + 1,
            posterior=posterior,
            verdict=verdict,
        )
        version.compute_hash(self._node.prev_hash)
        self._node.versions.append(version)
        self._node.prev_hash = version.hash
        return self

    def domain(self, domain: str) -> "TruthNodeBuilder":
        self._node.domain = domain
        return self

    def debate_id(self, debate_id: str) -> "TruthNodeBuilder":
        self._node.debate_id = debate_id
        return self

    def embedding(self, emb: list[float]) -> "TruthNodeBuilder":
        self._node.embedding = emb
        return self

    def build(self) -> TruthNode:
        return self._node

    def update(self, verdict: str, posterior: float) -> "TruthNodeBuilder":
        """Add a new version to the node."""
        return self.verdict(verdict, posterior)
