"""
DebateReadyBundle — packaged output from SEED pipeline.
"""

from __future__ import annotations

import uuid
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.seed.evidence_prepopulator import CDAGSkeleton


@dataclass
class PriorEstimate:
    """Estimated prior probability for a claim."""
    value: float = 0.5
    method: str = "evidence_ratio"  # evidence_ratio, heuristic, model
    confidence: float = 0.5


@dataclass
class ScoredClaim:
    """A claim with its debatability score and metadata."""
    claim_id: str = field(default_factory=lambda: f"sclaim_{uuid.uuid4().hex[:10]}")
    text: str = ""
    claim_type: str = "declarative"
    debatability_score: float = 0.0
    prior_estimate: Optional[PriorEstimate] = None
    entities: list[str] = field(default_factory=list)
    cdag_skeleton: Optional[CDAGSkeleton] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text[:200],
            "claim_type": self.claim_type,
            "debatability_score": round(self.debatability_score, 4),
            "prior": self.prior_estimate.value if self.prior_estimate else 0.5,
            "entities": self.entities,
        }


class DebateReadyBundle:
    """
    Complete bundle ready for RDCOrchestrator.debate_bundle().

    Contains ranked claims, C-DAG skeletons, evidence, and priors.

    Example:
        >>> bundle = DebateReadyBundle(source="Research paper")
        >>> bundle.add_claim(scored_claim)
        >>> result = orchestrator.debate_bundle(bundle.top_claim)
    """

    def __init__(
        self,
        source: str = "",
        source_type: str = "text",
    ):
        self.bundle_id = f"bundle_{uuid.uuid4().hex[:10]}"
        self.source = source
        self.source_type = source_type
        self._claims: list[ScoredClaim] = []

    def add_claim(self, claim: ScoredClaim) -> None:
        self._claims.append(claim)
        self._claims.sort(key=lambda c: c.debatability_score, reverse=True)

    @property
    def ranked_claims(self) -> list[ScoredClaim]:
        return self._claims

    @property
    def top_claim(self) -> Optional[ScoredClaim]:
        return self._claims[0] if self._claims else None

    @property
    def num_claims(self) -> int:
        return len(self._claims)

    def get_claims_above(self, threshold: float = 0.4) -> list[ScoredClaim]:
        return [c for c in self._claims if c.debatability_score >= threshold]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "source": self.source[:100],
            "source_type": self.source_type,
            "num_claims": self.num_claims,
            "top_score": self.top_claim.debatability_score if self.top_claim else 0,
            "claims": [c.to_dict() for c in self._claims],
        }
