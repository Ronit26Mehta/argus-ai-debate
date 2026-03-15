"""
Temporal C-DAG — time-aware extension of the Conceptual Debate Graph.

Adds timestamp-indexed evidence with exponential half-life decay functions.
Every evidence node carries a decay curve that reduces its effective
confidence as time passes, making posteriors time-dependent.

Core formula:
    temporal_weight(e, t) = confidence(e) × decay_fn(e.type, age(e, t))
    decay_fn(type, age) = exp(-λ × age)
    λ = ln(2) / half_life
"""

from __future__ import annotations

import math
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any

from dataclasses import dataclass, field
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvidenceCategory(str, Enum):
    """Evidence type categories with distinct half-lives."""
    EMPIRICAL = "empirical"       # RCTs, experiments — half-life ~5 years
    MARKET = "market"             # Financial signals — half-life ~24 hours
    EXPERT = "expert"             # Expert opinion — half-life ~2 years
    STATISTICAL = "statistical"   # Statistical analysis — half-life ~3 years
    LITERATURE = "literature"     # Published research — half-life ~4 years
    COMPUTATIONAL = "computational"  # Simulations — half-life ~3 years
    EMERGENT = "emergent"         # Live-stream evidence — half-life ~12 hours


# Default half-lives in HOURS for consistency
DEFAULT_HALF_LIVES: dict[EvidenceCategory, float] = {
    EvidenceCategory.EMPIRICAL: 5.0 * 365.25 * 24.0,       # ~5 years
    EvidenceCategory.MARKET: 24.0,                           # 24 hours
    EvidenceCategory.EXPERT: 2.0 * 365.25 * 24.0,           # ~2 years
    EvidenceCategory.STATISTICAL: 3.0 * 365.25 * 24.0,      # ~3 years
    EvidenceCategory.LITERATURE: 4.0 * 365.25 * 24.0,       # ~4 years
    EvidenceCategory.COMPUTATIONAL: 3.0 * 365.25 * 24.0,    # ~3 years
    EvidenceCategory.EMERGENT: 12.0,                          # 12 hours
}


@dataclass
class DecayFunction:
    """
    Exponential half-life decay function for evidence confidence.

    Computes temporal_weight = confidence × exp(-λ × age)
    where λ = ln(2) / half_life.

    Attributes:
        category: Evidence category determining half-life
        half_life_hours: Half-life in hours
        lambda_decay: Decay rate constant (computed from half-life)
    """

    category: EvidenceCategory
    half_life_hours: Optional[float] = None

    def __post_init__(self) -> None:
        if self.half_life_hours is None:
            self.half_life_hours = DEFAULT_HALF_LIVES.get(
                self.category,
                3.0 * 365.25 * 24.0,  # default 3 years
            )
        self.lambda_decay = math.log(2.0) / max(self.half_life_hours, 1e-10)

    def decay_factor(self, age_hours: float) -> float:
        """
        Compute decay factor for given age.

        Args:
            age_hours: Age of evidence in hours

        Returns:
            Decay factor in [0, 1] — multiplier for confidence
        """
        if age_hours <= 0:
            return 1.0
        return math.exp(-self.lambda_decay * age_hours)

    def compute_weight(
        self,
        confidence: float,
        age_hours: float,
    ) -> float:
        """
        Compute temporal weight of evidence.

        Args:
            confidence: Base confidence score (0-1)
            age_hours: Age of evidence in hours

        Returns:
            Temporal weight = confidence × decay_factor(age)
        """
        return confidence * self.decay_factor(age_hours)

    def effective_half_life_years(self) -> float:
        """Return half-life in years for display."""
        return self.half_life_hours / (365.25 * 24.0)


class TemporalEvidence(BaseModel):
    """
    Evidence node with temporal metadata for CHRONOS.

    Extends the standard Evidence concept with a timestamp and decay function
    so that its effective weight decreases over time.

    Attributes:
        evidence_id: Reference to original Evidence node ID
        text: Evidence text content
        confidence: Base confidence score (0-1)
        relevance: Relevance to proposition (0-1)
        polarity: Support (+1), Attack (-1), Neutral (0)
        category: Evidence category for decay
        timestamp: When the evidence was produced/published
        source_id: Source document/chunk ID
        weight: Base importance weight
    """

    model_config = {"frozen": False}

    evidence_id: str = Field(
        default_factory=lambda: f"tevid_{uuid.uuid4().hex[:12]}",
        description="Temporal evidence ID",
    )
    text: str = Field(default="", description="Evidence text")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    polarity: int = Field(default=0, ge=-1, le=1)
    category: EvidenceCategory = Field(default=EvidenceCategory.LITERATURE)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_id: Optional[str] = Field(default=None)
    weight: float = Field(default=1.0, ge=0.0)

    def age_hours(self, reference_time: Optional[datetime] = None) -> float:
        """
        Compute age in hours relative to reference time.

        Args:
            reference_time: Reference time (defaults to now)

        Returns:
            Age in hours
        """
        ref = reference_time or datetime.utcnow()
        delta = ref - self.timestamp
        return max(0.0, delta.total_seconds() / 3600.0)

    def temporal_weight(
        self,
        reference_time: Optional[datetime] = None,
        decay_fn: Optional[DecayFunction] = None,
    ) -> float:
        """
        Compute time-decayed effective weight.

        Args:
            reference_time: Time at which to evaluate weight
            decay_fn: Custom decay function (uses default for category if None)

        Returns:
            Effective weight after temporal decay
        """
        if decay_fn is None:
            decay_fn = DecayFunction(self.category)

        age = self.age_hours(reference_time)
        base_weight = self.confidence * self.relevance * self.weight
        return base_weight * decay_fn.decay_factor(age)

    def log_likelihood_ratio(
        self,
        reference_time: Optional[datetime] = None,
        decay_fn: Optional[DecayFunction] = None,
        temperature: float = 1.0,
    ) -> float:
        """
        Compute temporal log-likelihood ratio for Bayesian update.

        Args:
            reference_time: Time at which to evaluate
            decay_fn: Custom decay function
            temperature: Calibration temperature

        Returns:
            Signed LLR incorporating temporal decay
        """
        tw = self.temporal_weight(reference_time, decay_fn)
        tw = max(0.001, min(0.999, tw))
        lo = math.log(tw / (1.0 - tw))
        scaled = lo / temperature
        sign = 1.0 if self.polarity >= 0 else -1.0
        return sign * scaled


class TemporalCDAG:
    """
    Time-aware Conceptual Debate Graph.

    Wraps the standard CDAG with temporal evidence indexing and
    time-dependent posterior computation.

    Attributes:
        name: Graph name
        proposition_id: ID of the root proposition
        proposition_text: Text of the proposition
        prior: Prior probability
        evidence: List of temporal evidence nodes
        created_at: Graph creation time
    """

    def __init__(
        self,
        name: str = "temporal_debate",
        proposition_text: str = "",
        prior: float = 0.5,
    ):
        self.name = name
        self.proposition_id = f"tprop_{uuid.uuid4().hex[:12]}"
        self.proposition_text = proposition_text
        self.prior = prior
        self.evidence: list[TemporalEvidence] = []
        self.created_at = datetime.utcnow()
        self._decay_registry: dict[EvidenceCategory, DecayFunction] = {}

    def register_decay(
        self,
        category: EvidenceCategory,
        half_life_hours: float,
    ) -> None:
        """Register a custom decay function for an evidence category."""
        self._decay_registry[category] = DecayFunction(
            category=category,
            half_life_hours=half_life_hours,
        )

    def get_decay_fn(self, category: EvidenceCategory) -> DecayFunction:
        """Get decay function for category (custom or default)."""
        return self._decay_registry.get(
            category,
            DecayFunction(category),
        )

    def add_evidence(self, evidence: TemporalEvidence) -> None:
        """Add a temporal evidence node."""
        self.evidence.append(evidence)
        logger.debug(
            f"Added temporal evidence {evidence.evidence_id} "
            f"(category={evidence.category.value}, "
            f"timestamp={evidence.timestamp.isoformat()})"
        )

    def compute_posterior_at(self, time_point: datetime) -> float:
        """
        Compute Bayesian posterior at a specific time point.

        Uses log-odds aggregation with temporal decay:
            logit P(θ|E,t) = logit P(θ) + Σ_e temporal_weight(e, t) × sign(e) × LLR(e)

        Args:
            time_point: Time at which to compute posterior

        Returns:
            Posterior probability at given time
        """
        # Start with prior in log-odds
        p = max(0.001, min(0.999, self.prior))
        prior_lo = math.log(p / (1.0 - p))

        total_contribution = 0.0

        for ev in self.evidence:
            # Only include evidence published before this time point
            if ev.timestamp > time_point:
                continue

            decay_fn = self.get_decay_fn(ev.category)
            age = ev.age_hours(time_point)

            # Temporal decay factor
            decay = decay_fn.decay_factor(age)

            # Effective confidence after decay
            eff_conf = ev.confidence * decay
            eff_conf = max(0.001, min(0.999, eff_conf))

            # Log-likelihood ratio
            llr = math.log(eff_conf / (1.0 - eff_conf))

            # Sign by polarity
            sign = float(ev.polarity) if ev.polarity != 0 else 0.0

            # Weight by relevance and base weight
            contribution = ev.relevance * ev.weight * sign * llr
            total_contribution += contribution

        posterior_lo = prior_lo + total_contribution
        # Sigmoid to convert back to probability
        posterior_lo = max(-500, min(500, posterior_lo))
        posterior = 1.0 / (1.0 + math.exp(-posterior_lo))

        return posterior

    def compute_posterior_series(
        self,
        start_time: datetime,
        end_time: datetime,
        resolution: str = "month",
    ) -> list[tuple[datetime, float]]:
        """
        Compute posterior at regular intervals over a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            resolution: Time resolution ('day', 'week', 'month', 'year')

        Returns:
            List of (datetime, posterior) tuples
        """
        deltas = {
            "day": timedelta(days=1),
            "week": timedelta(weeks=1),
            "month": timedelta(days=30),
            "year": timedelta(days=365),
        }
        delta = deltas.get(resolution, timedelta(days=30))

        series = []
        current = start_time
        while current <= end_time:
            posterior = self.compute_posterior_at(current)
            series.append((current, posterior))
            current += delta

        return series

    @property
    def num_evidence(self) -> int:
        """Total number of evidence nodes."""
        return len(self.evidence)

    @property
    def evidence_by_category(self) -> dict[EvidenceCategory, list[TemporalEvidence]]:
        """Group evidence by category."""
        result: dict[EvidenceCategory, list[TemporalEvidence]] = {}
        for ev in self.evidence:
            result.setdefault(ev.category, []).append(ev)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "proposition_id": self.proposition_id,
            "proposition_text": self.proposition_text,
            "prior": self.prior,
            "num_evidence": self.num_evidence,
            "created_at": self.created_at.isoformat(),
            "evidence": [
                {
                    "id": ev.evidence_id,
                    "text": ev.text[:100],
                    "confidence": ev.confidence,
                    "polarity": ev.polarity,
                    "category": ev.category.value,
                    "timestamp": ev.timestamp.isoformat(),
                }
                for ev in self.evidence
            ],
        }
