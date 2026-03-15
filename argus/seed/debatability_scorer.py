"""
DebatabilityScorer — novel composite metric for debate-readiness.

DebatabilityScore = w1×BiPolarityRatio + w2×NoveltyQuotient + w3×EvidenceDensity

Novel: The first claim-worthiness metric designed specifically for
debate-native AI frameworks. No existing system quantifies whether
a claim DESERVES to be debated.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BiPolarityRatio:
    """
    Measures how balanced positive vs negative signals are for a claim.

    Ratio = min(pos, neg) / max(pos, neg)
    1.0 = perfectly balanced, 0.0 = entirely one-sided

    Attributes:
        positive_signals: Count of supporting indicators
        negative_signals: Count of opposing indicators
        ratio: Computed bipolarity ratio
    """
    positive_signals: int = 0
    negative_signals: int = 0
    ratio: float = 0.0

    def compute(self) -> float:
        mx = max(self.positive_signals, self.negative_signals)
        mn = min(self.positive_signals, self.negative_signals)
        self.ratio = mn / mx if mx > 0 else 0.0
        return self.ratio


@dataclass
class NoveltyQuotient:
    """
    Measures claim novelty relative to known claims.

    NQ = 1 - max_cosine_similarity(claim, known_claims)
    1.0 = completely novel, 0.0 = already well-established

    Attributes:
        max_similarity: Highest similarity to known claims
        quotient: Computed novelty quotient
    """
    max_similarity: float = 0.0
    quotient: float = 1.0

    def compute(self) -> float:
        self.quotient = 1.0 - self.max_similarity
        return self.quotient


class DebatabilityScorer:
    """
    Computes DebatabilityScore for extracted claims.

    DebatabilityScore = w1×BiPolarityRatio + w2×NoveltyQuotient + w3×EvidenceDensity

    Interpretation:
        Score > 0.7  → Excellent debate candidate
        Score 0.4-0.7 → Good debate candidate
        Score < 0.4  → Low-value; filter out

    Example:
        >>> scorer = DebatabilityScorer()
        >>> score = scorer.score_claim(
        ...     claim_text="Drug X reduces mortality by 15%",
        ...     source_chunks=["chunk1", "chunk2"],
        ...     total_chunks=10,
        ...     positive_signals=3,
        ...     negative_signals=2,
        ... )
        >>> print(f"Debatability: {score:.2f}")
    """

    # Polarity keywords for signal detection
    POSITIVE_MARKERS = [
        "effective", "beneficial", "improve", "increase", "support",
        "demonstrate", "confirm", "success", "significant", "advantage",
        "promising", "positive", "enhance", "efficient", "recommend",
    ]

    NEGATIVE_MARKERS = [
        "ineffective", "harmful", "worsen", "decrease", "oppose",
        "contradict", "fail", "insignificant", "disadvantage", "risk",
        "concern", "negative", "limit", "challenge", "question",
    ]

    def __init__(
        self,
        w_bipolarity: float = 0.35,
        w_novelty: float = 0.35,
        w_density: float = 0.30,
        min_debatability: float = 0.4,
    ):
        self.w_bipolarity = w_bipolarity
        self.w_novelty = w_novelty
        self.w_density = w_density
        self.min_debatability = min_debatability

    def score_claim(
        self,
        claim_text: str,
        source_chunks: Optional[list[str]] = None,
        total_chunks: int = 1,
        positive_signals: Optional[int] = None,
        negative_signals: Optional[int] = None,
        known_claim_similarity: float = 0.0,
    ) -> float:
        """
        Compute DebatabilityScore for a claim.

        Args:
            claim_text: The claim text
            source_chunks: Relevant source chunks
            total_chunks: Total chunks in source document
            positive_signals: Count of positive signals (auto-detected if None)
            negative_signals: Count of negative signals (auto-detected if None)
            known_claim_similarity: Max similarity to known claims

        Returns:
            DebatabilityScore in [0, 1]
        """
        # Auto-detect signals if not provided
        if positive_signals is None or negative_signals is None:
            pos, neg = self._count_signals(claim_text, source_chunks or [])
            positive_signals = positive_signals or pos
            negative_signals = negative_signals or neg

        # BiPolarityRatio
        bpr = BiPolarityRatio(positive_signals, negative_signals)
        bipolarity = bpr.compute()

        # NoveltyQuotient
        nq = NoveltyQuotient(max_similarity=known_claim_similarity)
        novelty = nq.compute()

        # EvidenceDensity
        relevant = len(source_chunks) if source_chunks else 0
        density = min(1.0, relevant / max(total_chunks, 1))

        # Composite score
        score = (
            self.w_bipolarity * bipolarity
            + self.w_novelty * novelty
            + self.w_density * density
        )

        return min(1.0, max(0.0, score))

    def score_claims(
        self,
        claims: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Score multiple claims and return sorted by debatability.

        Args:
            claims: List of claim dicts with 'text' and optional fields

        Returns:
            Sorted list of (claim, score) tuples
        """
        scored = []
        for claim in claims:
            text = claim.get("text", "")
            score = self.score_claim(
                claim_text=text,
                source_chunks=claim.get("source_chunks", []),
                total_chunks=claim.get("total_chunks", 1),
                known_claim_similarity=claim.get("similarity", 0.0),
            )
            scored.append((claim, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _count_signals(
        self,
        claim_text: str,
        context_chunks: list[str],
    ) -> tuple[int, int]:
        """Count positive and negative signals in claim and context."""
        full_text = (claim_text + " " + " ".join(context_chunks)).lower()

        pos = sum(1 for m in self.POSITIVE_MARKERS if m in full_text)
        neg = sum(1 for m in self.NEGATIVE_MARKERS if m in full_text)

        return max(pos, 1), max(neg, 0)  # Ensure at least 1 positive
