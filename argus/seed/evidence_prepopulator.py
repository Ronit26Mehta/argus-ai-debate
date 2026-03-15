"""
Evidence Pre-Populator — builds C-DAG skeleton from extracted evidence.
"""

from __future__ import annotations

import uuid
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvidenceFragment:
    """A fragment of evidence extracted from source document."""
    fragment_id: str = field(default_factory=lambda: f"frag_{uuid.uuid4().hex[:10]}")
    text: str = ""
    polarity: int = 0  # +1=supporting, -1=opposing, 0=neutral
    confidence: float = 0.5
    source_chunk_id: str = ""
    source_section: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id, "text": self.text[:100],
            "polarity": self.polarity, "confidence": round(self.confidence, 4),
        }


@dataclass
class CDAGSkeleton:
    """
    Pre-built C-DAG skeleton with evidence fragments.

    Attributes:
        proposition_text: Main proposition
        prior_estimate: Estimated prior probability
        supporting: Supporting evidence fragments
        opposing: Opposing evidence fragments
        neutral: Neutral/context fragments
    """
    proposition_text: str = ""
    prior_estimate: float = 0.5
    supporting: list[EvidenceFragment] = field(default_factory=list)
    opposing: list[EvidenceFragment] = field(default_factory=list)
    neutral: list[EvidenceFragment] = field(default_factory=list)

    @property
    def total_evidence(self) -> int:
        return len(self.supporting) + len(self.opposing) + len(self.neutral)

    @property
    def support_ratio(self) -> float:
        total = len(self.supporting) + len(self.opposing)
        return len(self.supporting) / max(total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition_text[:100],
            "prior_estimate": round(self.prior_estimate, 4),
            "num_supporting": len(self.supporting),
            "num_opposing": len(self.opposing),
            "num_neutral": len(self.neutral),
            "support_ratio": round(self.support_ratio, 4),
        }


class EvidencePrePopulator:
    """
    Builds C-DAG skeleton from source document chunks.

    For top-N claims: extracts supporting/opposing chunks and
    pre-classifies evidence polarity.

    Example:
        >>> populator = EvidencePrePopulator()
        >>> skeleton = populator.build_skeleton(
        ...     proposition="Drug X is effective",
        ...     chunks=["chunk1 text...", "chunk2 text..."],
        ... )
    """

    SUPPORT_MARKERS = [
        "support", "confirm", "demonstrate", "show", "indicate",
        "evidence for", "consistent with", "effective", "success",
        "positive", "beneficial", "significant improvement",
    ]

    OPPOSE_MARKERS = [
        "contradict", "oppose", "fail", "refute", "challenge",
        "evidence against", "inconsistent", "ineffective",
        "negative", "harmful", "no significant",
    ]

    def __init__(self, max_fragments_per_side: int = 10):
        self.max_fragments_per_side = max_fragments_per_side

    def build_skeleton(
        self,
        proposition: str,
        chunks: list[str],
        chunk_ids: Optional[list[str]] = None,
    ) -> CDAGSkeleton:
        """
        Build C-DAG skeleton from source chunks.

        Args:
            proposition: Proposition text
            chunks: Source text chunks
            chunk_ids: Optional chunk identifiers

        Returns:
            CDAGSkeleton with classified evidence
        """
        skeleton = CDAGSkeleton(proposition_text=proposition)
        chunk_ids = chunk_ids or [f"chunk_{i}" for i in range(len(chunks))]

        for chunk_text, chunk_id in zip(chunks, chunk_ids):
            polarity, confidence = self._classify_polarity(chunk_text, proposition)

            fragment = EvidenceFragment(
                text=chunk_text[:300],
                polarity=polarity,
                confidence=confidence,
                source_chunk_id=chunk_id,
            )

            if polarity > 0:
                if len(skeleton.supporting) < self.max_fragments_per_side:
                    skeleton.supporting.append(fragment)
            elif polarity < 0:
                if len(skeleton.opposing) < self.max_fragments_per_side:
                    skeleton.opposing.append(fragment)
            else:
                skeleton.neutral.append(fragment)

        # Estimate prior from support ratio
        skeleton.prior_estimate = 0.3 + 0.4 * skeleton.support_ratio

        logger.info(
            f"Built CDAG skeleton: {len(skeleton.supporting)} supporting, "
            f"{len(skeleton.opposing)} opposing, {len(skeleton.neutral)} neutral"
        )

        return skeleton

    def _classify_polarity(
        self,
        chunk_text: str,
        proposition: str,
    ) -> tuple[int, float]:
        """Classify evidence polarity relative to proposition."""
        text_lower = chunk_text.lower()

        support_count = sum(1 for m in self.SUPPORT_MARKERS if m in text_lower)
        oppose_count = sum(1 for m in self.OPPOSE_MARKERS if m in text_lower)

        total = support_count + oppose_count
        if total == 0:
            return 0, 0.3

        if support_count > oppose_count:
            confidence = min(0.9, 0.4 + 0.1 * support_count)
            return 1, confidence
        elif oppose_count > support_count:
            confidence = min(0.9, 0.4 + 0.1 * oppose_count)
            return -1, confidence
        else:
            return 0, 0.4
