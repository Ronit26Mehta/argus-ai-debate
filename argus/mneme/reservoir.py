"""
KnowledgeReservoir — vector store with recency-weighted retrieval.

Stores knowledge entries with embeddings and retrieves them using
cosine similarity weighted by recency decay.
"""

from __future__ import annotations

import uuid
import math
import logging
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecayFunction:
    """Exponential decay for knowledge recency."""
    half_life_days: float = 90.0

    @property
    def lambda_decay(self) -> float:
        return math.log(2.0) / max(self.half_life_days, 0.01)

    def weight(self, age_days: float) -> float:
        if age_days <= 0:
            return 1.0
        return math.exp(-self.lambda_decay * age_days)


@dataclass
class ReservoirEntry:
    """A single entry in the knowledge reservoir."""
    entry_id: str = field(default_factory=lambda: f"rentry_{uuid.uuid4().hex[:10]}")
    text: str = ""
    domain: str = "general"
    embedding: Optional[list[float]] = None
    confidence: float = 0.5
    source_debate_id: str = ""
    proposition_text: str = ""
    verdict: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    def age_days(self, reference: Optional[datetime] = None) -> float:
        ref = reference or datetime.utcnow()
        return max(0, (ref - self.created_at).total_seconds() / 86400.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "text": self.text[:100],
            "domain": self.domain,
            "confidence": round(self.confidence, 4),
            "verdict": self.verdict,
            "age_days": round(self.age_days(), 1),
            "access_count": self.access_count,
        }


class KnowledgeReservoir:
    """
    Vector store with recency-weighted retrieval.

    Stores knowledge from past debates and retrieves relevant entries
    using cosine similarity multiplied by recency decay.

    Example:
        >>> reservoir = KnowledgeReservoir(max_entries=5000)
        >>> reservoir.store(ReservoirEntry(
        ...     text="Drug X reduces mortality by 15%",
        ...     domain="clinical",
        ...     confidence=0.85,
        ... ))
        >>> results = reservoir.retrieve("mortality reduction", top_k=5)
    """

    def __init__(
        self,
        max_entries: int = 5000,
        decay: Optional[DecayFunction] = None,
        embedding_dim: int = 384,
    ):
        self.max_entries = max_entries
        self.decay = decay or DecayFunction()
        self.embedding_dim = embedding_dim
        self._entries: list[ReservoirEntry] = []

    def store(self, entry: ReservoirEntry) -> str:
        """Store an entry in the reservoir."""
        if entry.embedding is None:
            entry.embedding = self._generate_embedding(entry.text)

        self._entries.append(entry)

        # Evict oldest if over capacity
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: e.created_at)
            self._entries = self._entries[-self.max_entries:]

        logger.debug(f"Stored entry {entry.entry_id} in reservoir")
        return entry.entry_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> list[tuple[ReservoirEntry, float]]:
        """
        Retrieve entries by cosine similarity weighted by recency.

        Args:
            query: Query text
            top_k: Number of results
            domain_filter: Optional domain filter

        Returns:
            List of (entry, score) tuples
        """
        if not self._entries:
            return []

        query_embedding = self._generate_embedding(query)
        candidates = self._entries

        if domain_filter:
            candidates = [e for e in candidates if e.domain == domain_filter]

        scored: list[tuple[ReservoirEntry, float]] = []
        for entry in candidates:
            if entry.embedding is None:
                continue
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            recency = self.decay.weight(entry.age_days())
            score = sim * recency
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        # Update access counts
        for entry, _ in results:
            entry.access_count += 1

        return results

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate a simple hash-based pseudo-embedding."""
        np.random.seed(hash(text) % (2**31))
        emb = np.random.randn(self.embedding_dim).astype(float)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @property
    def size(self) -> int:
        return len(self._entries)

    def get_domains(self) -> list[str]:
        return list(set(e.domain for e in self._entries))

    def clear(self) -> None:
        self._entries.clear()
