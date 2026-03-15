"""
VERICHAIN Retriever — semantic search over past verdicts.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Any
from dataclasses import dataclass

import numpy as np

from argus.verichain.node import TruthNode

logger = logging.getLogger(__name__)


@dataclass
class PrecedentScorer:
    """Scores precedent relevance using authority and semantic similarity."""
    semantic_weight: float = 0.6
    authority_weight: float = 0.3
    recency_weight: float = 0.1

    def score(
        self,
        semantic_sim: float,
        authority: float,
        recency: float = 0.5,
    ) -> float:
        return (
            self.semantic_weight * semantic_sim
            + self.authority_weight * authority
            + self.recency_weight * recency
        )


class SemanticSearch:
    """Simple embedding-based semantic search."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def embed(self, text: str) -> list[float]:
        """Generate hash-based pseudo-embedding."""
        np.random.seed(hash(text) % (2**31))
        emb = np.random.randn(self.embedding_dim).astype(float)
        norm = np.linalg.norm(emb)
        return (emb / norm if norm > 0 else emb).tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        dot = np.dot(a_arr, b_arr)
        norms = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(dot / norms) if norms > 1e-10 else 0.0


class VERICHAINRetriever:
    """
    Retrieves relevant precedent verdicts for a new debate.

    Uses semantic similarity + authority score + recency to rank
    past verdicts from the VERICHAIN registry.

    Example:
        >>> retriever = VERICHAINRetriever(registry)
        >>> precedents = retriever.retrieve("new drug effectiveness question", top_k=5)
        >>> for node, score in precedents:
        ...     print(f"{node.proposition[:60]}  (auth={node.authority_score:.2f})")
    """

    def __init__(
        self,
        nodes: Optional[list[TruthNode]] = None,
        scorer: Optional[PrecedentScorer] = None,
    ):
        self.nodes = nodes or []
        self.scorer = scorer or PrecedentScorer()
        self._search = SemanticSearch()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> list[tuple[TruthNode, float]]:
        """Retrieve precedent verdicts."""
        query_embedding = self._search.embed(query)
        candidates = self.nodes

        if domain_filter:
            candidates = [n for n in candidates if n.domain == domain_filter]

        scored: list[tuple[TruthNode, float]] = []
        for node in candidates:
            if node.embedding is None:
                node.embedding = self._search.embed(node.proposition)

            sim = self._search.similarity(query_embedding, node.embedding)
            score = self.scorer.score(
                semantic_sim=sim,
                authority=node.authority_score,
            )
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
