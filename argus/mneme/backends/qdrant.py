"""
Qdrant backend for MNEME — native vector similarity search.
"""

from __future__ import annotations

import uuid
import logging
from typing import Optional, Any

from argus.mneme.reservoir import ReservoirEntry

logger = logging.getLogger(__name__)


class QdrantMemoryBackend:
    """
    Qdrant vector DB backend for MNEME.

    Uses Qdrant for native cosine similarity search over
    knowledge embeddings, replacing the in-memory brute-force approach.

    Requires: pip install qdrant-client

    Example:
        >>> backend = QdrantMemoryBackend(url="http://localhost:6333")
        >>> backend.save_entry(entry)
        >>> results = backend.search(query_vector, top_k=5)
    """

    COLLECTION_NAME = "mneme_knowledge"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "mneme_knowledge",
        embedding_dim: int = 384,
    ):
        self.url = url
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._client: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import (
                VectorParams, Distance,
            )
            self._client = QdrantClient(url=self.url)

            # Create collection if not exists
            collections = self._client.get_collections().collections
            names = [c.name for c in collections]
            if self.collection_name not in names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )

            logger.info(f"MNEME Qdrant backend initialized: {self.url}")
        except ImportError:
            logger.warning("qdrant-client not installed. Install with: pip install qdrant-client")
            self._client = None
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}")
            self._client = None

    def save_entry(self, entry: ReservoirEntry) -> None:
        if not self._client or not entry.embedding:
            return
        try:
            from qdrant_client.http.models import PointStruct
            point = PointStruct(
                id=entry.entry_id,
                vector=entry.embedding,
                payload={
                    "text": entry.text,
                    "domain": entry.domain,
                    "confidence": entry.confidence,
                    "source_debate_id": entry.source_debate_id,
                    "proposition_text": entry.proposition_text,
                    "verdict": entry.verdict,
                    "created_at": entry.created_at.isoformat(),
                },
            )
            self._client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
        except Exception as e:
            logger.warning(f"Qdrant save failed: {e}")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Search for similar entries.

        Returns list of (entry_id, score, payload) tuples.
        """
        if not self._client:
            return []
        try:
            filter_dict = None
            if domain_filter:
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                filter_dict = Filter(
                    must=[FieldCondition(key="domain", match=MatchValue(value=domain_filter))]
                )
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_dict,
            )
            return [
                (str(r.id), r.score, r.payload or {})
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Qdrant search failed: {e}")
            return []

    def count(self) -> int:
        if not self._client:
            return 0
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def close(self) -> None:
        self._client = None
