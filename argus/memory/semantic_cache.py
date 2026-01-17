"""
ARGUS Semantic Cache.

Cache LLM responses based on semantic similarity for efficiency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A semantic cache entry."""
    cache_id: str
    query: str
    response: str
    embedding: List[float]
    created_at: float
    ttl_seconds: int
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return time.time() > self.created_at + self.ttl_seconds


class SemanticCache:
    """Cache that retrieves based on semantic similarity."""
    
    def __init__(self, similarity_threshold: float = 0.85, max_entries: int = 1000,
                 ttl_seconds: int = 3600, embedding_generator: Optional[Any] = None,
                 dimension: int = 384):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.embedding_generator = embedding_generator
        self.dimension = dimension
        self._cache: dict[str, CacheEntry] = {}
        self._embeddings: List[np.ndarray] = []
        self._cache_ids: List[str] = []
        self._hits = 0
        self._misses = 0
    
    def _get_embedding(self, text: str) -> np.ndarray:
        if self.embedding_generator:
            emb = self.embedding_generator.embed(text)
            return np.array(emb, dtype=np.float32)
        return np.random.randn(self.dimension).astype(np.float32)
    
    def _generate_id(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response for query if similar enough."""
        self._cleanup_expired()
        if not self._cache:
            self._misses += 1
            return None
        query_emb = self._get_embedding(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        best_score = -1.0
        best_entry = None
        for i, emb in enumerate(self._embeddings):
            score = float(np.dot(query_emb, emb))
            if score > best_score:
                best_score = score
                cache_id = self._cache_ids[i]
                best_entry = self._cache.get(cache_id)
        if best_entry and best_score >= self.similarity_threshold:
            if not best_entry.is_expired:
                best_entry.hit_count += 1
                self._hits += 1
                logger.debug(f"Cache hit (score={best_score:.3f}): {best_entry.cache_id}")
                return best_entry.response
        self._misses += 1
        return None
    
    def set(self, query: str, response: str, **metadata: Any) -> str:
        """Cache a response for a query."""
        self._cleanup_expired()
        if len(self._cache) >= self.max_entries:
            self._evict_lru()
        cache_id = self._generate_id(query)
        embedding = self._get_embedding(query)
        embedding = embedding / np.linalg.norm(embedding)
        entry = CacheEntry(
            cache_id=cache_id, query=query, response=response,
            embedding=embedding.tolist(), created_at=time.time(),
            ttl_seconds=self.ttl_seconds, metadata=metadata,
        )
        self._cache[cache_id] = entry
        self._embeddings.append(embedding)
        self._cache_ids.append(cache_id)
        logger.debug(f"Cached response: {cache_id}")
        return cache_id
    
    def invalidate(self, cache_id: str) -> bool:
        """Remove entry from cache."""
        if cache_id in self._cache:
            del self._cache[cache_id]
            idx = self._cache_ids.index(cache_id)
            self._cache_ids.pop(idx)
            self._embeddings.pop(idx)
            return True
        return False
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        expired = [cid for cid, e in self._cache.items() if e.is_expired]
        for cid in expired:
            self.invalidate(cid)
        return len(expired)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        lru_id = min(self._cache.keys(), key=lambda k: self._cache[k].hit_count)
        self.invalidate(lru_id)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._embeddings.clear()
        self._cache_ids.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "max_entries": self.max_entries,
        }
