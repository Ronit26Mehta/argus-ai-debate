"""
ARGUS Long-Term Memory.

Vector store backed persistent memory for cross-session context.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A long-term memory entry."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    memory_type: str = "general"  # semantic, episodic, procedural
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    importance: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["accessed_at"] = self.accessed_at.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["accessed_at"] = datetime.fromisoformat(data["accessed_at"])
        return cls(**data)


class BaseLongTermMemory(ABC):
    """Abstract base for long-term memory."""
    
    @abstractmethod
    def add(self, content: str, memory_type: str = "general", **kwargs: Any) -> str:
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        pass


class VectorStoreMemory(BaseLongTermMemory):
    """Vector store backed long-term memory using FAISS."""
    
    def __init__(self, embedding_generator: Optional[Any] = None,
                 persistence_path: Optional[str] = None,
                 dimension: int = 384):
        self.embedding_generator = embedding_generator
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.dimension = dimension
        self._memories: dict[str, MemoryEntry] = {}
        self._index: Optional[Any] = None
        self._id_map: List[str] = []
        self._init_index()
        if self.persistence_path:
            self._load()
    
    def _init_index(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            logger.warning("FAISS not available, using simple numpy search")
            self._embeddings: List[np.ndarray] = []
    
    def _get_embedding(self, text: str) -> np.ndarray:
        if self.embedding_generator:
            emb = self.embedding_generator.embed(text)
            return np.array(emb, dtype=np.float32)
        return np.random.randn(self.dimension).astype(np.float32)
    
    def add(self, content: str, memory_type: str = "general", importance: float = 0.5, **kwargs: Any) -> str:
        embedding = self._get_embedding(content)
        embedding = embedding / np.linalg.norm(embedding)
        entry = MemoryEntry(
            content=content, embedding=embedding.tolist(),
            memory_type=memory_type, importance=importance, metadata=kwargs,
        )
        self._memories[entry.memory_id] = entry
        self._id_map.append(entry.memory_id)
        if self._index is not None:
            self._index.add(embedding.reshape(1, -1))
        else:
            self._embeddings.append(embedding)
        logger.debug(f"Added memory: {entry.memory_id}")
        return entry.memory_id
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        if not self._memories:
            return []
        query_emb = self._get_embedding(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        if self._index is not None:
            distances, indices = self._index.search(query_emb.reshape(1, -1), min(top_k, len(self._id_map)))
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self._id_map):
                    mem_id = self._id_map[idx]
                    entry = self._memories.get(mem_id)
                    if entry:
                        entry.accessed_at = datetime.utcnow()
                        entry.access_count += 1
                        results.append((entry, float(dist)))
            return results
        else:
            scores = [np.dot(query_emb, e) for e in self._embeddings]
            indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in indices:
                mem_id = self._id_map[idx]
                entry = self._memories.get(mem_id)
                if entry:
                    entry.accessed_at = datetime.utcnow()
                    entry.access_count += 1
                    results.append((entry, float(scores[idx])))
            return results
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._memories.get(memory_id)
    
    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False
    
    def save(self) -> None:
        if not self.persistence_path:
            return
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self._memories.values()]
        (self.persistence_path / "memories.json").write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        if not self.persistence_path:
            return
        mem_file = self.persistence_path / "memories.json"
        if mem_file.exists():
            data = json.loads(mem_file.read_text())
            for d in data:
                entry = MemoryEntry.from_dict(d)
                self._memories[entry.memory_id] = entry
                self._id_map.append(entry.memory_id)
                if entry.embedding:
                    emb = np.array(entry.embedding, dtype=np.float32)
                    if self._index is not None:
                        self._index.add(emb.reshape(1, -1))
                    else:
                        self._embeddings.append(emb)


class SemanticMemory(VectorStoreMemory):
    """Memory for facts and knowledge."""
    
    def add_fact(self, fact: str, source: Optional[str] = None, **kwargs: Any) -> str:
        return self.add(fact, memory_type="semantic", source=source, **kwargs)


class EpisodicMemory(VectorStoreMemory):
    """Memory for past interactions and events."""
    
    def add_episode(self, description: str, participants: Optional[List[str]] = None, **kwargs: Any) -> str:
        return self.add(description, memory_type="episodic", participants=participants or [], **kwargs)


class ProceduralMemory(VectorStoreMemory):
    """Memory for learned behaviors and patterns."""
    
    def add_procedure(self, procedure: str, success_rate: float = 0.0, **kwargs: Any) -> str:
        return self.add(procedure, memory_type="procedural", success_rate=success_rate, **kwargs)
