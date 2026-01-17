"""
ARGUS Memory Store Backends.

Abstract storage backends for memory persistence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, List, Dict

import numpy as np

logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def save(self, key: str, data: dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[dict[str, Any]]:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def list_keys(self) -> List[str]:
        pass
    
    @abstractmethod
    def clear(self) -> int:
        pass


class InMemoryStore(MemoryStore):
    """Simple in-memory storage for testing/development."""
    
    def __init__(self):
        self._data: Dict[str, dict[str, Any]] = {}
    
    def save(self, key: str, data: dict[str, Any]) -> bool:
        self._data[key] = data
        return True
    
    def load(self, key: str) -> Optional[dict[str, Any]]:
        return self._data.get(key)
    
    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    def list_keys(self) -> List[str]:
        return list(self._data.keys())
    
    def clear(self) -> int:
        count = len(self._data)
        self._data.clear()
        return count


class SQLiteStore(MemoryStore):
    """SQLite-based persistent storage."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()
    
    def save(self, key: str, data: dict[str, Any]) -> bool:
        try:
            json_data = json.dumps(data)
            self._conn.execute("""
                INSERT OR REPLACE INTO memories (key, data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, json_data))
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save to SQLite: {e}")
            return False
    
    def load(self, key: str) -> Optional[dict[str, Any]]:
        cursor = self._conn.execute("SELECT data FROM memories WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def delete(self, key: str) -> bool:
        cursor = self._conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0
    
    def list_keys(self) -> List[str]:
        cursor = self._conn.execute("SELECT key FROM memories")
        return [row[0] for row in cursor.fetchall()]
    
    def clear(self) -> int:
        cursor = self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        return cursor.rowcount
    
    def close(self) -> None:
        self._conn.close()


class FAISSStore(MemoryStore):
    """FAISS vector store for embeddings."""
    
    def __init__(self, dimension: int = 384, persistence_path: Optional[str] = None):
        self.dimension = dimension
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._metadata: Dict[str, dict[str, Any]] = {}
        self._embeddings: List[np.ndarray] = []
        self._keys: List[str] = []
        self._index: Optional[Any] = None
        self._init_index()
    
    def _init_index(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            logger.warning("FAISS not available")
    
    def save(self, key: str, data: dict[str, Any]) -> bool:
        embedding = data.get("embedding")
        if embedding is not None:
            emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
            if self._index is not None:
                self._index.add(emb)
            self._embeddings.append(emb.flatten())
            self._keys.append(key)
        self._metadata[key] = {k: v for k, v in data.items() if k != "embedding"}
        return True
    
    def load(self, key: str) -> Optional[dict[str, Any]]:
        data = self._metadata.get(key)
        if data and key in self._keys:
            idx = self._keys.index(key)
            data = data.copy()
            data["embedding"] = self._embeddings[idx].tolist()
        return data
    
    def delete(self, key: str) -> bool:
        if key in self._metadata:
            del self._metadata[key]
            if key in self._keys:
                idx = self._keys.index(key)
                self._keys.pop(idx)
                self._embeddings.pop(idx)
            return True
        return False
    
    def list_keys(self) -> List[str]:
        return list(self._metadata.keys())
    
    def clear(self) -> int:
        count = len(self._metadata)
        self._metadata.clear()
        self._embeddings.clear()
        self._keys.clear()
        self._init_index()
        return count
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple[str, float]]:
        if not self._embeddings:
            return []
        query = query_embedding.reshape(1, -1).astype(np.float32)
        if self._index is not None:
            D, I = self._index.search(query, min(top_k, len(self._keys)))
            return [(self._keys[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self._keys)]
        scores = [float(np.dot(query.flatten(), e)) for e in self._embeddings]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self._keys[i], scores[i]) for i in indices]


class FileSystemStore(MemoryStore):
    """File-based JSON storage."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.base_path / f"{safe_key}.json"
    
    def save(self, key: str, data: dict[str, Any]) -> bool:
        try:
            path = self._get_path(key)
            path.write_text(json.dumps(data, indent=2, default=str))
            return True
        except Exception as e:
            logger.error(f"Failed to save to file: {e}")
            return False
    
    def load(self, key: str) -> Optional[dict[str, Any]]:
        path = self._get_path(key)
        if path.exists():
            return json.loads(path.read_text())
        return None
    
    def delete(self, key: str) -> bool:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_keys(self) -> List[str]:
        return [p.stem for p in self.base_path.glob("*.json")]
    
    def clear(self) -> int:
        count = 0
        for p in self.base_path.glob("*.json"):
            p.unlink()
            count += 1
        return count
