"""
ARGUS Checkpointer.

State checkpointing for durable workflow execution.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A workflow checkpoint."""
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    step: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class BaseCheckpointer(ABC):
    """Abstract base class for checkpointers."""
    
    @abstractmethod
    def save(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str:
        pass
    
    @abstractmethod
    def load(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        pass
    
    @abstractmethod
    def list_checkpoints(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        pass
    
    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        pass


class MemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpointer for development/testing."""
    
    def __init__(self, max_checkpoints: int = 100):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._by_thread: Dict[str, List[str]] = {}
    
    def save(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str:
        parent_id = None
        if thread_id in self._by_thread and self._by_thread[thread_id]:
            parent_id = self._by_thread[thread_id][-1]
        checkpoint = Checkpoint(
            thread_id=thread_id, step=step, state=state,
            metadata=metadata, parent_id=parent_id
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        if thread_id not in self._by_thread:
            self._by_thread[thread_id] = []
        self._by_thread[thread_id].append(checkpoint.checkpoint_id)
        if len(self._checkpoints) > self.max_checkpoints:
            oldest = min(self._checkpoints.values(), key=lambda c: c.created_at)
            self.delete(oldest.checkpoint_id)
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} for thread {thread_id}")
        return checkpoint.checkpoint_id
    
    def load(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        if checkpoint_id:
            return self._checkpoints.get(checkpoint_id)
        if thread_id in self._by_thread and self._by_thread[thread_id]:
            latest_id = self._by_thread[thread_id][-1]
            return self._checkpoints.get(latest_id)
        return None
    
    def list_checkpoints(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        ids = self._by_thread.get(thread_id, [])
        return [self._checkpoints[cid] for cid in ids[-limit:] if cid in self._checkpoints]
    
    def delete(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self._checkpoints:
            cp = self._checkpoints.pop(checkpoint_id)
            if cp.thread_id in self._by_thread:
                self._by_thread[cp.thread_id] = [c for c in self._by_thread[cp.thread_id] if c != checkpoint_id]
            return True
        return False


class SQLiteCheckpointer(BaseCheckpointer):
    """SQLite-based persistent checkpointer."""
    
    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                state TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                parent_id TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_thread ON checkpoints(thread_id)")
        self._conn.commit()
    
    def save(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str:
        parent = self.load(thread_id)
        parent_id = parent.checkpoint_id if parent else None
        checkpoint = Checkpoint(
            thread_id=thread_id, step=step, state=state,
            metadata=metadata, parent_id=parent_id
        )
        self._conn.execute("""
            INSERT INTO checkpoints (checkpoint_id, thread_id, step, state, metadata, created_at, parent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (checkpoint.checkpoint_id, thread_id, step, json.dumps(state),
              json.dumps(metadata), checkpoint.created_at.isoformat(), parent_id))
        self._conn.commit()
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id}")
        return checkpoint.checkpoint_id
    
    def load(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        if checkpoint_id:
            cursor = self._conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
        else:
            cursor = self._conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1",
                (thread_id,)
            )
        row = cursor.fetchone()
        if row:
            return Checkpoint(
                checkpoint_id=row[0], thread_id=row[1], step=row[2],
                state=json.loads(row[3]), metadata=json.loads(row[4] or "{}"),
                created_at=datetime.fromisoformat(row[5]), parent_id=row[6]
            )
        return None
    
    def list_checkpoints(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        cursor = self._conn.execute(
            "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?",
            (thread_id, limit)
        )
        return [Checkpoint(
            checkpoint_id=r[0], thread_id=r[1], step=r[2],
            state=json.loads(r[3]), metadata=json.loads(r[4] or "{}"),
            created_at=datetime.fromisoformat(r[5]), parent_id=r[6]
        ) for r in cursor.fetchall()]
    
    def delete(self, checkpoint_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
        self._conn.commit()
        return cursor.rowcount > 0


class FileSystemCheckpointer(BaseCheckpointer):
    """File-based JSON checkpointer."""
    
    def __init__(self, base_path: str = "./checkpoints"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_thread_dir(self, thread_id: str) -> Path:
        return self.base_path / thread_id
    
    def save(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str:
        thread_dir = self._get_thread_dir(thread_id)
        thread_dir.mkdir(exist_ok=True)
        parent = self.load(thread_id)
        checkpoint = Checkpoint(
            thread_id=thread_id, step=step, state=state,
            metadata=metadata, parent_id=parent.checkpoint_id if parent else None
        )
        path = thread_dir / f"{checkpoint.checkpoint_id}.json"
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2))
        logger.debug(f"Saved checkpoint to {path}")
        return checkpoint.checkpoint_id
    
    def load(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        thread_dir = self._get_thread_dir(thread_id)
        if not thread_dir.exists():
            return None
        if checkpoint_id:
            path = thread_dir / f"{checkpoint_id}.json"
            if path.exists():
                return Checkpoint.from_dict(json.loads(path.read_text()))
        else:
            files = list(thread_dir.glob("*.json"))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                return Checkpoint.from_dict(json.loads(latest.read_text()))
        return None
    
    def list_checkpoints(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        thread_dir = self._get_thread_dir(thread_id)
        if not thread_dir.exists():
            return []
        files = sorted(thread_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [Checkpoint.from_dict(json.loads(f.read_text())) for f in files[:limit]]
    
    def delete(self, checkpoint_id: str) -> bool:
        for thread_dir in self.base_path.iterdir():
            if thread_dir.is_dir():
                path = thread_dir / f"{checkpoint_id}.json"
                if path.exists():
                    path.unlink()
                    return True
        return False
