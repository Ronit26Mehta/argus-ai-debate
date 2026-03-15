"""
SQLite backend for MNEME persistent memory.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Any
from pathlib import Path

from argus.mneme.reservoir import ReservoirEntry

logger = logging.getLogger(__name__)


class SQLiteMemoryBackend:
    """
    SQLite-based persistent memory backend.

    Stores ReservoirEntry objects in a SQLite database for
    persistence across sessions.

    Example:
        >>> backend = SQLiteMemoryBackend("./memory.db")
        >>> backend.save_entry(entry)
        >>> entries = backend.load_all()
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS knowledge_entries (
        entry_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        domain TEXT DEFAULT 'general',
        embedding TEXT,
        confidence REAL DEFAULT 0.5,
        source_debate_id TEXT,
        proposition_text TEXT,
        verdict TEXT,
        created_at TEXT,
        access_count INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS expertise_profiles (
        agent_id TEXT NOT NULL,
        domain TEXT NOT NULL,
        alpha REAL DEFAULT 2.0,
        beta REAL DEFAULT 2.0,
        total_evaluations INTEGER DEFAULT 0,
        PRIMARY KEY (agent_id, domain)
    );
    CREATE TABLE IF NOT EXISTS calibration_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id TEXT NOT NULL,
        domain TEXT NOT NULL,
        predicted REAL,
        actual REAL,
        recorded_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_entries_domain ON knowledge_entries(domain);
    CREATE INDEX IF NOT EXISTS idx_expertise_agent ON expertise_profiles(agent_id);
    """

    def __init__(self, db_path: str = "./mneme_memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(self.DDL)
        self._conn.commit()
        logger.info(f"MNEME SQLite backend initialized: {self.db_path}")

    def save_entry(self, entry: ReservoirEntry) -> None:
        """Save a ReservoirEntry to database."""
        self._conn.execute(
            """INSERT OR REPLACE INTO knowledge_entries
            (entry_id, text, domain, embedding, confidence, source_debate_id,
             proposition_text, verdict, created_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.entry_id, entry.text, entry.domain,
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.confidence, entry.source_debate_id,
                entry.proposition_text, entry.verdict,
                entry.created_at.isoformat(), entry.access_count,
            ),
        )
        self._conn.commit()

    def load_all(self, domain: Optional[str] = None) -> list[ReservoirEntry]:
        """Load all entries, optionally filtered by domain."""
        query = "SELECT * FROM knowledge_entries"
        params: list[Any] = []
        if domain:
            query += " WHERE domain = ?"
            params.append(domain)
        query += " ORDER BY created_at DESC"

        cursor = self._conn.execute(query, params)
        entries = []
        for row in cursor.fetchall():
            entry = ReservoirEntry(
                entry_id=row[0], text=row[1], domain=row[2],
                embedding=json.loads(row[3]) if row[3] else None,
                confidence=row[4], source_debate_id=row[5] or "",
                proposition_text=row[6] or "", verdict=row[7] or "",
                created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                access_count=row[9] or 0,
            )
            entries.append(entry)
        return entries

    def save_expertise(
        self, agent_id: str, domain: str,
        alpha: float, beta: float, total_evaluations: int,
    ) -> None:
        """Save expertise profile."""
        self._conn.execute(
            """INSERT OR REPLACE INTO expertise_profiles
            (agent_id, domain, alpha, beta, total_evaluations)
            VALUES (?, ?, ?, ?, ?)""",
            (agent_id, domain, alpha, beta, total_evaluations),
        )
        self._conn.commit()

    def save_calibration(
        self, agent_id: str, domain: str,
        predicted: float, actual: float,
    ) -> None:
        """Save calibration record."""
        self._conn.execute(
            """INSERT INTO calibration_records
            (agent_id, domain, predicted, actual, recorded_at)
            VALUES (?, ?, ?, ?, ?)""",
            (agent_id, domain, predicted, actual, datetime.utcnow().isoformat()),
        )
        self._conn.commit()

    def count_entries(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM knowledge_entries")
        return cursor.fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
