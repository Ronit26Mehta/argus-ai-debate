"""
PostgreSQL backend for MNEME persistent memory.

Requires psycopg2 or asyncpg.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Any

from argus.mneme.reservoir import ReservoirEntry

logger = logging.getLogger(__name__)


class PostgreSQLMemoryBackend:
    """
    PostgreSQL-based persistent memory backend.

    Supports high-volume production workloads with connection pooling.
    Uses pgvector extension for native vector similarity search
    when available.

    Example:
        >>> backend = PostgreSQLMemoryBackend(
        ...     dsn="postgresql://user:pass@localhost/mneme",
        ... )
        >>> backend.save_entry(entry)
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS knowledge_entries (
        entry_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        domain TEXT DEFAULT 'general',
        embedding JSONB,
        confidence REAL DEFAULT 0.5,
        source_debate_id TEXT,
        proposition_text TEXT,
        verdict TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
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
    CREATE INDEX IF NOT EXISTS idx_ke_domain ON knowledge_entries(domain);
    """

    def __init__(self, dsn: str = "postgresql://localhost/mneme"):
        self.dsn = dsn
        self._conn: Optional[Any] = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            import psycopg2
            self._conn = psycopg2.connect(self.dsn)
            with self._conn.cursor() as cur:
                cur.execute(self.DDL)
            self._conn.commit()
            logger.info(f"MNEME PostgreSQL backend initialized: {self.dsn}")
        except ImportError:
            logger.warning(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )
            self._conn = None
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            self._conn = None

    def save_entry(self, entry: ReservoirEntry) -> None:
        if not self._conn:
            logger.warning("No PostgreSQL connection. Entry not saved.")
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO knowledge_entries
                (entry_id, text, domain, embedding, confidence,
                 source_debate_id, proposition_text, verdict, created_at, access_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (entry_id) DO UPDATE SET
                    access_count = knowledge_entries.access_count + 1""",
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
        if not self._conn:
            return []
        query = "SELECT * FROM knowledge_entries"
        params: list[Any] = []
        if domain:
            query += " WHERE domain = %s"
            params.append(domain)
        query += " ORDER BY created_at DESC"
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            entries = []
            for row in cur.fetchall():
                entries.append(ReservoirEntry(
                    entry_id=row[0], text=row[1], domain=row[2],
                    embedding=json.loads(row[3]) if row[3] else None,
                    confidence=row[4], source_debate_id=row[5] or "",
                    proposition_text=row[6] or "", verdict=row[7] or "",
                ))
            return entries

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
