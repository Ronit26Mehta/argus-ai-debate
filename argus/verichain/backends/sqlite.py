"""
SQLite backend for VERICHAIN.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from typing import Optional, Any
from pathlib import Path

from argus.verichain.node import TruthNode, NodeVersion

logger = logging.getLogger(__name__)


class SQLiteVERICHAINBackend:
    """SQLite backend for VERICHAIN truth registry."""

    DDL = """
    CREATE TABLE IF NOT EXISTS truth_nodes (
        node_id TEXT PRIMARY KEY,
        proposition TEXT NOT NULL,
        domain TEXT DEFAULT 'general',
        current_posterior REAL DEFAULT 0.5,
        current_verdict TEXT DEFAULT 'undetermined',
        created_at TEXT,
        prev_hash TEXT,
        signature TEXT,
        debate_id TEXT,
        citation_count INTEGER DEFAULT 0,
        embedding TEXT,
        versions TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_tn_domain ON truth_nodes(domain);
    """

    def __init__(self, db_path: str = "./verichain.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self.DDL)
        self._conn.commit()

    def save_node(self, node: TruthNode) -> None:
        versions_json = json.dumps([
            {"version": v.version, "posterior": v.posterior,
             "verdict": v.verdict, "updated_at": v.updated_at, "hash": v.hash}
            for v in node.versions
        ])
        self._conn.execute(
            """INSERT OR REPLACE INTO truth_nodes
            (node_id, proposition, domain, current_posterior, current_verdict,
             created_at, prev_hash, signature, debate_id, citation_count,
             embedding, versions) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (node.node_id, node.proposition, node.domain,
             node.current_posterior, node.current_verdict, node.created_at,
             node.prev_hash, node.signature, node.debate_id,
             node.citation_count,
             json.dumps(node.embedding) if node.embedding else None,
             versions_json),
        )
        self._conn.commit()

    def load_node(self, node_id: str) -> Optional[TruthNode]:
        cursor = self._conn.execute(
            "SELECT * FROM truth_nodes WHERE node_id = ?", (node_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_node(row)

    def load_all(self) -> list[TruthNode]:
        cursor = self._conn.execute("SELECT * FROM truth_nodes ORDER BY created_at")
        return [self._row_to_node(row) for row in cursor.fetchall()]

    def search(self, query: str, top_k: int = 5) -> list[TruthNode]:
        cursor = self._conn.execute(
            "SELECT * FROM truth_nodes WHERE proposition LIKE ? LIMIT ?",
            (f"%{query}%", top_k),
        )
        return [self._row_to_node(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_node(row: tuple) -> TruthNode:
        versions = []
        if row[11]:
            for v in json.loads(row[11]):
                versions.append(NodeVersion(
                    version=v["version"], posterior=v["posterior"],
                    verdict=v["verdict"], updated_at=v["updated_at"],
                    hash=v.get("hash", ""),
                ))
        return TruthNode(
            node_id=row[0], proposition=row[1], domain=row[2],
            current_posterior=row[3], current_verdict=row[4],
            created_at=row[5] or "", prev_hash=row[6] or "",
            signature=row[7] or "", debate_id=row[8] or "",
            citation_count=row[9] or 0,
            embedding=json.loads(row[10]) if row[10] else None,
            versions=versions,
        )

    def close(self) -> None:
        self._conn.close()
