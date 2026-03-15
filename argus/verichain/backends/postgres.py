"""
PostgreSQL backend for VERICHAIN.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any

from argus.verichain.node import TruthNode, NodeVersion

logger = logging.getLogger(__name__)


class PostgreSQLVERICHAINBackend:
    """PostgreSQL backend for VERICHAIN truth registry."""

    DDL = """
    CREATE TABLE IF NOT EXISTS truth_nodes (
        node_id TEXT PRIMARY KEY,
        proposition TEXT NOT NULL,
        domain TEXT DEFAULT 'general',
        current_posterior REAL DEFAULT 0.5,
        current_verdict TEXT DEFAULT 'undetermined',
        created_at TIMESTAMP DEFAULT NOW(),
        prev_hash TEXT,
        signature TEXT,
        debate_id TEXT,
        citation_count INTEGER DEFAULT 0,
        embedding JSONB,
        versions JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_vtn_domain ON truth_nodes(domain);
    """

    def __init__(self, dsn: str = "postgresql://localhost/verichain"):
        self.dsn = dsn
        self._conn: Optional[Any] = None
        try:
            import psycopg2
            self._conn = psycopg2.connect(dsn)
            with self._conn.cursor() as cur:
                cur.execute(self.DDL)
            self._conn.commit()
        except ImportError:
            logger.warning("psycopg2 not installed")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")

    def save_node(self, node: TruthNode) -> None:
        if not self._conn:
            return
        versions_json = json.dumps([
            {"version": v.version, "posterior": v.posterior,
             "verdict": v.verdict, "updated_at": v.updated_at, "hash": v.hash}
            for v in node.versions
        ])
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO truth_nodes
                (node_id, proposition, domain, current_posterior, current_verdict,
                 created_at, prev_hash, debate_id, citation_count, versions)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (node_id) DO UPDATE SET
                    current_posterior = EXCLUDED.current_posterior,
                    current_verdict = EXCLUDED.current_verdict,
                    versions = EXCLUDED.versions""",
                (node.node_id, node.proposition, node.domain,
                 node.current_posterior, node.current_verdict,
                 node.created_at, node.prev_hash, node.debate_id,
                 node.citation_count, versions_json),
            )
        self._conn.commit()

    def load_all(self) -> list[TruthNode]:
        if not self._conn:
            return []
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM truth_nodes ORDER BY created_at")
            return [self._row_to_node(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_node(row: tuple) -> TruthNode:
        versions = []
        if row[11]:
            v_data = json.loads(row[11]) if isinstance(row[11], str) else row[11]
            for v in v_data:
                versions.append(NodeVersion(
                    version=v["version"], posterior=v["posterior"],
                    verdict=v["verdict"], updated_at=v.get("updated_at", ""),
                    hash=v.get("hash", ""),
                ))
        return TruthNode(
            node_id=row[0], proposition=row[1], domain=row[2],
            current_posterior=row[3] or 0.5, current_verdict=row[4] or "",
            versions=versions,
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
