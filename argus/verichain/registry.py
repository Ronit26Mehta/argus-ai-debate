"""
VERICHAIN Registry — persistent truth registry.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, Protocol
from dataclasses import dataclass

from argus.verichain.node import TruthNode, TruthNodeBuilder

logger = logging.getLogger(__name__)


class RegistryBackend(Protocol):
    """Protocol for truth registry backends."""
    def save_node(self, node: TruthNode) -> None: ...
    def load_node(self, node_id: str) -> Optional[TruthNode]: ...
    def load_all(self) -> list[TruthNode]: ...
    def search(self, query: str, top_k: int) -> list[TruthNode]: ...


class VERICHAINRegistry:
    """
    Persistent truth registry storing cross-debate verdicts.

    Supports in-memory, SQLite, and PostgreSQL backends.

    Example:
        >>> registry = VERICHAINRegistry(backend='sqlite', db_path='./truth.db')
        >>> registry.register_verdict(
        ...     proposition="Drug X is effective",
        ...     verdict="supported", posterior=0.78,
        ...     domain="clinical",
        ... )
        >>> results = registry.search("drug effectiveness", top_k=5)
    """

    def __init__(
        self,
        backend: str = "memory",
        db_path: str = "./verichain.db",
        **kwargs: Any,
    ):
        self.backend_type = backend
        self._nodes: dict[str, TruthNode] = {}
        self._db_backend: Optional[Any] = None

        if backend == "sqlite":
            from argus.verichain.backends.sqlite import SQLiteVERICHAINBackend
            self._db_backend = SQLiteVERICHAINBackend(db_path)

    def register_verdict(
        self,
        proposition: str,
        verdict: str,
        posterior: float,
        domain: str = "general",
        debate_id: str = "",
    ) -> TruthNode:
        """Register a new verdict in the chain."""
        # Build node
        prev_hash = ""
        if self._nodes:
            last_node = list(self._nodes.values())[-1]
            prev_hash = last_node.current_hash

        node = (TruthNodeBuilder()
                .proposition(proposition)
                .verdict(verdict, posterior)
                .domain(domain)
                .debate_id(debate_id)
                .build())
        node.prev_hash = prev_hash

        self._nodes[node.node_id] = node

        if self._db_backend:
            self._db_backend.save_node(node)

        logger.info(
            f"VERICHAIN registered: {node.node_id} "
            f"({verdict}, P={posterior:.3f})"
        )
        return node

    def update_verdict(
        self,
        node_id: str,
        verdict: str,
        posterior: float,
    ) -> Optional[TruthNode]:
        """Update an existing verdict with a new version."""
        node = self._nodes.get(node_id)
        if not node:
            return None

        builder = TruthNodeBuilder()
        builder._node = node
        builder.update(verdict, posterior)
        node.current_verdict = verdict
        node.current_posterior = posterior

        if self._db_backend:
            self._db_backend.save_node(node)

        return node

    def get_node(self, node_id: str) -> Optional[TruthNode]:
        return self._nodes.get(node_id)

    def search(self, query: str, top_k: int = 5) -> list[TruthNode]:
        """Simple text search over propositions."""
        query_lower = query.lower()
        scored = []
        for node in self._nodes.values():
            words = query_lower.split()
            match_count = sum(1 for w in words if w in node.proposition.lower())
            if match_count > 0:
                scored.append((node, match_count / max(len(words), 1)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:top_k]]

    @property
    def chain_length(self) -> int:
        return len(self._nodes)

    @property
    def all_nodes(self) -> list[TruthNode]:
        return list(self._nodes.values())

    def get_by_domain(self, domain: str) -> list[TruthNode]:
        return [n for n in self._nodes.values() if n.domain == domain]
