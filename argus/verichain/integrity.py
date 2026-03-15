"""
Chain Integrity — hash-chain verification and tamper detection.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.verichain.node import TruthNode, NodeVersion

logger = logging.getLogger(__name__)


@dataclass
class HashChain:
    """Represents the hash chain of the VERICHAIN."""
    chain_length: int = 0
    is_valid: bool = True
    first_hash: str = ""
    last_hash: str = ""
    broken_links: list[int] = field(default_factory=list)


class TamperDetector:
    """Detects tampering in node version history."""

    @staticmethod
    def verify_node(node: TruthNode) -> bool:
        """Verify internal hash chain consistency of a single node."""
        prev_hash = node.prev_hash
        for version in node.versions:
            expected_payload = (
                f"{version.version}:{version.posterior}:"
                f"{version.verdict}:{version.updated_at}:{prev_hash}"
            )
            expected_hash = hashlib.sha256(expected_payload.encode()).hexdigest()
            if version.hash and version.hash != expected_hash:
                return False
            prev_hash = version.hash or expected_hash
        return True


class ChainVerifier:
    """
    Verifies hash-chain integrity of the VERICHAIN.

    Walks the full chain and verifies each link. Reports any
    broken links for investigation.

    Example:
        >>> verifier = ChainVerifier()
        >>> result = verifier.verify_chain(registry.all_nodes)
        >>> print(f"Chain valid: {result.is_valid}")
    """

    def __init__(self):
        self._tamper_detector = TamperDetector()

    def verify_chain(self, nodes: list[TruthNode]) -> HashChain:
        """Verify the full VERICHAIN."""
        chain = HashChain(chain_length=len(nodes))

        if not nodes:
            return chain

        chain.first_hash = nodes[0].current_hash if nodes[0].current_hash else ""
        chain.last_hash = nodes[-1].current_hash if nodes[-1].current_hash else ""

        prev_hash = ""
        for i, node in enumerate(nodes):
            # Verify internal consistency
            if not self._tamper_detector.verify_node(node):
                chain.broken_links.append(i)
                chain.is_valid = False
                logger.warning(f"Tamper detected in node {node.node_id} at index {i}")

            # Verify chain link
            if node.prev_hash and prev_hash and node.prev_hash != prev_hash:
                chain.broken_links.append(i)
                chain.is_valid = False
                logger.warning(f"Chain break at index {i}: expected {prev_hash[:16]}, got {node.prev_hash[:16]}")

            prev_hash = node.current_hash

        if chain.is_valid:
            logger.info(f"VERICHAIN verified: {chain.chain_length} nodes, integrity OK")
        else:
            logger.warning(
                f"VERICHAIN integrity FAILED: {len(chain.broken_links)} broken links"
            )

        return chain

    def verify_single(self, node: TruthNode) -> bool:
        """Verify single node's internal integrity."""
        return self._tamper_detector.verify_node(node)
