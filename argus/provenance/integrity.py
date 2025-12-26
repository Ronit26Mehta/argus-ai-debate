"""
Integrity and Attestation for ARGUS.

Provides cryptographic integrity verification and attestation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass


def compute_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of content.
    
    Args:
        content: String content to hash
        algorithm: Hash algorithm (sha256, sha384, sha512)
        
    Returns:
        Hex-encoded hash
    """
    if algorithm == "sha256":
        return hashlib.sha256(content.encode()).hexdigest()
    elif algorithm == "sha384":
        return hashlib.sha384(content.encode()).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(content.encode()).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def verify_hash(content: str, expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify content matches expected hash.
    
    Args:
        content: Content to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm
        
    Returns:
        True if hash matches
    """
    actual = compute_hash(content, algorithm)
    return hmac.compare_digest(actual, expected_hash)


def compute_merkle_root(hashes: list[str]) -> str:
    """
    Compute Merkle root from list of hashes.
    
    Args:
        hashes: List of hex-encoded hashes
        
    Returns:
        Merkle root hash
    """
    if not hashes:
        return compute_hash("")
    
    if len(hashes) == 1:
        return hashes[0]
    
    # Pad to even length
    if len(hashes) % 2 == 1:
        hashes = hashes + [hashes[-1]]
    
    # Compute parent level
    parents = []
    for i in range(0, len(hashes), 2):
        combined = hashes[i] + hashes[i + 1]
        parents.append(compute_hash(combined))
    
    return compute_merkle_root(parents)


@dataclass
class Attestation:
    """
    Attestation for a piece of content.
    
    Attributes:
        content_hash: Hash of attested content
        timestamp: Attestation timestamp
        attester: Attester identifier
        algorithm: Hash algorithm used
        signature: Optional cryptographic signature
        metadata: Additional attestation data
    """
    content_hash: str
    timestamp: str
    attester: str
    algorithm: str = "sha256"
    signature: Optional[str] = None
    metadata: dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "attester": self.attester,
            "algorithm": self.algorithm,
            "signature": self.signature,
            "metadata": self.metadata,
        }
    
    def verify(self, content: str) -> bool:
        """
        Verify attestation against content.
        
        Args:
            content: Content to verify
            
        Returns:
            True if content matches attestation
        """
        return verify_hash(content, self.content_hash, self.algorithm)


def create_attestation(
    content: str,
    attester: str,
    algorithm: str = "sha256",
    metadata: Optional[dict[str, Any]] = None,
) -> Attestation:
    """
    Create an attestation for content.
    
    Args:
        content: Content to attest
        attester: Attester identifier (agent, system, etc.)
        algorithm: Hash algorithm
        metadata: Additional metadata
        
    Returns:
        Attestation object
    """
    content_hash = compute_hash(content, algorithm)
    
    return Attestation(
        content_hash=content_hash,
        timestamp=datetime.utcnow().isoformat(),
        attester=attester,
        algorithm=algorithm,
        metadata=metadata or {},
    )


def create_batch_attestation(
    contents: list[str],
    attester: str,
    algorithm: str = "sha256",
) -> tuple[Attestation, list[str]]:
    """
    Create attestation for multiple contents using Merkle tree.
    
    Args:
        contents: List of contents to attest
        attester: Attester identifier
        algorithm: Hash algorithm
        
    Returns:
        (Attestation with Merkle root, list of individual hashes)
    """
    individual_hashes = [compute_hash(c, algorithm) for c in contents]
    merkle_root = compute_merkle_root(individual_hashes)
    
    attestation = Attestation(
        content_hash=merkle_root,
        timestamp=datetime.utcnow().isoformat(),
        attester=attester,
        algorithm=algorithm,
        metadata={
            "type": "merkle_root",
            "num_items": len(contents),
        },
    )
    
    return attestation, individual_hashes


class IntegrityChecker:
    """
    Utility for checking content integrity.
    
    Maintains a registry of attestations and verifies content.
    """
    
    def __init__(self):
        self._attestations: dict[str, Attestation] = {}
    
    def register(
        self,
        content_id: str,
        content: str,
        attester: str,
    ) -> Attestation:
        """
        Register content and create attestation.
        
        Args:
            content_id: Unique content identifier
            content: Content to attest
            attester: Attester identifier
            
        Returns:
            Created attestation
        """
        attestation = create_attestation(content, attester)
        self._attestations[content_id] = attestation
        return attestation
    
    def verify(
        self,
        content_id: str,
        content: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Verify content against registered attestation.
        
        Args:
            content_id: Content identifier
            content: Content to verify
            
        Returns:
            (is_valid, error_message)
        """
        if content_id not in self._attestations:
            return False, "No attestation found"
        
        attestation = self._attestations[content_id]
        if attestation.verify(content):
            return True, None
        else:
            return False, "Content hash mismatch"
    
    def get_attestation(self, content_id: str) -> Optional[Attestation]:
        """Get attestation by content ID."""
        return self._attestations.get(content_id)
