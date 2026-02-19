"""
Epistemic Dead Reckoning (EDR) for CRUX Protocol.

Agents in a long-running ARGUS session may disconnect mid-debate —
network failure, restart, or provider timeout. Epistemic Dead Reckoning
allows a reconnecting agent to reconstruct its belief state without
replaying the full session, using only its last checkpoint hash and
a compact delta bundle.

Key Features:
    - Checkpoint creation and management
    - Delta bundle computation
    - Conflict detection on reconnect
    - Idempotent sync operations
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

from argus.crux.models import (
    generate_crux_id,
    compute_crux_hash,
    EDRDelta,
    BRPResolution,
    CRUXConfig,
)

if TYPE_CHECKING:
    from argus.crux.claim_bundle import ClaimBundle
    from argus.crux.ledger import CredibilityLedger, CredibilityEntry, CredibilityUpdate
    from argus.crux.brp import BeliefReconciliationProtocol

logger = logging.getLogger(__name__)


@dataclass
class EDRCheckpoint:
    """
    A checkpoint in the EDR system.
    
    Captures an agent's belief state at a point in time.
    
    Attributes:
        checkpoint_id: Unique checkpoint identifier
        agent_id: Agent this checkpoint belongs to
        checkpoint_hash: Hash of checkpoint state
        ledger_hash: Credibility ledger hash at checkpoint
        claim_bundle_ids: Active Claim Bundle IDs
        beliefs: Agent's beliefs (proposition_id -> posterior)
        timestamp: Checkpoint creation time
        metadata: Additional checkpoint data
    """
    checkpoint_id: str = field(default_factory=lambda: generate_crux_id("ckpt"))
    agent_id: str = ""
    checkpoint_hash: str = ""
    ledger_hash: str = ""
    claim_bundle_ids: list[str] = field(default_factory=list)
    beliefs: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.checkpoint_hash:
            self.checkpoint_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute checkpoint hash."""
        content = json.dumps({
            "agent_id": self.agent_id,
            "claim_bundle_ids": sorted(self.claim_bundle_ids),
            "beliefs": dict(sorted(self.beliefs.items())),
            "timestamp": self.timestamp.isoformat(),
        }, sort_keys=True)
        return compute_crux_hash(content)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "agent_id": self.agent_id,
            "checkpoint_hash": self.checkpoint_hash,
            "ledger_hash": self.ledger_hash,
            "claim_bundle_ids": self.claim_bundle_ids,
            "beliefs": self.beliefs,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EDRCheckpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            agent_id=data["agent_id"],
            checkpoint_hash=data["checkpoint_hash"],
            ledger_hash=data.get("ledger_hash", ""),
            claim_bundle_ids=data.get("claim_bundle_ids", []),
            beliefs=data.get("beliefs", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BeliefConflict:
    """
    A conflict between agent's checkpoint beliefs and current state.
    
    Attributes:
        proposition_id: Conflicting proposition
        checkpoint_belief: Agent's belief at checkpoint
        current_belief: Current system belief
        divergence: Absolute difference
        conflicting_bundles: Claim Bundles causing conflict
    """
    proposition_id: str
    checkpoint_belief: float
    current_belief: float
    divergence: float = 0.0
    conflicting_bundles: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.divergence = abs(self.checkpoint_belief - self.current_belief)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposition_id": self.proposition_id,
            "checkpoint_belief": self.checkpoint_belief,
            "current_belief": self.current_belief,
            "divergence": self.divergence,
            "conflicting_bundles": self.conflicting_bundles,
        }


@dataclass
class SyncResult:
    """
    Result of EDR synchronization.
    
    Attributes:
        success: Whether sync succeeded
        agent_id: Agent that synced
        delta: Delta bundle applied
        conflicts: Conflicts detected
        resolutions: BRP resolutions applied
        sync_duration_ms: Sync duration in milliseconds
    """
    success: bool
    agent_id: str
    delta: EDRDelta
    conflicts: list[BeliefConflict] = field(default_factory=list)
    resolutions: list[BRPResolution] = field(default_factory=list)
    sync_duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "agent_id": self.agent_id,
            "delta": self.delta.to_dict(),
            "num_conflicts": len(self.conflicts),
            "num_resolutions": len(self.resolutions),
            "sync_duration_ms": self.sync_duration_ms,
        }


class EDRSynchronizer:
    """
    Synchronizer for Epistemic Dead Reckoning.
    
    Manages delta computation and conflict resolution for
    reconnecting agents.
    
    Example:
        >>> sync = EDRSynchronizer(ledger, brp)
        >>> 
        >>> # Agent reconnects with checkpoint
        >>> delta = sync.compute_delta(agent_id, last_checkpoint_hash)
        >>> 
        >>> # Apply delta with conflict resolution
        >>> result = sync.apply_delta(agent_id, delta)
    """
    
    def __init__(
        self,
        ledger: Optional["CredibilityLedger"] = None,
        brp: Optional["BeliefReconciliationProtocol"] = None,
        config: Optional[CRUXConfig] = None,
    ):
        """
        Initialize synchronizer.
        
        Args:
            ledger: Credibility ledger for state lookup
            brp: BRP for conflict resolution
            config: CRUX configuration
        """
        self.ledger = ledger
        self.brp = brp
        self.config = config or CRUXConfig()
        
        # Checkpoint storage
        self._checkpoints: dict[str, list[EDRCheckpoint]] = defaultdict(list)
        
        # Active Claim Bundles
        self._active_bundles: dict[str, "ClaimBundle"] = {}
        
        # Current beliefs by proposition
        self._current_beliefs: dict[str, float] = {}
    
    def create_checkpoint(
        self,
        agent_id: str,
        claim_bundles: Optional[list["ClaimBundle"]] = None,
    ) -> EDRCheckpoint:
        """
        Create a checkpoint for an agent.
        
        Args:
            agent_id: Agent ID
            claim_bundles: Current active Claim Bundles
            
        Returns:
            New EDRCheckpoint
        """
        # Get agent's current beliefs
        beliefs = {}
        bundle_ids = []
        
        if claim_bundles:
            for cb in claim_bundles:
                bundle_ids.append(cb.cb_id)
                if cb.proposition_id:
                    beliefs[cb.proposition_id] = cb.posterior
        
        # Get ledger hash
        ledger_hash = ""
        if self.ledger:
            ledger_hash = self.ledger.head_hash
        
        checkpoint = EDRCheckpoint(
            agent_id=agent_id,
            ledger_hash=ledger_hash,
            claim_bundle_ids=bundle_ids,
            beliefs=beliefs,
        )
        
        # Store checkpoint
        self._checkpoints[agent_id].append(checkpoint)
        
        logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for {agent_id}")
        
        return checkpoint
    
    def find_checkpoint(
        self,
        checkpoint_hash: str,
    ) -> Optional[EDRCheckpoint]:
        """
        Find a checkpoint by hash.
        
        Args:
            checkpoint_hash: Hash to search for
            
        Returns:
            Checkpoint if found
        """
        for agent_checkpoints in self._checkpoints.values():
            for ckpt in agent_checkpoints:
                if ckpt.checkpoint_hash == checkpoint_hash:
                    return ckpt
        return None
    
    def get_latest_checkpoint(
        self,
        agent_id: str,
    ) -> Optional[EDRCheckpoint]:
        """Get most recent checkpoint for an agent."""
        agent_ckpts = self._checkpoints.get(agent_id, [])
        if not agent_ckpts:
            return None
        return max(agent_ckpts, key=lambda c: c.timestamp)
    
    def register_bundle(self, bundle: "ClaimBundle") -> None:
        """Register an active Claim Bundle."""
        self._active_bundles[bundle.cb_id] = bundle
        
        # Update current beliefs
        if bundle.proposition_id:
            self._current_beliefs[bundle.proposition_id] = bundle.posterior
    
    def compute_delta(
        self,
        agent_id: str,
        last_checkpoint_hash: str,
    ) -> EDRDelta:
        """
        Compute delta from a checkpoint to current state.
        
        This is the core EDR algorithm:
        1. Locate checkpoint in history
        2. Collect all Claim Bundles issued after checkpoint
        3. Include relevant BRP resolutions
        4. Include credibility updates
        
        Args:
            agent_id: Reconnecting agent ID
            last_checkpoint_hash: Agent's last known checkpoint hash
            
        Returns:
            EDRDelta with all changes since checkpoint
        """
        # Find checkpoint
        checkpoint = self.find_checkpoint(last_checkpoint_hash)
        if not checkpoint:
            logger.warning(f"Checkpoint not found: {last_checkpoint_hash}")
            # Return delta with all current state
            return EDRDelta(
                checkpoint_hash=last_checkpoint_hash,
                claim_bundles=list(self._active_bundles.values()),
                current_head_hash=self.ledger.head_hash if self.ledger else "",
            )
        
        # Get bundles issued after checkpoint
        new_bundles = [
            cb for cb in self._active_bundles.values()
            if cb.issued_at > checkpoint.timestamp
        ]
        
        # Get BRP resolutions since checkpoint
        resolutions = []
        if self.brp:
            for session in self.brp._history:
                if session.completed_at and session.completed_at > checkpoint.timestamp:
                    if session.resolution:
                        resolutions.append(session.resolution)
        
        # Get credibility updates since checkpoint
        cred_updates = []
        if self.ledger:
            entries = self.ledger.get_entries_since(checkpoint.timestamp)
            for entry in entries:
                if entry.agent_id == agent_id and entry.ground_truth is not None:
                    cred_updates.append(entry)
        
        delta = EDRDelta(
            checkpoint_hash=last_checkpoint_hash,
            claim_bundles=new_bundles,
            resolutions=resolutions,
            credibility_updates=cred_updates,
            current_head_hash=self.ledger.head_hash if self.ledger else "",
        )
        
        logger.info(
            f"Computed delta for {agent_id}: "
            f"{len(new_bundles)} bundles, {len(resolutions)} resolutions"
        )
        
        return delta
    
    def detect_conflicts(
        self,
        checkpoint: EDRCheckpoint,
        threshold: float = 0.20,
    ) -> list[BeliefConflict]:
        """
        Detect conflicts between checkpoint beliefs and current state.
        
        Args:
            checkpoint: Agent's checkpoint
            threshold: Divergence threshold for conflict
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        for prop_id, ckpt_belief in checkpoint.beliefs.items():
            current_belief = self._current_beliefs.get(prop_id)
            
            if current_belief is None:
                continue
            
            divergence = abs(ckpt_belief - current_belief)
            if divergence > threshold:
                # Find conflicting bundles
                conflicting = [
                    cb.cb_id for cb in self._active_bundles.values()
                    if cb.proposition_id == prop_id and 
                    cb.issued_at > checkpoint.timestamp
                ]
                
                conflicts.append(BeliefConflict(
                    proposition_id=prop_id,
                    checkpoint_belief=ckpt_belief,
                    current_belief=current_belief,
                    conflicting_bundles=conflicting,
                ))
        
        return conflicts
    
    def sync(
        self,
        agent_id: str,
        last_checkpoint_hash: str,
    ) -> SyncResult:
        """
        Full synchronization for a reconnecting agent.
        
        This is the main EDR entry point:
        1. Compute delta from checkpoint
        2. Detect conflicts with agent's last beliefs
        3. Trigger BRP for each conflict
        4. Return sync result
        
        EDR is designed to be idempotent — calling sync() multiple times
        with the same checkpoint hash returns the same delta.
        
        Args:
            agent_id: Reconnecting agent
            last_checkpoint_hash: Last known checkpoint hash
            
        Returns:
            SyncResult with delta and resolutions
        """
        import time
        start_time = time.time()
        
        # Compute delta
        delta = self.compute_delta(agent_id, last_checkpoint_hash)
        
        # Find checkpoint for conflict detection
        checkpoint = self.find_checkpoint(last_checkpoint_hash)
        conflicts = []
        resolutions = []
        
        if checkpoint:
            # Detect conflicts
            conflicts = self.detect_conflicts(
                checkpoint,
                self.config.contradiction_threshold,
            )
            
            # Resolve conflicts via BRP (if BRP available)
            if self.brp and conflicts:
                for conflict in conflicts:
                    # Get conflicting bundles for BRP
                    conflicting_bundles = [
                        self._active_bundles[cb_id]
                        for cb_id in conflict.conflicting_bundles
                        if cb_id in self._active_bundles
                    ]
                    
                    # This would trigger BRP reconciliation
                    # Simplified here - in full implementation would spawn BRP sessions
                    logger.info(
                        f"Conflict on {conflict.proposition_id}: "
                        f"divergence={conflict.divergence:.3f}"
                    )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = SyncResult(
            success=True,
            agent_id=agent_id,
            delta=delta,
            conflicts=conflicts,
            resolutions=resolutions,
            sync_duration_ms=elapsed_ms,
        )
        
        # Create new checkpoint for reconnected agent
        self.create_checkpoint(
            agent_id,
            list(self._active_bundles.values()),
        )
        
        logger.info(
            f"Sync completed for {agent_id}: "
            f"{delta.num_changes} changes, {len(conflicts)} conflicts"
        )
        
        return result


class EpistemicDeadReckoning:
    """
    Epistemic Dead Reckoning (EDR) - CRUX Primitive 6.
    
    High-level interface for EDR operations.
    
    Example:
        >>> edr = EpistemicDeadReckoning(ledger, brp, config)
        >>> 
        >>> # Create checkpoint before agent goes offline
        >>> checkpoint = edr.checkpoint(agent_id, current_bundles)
        >>> 
        >>> # Later, when agent reconnects
        >>> result = edr.reconnect(agent_id, checkpoint.checkpoint_hash)
        >>> 
        >>> # Apply delta to update agent's state
        >>> for bundle in result.delta.claim_bundles:
        ...     agent.process_bundle(bundle)
    """
    
    def __init__(
        self,
        ledger: Optional["CredibilityLedger"] = None,
        brp: Optional["BeliefReconciliationProtocol"] = None,
        config: Optional[CRUXConfig] = None,
    ):
        """
        Initialize EDR.
        
        Args:
            ledger: Credibility ledger
            brp: Belief Reconciliation Protocol
            config: CRUX configuration
        """
        self.config = config or CRUXConfig()
        self.synchronizer = EDRSynchronizer(ledger, brp, config)
        self._enabled = config.enable_edr if config else True
    
    @property
    def enabled(self) -> bool:
        """Check if EDR is enabled."""
        return self._enabled
    
    def checkpoint(
        self,
        agent_id: str,
        claim_bundles: Optional[list["ClaimBundle"]] = None,
    ) -> EDRCheckpoint:
        """
        Create a checkpoint for an agent.
        
        Args:
            agent_id: Agent ID
            claim_bundles: Current active bundles
            
        Returns:
            EDRCheckpoint
        """
        if not self._enabled:
            raise RuntimeError("EDR is disabled")
        
        return self.synchronizer.create_checkpoint(agent_id, claim_bundles)
    
    def reconnect(
        self,
        agent_id: str,
        last_checkpoint_hash: str,
    ) -> SyncResult:
        """
        Reconnect an agent using EDR.
        
        Args:
            agent_id: Reconnecting agent
            last_checkpoint_hash: Last known checkpoint hash
            
        Returns:
            SyncResult with delta and any resolutions
        """
        if not self._enabled:
            raise RuntimeError("EDR is disabled")
        
        return self.synchronizer.sync(agent_id, last_checkpoint_hash)
    
    def register_bundle(self, bundle: "ClaimBundle") -> None:
        """Register a new Claim Bundle."""
        self.synchronizer.register_bundle(bundle)
    
    def get_checkpoint(self, agent_id: str) -> Optional[EDRCheckpoint]:
        """Get latest checkpoint for an agent."""
        return self.synchronizer.get_latest_checkpoint(agent_id)
