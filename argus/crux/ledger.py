"""
Credibility Ledger (CL) for CRUX Protocol.

The Credibility Ledger is a distributed, append-only log — compatible
with ARGUS's existing SHA-256 hash chain — that records every agent's
calibration history. It is the statistical trust layer of CRUX, replacing
binary API-key authentication with probabilistic credibility scores.

Key Features:
    - Append-only hash-chained entries
    - Brier score tracking per prediction
    - Exponential moving average credibility updates
    - Sybil resistance via track record requirements
    - Adversarial agent detection and suspension
"""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
import threading

from argus.crux.models import generate_crux_id, CRUXConfig

if TYPE_CHECKING:
    from argus.crux.claim_bundle import ClaimBundle
    from argus.crux.agent_card import EpistemicAgentCard

logger = logging.getLogger(__name__)


@dataclass
class CredibilityEntry:
    """
    A single entry in the Credibility Ledger.
    
    Records an agent's prediction and outcome for calibration tracking.
    
    Attributes:
        entry_id: Unique entry identifier
        agent_id: Agent making the prediction
        session_id: Debate session ID
        proposition_id: Proposition being evaluated
        predicted_posterior: Agent's predicted probability
        ground_truth: Actual outcome (0 or 1)
        brier_contribution: Brier score for this prediction
        timestamp: Entry timestamp
        prev_entry_hash: Hash of previous entry (chain)
        entry_hash: Hash of this entry
        metadata: Additional entry data
    """
    entry_id: str = field(default_factory=lambda: generate_crux_id("cl"))
    agent_id: str = ""
    session_id: str = ""
    proposition_id: str = ""
    predicted_posterior: float = 0.5
    ground_truth: Optional[int] = None
    brier_contribution: float = 0.25
    timestamp: datetime = field(default_factory=datetime.utcnow)
    prev_entry_hash: str = ""
    entry_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash for this entry."""
        content = json.dumps({
            "entry_id": self.entry_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "proposition_id": self.proposition_id,
            "predicted_posterior": self.predicted_posterior,
            "ground_truth": self.ground_truth,
            "brier_contribution": self.brier_contribution,
            "timestamp": self.timestamp.isoformat(),
            "prev_entry_hash": self.prev_entry_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "proposition_id": self.proposition_id,
            "predicted_posterior": self.predicted_posterior,
            "ground_truth": self.ground_truth,
            "brier_contribution": self.brier_contribution,
            "timestamp": self.timestamp.isoformat(),
            "prev_entry_hash": self.prev_entry_hash,
            "entry_hash": self.entry_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CredibilityEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            agent_id=data["agent_id"],
            session_id=data.get("session_id", ""),
            proposition_id=data.get("proposition_id", ""),
            predicted_posterior=data["predicted_posterior"],
            ground_truth=data.get("ground_truth"),
            brier_contribution=data["brier_contribution"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            prev_entry_hash=data.get("prev_entry_hash", ""),
            entry_hash=data.get("entry_hash", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CredibilityUpdate:
    """
    Update to an agent's credibility.
    
    Attributes:
        agent_id: Agent being updated
        old_credibility: Previous credibility rating
        new_credibility: Updated credibility rating
        brier_contribution: Brier score that triggered update
        recency_weight: Weight used for EMA (λ)
        timestamp: Update timestamp
    """
    agent_id: str
    old_credibility: float
    new_credibility: float
    brier_contribution: float
    recency_weight: float = 0.15
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def delta(self) -> float:
        """Change in credibility."""
        return self.new_credibility - self.old_credibility
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "old_credibility": self.old_credibility,
            "new_credibility": self.new_credibility,
            "delta": self.delta,
            "brier_contribution": self.brier_contribution,
            "recency_weight": self.recency_weight,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentCredibilityState:
    """
    Current credibility state for an agent.
    
    Attributes:
        agent_id: Agent identifier
        credibility_rating: Current credibility (0-1)
        sample_size: Number of evaluated predictions
        total_brier: Sum of Brier contributions
        last_activity: Last activity timestamp
        is_suspended: Whether agent is suspended
        suspension_reason: Reason for suspension
    """
    agent_id: str
    credibility_rating: float = 0.50
    sample_size: int = 0
    total_brier: float = 0.0
    last_activity: Optional[datetime] = None
    is_suspended: bool = False
    suspension_reason: str = ""
    
    @property
    def average_brier(self) -> float:
        """Average Brier score."""
        if self.sample_size == 0:
            return 0.25
        return self.total_brier / self.sample_size
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "credibility_rating": self.credibility_rating,
            "sample_size": self.sample_size,
            "average_brier": self.average_brier,
            "is_suspended": self.is_suspended,
            "suspension_reason": self.suspension_reason,
            "last_activity": (
                self.last_activity.isoformat() 
                if self.last_activity else None
            ),
        }


class CredibilityLedger:
    """
    Credibility Ledger (CL) - CRUX Primitive 5.
    
    Maintains a hash-chained log of agent calibration history
    and computes credibility ratings using exponential moving average.
    
    Credibility Update Rule:
        new_credibility = (1 - λ) * old_credibility + λ * (1 - brier_contribution)
    
    Where:
        λ = recency_weight (default 0.15)
        brier_contribution in [0, 1], lower = better
    
    Example:
        >>> ledger = CredibilityLedger()
        >>> 
        >>> # Record a prediction
        >>> entry = ledger.record_prediction(
        ...     agent_id="specialist-001",
        ...     proposition_id="prop_abc",
        ...     predicted_posterior=0.73,
        ... )
        >>> 
        >>> # Later, record outcome
        >>> ledger.record_outcome(entry.entry_id, ground_truth=1)
        >>> 
        >>> # Get credibility
        >>> cred = ledger.get_credibility("specialist-001")
    """
    
    def __init__(
        self,
        config: Optional[CRUXConfig] = None,
        path: Optional[str] = None,
        backend: str = "memory",
    ):
        """
        Initialize Credibility Ledger.
        
        Args:
            config: CRUX configuration
            path: Path to ledger file (for persistence)
            backend: Backend type ("memory", "prov-o", "file")
        """
        self.config = config or CRUXConfig()
        self.path = Path(path) if path else None
        self.backend = backend
        
        # Hash chain
        self._entries: list[CredibilityEntry] = []
        self._lock = threading.Lock()
        
        # Agent state tracking
        self._agent_states: dict[str, AgentCredibilityState] = {}
        
        # Pending predictions (awaiting outcomes)
        self._pending: dict[str, CredibilityEntry] = {}
        
        # Update history
        self._updates: list[CredibilityUpdate] = []
        
        # Load from file if exists
        if self.path and self.path.exists():
            self._load()
    
    @property
    def head_hash(self) -> str:
        """Get current head hash of the chain."""
        if not self._entries:
            return ""
        return self._entries[-1].entry_hash
    
    @property
    def chain_length(self) -> int:
        """Get number of entries in chain."""
        return len(self._entries)
    
    def _get_or_create_state(self, agent_id: str) -> AgentCredibilityState:
        """Get or create agent state."""
        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = AgentCredibilityState(agent_id=agent_id)
        return self._agent_states[agent_id]
    
    def record_prediction(
        self,
        agent_id: str,
        proposition_id: str,
        predicted_posterior: float,
        session_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> CredibilityEntry:
        """
        Record a prediction (awaiting outcome).
        
        Args:
            agent_id: Agent making prediction
            proposition_id: Proposition being evaluated
            predicted_posterior: Predicted probability
            session_id: Debate session ID
            metadata: Additional data
            
        Returns:
            CredibilityEntry (pending outcome)
        """
        with self._lock:
            entry = CredibilityEntry(
                agent_id=agent_id,
                session_id=session_id,
                proposition_id=proposition_id,
                predicted_posterior=predicted_posterior,
                prev_entry_hash=self.head_hash,
                metadata=metadata or {},
            )
            entry.entry_hash = entry.compute_hash()
            
            self._pending[entry.entry_id] = entry
            
            # Update agent activity
            state = self._get_or_create_state(agent_id)
            state.last_activity = datetime.utcnow()
            
            logger.debug(f"Recorded prediction {entry.entry_id} for {agent_id}")
            
            return entry
    
    def record_outcome(
        self,
        entry_id: str,
        ground_truth: int,
    ) -> Optional[CredibilityUpdate]:
        """
        Record outcome for a pending prediction.
        
        Computes Brier contribution and updates credibility.
        
        Args:
            entry_id: ID of pending entry
            ground_truth: Actual outcome (0 or 1)
            
        Returns:
            CredibilityUpdate if entry found
        """
        with self._lock:
            if entry_id not in self._pending:
                logger.warning(f"No pending entry found: {entry_id}")
                return None
            
            entry = self._pending.pop(entry_id)
            
            # Compute Brier contribution
            # Brier = (predicted - truth)^2, range [0, 1]
            brier = (entry.predicted_posterior - ground_truth) ** 2
            entry.ground_truth = ground_truth
            entry.brier_contribution = brier
            
            # Update entry hash with outcome
            entry.prev_entry_hash = self.head_hash
            entry.entry_hash = entry.compute_hash()
            
            # Append to chain
            self._entries.append(entry)
            
            # Update agent credibility
            update = self._update_credibility(entry.agent_id, brier)
            
            # Persist if configured
            if self.path:
                self._append_to_file(entry)
            
            logger.debug(
                f"Recorded outcome for {entry_id}: "
                f"brier={brier:.4f}, new_cred={update.new_credibility:.4f}"
            )
            
            return update
    
    def record_prediction_with_outcome(
        self,
        agent_id: str,
        proposition_id: str,
        predicted_posterior: float,
        ground_truth: int,
        session_id: str = "",
    ) -> CredibilityUpdate:
        """
        Record prediction and outcome together.
        
        Convenience method for when outcome is already known.
        
        Args:
            agent_id: Agent ID
            proposition_id: Proposition ID
            predicted_posterior: Predicted probability
            ground_truth: Actual outcome
            session_id: Session ID
            
        Returns:
            CredibilityUpdate
        """
        with self._lock:
            # Compute Brier
            brier = (predicted_posterior - ground_truth) ** 2
            
            entry = CredibilityEntry(
                agent_id=agent_id,
                session_id=session_id,
                proposition_id=proposition_id,
                predicted_posterior=predicted_posterior,
                ground_truth=ground_truth,
                brier_contribution=brier,
                prev_entry_hash=self.head_hash,
            )
            entry.entry_hash = entry.compute_hash()
            
            # Append to chain
            self._entries.append(entry)
            
            # Update credibility
            update = self._update_credibility(agent_id, brier)
            
            # Persist
            if self.path:
                self._append_to_file(entry)
            
            return update
    
    def _update_credibility(
        self,
        agent_id: str,
        brier_contribution: float,
    ) -> CredibilityUpdate:
        """
        Update agent credibility using EMA.
        
        Formula: new_cred = (1-λ) * old_cred + λ * (1 - brier)
        """
        state = self._get_or_create_state(agent_id)
        old_cred = state.credibility_rating
        
        λ = self.config.credibility_recency_weight
        
        # EMA update
        new_cred = (1 - λ) * old_cred + λ * (1 - brier_contribution)
        new_cred = max(0.0, min(1.0, new_cred))
        
        # Update state
        state.credibility_rating = new_cred
        state.sample_size += 1
        state.total_brier += brier_contribution
        state.last_activity = datetime.utcnow()
        
        # Check for suspension
        self._check_suspension(state)
        
        # Create update record
        update = CredibilityUpdate(
            agent_id=agent_id,
            old_credibility=old_cred,
            new_credibility=new_cred,
            brier_contribution=brier_contribution,
            recency_weight=λ,
        )
        self._updates.append(update)
        
        return update
    
    def _check_suspension(self, state: AgentCredibilityState) -> bool:
        """
        Check if agent should be suspended.
        
        Suspends agents whose credibility drops below configured floor.
        """
        if state.is_suspended:
            return True
        
        if state.credibility_rating < self.config.min_credibility_floor:
            state.is_suspended = True
            state.suspension_reason = (
                f"Credibility {state.credibility_rating:.3f} below "
                f"floor {self.config.min_credibility_floor}"
            )
            logger.warning(f"Agent {state.agent_id} suspended: {state.suspension_reason}")
            return True
        
        return False
    
    def get_credibility(self, agent_id: str) -> Optional[float]:
        """
        Get current credibility for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Credibility rating or None if unknown
        """
        state = self._agent_states.get(agent_id)
        if state is None:
            return None
        return state.credibility_rating
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentCredibilityState]:
        """Get full state for an agent."""
        return self._agent_states.get(agent_id)
    
    def is_suspended(self, agent_id: str) -> bool:
        """Check if agent is suspended."""
        state = self._agent_states.get(agent_id)
        return state.is_suspended if state else False
    
    def reinstate_agent(self, agent_id: str) -> bool:
        """
        Reinstate a suspended agent.
        
        Args:
            agent_id: Agent to reinstate
            
        Returns:
            True if reinstated
        """
        state = self._agent_states.get(agent_id)
        if state is None or not state.is_suspended:
            return False
        
        state.is_suspended = False
        state.suspension_reason = ""
        logger.info(f"Agent {agent_id} reinstated")
        return True
    
    def get_entries_for_agent(
        self,
        agent_id: str,
        limit: Optional[int] = None,
    ) -> list[CredibilityEntry]:
        """
        Get ledger entries for an agent.
        
        Args:
            agent_id: Agent ID
            limit: Maximum entries to return
            
        Returns:
            List of entries (most recent first)
        """
        entries = [e for e in self._entries if e.agent_id == agent_id]
        entries.reverse()  # Most recent first
        
        if limit:
            entries = entries[:limit]
        
        return entries
    
    def get_entries_since(
        self,
        timestamp: datetime,
    ) -> list[CredibilityEntry]:
        """Get all entries since a timestamp."""
        return [e for e in self._entries if e.timestamp >= timestamp]
    
    def get_entries_after_hash(
        self,
        hash_value: str,
    ) -> list[CredibilityEntry]:
        """Get all entries after a specific hash."""
        found = False
        result = []
        
        for entry in self._entries:
            if found:
                result.append(entry)
            elif entry.entry_hash == hash_value:
                found = True
        
        return result
    
    def verify_chain_integrity(self) -> tuple[bool, Optional[str]]:
        """
        Verify hash chain integrity.
        
        Returns:
            (is_valid, error_message)
        """
        prev_hash = ""
        
        for i, entry in enumerate(self._entries):
            # Check prev_hash reference
            if entry.prev_entry_hash != prev_hash:
                return False, f"Entry {i}: prev_hash mismatch"
            
            # Verify entry hash
            computed = entry.compute_hash()
            if entry.entry_hash != computed:
                return False, f"Entry {i}: hash mismatch"
            
            prev_hash = entry.entry_hash
        
        return True, None
    
    def detect_sybil_agents(
        self,
        min_samples: int = 10,
        suspicion_threshold: float = 0.95,
    ) -> list[str]:
        """
        Detect potential Sybil agents.
        
        Flags agents with identical belief domains and suspiciously
        similar calibration patterns.
        
        Args:
            min_samples: Minimum samples for analysis
            suspicion_threshold: Similarity threshold
            
        Returns:
            List of suspicious agent IDs
        """
        suspicious = []
        
        # Group by similar credibility (could be more sophisticated)
        by_credibility = defaultdict(list)
        for agent_id, state in self._agent_states.items():
            if state.sample_size >= min_samples:
                # Round to bucket
                bucket = round(state.credibility_rating, 1)
                by_credibility[bucket].append(agent_id)
        
        # Flag clusters of new agents with identical ratings
        for bucket, agents in by_credibility.items():
            if len(agents) >= 3:
                # Check for similar patterns (simplified)
                suspicious.extend(agents)
        
        return list(set(suspicious))
    
    def _append_to_file(self, entry: CredibilityEntry) -> None:
        """Append entry to ledger file."""
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
    
    def _load(self) -> None:
        """Load ledger from file."""
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entry = CredibilityEntry.from_dict(data)
                    self._entries.append(entry)
                    
                    # Rebuild agent state
                    if entry.ground_truth is not None:
                        state = self._get_or_create_state(entry.agent_id)
                        state.sample_size += 1
                        state.total_brier += entry.brier_contribution
                        state.last_activity = entry.timestamp
        
        # Recompute credibility ratings
        for agent_id, state in self._agent_states.items():
            entries = self.get_entries_for_agent(agent_id)
            cred = 0.5
            λ = self.config.credibility_recency_weight
            
            for entry in reversed(entries):  # Oldest first
                if entry.ground_truth is not None:
                    cred = (1 - λ) * cred + λ * (1 - entry.brier_contribution)
            
            state.credibility_rating = max(0.0, min(1.0, cred))
        
        logger.info(f"Loaded {len(self._entries)} entries from {self.path}")
    
    def get_all_agent_ids(self) -> list[str]:
        """Get all agent IDs with credibility records."""
        return list(self._agent_states.keys())
    
    def get_credibility_history(
        self,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get credibility history for an agent.
        
        Returns a list of {timestamp, credibility} dicts showing
        how credibility evolved over time.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of history points
        """
        entries = self.get_entries_for_agent(agent_id)
        entries.reverse()  # Oldest first
        
        history = []
        current_cred = 0.5
        λ = self.config.credibility_recency_weight
        
        for entry in entries:
            if entry.ground_truth is not None:
                current_cred = (1 - λ) * current_cred + λ * (1 - entry.brier_contribution)
                current_cred = max(0.0, min(1.0, current_cred))
                history.append({
                    "timestamp": entry.timestamp,
                    "credibility": current_cred,
                    "brier": entry.brier_contribution,
                })
        
        return history
    
    def export_statistics(self) -> dict[str, Any]:
        """Export ledger statistics."""
        return {
            "chain_length": len(self._entries),
            "head_hash": self.head_hash,
            "num_agents": len(self._agent_states),
            "num_suspended": sum(
                1 for s in self._agent_states.values() if s.is_suspended
            ),
            "agents": {
                aid: state.to_dict()
                for aid, state in self._agent_states.items()
            },
        }
