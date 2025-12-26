"""
Provenance Ledger for ARGUS.

PROV-O compatible append-only event ledger for tracking
all system activities with hash-chained integrity.
"""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

import threading

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of provenance events."""
    # Entity events
    ENTITY_CREATED = "entity_created"
    ENTITY_MODIFIED = "entity_modified"
    ENTITY_DELETED = "entity_deleted"
    
    # Activity events
    AGENT_ACTION = "agent_action"
    RETRIEVAL = "retrieval"
    INFERENCE = "inference"
    DECISION = "decision"
    
    # Derivation events
    DERIVED_FROM = "derived_from"
    GENERATED_BY = "generated_by"
    USED = "used"
    
    # Debate events
    PROPOSITION_ADDED = "proposition_added"
    EVIDENCE_ADDED = "evidence_added"
    REBUTTAL_ADDED = "rebuttal_added"
    VERDICT_RENDERED = "verdict_rendered"
    
    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"


@dataclass
class ProvenanceEvent:
    """
    A provenance event in the ledger.
    
    Follows PROV-O data model with extensions for ARGUS.
    
    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        timestamp: Event timestamp (ISO format)
        agent_id: ID of agent/actor
        entity_id: ID of affected entity
        activity_id: ID of related activity
        attributes: Event-specific attributes
        prev_hash: Hash of previous event (chain)
        hash: Hash of this event
    """
    event_id: str
    event_type: EventType
    timestamp: str
    agent_id: Optional[str] = None
    entity_id: Optional[str] = None
    activity_id: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = ""
    hash: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "entity_id": self.entity_id,
            "activity_id": self.activity_id,
            "attributes": self.attributes,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }
    
    def to_prov(self) -> dict[str, Any]:
        """Convert to PROV-O compatible format."""
        prov = {
            "@type": "prov:Activity",
            "prov:id": self.event_id,
            "prov:startedAtTime": self.timestamp,
            "prov:type": f"argus:{self.event_type.value}",
        }
        
        if self.agent_id:
            prov["prov:wasAssociatedWith"] = self.agent_id
        if self.entity_id:
            prov["prov:used"] = self.entity_id
        
        return prov
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            agent_id=data.get("agent_id"),
            entity_id=data.get("entity_id"),
            activity_id=data.get("activity_id"),
            attributes=data.get("attributes", {}),
            prev_hash=data.get("prev_hash", ""),
            hash=data.get("hash", ""),
        )


class ProvenanceLedger:
    """
    Append-only provenance ledger.
    
    Maintains a hash-chained log of all system events
    with PROV-O compatibility and persistence.
    
    Example:
        >>> ledger = ProvenanceLedger(path="provenance.jsonl")
        >>> ledger.record(
        ...     EventType.PROPOSITION_ADDED,
        ...     entity_id="prop_123",
        ...     attributes={"text": "Drug X is effective"}
        ... )
        >>> events = ledger.query(entity_id="prop_123")
    """
    
    def __init__(
        self,
        path: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize ledger.
        
        Args:
            path: Path to ledger file (in-memory if None)
            enabled: Whether to record events
        """
        self.path = Path(path) if path else None
        self.enabled = enabled
        
        self._events: list[ProvenanceEvent] = []
        self._lock = threading.Lock()
        self._counter = 0
        
        # Load existing events if file exists
        if self.path and self.path.exists():
            self._load()
    
    def _generate_id(self) -> str:
        """Generate unique event ID."""
        self._counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"event_{timestamp}_{self._counter:06d}"
    
    def _compute_hash(self, event: ProvenanceEvent) -> str:
        """Compute SHA-256 hash for event."""
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "agent_id": event.agent_id,
            "entity_id": event.entity_id,
            "activity_id": event.activity_id,
            "attributes": event.attributes,
            "prev_hash": event.prev_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def record(
        self,
        event_type: EventType,
        agent_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        activity_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Optional[ProvenanceEvent]:
        """
        Record a provenance event.
        
        Args:
            event_type: Type of event
            agent_id: Agent/actor ID
            entity_id: Affected entity ID
            activity_id: Related activity ID
            attributes: Event attributes
            
        Returns:
            ProvenanceEvent if recorded, None if disabled
        """
        if not self.enabled:
            return None
        
        with self._lock:
            # Get previous hash
            prev_hash = ""
            if self._events:
                prev_hash = self._events[-1].hash
            
            # Create event
            event = ProvenanceEvent(
                event_id=self._generate_id(),
                event_type=event_type,
                timestamp=datetime.utcnow().isoformat(),
                agent_id=agent_id,
                entity_id=entity_id,
                activity_id=activity_id,
                attributes=attributes or {},
                prev_hash=prev_hash,
            )
            
            # Compute hash
            event.hash = self._compute_hash(event)
            
            # Append to ledger
            self._events.append(event)
            
            # Persist if path configured
            if self.path:
                self._append_to_file(event)
            
            logger.debug(f"Recorded event: {event.event_id} ({event.event_type.value})")
            
            return event
    
    def query(
        self,
        event_type: Optional[EventType] = None,
        agent_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[ProvenanceEvent]:
        """
        Query events from the ledger.
        
        Args:
            event_type: Filter by event type
            agent_id: Filter by agent
            entity_id: Filter by entity
            since: Filter by start time
            until: Filter by end time
            limit: Maximum results
            
        Returns:
            List of matching events
        """
        results = []
        
        for event in self._events:
            if event_type and event.event_type != event_type:
                continue
            if agent_id and event.agent_id != agent_id:
                continue
            if entity_id and event.entity_id != entity_id:
                continue
            if since:
                if datetime.fromisoformat(event.timestamp) < since:
                    continue
            if until:
                if datetime.fromisoformat(event.timestamp) > until:
                    continue
            
            results.append(event)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_entity_history(
        self,
        entity_id: str,
    ) -> list[ProvenanceEvent]:
        """
        Get complete history for an entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            List of events affecting the entity
        """
        return self.query(entity_id=entity_id, limit=1000)
    
    def verify_integrity(self) -> tuple[bool, list[str]]:
        """
        Verify hash chain integrity.
        
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        for i, event in enumerate(self._events):
            # Check hash
            computed = self._compute_hash(event)
            if computed != event.hash:
                errors.append(f"Event {event.event_id}: hash mismatch")
            
            # Check chain
            if i > 0:
                if event.prev_hash != self._events[i - 1].hash:
                    errors.append(f"Event {event.event_id}: chain broken")
        
        return len(errors) == 0, errors
    
    def _append_to_file(self, event: ProvenanceEvent) -> None:
        """Append event to file."""
        with open(self.path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    
    def _load(self) -> None:
        """Load events from file."""
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    event = ProvenanceEvent.from_dict(data)
                    self._events.append(event)
                    self._counter += 1
        
        logger.info(f"Loaded {len(self._events)} events from {self.path}")
    
    def export_prov(self) -> dict[str, Any]:
        """
        Export ledger in PROV-O JSON format.
        
        Returns:
            PROV-O compatible document
        """
        return {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "argus": "http://argus.ai/ns/",
            },
            "@graph": [event.to_prov() for event in self._events],
        }
    
    def __len__(self) -> int:
        return len(self._events)
