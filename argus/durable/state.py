"""
ARGUS Durable State.

Debate state serialization for checkpointing and recovery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from argus.cdag.graph import CDAG

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """A versioned state snapshot."""
    snapshot_id: str
    version: int
    state: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSnapshot":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class DebateState:
    """Complete debate state for checkpointing."""
    debate_id: str
    proposition: str
    current_round: int = 0
    max_rounds: int = 5
    graph_state: Optional[dict[str, Any]] = None
    agent_states: Dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    rebuttal_ids: List[str] = field(default_factory=list)
    history: List[dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebateState":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


def serialize_state(state: DebateState) -> str:
    """Serialize debate state to JSON string."""
    return json.dumps(state.to_dict(), indent=2)


def deserialize_state(json_str: str) -> DebateState:
    """Deserialize debate state from JSON string."""
    return DebateState.from_dict(json.loads(json_str))


def serialize_graph(graph: "CDAG") -> dict[str, Any]:
    """Serialize CDAG graph to dictionary."""
    nodes = []
    edges = []
    if hasattr(graph, 'nodes'):
        for node_id, node in graph.nodes.items():
            node_data = {"id": node_id}
            for attr in ["text", "prior", "posterior", "confidence", "status"]:
                if hasattr(node, attr):
                    val = getattr(node, attr)
                    if hasattr(val, "value"):
                        val = val.value
                    node_data[attr] = val
            if hasattr(node, "node_type"):
                node_data["type"] = node.node_type.value
            nodes.append(node_data)
    if hasattr(graph, 'edges'):
        for edge in graph.edges:
            edge_data = {"source": edge.source, "target": edge.target}
            if hasattr(edge, "edge_type"):
                edge_data["type"] = edge.edge_type.value
            if hasattr(edge, "polarity"):
                edge_data["polarity"] = edge.polarity.value if hasattr(edge.polarity, "value") else edge.polarity
            if hasattr(edge, "weight"):
                edge_data["weight"] = edge.weight
            edges.append(edge_data)
    return {"nodes": nodes, "edges": edges}


class StateManager:
    """Manage debate state for durable execution."""
    
    def __init__(self):
        self._current_state: Optional[DebateState] = None
        self._snapshots: List[StateSnapshot] = []
        self._version = 0
    
    def initialize(self, debate_id: str, proposition: str, max_rounds: int = 5) -> DebateState:
        """Initialize a new debate state."""
        self._current_state = DebateState(
            debate_id=debate_id, proposition=proposition, max_rounds=max_rounds
        )
        return self._current_state
    
    def get_state(self) -> Optional[DebateState]:
        return self._current_state
    
    def update(self, **updates: Any) -> DebateState:
        """Update current state."""
        if not self._current_state:
            raise RuntimeError("No state initialized")
        for key, value in updates.items():
            if hasattr(self._current_state, key):
                setattr(self._current_state, key, value)
        self._current_state.updated_at = datetime.utcnow()
        return self._current_state
    
    def snapshot(self, description: str = "") -> StateSnapshot:
        """Create a snapshot of current state."""
        if not self._current_state:
            raise RuntimeError("No state to snapshot")
        import uuid
        self._version += 1
        snap = StateSnapshot(
            snapshot_id=str(uuid.uuid4()), version=self._version,
            state=self._current_state.to_dict(), description=description
        )
        self._snapshots.append(snap)
        return snap
    
    def restore(self, snapshot: StateSnapshot) -> DebateState:
        """Restore state from snapshot."""
        self._current_state = DebateState.from_dict(snapshot.state)
        return self._current_state
    
    def get_snapshots(self) -> List[StateSnapshot]:
        return self._snapshots.copy()
