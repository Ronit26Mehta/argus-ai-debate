"""
AgentOps Tool for ARGUS.

Session replays, metrics, and monitoring for ADK agents.
Provides comprehensive observability for AI agent behaviors.
"""

from __future__ import annotations

import os
import uuid
import json
import logging
import time
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """Represents a single agent event."""
    event_id: str
    event_type: str
    agent_id: str
    session_id: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class AgentSession:
    """Represents an agent session for replay and analysis."""
    session_id: str
    agent_id: str
    agent_name: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    events: list[AgentEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        return self.ended_at is None
    
    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
    
    def add_event(self, event: AgentEvent) -> None:
        self.events.append(event)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "event_count": len(self.events),
            "is_active": self.is_active,
            "metadata": self.metadata,
            "tags": self.tags,
        }


class AgentOpsTool(BaseTool):
    """
    AgentOps - Comprehensive observability for AI agents.
    
    Features:
    - Session recording and replay
    - Event tracking and metrics
    - Performance monitoring
    - Error tracking and analysis
    - Agent behavior analytics
    
    Example:
        >>> tool = AgentOpsTool(api_key="...")
        >>> result = tool(action="start_session", agent_name="researcher")
        >>> result = tool(action="track_event", session_id="...", 
        ...               event_type="llm_call", data={...})
        >>> result = tool(action="end_session", session_id="...")
        >>> result = tool(action="get_replay", session_id="...")
    """
    
    name = "agentops"
    description = "Session replays, metrics, and monitoring for AI agents"
    category = ToolCategory.EXTERNAL_API
    version = "1.0.0"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.agentops.ai",
        enable_cloud_sync: bool = False,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("AGENTOPS_API_KEY")
        self.api_url = api_url
        self.enable_cloud_sync = enable_cloud_sync and self.api_key is not None
        
        # Local storage
        self._sessions: dict[str, AgentSession] = {}
        self._agents: dict[str, dict] = {}
        self._metrics: dict[str, list[float]] = defaultdict(list)
        
        logger.debug(f"AgentOps initialized (cloud_sync={self.enable_cloud_sync})")
    
    def execute(
        self,
        action: str = "get_metrics",
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[str] = None,
        data: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute AgentOps operations.
        
        Args:
            action: Operation to perform
            session_id: Session identifier
            agent_id: Agent identifier
            agent_name: Agent display name
            event_type: Type of event to track
            data: Event data payload
            tags: Session tags for filtering
            metadata: Additional metadata
            limit: Max results to return
            
        Returns:
            ToolResult with operation result
        """
        actions = {
            "start_session": self._start_session,
            "end_session": self._end_session,
            "track_event": self._track_event,
            "get_replay": self._get_replay,
            "list_sessions": self._list_sessions,
            "get_session": self._get_session,
            "get_metrics": self._get_metrics,
            "get_agent_stats": self._get_agent_stats,
            "search_events": self._search_events,
            "register_agent": self._register_agent,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                session_id=session_id,
                agent_id=agent_id,
                agent_name=agent_name,
                event_type=event_type,
                data=data or {},
                tags=tags or [],
                metadata=metadata or {},
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"AgentOps error: {e}")
            return ToolResult.from_error(f"AgentOps error: {e}")
    
    def _register_agent(
        self,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Register an agent for tracking."""
        if not agent_name:
            return ToolResult.from_error("agent_name is required")
        
        agent_id = agent_id or str(uuid.uuid4())[:8]
        
        self._agents[agent_id] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "session_count": 0,
            "total_events": 0,
        }
        
        return ToolResult.from_data({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "message": "Agent registered successfully",
        })
    
    def _start_session(
        self,
        agent_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Start a new agent session."""
        if not agent_name:
            return ToolResult.from_error("agent_name is required")
        
        session_id = str(uuid.uuid4())[:12]
        agent_id = agent_id or str(uuid.uuid4())[:8]
        
        session = AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        self._sessions[session_id] = session
        
        # Update agent stats
        if agent_id in self._agents:
            self._agents[agent_id]["session_count"] += 1
        
        # Track session start event
        self._track_event_internal(
            session=session,
            event_type="session_start",
            data={"agent_name": agent_name},
        )
        
        logger.info(f"Started session {session_id} for agent {agent_name}")
        
        return ToolResult.from_data({
            "session_id": session_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "started_at": session.started_at.isoformat(),
        })
    
    def _end_session(
        self,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """End an agent session."""
        if not session_id or session_id not in self._sessions:
            return ToolResult.from_error("Invalid session_id")
        
        session = self._sessions[session_id]
        session.ended_at = datetime.utcnow()
        
        # Track session end event
        self._track_event_internal(
            session=session,
            event_type="session_end",
            data={"duration_seconds": session.duration_seconds},
        )
        
        # Record metrics
        self._metrics["session_duration"].append(session.duration_seconds)
        self._metrics["events_per_session"].append(len(session.events))
        
        return ToolResult.from_data({
            "session_id": session_id,
            "duration_seconds": session.duration_seconds,
            "event_count": len(session.events),
            "ended_at": session.ended_at.isoformat(),
        })
    
    def _track_event_internal(
        self,
        session: AgentSession,
        event_type: str,
        data: dict,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> AgentEvent:
        """Internal method to track an event."""
        event = AgentEvent(
            event_id=str(uuid.uuid4())[:12],
            event_type=event_type,
            agent_id=session.agent_id,
            session_id=session.session_id,
            timestamp=datetime.utcnow(),
            data=data,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        session.add_event(event)
        return event
    
    def _track_event(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        data: Optional[dict] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Track an event in a session."""
        if not session_id or session_id not in self._sessions:
            return ToolResult.from_error("Invalid session_id")
        if not event_type:
            return ToolResult.from_error("event_type is required")
        
        session = self._sessions[session_id]
        
        if not session.is_active:
            return ToolResult.from_error("Session has ended")
        
        event = self._track_event_internal(
            session=session,
            event_type=event_type,
            data=data or {},
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        
        # Update metrics
        if duration_ms:
            self._metrics[f"{event_type}_duration"].append(duration_ms)
        if not success:
            self._metrics["error_count"].append(1)
        
        return ToolResult.from_data({
            "event_id": event.event_id,
            "event_type": event_type,
            "session_id": session_id,
            "timestamp": event.timestamp.isoformat(),
        })
    
    def _get_replay(
        self,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get full session replay data."""
        if not session_id or session_id not in self._sessions:
            return ToolResult.from_error("Invalid session_id")
        
        session = self._sessions[session_id]
        
        replay_data = {
            "session": session.to_dict(),
            "events": [e.to_dict() for e in session.events],
            "timeline": [
                {
                    "time_offset_ms": (e.timestamp - session.started_at).total_seconds() * 1000,
                    "event_type": e.event_type,
                    "event_id": e.event_id,
                    "success": e.success,
                }
                for e in session.events
            ],
        }
        
        return ToolResult.from_data(replay_data)
    
    def _list_sessions(
        self,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        active_only: bool = False,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List sessions with optional filtering."""
        sessions = list(self._sessions.values())
        
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        
        if tags:
            sessions = [s for s in sessions if any(t in s.tags for t in tags)]
        
        if active_only:
            sessions = [s for s in sessions if s.is_active]
        
        sessions = sessions[-limit:]
        
        return ToolResult.from_data({
            "sessions": [s.to_dict() for s in sessions],
            "count": len(sessions),
        })
    
    def _get_session(
        self,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get details for a specific session."""
        if not session_id or session_id not in self._sessions:
            return ToolResult.from_error("Invalid session_id")
        
        session = self._sessions[session_id]
        return ToolResult.from_data({"session": session.to_dict()})
    
    def _get_metrics(self, limit: int = 100, **kwargs) -> ToolResult:
        """Get aggregated metrics."""
        metrics = {}
        
        for name, values in self._metrics.items():
            if values:
                metrics[name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        
        return ToolResult.from_data({
            "metrics": metrics,
            "total_sessions": len(self._sessions),
            "active_sessions": len([s for s in self._sessions.values() if s.is_active]),
            "total_events": sum(len(s.events) for s in self._sessions.values()),
        })
    
    def _get_agent_stats(
        self,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get statistics for an agent or all agents."""
        if agent_id:
            if agent_id not in self._agents:
                # Calculate from sessions
                sessions = [s for s in self._sessions.values() if s.agent_id == agent_id]
                if not sessions:
                    return ToolResult.from_error("Agent not found")
                
                total_events = sum(len(s.events) for s in sessions)
                total_duration = sum(s.duration_seconds for s in sessions)
                
                return ToolResult.from_data({
                    "agent_id": agent_id,
                    "session_count": len(sessions),
                    "total_events": total_events,
                    "total_duration_seconds": total_duration,
                    "avg_session_duration": total_duration / len(sessions) if sessions else 0,
                })
            
            return ToolResult.from_data({"agent": self._agents[agent_id]})
        
        return ToolResult.from_data({
            "agents": list(self._agents.values()),
            "count": len(self._agents),
        })
    
    def _search_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Search events across sessions."""
        events = []
        
        sessions = [self._sessions[session_id]] if session_id else self._sessions.values()
        
        for session in sessions:
            for event in session.events:
                if event_type and event.event_type != event_type:
                    continue
                events.append(event.to_dict())
        
        events = events[-limit:]
        
        return ToolResult.from_data({
            "events": events,
            "count": len(events),
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start_session", "end_session", "track_event", 
                                "get_replay", "list_sessions", "get_session",
                                "get_metrics", "get_agent_stats", "search_events",
                                "register_agent"],
                    },
                    "session_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "event_type": {"type": "string"},
                    "data": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "limit": {"type": "integer"},
                },
                "required": ["action"],
            },
        }
