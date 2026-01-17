"""
ARGUS Human-in-the-Loop Middleware.

Middleware that intercepts agent actions and tool calls for human review.
Supports state preservation during interruptions and workflow resumption.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable, TYPE_CHECKING

from pydantic import BaseModel, Field

from argus.hitl.config import (
    HITLConfig,
    ApprovalMode,
    InterruptionPoint,
    SensitivityLevel,
)

if TYPE_CHECKING:
    from argus.tools.base import BaseTool, ToolResult
    from argus.agents.base import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class InterruptStatus(str, Enum):
    """Status of an interrupt request."""
    PENDING = "pending"       # Waiting for human response
    APPROVED = "approved"     # Action approved
    REJECTED = "rejected"     # Action rejected
    MODIFIED = "modified"     # Action modified by human
    TIMEOUT = "timeout"       # Timed out waiting
    ESCALATED = "escalated"   # Escalated to higher authority
    CANCELLED = "cancelled"   # Request cancelled


@dataclass
class InterruptRequest:
    """Represents a pending action awaiting human decision.
    
    Captures all information about an action that has been
    paused for human review.
    
    Attributes:
        request_id: Unique identifier for this request
        interruption_point: Where the workflow was paused
        action_type: Type of action (tool_call, agent_action, etc.)
        action_name: Name of the action/tool
        action_args: Arguments for the action
        sensitivity: Sensitivity level of the action
        context: Additional context for the reviewer
        timestamp: When the request was created
        status: Current status of the request
        human_response: Response from human reviewer
        modified_action: Modified action if applicable
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    interruption_point: InterruptionPoint = InterruptionPoint.BEFORE_TOOL_CALL
    action_type: str = "tool_call"
    action_name: str = ""
    action_args: dict[str, Any] = field(default_factory=dict)
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: InterruptStatus = InterruptStatus.PENDING
    human_response: Optional[str] = None
    modified_action: Optional[dict[str, Any]] = None
    timeout_at: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["timeout_at"] = self.timeout_at.isoformat() if self.timeout_at else None
        data["interruption_point"] = self.interruption_point.value
        data["status"] = self.status.value
        data["sensitivity"] = self.sensitivity.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterruptRequest":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("timeout_at"):
            data["timeout_at"] = datetime.fromisoformat(data["timeout_at"])
        data["interruption_point"] = InterruptionPoint(data["interruption_point"])
        data["status"] = InterruptStatus(data["status"])
        data["sensitivity"] = SensitivityLevel(data["sensitivity"])
        return cls(**data)


@dataclass
class MiddlewareState:
    """State preserved during workflow interruption.
    
    Enables the workflow to resume from exactly where it paused.
    """
    session_id: str
    interrupt_request: InterruptRequest
    preserved_context: dict[str, Any] = field(default_factory=dict)
    agent_state: Optional[dict[str, Any]] = None
    graph_state: Optional[dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "session_id": self.session_id,
            "interrupt_request": self.interrupt_request.to_dict(),
            "preserved_context": self.preserved_context,
            "agent_state": self.agent_state,
            "graph_state": self.graph_state,
            "timestamp": self.timestamp.isoformat(),
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "MiddlewareState":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        data["interrupt_request"] = InterruptRequest.from_dict(data["interrupt_request"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class HITLMiddleware:
    """Middleware for human-in-the-loop intervention.
    
    Intercepts workflow execution at configured points to allow
    human review, approval, modification, or rejection of actions.
    
    Example:
        >>> middleware = HITLMiddleware(config)
        >>> 
        >>> # Check if action should be intercepted
        >>> if middleware.should_intercept(action, InterruptionPoint.BEFORE_TOOL_CALL):
        ...     request = middleware.create_interrupt(action)
        ...     # Wait for human response
        ...     response = middleware.wait_for_response(request)
        ...     if response.status == InterruptStatus.APPROVED:
        ...         # Proceed with action
        ...         pass
    """
    
    def __init__(
        self,
        config: Optional[HITLConfig] = None,
        response_handler: Optional[Callable[[InterruptRequest], InterruptStatus]] = None,
    ):
        """Initialize middleware.
        
        Args:
            config: HITL configuration
            response_handler: Callback to handle interrupt responses
        """
        self.config = config or HITLConfig()
        self.response_handler = response_handler
        
        self._pending_requests: dict[str, InterruptRequest] = {}
        self._states: dict[str, MiddlewareState] = {}
        self._decision_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        
        logger.debug(f"HITLMiddleware initialized with config: {self.config}")
    
    def should_intercept(
        self,
        action_name: str,
        point: InterruptionPoint,
        sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
    ) -> bool:
        """Check if an action should be intercepted at given point.
        
        Args:
            action_name: Name of the action
            point: Current interruption point
            sensitivity: Sensitivity level of the action
            
        Returns:
            True if action should be intercepted for review
        """
        if not self.config.enabled:
            return False
            
        if not self.config.should_interrupt_at(point):
            return False
            
        return self.config.should_require_approval(sensitivity)
    
    def create_interrupt(
        self,
        action_name: str,
        action_args: dict[str, Any],
        point: InterruptionPoint,
        sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
        context: Optional[dict[str, Any]] = None,
    ) -> InterruptRequest:
        """Create an interrupt request for human review.
        
        Args:
            action_name: Name of the action
            action_args: Arguments for the action
            point: Where the workflow was paused
            sensitivity: Sensitivity level
            context: Additional context for reviewer
            
        Returns:
            InterruptRequest awaiting human decision
        """
        timeout_at = None
        if self.config.timeout:
            from datetime import timedelta
            timeout_at = datetime.utcnow() + timedelta(seconds=self.config.timeout)
        
        request = InterruptRequest(
            interruption_point=point,
            action_type="tool_call" if "tool" in point.value else "agent_action",
            action_name=action_name,
            action_args=action_args,
            sensitivity=sensitivity,
            context=context or {},
            timeout_at=timeout_at,
        )
        
        with self._lock:
            self._pending_requests[request.request_id] = request
        
        logger.info(
            f"Created interrupt request {request.request_id} for action '{action_name}'"
        )
        
        return request
    
    def save_state(
        self,
        session_id: str,
        request: InterruptRequest,
        context: Optional[dict[str, Any]] = None,
        agent_state: Optional[dict[str, Any]] = None,
        graph_state: Optional[dict[str, Any]] = None,
    ) -> MiddlewareState:
        """Save workflow state during interruption.
        
        Args:
            session_id: Unique session identifier
            request: The pending interrupt request
            context: Additional context to preserve
            agent_state: Current agent state
            graph_state: Current graph state
            
        Returns:
            Saved MiddlewareState
        """
        state = MiddlewareState(
            session_id=session_id,
            interrupt_request=request,
            preserved_context=context or {},
            agent_state=agent_state,
            graph_state=graph_state,
        )
        
        with self._lock:
            self._states[session_id] = state
        
        logger.debug(f"Saved middleware state for session {session_id}")
        
        return state
    
    def restore_state(self, session_id: str) -> Optional[MiddlewareState]:
        """Restore workflow state after interruption.
        
        Args:
            session_id: Session identifier to restore
            
        Returns:
            Restored MiddlewareState or None if not found
        """
        with self._lock:
            state = self._states.get(session_id)
            if state:
                del self._states[session_id]
                
        if state:
            logger.debug(f"Restored middleware state for session {session_id}")
        else:
            logger.warning(f"No saved state found for session {session_id}")
            
        return state
    
    def submit_response(
        self,
        request_id: str,
        status: InterruptStatus,
        response: Optional[str] = None,
        modified_action: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Submit human response to an interrupt request.
        
        Args:
            request_id: ID of the interrupt request
            status: Human's decision
            response: Optional response message
            modified_action: Modified action if status is MODIFIED
            
        Returns:
            True if response was recorded successfully
        """
        with self._lock:
            request = self._pending_requests.get(request_id)
            if not request:
                logger.warning(f"No pending request found with ID {request_id}")
                return False
            
            request.status = status
            request.human_response = response
            request.modified_action = modified_action
            
            # Record decision for audit
            if self.config.record_all_decisions:
                self._decision_history.append({
                    "request_id": request_id,
                    "action_name": request.action_name,
                    "status": status.value,
                    "response": response,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            
            del self._pending_requests[request_id]
        
        logger.info(
            f"Recorded {status.value} response for request {request_id}"
        )
        
        return True
    
    def get_pending_requests(self) -> list[InterruptRequest]:
        """Get all pending interrupt requests.
        
        Returns:
            List of pending requests
        """
        with self._lock:
            return list(self._pending_requests.values())
    
    def get_request(self, request_id: str) -> Optional[InterruptRequest]:
        """Get a specific interrupt request.
        
        Args:
            request_id: Request ID to retrieve
            
        Returns:
            InterruptRequest or None if not found
        """
        with self._lock:
            return self._pending_requests.get(request_id)
    
    def check_timeout(self, request_id: str) -> bool:
        """Check if a request has timed out.
        
        Args:
            request_id: Request to check
            
        Returns:
            True if request has timed out
        """
        with self._lock:
            request = self._pending_requests.get(request_id)
            if not request or not request.timeout_at:
                return False
            
            if datetime.utcnow() >= request.timeout_at:
                if self.config.auto_approve_on_timeout:
                    request.status = InterruptStatus.APPROVED
                else:
                    request.status = InterruptStatus.TIMEOUT
                
                del self._pending_requests[request_id]
                return True
        
        return False
    
    def get_decision_history(
        self,
        limit: int = 100,
        action_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get history of human decisions.
        
        Args:
            limit: Maximum records to return
            action_name: Filter by action name
            
        Returns:
            List of decision records
        """
        with self._lock:
            history = self._decision_history.copy()
        
        if action_name:
            history = [h for h in history if h["action_name"] == action_name]
        
        return history[-limit:]
    
    def clear_pending(self) -> int:
        """Clear all pending requests.
        
        Returns:
            Number of cleared requests
        """
        with self._lock:
            count = len(self._pending_requests)
            self._pending_requests.clear()
            
        logger.info(f"Cleared {count} pending interrupt requests")
        return count


class ToolInterceptor:
    """Interceptor that wraps tool execution with HITL middleware.
    
    Provides a decorator-style interface for wrapping tools with
    human oversight.
    
    Example:
        >>> interceptor = ToolInterceptor(middleware)
        >>> 
        >>> @interceptor.wrap(sensitivity=SensitivityLevel.HIGH)
        ... def my_sensitive_tool(**kwargs):
        ...     return result
    """
    
    def __init__(self, middleware: HITLMiddleware):
        """Initialize interceptor.
        
        Args:
            middleware: HITL middleware instance
        """
        self.middleware = middleware
    
    def wrap(
        self,
        sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
        point: InterruptionPoint = InterruptionPoint.BEFORE_TOOL_CALL,
    ) -> Callable:
        """Decorator to wrap a function with HITL interception.
        
        Args:
            sensitivity: Sensitivity level of the wrapped function
            point: Interruption point for the function
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                action_name = func.__name__
                
                if not self.middleware.should_intercept(action_name, point, sensitivity):
                    return func(*args, **kwargs)
                
                # Create interrupt request
                request = self.middleware.create_interrupt(
                    action_name=action_name,
                    action_args=kwargs,
                    point=point,
                    sensitivity=sensitivity,
                )
                
                # In a real implementation, this would wait for human response
                # For now, we just log and proceed
                logger.info(
                    f"Action '{action_name}' intercepted for review "
                    f"(request_id={request.request_id})"
                )
                
                # Check if auto-approved (for testing without actual human)
                if self.middleware.check_timeout(request.request_id):
                    if request.status == InterruptStatus.APPROVED:
                        return func(*args, **kwargs)
                    else:
                        raise RuntimeError(f"Action '{action_name}' timed out")
                
                # Proceed with original function (in sync mode)
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
