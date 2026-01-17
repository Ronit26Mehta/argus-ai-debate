"""
ARGUS Human-in-the-Loop Handlers.

Handlers for processing human decisions on intercepted actions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable

from argus.hitl.config import HITLConfig
from argus.hitl.middleware import InterruptRequest, InterruptStatus

logger = logging.getLogger(__name__)


@dataclass
class HandlerResult:
    """Result from a decision handler."""
    success: bool
    status: InterruptStatus
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    should_proceed: bool = True
    modified_args: Optional[dict[str, Any]] = None
    
    @classmethod
    def approved(cls, message: str = "Action approved") -> "HandlerResult":
        return cls(success=True, status=InterruptStatus.APPROVED, message=message)
    
    @classmethod
    def rejected(cls, message: str = "Action rejected") -> "HandlerResult":
        return cls(success=True, status=InterruptStatus.REJECTED, message=message, should_proceed=False)
    
    @classmethod
    def modified(cls, modified_args: dict[str, Any], message: str = "Action modified") -> "HandlerResult":
        return cls(success=True, status=InterruptStatus.MODIFIED, message=message, modified_args=modified_args)


class BaseHandler(ABC):
    """Abstract base class for HITL decision handlers."""
    
    def __init__(self, config: Optional[HITLConfig] = None):
        self.config = config or HITLConfig()
        self._handled_count = 0
        self._history: list[dict[str, Any]] = []
    
    @abstractmethod
    def handle(self, request: InterruptRequest) -> HandlerResult:
        pass
    
    def record(self, request: InterruptRequest, result: HandlerResult) -> None:
        self._handled_count += 1
        self._history.append({
            "request_id": request.request_id,
            "action_name": request.action_name,
            "status": result.status.value,
            "timestamp": datetime.utcnow().isoformat(),
        })


class ApprovalHandler(BaseHandler):
    """Handler for approved actions."""
    
    def __init__(self, config: Optional[HITLConfig] = None, pre_execute_hook: Optional[Callable] = None):
        super().__init__(config)
        self.pre_execute_hook = pre_execute_hook
    
    def handle(self, request: InterruptRequest) -> HandlerResult:
        logger.info(f"Processing approval for '{request.action_name}'")
        if self.pre_execute_hook and not self.pre_execute_hook(request):
            result = HandlerResult.rejected("Pre-execute validation failed")
            self.record(request, result)
            return result
        result = HandlerResult.approved(f"Action '{request.action_name}' approved")
        self.record(request, result)
        return result


class RejectionHandler(BaseHandler):
    """Handler for rejected actions."""
    
    def __init__(self, config: Optional[HITLConfig] = None, require_reason: bool = False):
        super().__init__(config)
        self.require_reason = require_reason
        self._rejection_reasons: dict[str, str] = {}
    
    def handle(self, request: InterruptRequest, reason: Optional[str] = None) -> HandlerResult:
        logger.info(f"Processing rejection for '{request.action_name}'")
        if self.require_reason and not reason:
            return HandlerResult(success=False, status=InterruptStatus.PENDING, 
                                message="Rejection reason required", should_proceed=False)
        if reason:
            self._rejection_reasons[request.request_id] = reason
        result = HandlerResult.rejected(f"Action '{request.action_name}' rejected" + (f": {reason}" if reason else ""))
        self.record(request, result)
        return result


class ModificationHandler(BaseHandler):
    """Handler for modified actions."""
    
    def __init__(self, config: Optional[HITLConfig] = None, allowed_fields: Optional[set[str]] = None):
        super().__init__(config)
        self.allowed_fields = allowed_fields
    
    def handle(self, request: InterruptRequest, modified_args: Optional[dict[str, Any]] = None) -> HandlerResult:
        if not self.config.allow_modifications:
            return HandlerResult.rejected("Modifications not allowed")
        modified_args = modified_args or request.modified_action or request.action_args.copy()
        logger.info(f"Processing modification for '{request.action_name}'")
        result = HandlerResult.modified(modified_args, f"Action '{request.action_name}' modified")
        self.record(request, result)
        return result


class EscalationHandler(BaseHandler):
    """Handler for escalated actions."""
    
    def __init__(self, config: Optional[HITLConfig] = None, escalation_callback: Optional[Callable] = None):
        super().__init__(config)
        self.escalation_callback = escalation_callback
        self._escalation_levels: dict[str, int] = {}
    
    def handle(self, request: InterruptRequest, reason: str = "") -> HandlerResult:
        if not self.config.escalation_enabled:
            return HandlerResult.rejected("Escalation not enabled")
        level = self._escalation_levels.get(request.request_id, 0) + 1
        self._escalation_levels[request.request_id] = level
        if self.escalation_callback:
            self.escalation_callback(request, reason)
        result = HandlerResult(success=True, status=InterruptStatus.ESCALATED,
                              message=f"Escalated to level {level}", should_proceed=False,
                              data={"level": level, "reason": reason})
        self.record(request, result)
        return result


class DecisionRouter:
    """Routes interrupt requests to appropriate handlers."""
    
    def __init__(self, config: Optional[HITLConfig] = None):
        self.config = config or HITLConfig()
        self.approval_handler = ApprovalHandler(self.config)
        self.rejection_handler = RejectionHandler(self.config)
        self.modification_handler = ModificationHandler(self.config)
        self.escalation_handler = EscalationHandler(self.config)
    
    def route(self, request: InterruptRequest, decision: InterruptStatus, **kwargs: Any) -> HandlerResult:
        if decision == InterruptStatus.APPROVED:
            return self.approval_handler.handle(request)
        elif decision == InterruptStatus.REJECTED:
            return self.rejection_handler.handle(request, kwargs.get("reason"))
        elif decision == InterruptStatus.MODIFIED:
            return self.modification_handler.handle(request, kwargs.get("modified_args"))
        elif decision == InterruptStatus.ESCALATED:
            return self.escalation_handler.handle(request, kwargs.get("reason", ""))
        return HandlerResult(success=False, status=request.status, 
                            message=f"No handler for: {decision}", should_proceed=False)
