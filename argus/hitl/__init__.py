"""
ARGUS Human-in-the-Loop Module.

Provides human oversight for agent actions with approval workflows,
rejection handling, modification support, and feedback collection.

Example:
    >>> from argus.hitl import HITLMiddleware, HITLConfig, ApprovalHandler
    >>> 
    >>> config = HITLConfig(enabled=True)
    >>> middleware = HITLMiddleware(config)
    >>> 
    >>> if middleware.should_intercept("sensitive_action", InterruptionPoint.BEFORE_TOOL_CALL):
    ...     request = middleware.create_interrupt("sensitive_action", args, ...)
"""

from argus.hitl.config import (
    HITLConfig,
    HITLCallbackConfig,
    ApprovalMode,
    InterruptionPoint,
    SensitivityLevel,
    FeedbackType,
    get_default_hitl_config,
)

from argus.hitl.middleware import (
    HITLMiddleware,
    InterruptRequest,
    InterruptStatus,
    MiddlewareState,
    ToolInterceptor,
)

from argus.hitl.handlers import (
    HandlerResult,
    BaseHandler,
    ApprovalHandler,
    RejectionHandler,
    ModificationHandler,
    EscalationHandler,
    DecisionRouter,
)

from argus.hitl.callbacks import (
    Feedback,
    BaseCallback,
    FeedbackCallback,
    RatingCallback,
    AnnotationCallback,
    CorrectionCallback,
    CallbackManager,
)

__all__ = [
    # Config
    "HITLConfig",
    "HITLCallbackConfig",
    "ApprovalMode",
    "InterruptionPoint",
    "SensitivityLevel",
    "FeedbackType",
    "get_default_hitl_config",
    # Middleware
    "HITLMiddleware",
    "InterruptRequest",
    "InterruptStatus",
    "MiddlewareState",
    "ToolInterceptor",
    # Handlers
    "HandlerResult",
    "BaseHandler",
    "ApprovalHandler",
    "RejectionHandler",
    "ModificationHandler",
    "EscalationHandler",
    "DecisionRouter",
    # Callbacks
    "Feedback",
    "BaseCallback",
    "FeedbackCallback",
    "RatingCallback",
    "AnnotationCallback",
    "CorrectionCallback",
    "CallbackManager",
]
