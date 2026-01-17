"""
Unit tests for ARGUS Human-in-the-Loop module.
"""

import pytest
from datetime import datetime

from argus.hitl.config import (
    HITLConfig, ApprovalMode, InterruptionPoint, SensitivityLevel
)
from argus.hitl.middleware import (
    HITLMiddleware, InterruptRequest, InterruptStatus, MiddlewareState
)
from argus.hitl.handlers import (
    ApprovalHandler, RejectionHandler, ModificationHandler, DecisionRouter, HandlerResult
)
from argus.hitl.callbacks import (
    FeedbackCallback, RatingCallback, AnnotationCallback, CorrectionCallback, FeedbackType
)


class TestHITLConfig:
    """Tests for HITL configuration."""
    
    def test_default_config(self):
        config = HITLConfig()
        assert config.enabled is False
        assert config.approval_mode == ApprovalMode.SENSITIVE_ONLY
        assert config.timeout == 300.0
    
    def test_should_require_approval_disabled(self):
        config = HITLConfig(enabled=False)
        assert config.should_require_approval(SensitivityLevel.CRITICAL) is False
    
    def test_should_require_approval_all_mode(self):
        config = HITLConfig(enabled=True, approval_mode=ApprovalMode.ALL)
        assert config.should_require_approval(SensitivityLevel.LOW) is True
    
    def test_should_require_approval_sensitive_mode(self):
        config = HITLConfig(
            enabled=True, 
            approval_mode=ApprovalMode.SENSITIVE_ONLY,
            min_sensitivity_for_approval=SensitivityLevel.HIGH
        )
        assert config.should_require_approval(SensitivityLevel.LOW) is False
        assert config.should_require_approval(SensitivityLevel.HIGH) is True
        assert config.should_require_approval(SensitivityLevel.CRITICAL) is True
    
    def test_should_interrupt_at(self):
        config = HITLConfig(
            enabled=True,
            interruption_points=[InterruptionPoint.BEFORE_VERDICT]
        )
        assert config.should_interrupt_at(InterruptionPoint.BEFORE_VERDICT) is True
        assert config.should_interrupt_at(InterruptionPoint.BEFORE_TOOL_CALL) is False


class TestHITLMiddleware:
    """Tests for HITL middleware."""
    
    def test_middleware_initialization(self):
        config = HITLConfig(enabled=True)
        middleware = HITLMiddleware(config)
        assert middleware.config.enabled is True
    
    def test_should_intercept_disabled(self):
        config = HITLConfig(enabled=False)
        middleware = HITLMiddleware(config)
        assert middleware.should_intercept("action", InterruptionPoint.BEFORE_TOOL_CALL) is False
    
    def test_create_interrupt(self):
        config = HITLConfig(enabled=True, approval_mode=ApprovalMode.ALL)
        middleware = HITLMiddleware(config)
        request = middleware.create_interrupt(
            action_name="test_action",
            action_args={"arg1": "value1"},
            point=InterruptionPoint.BEFORE_TOOL_CALL,
        )
        assert request.action_name == "test_action"
        assert request.action_args == {"arg1": "value1"}
        assert request.status == InterruptStatus.PENDING
    
    def test_submit_response(self):
        config = HITLConfig(enabled=True)
        middleware = HITLMiddleware(config)
        request = middleware.create_interrupt("action", {}, InterruptionPoint.BEFORE_TOOL_CALL)
        success = middleware.submit_response(request.request_id, InterruptStatus.APPROVED)
        assert success is True
        assert middleware.get_request(request.request_id) is None
    
    def test_save_restore_state(self):
        config = HITLConfig(enabled=True)
        middleware = HITLMiddleware(config)
        request = middleware.create_interrupt("action", {}, InterruptionPoint.BEFORE_TOOL_CALL)
        state = middleware.save_state("session-1", request, context={"key": "value"})
        assert state.session_id == "session-1"
        restored = middleware.restore_state("session-1")
        assert restored is not None
        assert restored.preserved_context == {"key": "value"}


class TestHandlers:
    """Tests for decision handlers."""
    
    def test_approval_handler(self):
        handler = ApprovalHandler()
        request = InterruptRequest(action_name="test")
        result = handler.handle(request)
        assert result.success is True
        assert result.status == InterruptStatus.APPROVED
        assert result.should_proceed is True
    
    def test_rejection_handler(self):
        handler = RejectionHandler()
        request = InterruptRequest(action_name="test")
        result = handler.handle(request, reason="Not allowed")
        assert result.status == InterruptStatus.REJECTED
        assert result.should_proceed is False
        assert "Not allowed" in result.message
    
    def test_modification_handler(self):
        config = HITLConfig(allow_modifications=True)
        handler = ModificationHandler(config)
        request = InterruptRequest(action_name="test", action_args={"a": 1})
        result = handler.handle(request, modified_args={"a": 2})
        assert result.status == InterruptStatus.MODIFIED
        assert result.modified_args == {"a": 2}
    
    def test_decision_router(self):
        router = DecisionRouter()
        request = InterruptRequest(action_name="test")
        result = router.route(request, InterruptStatus.APPROVED)
        assert result.status == InterruptStatus.APPROVED


class TestCallbacks:
    """Tests for feedback callbacks."""
    
    def test_feedback_callback(self):
        callback = FeedbackCallback()
        feedback = callback.collect(FeedbackType.APPROVAL, content=True, action_name="test")
        assert feedback.feedback_type == FeedbackType.APPROVAL
        assert feedback.content is True
    
    def test_rating_callback(self):
        callback = RatingCallback()
        feedback = callback.collect(rating=5, action_name="test")
        assert feedback.content["rating"] == 5
        assert callback.get_average_rating() == 5.0
    
    def test_rating_callback_invalid(self):
        callback = RatingCallback(min_rating=1, max_rating=5)
        with pytest.raises(ValueError):
            callback.collect(rating=10)
    
    def test_annotation_callback(self):
        callback = AnnotationCallback()
        feedback = callback.collect(annotation="Good response", target_id="node-1")
        assert feedback.content["annotation"] == "Good response"
    
    def test_correction_callback(self):
        callback = CorrectionCallback()
        feedback = callback.collect(
            original="The answer is 42",
            corrected="The answer is 43",
            reason="Calculation error"
        )
        assert feedback.content["original"] == "The answer is 42"
        assert feedback.content["corrected"] == "The answer is 43"
