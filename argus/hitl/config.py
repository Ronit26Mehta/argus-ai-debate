"""
ARGUS Human-in-the-Loop (HITL) Configuration.

Configuration models for human oversight and approval workflows
in the ARGUS debate system.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ApprovalMode(str, Enum):
    """Approval mode for agent actions.
    
    Determines which actions require human approval before execution.
    """
    NONE = "none"                    # No approval required
    SENSITIVE_ONLY = "sensitive"     # Only sensitive actions need approval
    ALL = "all"                      # All actions require approval


class InterruptionPoint(str, Enum):
    """Points where workflow can be interrupted for human input.
    
    Defines strategic locations in the debate workflow where
    human oversight can be inserted.
    """
    BEFORE_TOOL_CALL = "before_tool"       # Before any tool execution
    AFTER_TOOL_CALL = "after_tool"         # After tool execution for review
    BEFORE_AGENT_ACTION = "before_agent"   # Before agent performs action
    AFTER_AGENT_ACTION = "after_agent"     # After agent action for review
    BEFORE_VERDICT = "before_verdict"      # Before jury renders verdict
    ON_EVIDENCE_SUBMISSION = "on_evidence" # When new evidence is submitted
    ON_REBUTTAL = "on_rebuttal"           # When rebuttal is proposed
    CUSTOM = "custom"                      # Custom interruption logic


class SensitivityLevel(str, Enum):
    """Sensitivity level of actions for approval filtering.
    
    Higher sensitivity actions require more stringent approval.
    """
    LOW = "low"           # Routine operations
    MEDIUM = "medium"     # Operations with moderate impact
    HIGH = "high"         # Operations with significant impact
    CRITICAL = "critical" # Operations requiring mandatory review


class FeedbackType(str, Enum):
    """Types of feedback that can be collected from humans."""
    APPROVAL = "approval"         # Simple approve/reject
    RATING = "rating"             # Numeric rating (1-5)
    ANNOTATION = "annotation"     # Text annotation
    CORRECTION = "correction"     # Correction to output
    ESCALATION = "escalation"     # Escalate to higher authority


class HITLCallbackConfig(BaseModel):
    """Configuration for HITL callbacks.
    
    Attributes:
        enabled: Whether callbacks are active
        async_mode: Run callbacks asynchronously
        timeout: Timeout for callback responses in seconds
    """
    enabled: bool = Field(
        default=True,
        description="Enable HITL callbacks",
    )
    async_mode: bool = Field(
        default=False,
        description="Run callbacks asynchronously",
    )
    timeout: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Callback timeout in seconds",
    )


class HITLConfig(BaseModel):
    """Main configuration for Human-in-the-Loop functionality.
    
    Controls how and when human oversight is applied to the
    ARGUS debate workflow.
    
    Attributes:
        enabled: Master switch for HITL functionality
        approval_mode: When to require approval
        interruption_points: Where to pause for human input
        timeout: Default timeout waiting for human response
        auto_approve_on_timeout: Auto-approve if timeout reached
        min_sensitivity_for_approval: Minimum sensitivity requiring approval
        callback_config: Configuration for feedback callbacks
        
    Example:
        >>> config = HITLConfig(
        ...     enabled=True,
        ...     approval_mode=ApprovalMode.SENSITIVE_ONLY,
        ...     timeout=300.0,
        ... )
    """
    enabled: bool = Field(
        default=False,
        description="Enable HITL functionality",
    )
    
    approval_mode: ApprovalMode = Field(
        default=ApprovalMode.SENSITIVE_ONLY,
        description="When to require human approval",
    )
    
    interruption_points: list[InterruptionPoint] = Field(
        default_factory=lambda: [InterruptionPoint.BEFORE_VERDICT],
        description="Where to pause for human input",
    )
    
    timeout: float = Field(
        default=300.0,
        ge=1.0,
        le=7200.0,
        description="Timeout waiting for human response (seconds)",
    )
    
    auto_approve_on_timeout: bool = Field(
        default=False,
        description="Auto-approve action if timeout is reached",
    )
    
    min_sensitivity_for_approval: SensitivityLevel = Field(
        default=SensitivityLevel.MEDIUM,
        description="Minimum sensitivity level requiring approval",
    )
    
    callback_config: HITLCallbackConfig = Field(
        default_factory=HITLCallbackConfig,
        description="Callback configuration",
    )
    
    record_all_decisions: bool = Field(
        default=True,
        description="Record all human decisions for audit",
    )
    
    allow_modifications: bool = Field(
        default=True,
        description="Allow humans to modify proposed actions",
    )
    
    escalation_enabled: bool = Field(
        default=False,
        description="Enable escalation to higher authority",
    )
    
    def should_require_approval(self, sensitivity: SensitivityLevel) -> bool:
        """Check if an action with given sensitivity requires approval.
        
        Args:
            sensitivity: Sensitivity level of the action
            
        Returns:
            True if approval is required
        """
        if not self.enabled:
            return False
            
        if self.approval_mode == ApprovalMode.NONE:
            return False
            
        if self.approval_mode == ApprovalMode.ALL:
            return True
            
        # SENSITIVE_ONLY mode
        sensitivity_order = [
            SensitivityLevel.LOW,
            SensitivityLevel.MEDIUM,
            SensitivityLevel.HIGH,
            SensitivityLevel.CRITICAL,
        ]
        
        action_index = sensitivity_order.index(sensitivity)
        threshold_index = sensitivity_order.index(self.min_sensitivity_for_approval)
        
        return action_index >= threshold_index
    
    def should_interrupt_at(self, point: InterruptionPoint) -> bool:
        """Check if workflow should be interrupted at given point.
        
        Args:
            point: The interruption point to check
            
        Returns:
            True if workflow should pause at this point
        """
        if not self.enabled:
            return False
            
        return point in self.interruption_points


def get_default_hitl_config() -> HITLConfig:
    """Get default HITL configuration.
    
    Returns:
        Default HITLConfig instance
    """
    return HITLConfig()
