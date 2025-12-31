"""
Guardrails for ARGUS Tools.

Provides content filtering and policy enforcement for tool execution.
Users can create custom guardrails by subclassing the Guardrail base class.

Example:
    >>> # Create a custom guardrail
    >>> class PIIFilter(Guardrail):
    ...     def check(self, tool_name, arguments):
    ...         if contains_pii(arguments):
    ...             return GuardrailResult(allowed=False, reason="PII detected")
    ...         return GuardrailResult(allowed=True)
    >>> 
    >>> # Add to executor
    >>> executor.add_guardrail(PIIFilter())
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Pattern
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GuardrailConfig(BaseModel):
    """Configuration for guardrails.
    
    Attributes:
        enabled: Whether guardrails are active
        log_violations: Log when guardrails block execution
        strict_mode: Fail closed on errors
    """
    enabled: bool = Field(
        default=True,
        description="Enable guardrail checks",
    )
    log_violations: bool = Field(
        default=True,
        description="Log blocked executions",
    )
    strict_mode: bool = Field(
        default=False,
        description="Fail closed on guardrail errors",
    )


@dataclass
class GuardrailResult:
    """Result from a guardrail check.
    
    Attributes:
        allowed: Whether the operation is allowed
        reason: Reason if blocked
        metadata: Additional information
    """
    allowed: bool
    reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Guardrail(ABC):
    """Abstract base class for guardrails.
    
    Subclass this to create custom guardrails that filter
    tool inputs and outputs based on policies.
    
    Example:
        >>> class MyGuardrail(Guardrail):
        ...     name = "my_guardrail"
        ...     
        ...     def check(self, tool_name, arguments):
        ...         # Check logic here
        ...         return GuardrailResult(allowed=True)
    """
    
    name: str = "base_guardrail"
    description: str = "Base guardrail"
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        """Initialize guardrail.
        
        Args:
            config: Guardrail configuration
        """
        self.config = config or GuardrailConfig()
    
    @abstractmethod
    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardrailResult:
        """Check if tool execution should be allowed.
        
        Args:
            tool_name: Name of the tool being executed
            arguments: Arguments to the tool
            
        Returns:
            GuardrailResult indicating if allowed
        """
        pass
    
    def check_output(
        self,
        tool_name: str,
        result: Any,
    ) -> GuardrailResult:
        """Check tool output (optional).
        
        Override to filter outputs as well as inputs.
        
        Args:
            tool_name: Name of the tool
            result: Tool result to check
            
        Returns:
            GuardrailResult
        """
        return GuardrailResult(allowed=True)


# =============================================================================
# Built-in Guardrails
# =============================================================================

class ContentFilter(Guardrail):
    """Filters content based on keyword/pattern matching.
    
    Blocks execution if arguments contain blocked patterns.
    
    Example:
        >>> filter = ContentFilter(
        ...     blocked_patterns=[r"password", r"secret"],
        ...     blocked_keywords=["DELETE", "DROP"],
        ... )
    """
    
    name = "content_filter"
    description = "Filter content based on patterns and keywords"
    
    def __init__(
        self,
        blocked_patterns: Optional[list[str]] = None,
        blocked_keywords: Optional[list[str]] = None,
        case_sensitive: bool = False,
        config: Optional[GuardrailConfig] = None,
    ):
        """Initialize content filter.
        
        Args:
            blocked_patterns: Regex patterns to block
            blocked_keywords: Keywords to block
            case_sensitive: Whether matching is case-sensitive
            config: Guardrail configuration
        """
        super().__init__(config)
        self.blocked_keywords = blocked_keywords or []
        self.case_sensitive = case_sensitive
        
        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self.blocked_patterns: list[Pattern] = [
            re.compile(p, flags)
            for p in (blocked_patterns or [])
        ]
    
    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardrailResult:
        """Check arguments for blocked content.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            GuardrailResult
        """
        # Convert arguments to text for checking
        text = self._arguments_to_text(arguments)
        
        # Check keywords
        check_text = text if self.case_sensitive else text.lower()
        for keyword in self.blocked_keywords:
            check_keyword = keyword if self.case_sensitive else keyword.lower()
            if check_keyword in check_text:
                reason = f"Blocked keyword detected: {keyword}"
                if self.config.log_violations:
                    logger.warning(f"ContentFilter blocked {tool_name}: {reason}")
                return GuardrailResult(allowed=False, reason=reason)
        
        # Check patterns
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                reason = f"Blocked pattern detected: {pattern.pattern}"
                if self.config.log_violations:
                    logger.warning(f"ContentFilter blocked {tool_name}: {reason}")
                return GuardrailResult(allowed=False, reason=reason)
        
        return GuardrailResult(allowed=True)
    
    def _arguments_to_text(self, arguments: dict[str, Any]) -> str:
        """Convert arguments to searchable text."""
        parts = []
        for key, value in arguments.items():
            parts.append(f"{key}: {value}")
        return " ".join(parts)


class PolicyEnforcer(Guardrail):
    """Enforces tool-specific policies.
    
    Define allowed/blocked tools and argument constraints.
    
    Example:
        >>> enforcer = PolicyEnforcer(
        ...     allowed_tools=["search", "calculate"],
        ...     tool_limits={"search": 10},  # Max 10 calls
        ... )
    """
    
    name = "policy_enforcer"
    description = "Enforce tool usage policies"
    
    def __init__(
        self,
        allowed_tools: Optional[list[str]] = None,
        blocked_tools: Optional[list[str]] = None,
        tool_limits: Optional[dict[str, int]] = None,
        config: Optional[GuardrailConfig] = None,
    ):
        """Initialize policy enforcer.
        
        Args:
            allowed_tools: Whitelist of allowed tools (None = all allowed)
            blocked_tools: Blacklist of blocked tools
            tool_limits: Per-tool call limits
            config: Guardrail configuration
        """
        super().__init__(config)
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.blocked_tools = set(blocked_tools) if blocked_tools else set()
        self.tool_limits = tool_limits or {}
        self._call_counts: dict[str, int] = {}
    
    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardrailResult:
        """Check if tool execution is allowed by policy.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            GuardrailResult
        """
        # Check blocklist
        if tool_name in self.blocked_tools:
            return GuardrailResult(
                allowed=False,
                reason=f"Tool '{tool_name}' is blocked by policy",
            )
        
        # Check allowlist
        if self.allowed_tools is not None:
            if tool_name not in self.allowed_tools:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Tool '{tool_name}' is not in allowed list",
                )
        
        # Check limits
        if tool_name in self.tool_limits:
            count = self._call_counts.get(tool_name, 0)
            if count >= self.tool_limits[tool_name]:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Tool '{tool_name}' has reached call limit",
                )
            self._call_counts[tool_name] = count + 1
        
        return GuardrailResult(allowed=True)
    
    def reset_counts(self) -> None:
        """Reset call counts for all tools."""
        self._call_counts.clear()
    
    def get_counts(self) -> dict[str, int]:
        """Get current call counts."""
        return dict(self._call_counts)


class RateLimiter(Guardrail):
    """Rate limiting guardrail.
    
    Limits tool calls per time window.
    
    Example:
        >>> limiter = RateLimiter(
        ...     calls_per_minute=60,
        ...     calls_per_hour=1000,
        ... )
    """
    
    name = "rate_limiter"
    description = "Limit tool call rate"
    
    def __init__(
        self,
        calls_per_minute: int = 60,
        calls_per_hour: int = 1000,
        config: Optional[GuardrailConfig] = None,
    ):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls per minute
            calls_per_hour: Maximum calls per hour
            config: Guardrail configuration
        """
        super().__init__(config)
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self._minute_timestamps: list[float] = []
        self._hour_timestamps: list[float] = []
    
    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardrailResult:
        """Check if within rate limits.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            GuardrailResult
        """
        import time
        now = time.time()
        
        # Clean old timestamps
        minute_ago = now - 60
        hour_ago = now - 3600
        
        self._minute_timestamps = [
            t for t in self._minute_timestamps if t > minute_ago
        ]
        self._hour_timestamps = [
            t for t in self._hour_timestamps if t > hour_ago
        ]
        
        # Check limits
        if len(self._minute_timestamps) >= self.calls_per_minute:
            return GuardrailResult(
                allowed=False,
                reason="Rate limit exceeded (per minute)",
            )
        
        if len(self._hour_timestamps) >= self.calls_per_hour:
            return GuardrailResult(
                allowed=False,
                reason="Rate limit exceeded (per hour)",
            )
        
        # Record call
        self._minute_timestamps.append(now)
        self._hour_timestamps.append(now)
        
        return GuardrailResult(allowed=True)


class InputValidator(Guardrail):
    """Validates tool inputs against schemas.
    
    Example:
        >>> validator = InputValidator(
        ...     required_fields={"search": ["query"]},
        ...     max_lengths={"query": 1000},
        ... )
    """
    
    name = "input_validator"
    description = "Validate tool inputs"
    
    def __init__(
        self,
        required_fields: Optional[dict[str, list[str]]] = None,
        max_lengths: Optional[dict[str, int]] = None,
        config: Optional[GuardrailConfig] = None,
    ):
        """Initialize input validator.
        
        Args:
            required_fields: Required fields per tool
            max_lengths: Maximum string lengths per field
            config: Configuration
        """
        super().__init__(config)
        self.required_fields = required_fields or {}
        self.max_lengths = max_lengths or {}
    
    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> GuardrailResult:
        """Validate inputs.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            GuardrailResult
        """
        # Check required fields
        if tool_name in self.required_fields:
            for field in self.required_fields[tool_name]:
                if field not in arguments:
                    return GuardrailResult(
                        allowed=False,
                        reason=f"Missing required field: {field}",
                    )
        
        # Check lengths
        for field, max_len in self.max_lengths.items():
            if field in arguments:
                value = arguments[field]
                if isinstance(value, str) and len(value) > max_len:
                    return GuardrailResult(
                        allowed=False,
                        reason=f"Field '{field}' exceeds max length {max_len}",
                    )
        
        return GuardrailResult(allowed=True)
