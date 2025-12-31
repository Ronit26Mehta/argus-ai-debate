"""
Tracing for ARGUS.

Provides distributed tracing capabilities for tracking
execution flow across ARGUS components.

Example:
    >>> tracer = Tracer("argus")
    >>> 
    >>> with tracer.span("debate") as span:
    ...     span.set_attribute("proposition_id", "p1")
    ...     # Do debate work
    ...     with tracer.span("evidence_retrieval") as child:
    ...         # Get evidence
    ...         child.set_attribute("count", 5)
"""

from __future__ import annotations

import logging
import time
import uuid
import contextvars
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from contextlib import contextmanager

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Context variable for current span
_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "_current_span", default=None
)


class SpanStatus(str, Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class TraceConfig(BaseModel):
    """Configuration for tracing.
    
    Attributes:
        enabled: Whether tracing is active
        sample_rate: Fraction of traces to record
        max_spans: Maximum spans to keep
    """
    enabled: bool = Field(
        default=True,
        description="Enable tracing",
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate",
    )
    max_spans: int = Field(
        default=10000,
        ge=100,
        description="Maximum spans to keep",
    )
    export_on_complete: bool = Field(
        default=False,
        description="Export spans when completed",
    )


@dataclass
class SpanContext:
    """Context for a trace span."""
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
        }


@dataclass
class Span:
    """A trace span representing a unit of work.
    
    Spans track the start/end time, attributes, and events
    for a specific operation.
    """
    name: str
    context: SpanContext
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on the span.
        
        Args:
            key: Attribute key
            value: Attribute value
            
        Returns:
            Self for chaining
        """
        self.attributes[key] = value
        return self
    
    def set_attributes(self, attributes: dict[str, Any]) -> "Span":
        """Set multiple attributes.
        
        Args:
            attributes: Dict of attributes
            
        Returns:
            Self for chaining
        """
        self.attributes.update(attributes)
        return self
    
    def add_event(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> "Span":
        """Add an event to the span.
        
        Args:
            name: Event name
            attributes: Event attributes
            
        Returns:
            Self for chaining
        """
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })
        return self
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> "Span":
        """Set the span status.
        
        Args:
            status: Span status
            message: Optional status message
            
        Returns:
            Self for chaining
        """
        self.status = status
        if message:
            self.attributes["status_message"] = message
        return self
    
    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            end = datetime.utcnow()
        else:
            end = self.end_time
        return (end - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """Creates and manages trace spans.
    
    Use the tracer to create spans around operations you want to trace.
    Spans can be nested to create a tree structure.
    
    Example:
        >>> tracer = Tracer("debate_service")
        >>> 
        >>> with tracer.span("run_debate") as span:
        ...     span.set_attribute("proposition", "test")
        ...     
        ...     with tracer.span("gather_evidence") as child:
        ...         # Child span automatically linked to parent
        ...         child.add_event("found_evidence", {"count": 5})
    """
    
    def __init__(
        self,
        service_name: str = "argus",
        config: Optional[TraceConfig] = None,
    ):
        """Initialize tracer.
        
        Args:
            service_name: Name of the service
            config: Trace configuration
        """
        self.service_name = service_name
        self.config = config or TraceConfig()
        self._lock = RLock()
        self._spans: list[Span] = []
        self._active_traces: dict[str, list[Span]] = {}
        
        logger.debug(f"Initialized Tracer for {service_name}")
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return uuid.uuid4().hex[:16]
    
    def _should_sample(self) -> bool:
        """Check if trace should be sampled."""
        if not self.config.enabled:
            return False
        if self.config.sample_rate >= 1.0:
            return True
        import random
        return random.random() < self.config.sample_rate
    
    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """Create a new span as a context manager.
        
        Args:
            name: Span name
            attributes: Initial attributes
            
        Yields:
            Span object
        """
        if not self._should_sample():
            # Create a no-op span if not sampling
            yield _NoOpSpan()
            return
        
        # Get parent span from context
        parent_span = _current_span.get()
        
        # Create context
        if parent_span:
            context = SpanContext(
                trace_id=parent_span.context.trace_id,
                span_id=self._generate_id(),
                parent_id=parent_span.context.span_id,
            )
        else:
            context = SpanContext(
                trace_id=self._generate_id(),
                span_id=self._generate_id(),
            )
        
        # Create span
        span = Span(name=name, context=context)
        if attributes:
            span.set_attributes(attributes)
        
        # Set as current
        token = _current_span.set(span)
        
        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"message": str(e), "type": type(e).__name__})
            raise
        finally:
            # End span
            span.end()
            
            # Restore parent
            _current_span.reset(token)
            
            # Store span
            with self._lock:
                self._spans.append(span)
                
                # Track in active traces
                trace_id = context.trace_id
                if trace_id not in self._active_traces:
                    self._active_traces[trace_id] = []
                self._active_traces[trace_id].append(span)
                
                # Limit stored spans
                if len(self._spans) > self.config.max_spans:
                    self._spans = self._spans[-self.config.max_spans:]
    
    def start_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Span:
        """Start a span manually (must call end()).
        
        Args:
            name: Span name
            attributes: Initial attributes
            
        Returns:
            Span object
        """
        parent_span = _current_span.get()
        
        if parent_span:
            context = SpanContext(
                trace_id=parent_span.context.trace_id,
                span_id=self._generate_id(),
                parent_id=parent_span.context.span_id,
            )
        else:
            context = SpanContext(
                trace_id=self._generate_id(),
                span_id=self._generate_id(),
            )
        
        span = Span(name=name, context=context)
        if attributes:
            span.set_attributes(attributes)
        
        return span
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span.
        
        Returns:
            Current Span or None
        """
        return _current_span.get()
    
    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            List of spans
        """
        with self._lock:
            return list(self._active_traces.get(trace_id, []))
    
    def get_recent_spans(self, limit: int = 100) -> list[Span]:
        """Get recent spans.
        
        Args:
            limit: Maximum spans to return
            
        Returns:
            List of spans
        """
        with self._lock:
            return self._spans[-limit:]
    
    def get_stats(self) -> dict[str, Any]:
        """Get tracer statistics.
        
        Returns:
            Dict with span counts and timing
        """
        with self._lock:
            completed = [s for s in self._spans if s.end_time]
            durations = [s.duration_ms for s in completed]
            
            return {
                "total_spans": len(self._spans),
                "active_traces": len(self._active_traces),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "error_count": sum(1 for s in self._spans if s.status == SpanStatus.ERROR),
            }
    
    def export(self) -> list[dict[str, Any]]:
        """Export all spans as dicts.
        
        Returns:
            List of span dicts
        """
        with self._lock:
            return [s.to_dict() for s in self._spans]
    
    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self._spans.clear()
            self._active_traces.clear()


class _NoOpSpan:
    """No-op span for when sampling is disabled."""
    
    def set_attribute(self, key: str, value: Any) -> "_NoOpSpan":
        return self
    
    def set_attributes(self, attributes: dict[str, Any]) -> "_NoOpSpan":
        return self
    
    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> "_NoOpSpan":
        return self
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> "_NoOpSpan":
        return self


# =============================================================================
# Global Default Tracer
# =============================================================================

_default_tracer: Optional[Tracer] = None
_tracer_lock = RLock()


def get_tracer(service_name: str = "argus") -> Tracer:
    """Get the default tracer.
    
    Args:
        service_name: Service name
        
    Returns:
        Tracer instance
    """
    global _default_tracer
    
    with _tracer_lock:
        if _default_tracer is None:
            _default_tracer = Tracer(service_name)
        return _default_tracer
