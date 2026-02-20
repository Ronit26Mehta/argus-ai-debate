"""
Cloud Trace Tool for ARGUS.

Monitor, debug, and trace ADK agent interactions.
Integration with Google Cloud Trace for distributed tracing.
"""

from __future__ import annotations

import os
import uuid
import logging
import time
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A trace span representing a unit of work."""
    span_id: str
    trace_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0
    
    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class Trace:
    """A complete trace representing a request lifecycle."""
    trace_id: str
    name: str
    spans: list[Span] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def root_span(self) -> Optional[Span]:
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None
    
    @property
    def duration_ms(self) -> float:
        if self.root_span:
            return self.root_span.duration_ms
        return 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }


class CloudTraceTool(BaseTool):
    """
    Google Cloud Trace - Distributed tracing for AI agents.
    
    Features:
    - Create and manage traces
    - Span instrumentation
    - Automatic context propagation
    - Export to Cloud Trace service
    - Agent interaction debugging
    
    Example:
        >>> tool = CloudTraceTool(project_id="my-project")
        >>> result = tool(action="start_trace", name="agent_request")
        >>> result = tool(action="start_span", trace_id="...", name="llm_call")
        >>> result = tool(action="end_span", span_id="...")
        >>> result = tool(action="end_trace", trace_id="...")
    """
    
    name = "cloud_trace"
    description = "Monitor, debug, and trace ADK agent interactions"
    category = ToolCategory.EXTERNAL_API
    version = "1.0.0"
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        enable_cloud_export: bool = False,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.enable_cloud_export = enable_cloud_export
        
        # Local storage
        self._traces: dict[str, Trace] = {}
        self._spans: dict[str, Span] = {}
        self._active_spans: dict[str, str] = {}  # trace_id -> current span_id
        
        # Cloud Trace client (lazy loaded)
        self._trace_client = None
        
        logger.debug(f"CloudTrace initialized (cloud_export={self.enable_cloud_export})")
    
    def _get_trace_client(self):
        """Lazy-load Cloud Trace client."""
        if self._trace_client is None and self.enable_cloud_export:
            try:
                from google.cloud import trace_v1
                self._trace_client = trace_v1.TraceServiceClient()
            except ImportError:
                logger.warning("google-cloud-trace not installed, using local tracing only")
        return self._trace_client
    
    def execute(
        self,
        action: str = "list_traces",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[dict] = None,
        status: Optional[str] = None,
        event_name: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Cloud Trace operations."""
        actions = {
            "start_trace": self._start_trace,
            "end_trace": self._end_trace,
            "start_span": self._start_span,
            "end_span": self._end_span,
            "add_event": self._add_event,
            "set_status": self._set_status,
            "set_attributes": self._set_attributes,
            "get_trace": self._get_trace,
            "get_span": self._get_span,
            "list_traces": self._list_traces,
            "export_trace": self._export_trace,
            "get_active_span": self._get_active_span,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                trace_id=trace_id,
                span_id=span_id,
                name=name,
                attributes=attributes or {},
                status=status,
                event_name=event_name,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"CloudTrace error: {e}")
            return ToolResult.from_error(f"CloudTrace error: {e}")
    
    def _start_trace(
        self,
        name: Optional[str] = None,
        attributes: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Start a new trace."""
        if not name:
            return ToolResult.from_error("name is required")
        
        trace_id = str(uuid.uuid4()).replace("-", "")[:32]
        
        trace = Trace(
            trace_id=trace_id,
            name=name,
        )
        
        # Create root span
        root_span = Span(
            span_id=str(uuid.uuid4())[:16],
            trace_id=trace_id,
            name=name,
            attributes=attributes or {},
        )
        
        trace.spans.append(root_span)
        self._traces[trace_id] = trace
        self._spans[root_span.span_id] = root_span
        self._active_spans[trace_id] = root_span.span_id
        
        logger.debug(f"Started trace {trace_id} with root span {root_span.span_id}")
        
        return ToolResult.from_data({
            "trace_id": trace_id,
            "span_id": root_span.span_id,
            "name": name,
        })
    
    def _end_trace(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """End a trace."""
        if not trace_id or trace_id not in self._traces:
            return ToolResult.from_error("Invalid trace_id")
        
        trace = self._traces[trace_id]
        
        # End all open spans
        for span in trace.spans:
            if span.end_time is None:
                span.end_time = datetime.utcnow()
        
        # Remove from active
        if trace_id in self._active_spans:
            del self._active_spans[trace_id]
        
        # Export to Cloud Trace if enabled
        if self.enable_cloud_export:
            self._export_to_cloud(trace)
        
        return ToolResult.from_data({
            "trace_id": trace_id,
            "name": trace.name,
            "duration_ms": trace.duration_ms,
            "span_count": len(trace.spans),
        })
    
    def _start_span(
        self,
        trace_id: Optional[str] = None,
        name: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Start a new span within a trace."""
        if not trace_id or trace_id not in self._traces:
            return ToolResult.from_error("Invalid trace_id")
        if not name:
            return ToolResult.from_error("name is required")
        
        trace = self._traces[trace_id]
        
        # Use active span as parent if not specified
        if parent_span_id is None:
            parent_span_id = self._active_spans.get(trace_id)
        
        span = Span(
            span_id=str(uuid.uuid4())[:16],
            trace_id=trace_id,
            name=name,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        
        trace.spans.append(span)
        self._spans[span.span_id] = span
        self._active_spans[trace_id] = span.span_id
        
        return ToolResult.from_data({
            "span_id": span.span_id,
            "trace_id": trace_id,
            "name": name,
            "parent_span_id": parent_span_id,
        })
    
    def _end_span(
        self,
        span_id: Optional[str] = None,
        status: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """End a span."""
        if not span_id or span_id not in self._spans:
            return ToolResult.from_error("Invalid span_id")
        
        span = self._spans[span_id]
        span.end_time = datetime.utcnow()
        
        if status:
            span.status = status
        
        # Update active span to parent
        if span.trace_id in self._active_spans:
            if self._active_spans[span.trace_id] == span_id:
                self._active_spans[span.trace_id] = span.parent_span_id
        
        return ToolResult.from_data({
            "span_id": span_id,
            "name": span.name,
            "duration_ms": span.duration_ms,
            "status": span.status,
        })
    
    def _add_event(
        self,
        span_id: Optional[str] = None,
        event_name: Optional[str] = None,
        attributes: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Add an event to a span."""
        if not span_id or span_id not in self._spans:
            return ToolResult.from_error("Invalid span_id")
        if not event_name:
            return ToolResult.from_error("event_name is required")
        
        span = self._spans[span_id]
        span.add_event(event_name, attributes)
        
        return ToolResult.from_data({
            "span_id": span_id,
            "event_name": event_name,
            "event_count": len(span.events),
        })
    
    def _set_status(
        self,
        span_id: Optional[str] = None,
        status: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Set span status."""
        if not span_id or span_id not in self._spans:
            return ToolResult.from_error("Invalid span_id")
        if not status:
            return ToolResult.from_error("status is required")
        
        span = self._spans[span_id]
        span.status = status
        
        return ToolResult.from_data({
            "span_id": span_id,
            "status": status,
        })
    
    def _set_attributes(
        self,
        span_id: Optional[str] = None,
        attributes: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Set span attributes."""
        if not span_id or span_id not in self._spans:
            return ToolResult.from_error("Invalid span_id")
        
        span = self._spans[span_id]
        span.attributes.update(attributes or {})
        
        return ToolResult.from_data({
            "span_id": span_id,
            "attributes": span.attributes,
        })
    
    def _get_trace(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get trace details."""
        if not trace_id or trace_id not in self._traces:
            return ToolResult.from_error("Invalid trace_id")
        
        trace = self._traces[trace_id]
        return ToolResult.from_data({"trace": trace.to_dict()})
    
    def _get_span(
        self,
        span_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get span details."""
        if not span_id or span_id not in self._spans:
            return ToolResult.from_error("Invalid span_id")
        
        span = self._spans[span_id]
        return ToolResult.from_data({"span": span.to_dict()})
    
    def _list_traces(
        self,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List all traces."""
        traces = list(self._traces.values())[-limit:]
        
        return ToolResult.from_data({
            "traces": [
                {
                    "trace_id": t.trace_id,
                    "name": t.name,
                    "started_at": t.started_at.isoformat(),
                    "duration_ms": t.duration_ms,
                    "span_count": len(t.spans),
                }
                for t in traces
            ],
            "count": len(traces),
        })
    
    def _export_trace(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Export a trace to Cloud Trace."""
        if not trace_id or trace_id not in self._traces:
            return ToolResult.from_error("Invalid trace_id")
        
        trace = self._traces[trace_id]
        
        if self.enable_cloud_export:
            try:
                self._export_to_cloud(trace)
                return ToolResult.from_data({
                    "trace_id": trace_id,
                    "exported": True,
                    "destination": "Cloud Trace",
                })
            except Exception as e:
                return ToolResult.from_error(f"Export failed: {e}")
        
        return ToolResult.from_data({
            "trace_id": trace_id,
            "exported": False,
            "reason": "Cloud export not enabled",
            "trace": trace.to_dict(),
        })
    
    def _export_to_cloud(self, trace: Trace) -> None:
        """Export trace to Google Cloud Trace."""
        client = self._get_trace_client()
        if not client:
            return
        
        from google.cloud.trace_v1 import types
        
        # Convert to Cloud Trace format
        cloud_trace = types.Trace(
            project_id=self.project_id,
            trace_id=trace.trace_id,
            spans=[
                types.TraceSpan(
                    span_id=int(span.span_id[:16], 16) & 0x7FFFFFFFFFFFFFFF,
                    name=span.name,
                    start_time=span.start_time,
                    end_time=span.end_time or datetime.utcnow(),
                    parent_span_id=int(span.parent_span_id[:16], 16) & 0x7FFFFFFFFFFFFFFF if span.parent_span_id else 0,
                )
                for span in trace.spans
            ],
        )
        
        client.patch_traces(project_id=self.project_id, traces=types.Traces(traces=[cloud_trace]))
    
    def _get_active_span(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get the currently active span for a trace."""
        if not trace_id or trace_id not in self._traces:
            return ToolResult.from_error("Invalid trace_id")
        
        active_span_id = self._active_spans.get(trace_id)
        if not active_span_id or active_span_id not in self._spans:
            return ToolResult.from_data({"active_span": None})
        
        return ToolResult.from_data({"active_span": self._spans[active_span_id].to_dict()})
    
    @contextmanager
    def trace_context(self, name: str, attributes: Optional[dict] = None):
        """Context manager for tracing a code block."""
        result = self.execute(action="start_trace", name=name, attributes=attributes)
        trace_id = result.data["trace_id"]
        span_id = result.data["span_id"]
        
        try:
            yield trace_id, span_id
        except Exception as e:
            self.execute(action="set_status", span_id=span_id, status="ERROR")
            self.execute(action="add_event", span_id=span_id, event_name="exception",
                        attributes={"message": str(e)})
            raise
        finally:
            self.execute(action="end_trace", trace_id=trace_id)
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start_trace", "end_trace", "start_span", "end_span",
                                "add_event", "set_status", "set_attributes",
                                "get_trace", "get_span", "list_traces",
                                "export_trace", "get_active_span"],
                    },
                    "trace_id": {"type": "string"},
                    "span_id": {"type": "string"},
                    "name": {"type": "string"},
                    "attributes": {"type": "object"},
                    "status": {"type": "string"},
                    "event_name": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["action"],
            },
        }
