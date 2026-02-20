"""
Monocle Tool for ARGUS.

GenAI application observability and tracing.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class MonocleTool(BaseTool):
    """
    Monocle - GenAI application observability.
    
    Features:
    - Distributed tracing
    - LLM call monitoring
    - Cost tracking
    - Performance analytics
    - OpenTelemetry integration
    
    Example:
        >>> tool = MonocleTool()
        >>> result = tool(action="log_trace", name="my-trace")
        >>> result = tool(action="list_traces")
    """
    
    name = "monocle"
    description = "GenAI application observability"
    category = ToolCategory.OBSERVABILITY
    version = "1.0.0"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.base_url = (base_url or os.getenv("MONOCLE_URL", "http://localhost:8080")).rstrip("/")
        self.api_key = api_key or os.getenv("MONOCLE_API_KEY")
        
        self._session = None
        self._tracer = None
        
        logger.debug(f"Monocle tool initialized (url={self.base_url})")
    
    def _get_session(self):
        """Get HTTP session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session.headers.update(headers)
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict | list:
        """Make API request to Monocle."""
        session = self._get_session()
        url = f"{self.base_url}/api/v1{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            json=data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("message", error_data.get("error", f"HTTP {response.status_code}"))
                raise Exception(str(message))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        if response.status_code == 204:
            return {}
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_traces",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Monocle operations."""
        actions = {
            # Trace operations
            "list_traces": self._list_traces,
            "get_trace": self._get_trace,
            "log_trace": self._log_trace,
            
            # Span operations
            "list_spans": self._list_spans,
            "get_span": self._get_span,
            "log_span": self._log_span,
            
            # Workflow operations
            "list_workflows": self._list_workflows,
            "get_workflow": self._get_workflow,
            "log_workflow": self._log_workflow,
            
            # LLM call logging
            "log_llm_call": self._log_llm_call,
            "log_retrieval": self._log_retrieval,
            "log_embedding": self._log_embedding,
            
            # Analytics
            "get_latency_stats": self._get_latency_stats,
            "get_token_usage": self._get_token_usage,
            "get_cost_analysis": self._get_cost_analysis,
            
            # Configuration
            "setup_tracing": self._setup_tracing,
            "get_config": self._get_config,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                trace_id=trace_id,
                span_id=span_id,
                workflow_name=workflow_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Monocle error: {e}")
            return ToolResult.from_error(f"Monocle error: {e}")
    
    # Trace operations
    def _list_traces(
        self,
        workflow_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List traces."""
        params = {"limit": min(limit, 1000)}
        
        if workflow_name:
            params["workflow_name"] = workflow_name
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = self._request("GET", "/traces", params=params)
        
        traces = response if isinstance(response, list) else response.get("traces", [])
        
        return ToolResult.from_data({
            "traces": traces,
            "count": len(traces),
        })
    
    def _get_trace(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get trace details."""
        if not trace_id:
            return ToolResult.from_error("trace_id is required")
        
        response = self._request("GET", f"/traces/{trace_id}")
        
        return ToolResult.from_data({
            "trace": response,
        })
    
    def _log_trace(
        self,
        **kwargs,
    ) -> ToolResult:
        """Log a trace."""
        name = kwargs.get("name", "trace")
        
        data = {
            "name": name,
            "timestamp": kwargs.get("timestamp", datetime.utcnow().isoformat()),
        }
        
        trace_id = kwargs.get("trace_id")
        if trace_id:
            data["trace_id"] = trace_id
        
        workflow_name = kwargs.get("workflow_name")
        if workflow_name:
            data["workflow_name"] = workflow_name
        
        metadata = kwargs.get("metadata")
        if metadata:
            data["metadata"] = metadata
        
        spans = kwargs.get("spans", [])
        if spans:
            data["spans"] = spans
        
        response = self._request("POST", "/traces", data=data)
        
        return ToolResult.from_data({
            "trace_id": response.get("trace_id"),
            "logged": True,
        })
    
    # Span operations
    def _list_spans(
        self,
        trace_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List spans."""
        params = {"limit": min(limit, 1000)}
        
        if trace_id:
            params["trace_id"] = trace_id
        
        span_kind = kwargs.get("span_kind")
        if span_kind:
            params["span_kind"] = span_kind
        
        response = self._request("GET", "/spans", params=params)
        
        spans = response if isinstance(response, list) else response.get("spans", [])
        
        return ToolResult.from_data({
            "spans": spans,
            "count": len(spans),
        })
    
    def _get_span(
        self,
        span_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get span details."""
        if not span_id:
            return ToolResult.from_error("span_id is required")
        
        response = self._request("GET", f"/spans/{span_id}")
        
        return ToolResult.from_data({
            "span": response,
        })
    
    def _log_span(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log a span."""
        if not trace_id:
            return ToolResult.from_error("trace_id is required")
        
        name = kwargs.get("name", "span")
        span_kind = kwargs.get("span_kind", "INTERNAL")
        
        data = {
            "trace_id": trace_id,
            "name": name,
            "span_kind": span_kind,
            "start_time": kwargs.get("start_time", datetime.utcnow().isoformat()),
        }
        
        end_time = kwargs.get("end_time")
        if end_time:
            data["end_time"] = end_time
        
        parent_span_id = kwargs.get("parent_span_id")
        if parent_span_id:
            data["parent_span_id"] = parent_span_id
        
        attributes = kwargs.get("attributes")
        if attributes:
            data["attributes"] = attributes
        
        status = kwargs.get("status", "OK")
        data["status"] = status
        
        response = self._request("POST", "/spans", data=data)
        
        return ToolResult.from_data({
            "span_id": response.get("span_id"),
            "logged": True,
        })
    
    # Workflow operations
    def _list_workflows(
        self,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List workflows."""
        params = {"limit": min(limit, 1000)}
        
        response = self._request("GET", "/workflows", params=params)
        
        workflows = response if isinstance(response, list) else response.get("workflows", [])
        
        return ToolResult.from_data({
            "workflows": workflows,
            "count": len(workflows),
        })
    
    def _get_workflow(
        self,
        workflow_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get workflow details."""
        if not workflow_name:
            return ToolResult.from_error("workflow_name is required")
        
        response = self._request("GET", f"/workflows/{workflow_name}")
        
        return ToolResult.from_data({
            "workflow": response,
        })
    
    def _log_workflow(
        self,
        workflow_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log a workflow execution."""
        if not workflow_name:
            return ToolResult.from_error("workflow_name is required")
        
        data = {
            "workflow_name": workflow_name,
            "timestamp": kwargs.get("timestamp", datetime.utcnow().isoformat()),
        }
        
        version = kwargs.get("version")
        if version:
            data["version"] = version
        
        inputs = kwargs.get("inputs")
        if inputs:
            data["inputs"] = inputs
        
        outputs = kwargs.get("outputs")
        if outputs:
            data["outputs"] = outputs
        
        metadata = kwargs.get("metadata")
        if metadata:
            data["metadata"] = metadata
        
        response = self._request("POST", "/workflows", data=data)
        
        return ToolResult.from_data({
            "workflow_id": response.get("id"),
            "logged": True,
        })
    
    # LLM call logging
    def _log_llm_call(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log an LLM call."""
        if not trace_id:
            return ToolResult.from_error("trace_id is required")
        
        model = kwargs.get("model")
        if not model:
            return ToolResult.from_error("model is required")
        
        data = {
            "trace_id": trace_id,
            "span_kind": "LLM",
            "name": kwargs.get("name", f"llm.{model}"),
            "model": model,
            "start_time": kwargs.get("start_time", datetime.utcnow().isoformat()),
        }
        
        prompt = kwargs.get("prompt")
        if prompt:
            data["input"] = prompt
        
        completion = kwargs.get("completion")
        if completion:
            data["output"] = completion
        
        # Token counts
        prompt_tokens = kwargs.get("prompt_tokens")
        completion_tokens = kwargs.get("completion_tokens")
        
        if prompt_tokens or completion_tokens:
            data["token_usage"] = {}
            if prompt_tokens:
                data["token_usage"]["prompt_tokens"] = prompt_tokens
            if completion_tokens:
                data["token_usage"]["completion_tokens"] = completion_tokens
            data["token_usage"]["total_tokens"] = (prompt_tokens or 0) + (completion_tokens or 0)
        
        temperature = kwargs.get("temperature")
        if temperature is not None:
            data["temperature"] = temperature
        
        end_time = kwargs.get("end_time")
        if end_time:
            data["end_time"] = end_time
        
        response = self._request("POST", "/spans", data=data)
        
        return ToolResult.from_data({
            "span_id": response.get("span_id"),
            "logged": True,
        })
    
    def _log_retrieval(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log a retrieval operation."""
        if not trace_id:
            return ToolResult.from_error("trace_id is required")
        
        data = {
            "trace_id": trace_id,
            "span_kind": "RETRIEVAL",
            "name": kwargs.get("name", "retrieval"),
            "start_time": kwargs.get("start_time", datetime.utcnow().isoformat()),
        }
        
        query = kwargs.get("query")
        if query:
            data["input"] = query
        
        documents = kwargs.get("documents", [])
        if documents:
            data["output"] = documents
        
        top_k = kwargs.get("top_k")
        if top_k:
            data["top_k"] = top_k
        
        vector_db = kwargs.get("vector_db")
        if vector_db:
            data["vector_db"] = vector_db
        
        end_time = kwargs.get("end_time")
        if end_time:
            data["end_time"] = end_time
        
        response = self._request("POST", "/spans", data=data)
        
        return ToolResult.from_data({
            "span_id": response.get("span_id"),
            "logged": True,
        })
    
    def _log_embedding(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log an embedding operation."""
        if not trace_id:
            return ToolResult.from_error("trace_id is required")
        
        data = {
            "trace_id": trace_id,
            "span_kind": "EMBEDDING",
            "name": kwargs.get("name", "embedding"),
            "start_time": kwargs.get("start_time", datetime.utcnow().isoformat()),
        }
        
        model = kwargs.get("model")
        if model:
            data["model"] = model
        
        texts = kwargs.get("texts", [])
        if texts:
            data["input"] = texts
            data["input_count"] = len(texts)
        
        dimensions = kwargs.get("dimensions")
        if dimensions:
            data["dimensions"] = dimensions
        
        end_time = kwargs.get("end_time")
        if end_time:
            data["end_time"] = end_time
        
        response = self._request("POST", "/spans", data=data)
        
        return ToolResult.from_data({
            "span_id": response.get("span_id"),
            "logged": True,
        })
    
    # Analytics
    def _get_latency_stats(
        self,
        workflow_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get latency statistics."""
        params = {}
        
        if workflow_name:
            params["workflow_name"] = workflow_name
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = self._request(
            "GET",
            "/analytics/latency",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "latency_stats": response,
        })
    
    def _get_token_usage(
        self,
        workflow_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get token usage statistics."""
        params = {}
        
        if workflow_name:
            params["workflow_name"] = workflow_name
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = self._request(
            "GET",
            "/analytics/tokens",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "token_usage": response,
        })
    
    def _get_cost_analysis(
        self,
        workflow_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get cost analysis."""
        params = {}
        
        if workflow_name:
            params["workflow_name"] = workflow_name
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = self._request(
            "GET",
            "/analytics/cost",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "cost_analysis": response,
        })
    
    # Configuration
    def _setup_tracing(
        self,
        **kwargs,
    ) -> ToolResult:
        """Setup tracing configuration."""
        workflow_name = kwargs.get("workflow_name", "default")
        
        data = {
            "workflow_name": workflow_name,
            "exporter": kwargs.get("exporter", "otlp"),
            "endpoint": kwargs.get("endpoint", self.base_url),
        }
        
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is not None:
            data["sample_rate"] = sample_rate
        
        response = self._request("POST", "/config/tracing", data=data)
        
        return ToolResult.from_data({
            "config": response,
            "setup": True,
        })
    
    def _get_config(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get current configuration."""
        response = self._request("GET", "/config")
        
        return ToolResult.from_data({
            "config": response,
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_traces", "get_trace", "log_trace",
                            "list_spans", "get_span", "log_span",
                            "list_workflows", "get_workflow", "log_workflow",
                            "log_llm_call", "log_retrieval", "log_embedding",
                            "get_latency_stats", "get_token_usage", "get_cost_analysis",
                            "setup_tracing", "get_config",
                        ],
                    },
                    "trace_id": {"type": "string"},
                    "span_id": {"type": "string"},
                    "workflow_name": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
