"""
Phoenix Tool for ARGUS.

Arize Phoenix - ML observability for LLMs and embeddings.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class PhoenixTool(BaseTool):
    """
    Arize Phoenix - LLM observability and evaluation.
    
    Features:
    - LLM trace logging
    - Embedding analysis
    - Span management
    - Evaluation metrics
    - Dataset management
    
    Example:
        >>> tool = PhoenixTool()
        >>> result = tool(action="log_trace", trace_id="123")
        >>> result = tool(action="list_traces")
    """
    
    name = "phoenix"
    description = "LLM observability and evaluation"
    category = ToolCategory.OBSERVABILITY
    version = "1.0.0"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.base_url = (base_url or os.getenv("PHOENIX_URL", "http://localhost:6006")).rstrip("/")
        self.api_key = api_key or os.getenv("PHOENIX_API_KEY")
        
        self._session = None
        
        logger.debug(f"Phoenix tool initialized (url={self.base_url})")
    
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
        """Make API request to Phoenix."""
        session = self._get_session()
        url = f"{self.base_url}/api{endpoint}"
        
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
        project_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Phoenix operations."""
        actions = {
            # Trace operations
            "list_traces": self._list_traces,
            "get_trace": self._get_trace,
            "log_trace": self._log_trace,
            "delete_trace": self._delete_trace,
            
            # Span operations
            "list_spans": self._list_spans,
            "get_span": self._get_span,
            "log_span": self._log_span,
            
            # Evaluation
            "list_evaluations": self._list_evaluations,
            "log_evaluation": self._log_evaluation,
            "get_evaluation_metrics": self._get_evaluation_metrics,
            
            # Dataset operations
            "list_datasets": self._list_datasets,
            "get_dataset": self._get_dataset,
            "create_dataset": self._create_dataset,
            
            # Project operations
            "list_projects": self._list_projects,
            "get_project": self._get_project,
            
            # Embeddings
            "get_embedding_drift": self._get_embedding_drift,
            "get_clusters": self._get_clusters,
            
            # Export
            "export_traces": self._export_traces,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                trace_id=trace_id,
                span_id=span_id,
                project_name=project_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Phoenix error: {e}")
            return ToolResult.from_error(f"Phoenix error: {e}")
    
    # Trace operations
    def _list_traces(
        self,
        project_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List traces."""
        params = {"limit": min(limit, 1000)}
        
        if project_name:
            params["project_name"] = project_name
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
        trace_id = kwargs.get("trace_id")
        name = kwargs.get("name", "trace")
        
        data = {
            "name": name,
            "timestamp": kwargs.get("timestamp", datetime.utcnow().isoformat()),
        }
        
        if trace_id:
            data["trace_id"] = trace_id
        
        project_name = kwargs.get("project_name")
        if project_name:
            data["project_name"] = project_name
        
        metadata = kwargs.get("metadata")
        if metadata:
            data["metadata"] = metadata
        
        spans = kwargs.get("spans", [])
        if spans:
            data["spans"] = spans
        
        response = self._request("POST", "/traces", data=data)
        
        return ToolResult.from_data({
            "trace_id": response.get("trace_id") or trace_id,
            "logged": True,
        })
    
    def _delete_trace(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a trace."""
        if not trace_id:
            return ToolResult.from_error("trace_id is required")
        
        self._request("DELETE", f"/traces/{trace_id}")
        
        return ToolResult.from_data({
            "trace_id": trace_id,
            "deleted": True,
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
        span_kind = kwargs.get("span_kind", "LLM")
        
        data = {
            "trace_id": trace_id,
            "name": name,
            "span_kind": span_kind,
            "start_time": kwargs.get("start_time", datetime.utcnow().isoformat()),
        }
        
        end_time = kwargs.get("end_time")
        if end_time:
            data["end_time"] = end_time
        
        parent_id = kwargs.get("parent_id")
        if parent_id:
            data["parent_id"] = parent_id
        
        # LLM-specific fields
        input_value = kwargs.get("input")
        if input_value:
            data["input"] = {"value": input_value}
        
        output_value = kwargs.get("output")
        if output_value:
            data["output"] = {"value": output_value}
        
        attributes = kwargs.get("attributes")
        if attributes:
            data["attributes"] = attributes
        
        response = self._request("POST", "/spans", data=data)
        
        return ToolResult.from_data({
            "span_id": response.get("span_id"),
            "logged": True,
        })
    
    # Evaluation operations
    def _list_evaluations(
        self,
        project_name: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List evaluations."""
        params = {"limit": min(limit, 1000)}
        
        if project_name:
            params["project_name"] = project_name
        
        response = self._request("GET", "/evaluations", params=params)
        
        evaluations = response if isinstance(response, list) else response.get("evaluations", [])
        
        return ToolResult.from_data({
            "evaluations": evaluations,
            "count": len(evaluations),
        })
    
    def _log_evaluation(
        self,
        span_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log an evaluation."""
        if not span_id:
            return ToolResult.from_error("span_id is required")
        
        name = kwargs.get("name", "evaluation")
        score = kwargs.get("score")
        label = kwargs.get("label")
        
        if score is None and label is None:
            return ToolResult.from_error("score or label is required")
        
        data = {
            "span_id": span_id,
            "name": name,
        }
        
        if score is not None:
            data["score"] = score
        if label is not None:
            data["label"] = label
        
        explanation = kwargs.get("explanation")
        if explanation:
            data["explanation"] = explanation
        
        response = self._request("POST", "/evaluations", data=data)
        
        return ToolResult.from_data({
            "evaluation_id": response.get("id"),
            "logged": True,
        })
    
    def _get_evaluation_metrics(
        self,
        project_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get evaluation metrics summary."""
        params = {}
        if project_name:
            params["project_name"] = project_name
        
        response = self._request(
            "GET",
            "/evaluations/metrics",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "metrics": response,
        })
    
    # Dataset operations
    def _list_datasets(
        self,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List datasets."""
        params = {"limit": min(limit, 1000)}
        
        response = self._request("GET", "/datasets", params=params)
        
        datasets = response if isinstance(response, list) else response.get("datasets", [])
        
        return ToolResult.from_data({
            "datasets": datasets,
            "count": len(datasets),
        })
    
    def _get_dataset(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get dataset details."""
        dataset_id = kwargs.get("dataset_id")
        if not dataset_id:
            return ToolResult.from_error("dataset_id is required")
        
        response = self._request("GET", f"/datasets/{dataset_id}")
        
        return ToolResult.from_data({
            "dataset": response,
        })
    
    def _create_dataset(
        self,
        **kwargs,
    ) -> ToolResult:
        """Create a dataset."""
        name = kwargs.get("name")
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {"name": name}
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        examples = kwargs.get("examples", [])
        if examples:
            data["examples"] = examples
        
        response = self._request("POST", "/datasets", data=data)
        
        return ToolResult.from_data({
            "dataset_id": response.get("id"),
            "created": True,
        })
    
    # Project operations
    def _list_projects(
        self,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List projects."""
        params = {"limit": min(limit, 1000)}
        
        response = self._request("GET", "/projects", params=params)
        
        projects = response if isinstance(response, list) else response.get("projects", [])
        
        return ToolResult.from_data({
            "projects": projects,
            "count": len(projects),
        })
    
    def _get_project(
        self,
        project_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get project details."""
        if not project_name:
            return ToolResult.from_error("project_name is required")
        
        response = self._request("GET", f"/projects/{project_name}")
        
        return ToolResult.from_data({
            "project": response,
        })
    
    # Embeddings
    def _get_embedding_drift(
        self,
        project_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get embedding drift metrics."""
        params = {}
        if project_name:
            params["project_name"] = project_name
        
        response = self._request(
            "GET",
            "/embeddings/drift",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "drift": response,
        })
    
    def _get_clusters(
        self,
        project_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get embedding clusters."""
        params = {}
        if project_name:
            params["project_name"] = project_name
        
        n_clusters = kwargs.get("n_clusters")
        if n_clusters:
            params["n_clusters"] = n_clusters
        
        response = self._request(
            "GET",
            "/embeddings/clusters",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "clusters": response.get("clusters", []),
        })
    
    # Export
    def _export_traces(
        self,
        project_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Export traces."""
        params = {}
        if project_name:
            params["project_name"] = project_name
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        format_type = kwargs.get("format", "json")
        params["format"] = format_type
        
        response = self._request(
            "GET",
            "/traces/export",
            params=params,
        )
        
        return ToolResult.from_data({
            "export": response,
            "format": format_type,
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
                            "list_traces", "get_trace", "log_trace", "delete_trace",
                            "list_spans", "get_span", "log_span",
                            "list_evaluations", "log_evaluation", "get_evaluation_metrics",
                            "list_datasets", "get_dataset", "create_dataset",
                            "list_projects", "get_project",
                            "get_embedding_drift", "get_clusters",
                            "export_traces",
                        ],
                    },
                    "trace_id": {"type": "string"},
                    "span_id": {"type": "string"},
                    "project_name": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
