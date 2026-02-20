"""
Weights & Biases Weave Tool for ARGUS.

LLM tracing, evaluation, and observability.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class WandBWeaveTool(BaseTool):
    """
    Weights & Biases Weave - LLM observability and evaluation.
    
    Features:
    - LLM call tracing
    - Evaluation tracking
    - Dataset management
    - Model monitoring
    - Experiment tracking
    - Cost analysis
    
    Example:
        >>> tool = WandBWeaveTool(api_key="your-api-key")
        >>> result = tool(action="list_calls")
        >>> result = tool(action="create_evaluation", name="my-eval")
    """
    
    name = "wandb_weave"
    description = "LLM tracing and evaluation with W&B Weave"
    category = ToolCategory.OBSERVABILITY
    version = "1.0.0"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("WANDB_API_KEY")
        self.entity = entity or os.getenv("WANDB_ENTITY")
        self.project = project or os.getenv("WANDB_PROJECT", "weave")
        self.base_url = "https://api.wandb.ai"
        self.weave_url = "https://trace.wandb.ai"
        
        self._session = None
        
        logger.debug(f"W&B Weave tool initialized (entity={self.entity}, project={self.project})")
    
    def _get_session(self):
        """Get HTTP session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Basic api:{self.api_key}"
            self._session.headers.update(headers)
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        base_url: Optional[str] = None,
    ) -> dict | list:
        """Make API request to W&B/Weave."""
        session = self._get_session()
        url = f"{base_url or self.weave_url}{endpoint}"
        
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
    
    def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute GraphQL query against W&B API."""
        session = self._get_session()
        url = f"{self.base_url}/graphql"
        
        data = {"query": query}
        if variables:
            data["variables"] = variables
        
        response = session.post(url, json=data)
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("message", str(error_data.get("errors", f"HTTP {response.status_code}")))
                raise Exception(str(message))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        result = response.json()
        if "errors" in result:
            raise Exception(str(result["errors"]))
        
        return result.get("data", {})
    
    def execute(
        self,
        action: str = "list_calls",
        call_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        evaluation_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute W&B Weave operations."""
        actions = {
            # Call/Trace operations
            "list_calls": self._list_calls,
            "get_call": self._get_call,
            "log_call": self._log_call,
            "list_traces": self._list_traces,
            "get_trace": self._get_trace,
            
            # Evaluation operations
            "list_evaluations": self._list_evaluations,
            "get_evaluation": self._get_evaluation,
            "create_evaluation": self._create_evaluation,
            "log_evaluation_result": self._log_evaluation_result,
            
            # Dataset operations
            "list_datasets": self._list_datasets,
            "get_dataset": self._get_dataset,
            "create_dataset": self._create_dataset,
            "add_dataset_rows": self._add_dataset_rows,
            
            # Model operations
            "list_models": self._list_models,
            "get_model": self._get_model,
            "log_model": self._log_model,
            
            # Ops (functions) operations
            "list_ops": self._list_ops,
            "get_op": self._get_op,
            
            # Analytics
            "get_call_stats": self._get_call_stats,
            "get_cost_summary": self._get_cost_summary,
            "get_latency_stats": self._get_latency_stats,
            
            # Project operations
            "list_projects": self._list_projects,
            "get_project": self._get_project,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                call_id=call_id,
                trace_id=trace_id,
                evaluation_id=evaluation_id,
                dataset_id=dataset_id,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"W&B Weave error: {e}")
            return ToolResult.from_error(f"W&B Weave error: {e}")
    
    # Call/Trace operations
    def _list_calls(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List calls/spans."""
        limit = kwargs.get("limit", 100)
        
        params = {
            "entity": self.entity,
            "project": self.project,
            "limit": min(limit, 1000),
        }
        
        if trace_id:
            params["trace_id"] = trace_id
        
        op_name = kwargs.get("op_name")
        if op_name:
            params["op_name"] = op_name
        
        start_time = kwargs.get("start_time")
        if start_time:
            params["start_time"] = start_time
        
        end_time = kwargs.get("end_time")
        if end_time:
            params["end_time"] = end_time
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/calls",
            params=params,
        )
        
        calls = response if isinstance(response, list) else response.get("calls", [])
        
        return ToolResult.from_data({
            "calls": calls,
            "count": len(calls),
        })
    
    def _get_call(
        self,
        call_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get call details."""
        if not call_id:
            return ToolResult.from_error("call_id is required")
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/calls/{call_id}",
        )
        
        return ToolResult.from_data({
            "call": response,
        })
    
    def _log_call(
        self,
        **kwargs,
    ) -> ToolResult:
        """Log a call/span."""
        op_name = kwargs.get("op_name")
        if not op_name:
            return ToolResult.from_error("op_name is required")
        
        data = {
            "op_name": op_name,
            "started_at": kwargs.get("started_at", datetime.utcnow().isoformat()),
        }
        
        trace_id = kwargs.get("trace_id")
        if trace_id:
            data["trace_id"] = trace_id
        
        parent_id = kwargs.get("parent_id")
        if parent_id:
            data["parent_id"] = parent_id
        
        inputs = kwargs.get("inputs")
        if inputs:
            data["inputs"] = inputs
        
        output = kwargs.get("output")
        if output:
            data["output"] = output
        
        ended_at = kwargs.get("ended_at")
        if ended_at:
            data["ended_at"] = ended_at
        
        exception = kwargs.get("exception")
        if exception:
            data["exception"] = exception
        
        # LLM-specific attributes
        model = kwargs.get("model")
        if model:
            data["attributes"] = data.get("attributes", {})
            data["attributes"]["model"] = model
        
        token_usage = kwargs.get("token_usage")
        if token_usage:
            data["attributes"] = data.get("attributes", {})
            data["attributes"]["token_usage"] = token_usage
        
        response = self._request(
            "POST",
            f"/{self.entity}/{self.project}/calls",
            data=data,
        )
        
        return ToolResult.from_data({
            "call_id": response.get("id"),
            "trace_id": response.get("trace_id"),
            "logged": True,
        })
    
    def _list_traces(
        self,
        **kwargs,
    ) -> ToolResult:
        """List traces."""
        limit = kwargs.get("limit", 100)
        
        params = {
            "entity": self.entity,
            "project": self.project,
            "limit": min(limit, 1000),
        }
        
        start_time = kwargs.get("start_time")
        if start_time:
            params["start_time"] = start_time
        
        end_time = kwargs.get("end_time")
        if end_time:
            params["end_time"] = end_time
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/traces",
            params=params,
        )
        
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
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/traces/{trace_id}",
        )
        
        return ToolResult.from_data({
            "trace": response,
        })
    
    # Evaluation operations
    def _list_evaluations(
        self,
        **kwargs,
    ) -> ToolResult:
        """List evaluations."""
        limit = kwargs.get("limit", 100)
        
        params = {
            "entity": self.entity,
            "project": self.project,
            "limit": min(limit, 1000),
        }
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/evaluations",
            params=params,
        )
        
        evaluations = response if isinstance(response, list) else response.get("evaluations", [])
        
        return ToolResult.from_data({
            "evaluations": evaluations,
            "count": len(evaluations),
        })
    
    def _get_evaluation(
        self,
        evaluation_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get evaluation details."""
        if not evaluation_id:
            return ToolResult.from_error("evaluation_id is required")
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/evaluations/{evaluation_id}",
        )
        
        return ToolResult.from_data({
            "evaluation": response,
        })
    
    def _create_evaluation(
        self,
        **kwargs,
    ) -> ToolResult:
        """Create an evaluation."""
        name = kwargs.get("name")
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {
            "name": name,
        }
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        dataset_id = kwargs.get("dataset_id")
        if dataset_id:
            data["dataset_id"] = dataset_id
        
        model_id = kwargs.get("model_id")
        if model_id:
            data["model_id"] = model_id
        
        scorers = kwargs.get("scorers", [])
        if scorers:
            data["scorers"] = scorers
        
        response = self._request(
            "POST",
            f"/{self.entity}/{self.project}/evaluations",
            data=data,
        )
        
        return ToolResult.from_data({
            "evaluation_id": response.get("id"),
            "created": True,
        })
    
    def _log_evaluation_result(
        self,
        evaluation_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log an evaluation result."""
        if not evaluation_id:
            return ToolResult.from_error("evaluation_id is required")
        
        data = {
            "evaluation_id": evaluation_id,
        }
        
        example_id = kwargs.get("example_id")
        if example_id:
            data["example_id"] = example_id
        
        prediction = kwargs.get("prediction")
        if prediction:
            data["prediction"] = prediction
        
        scores = kwargs.get("scores", {})
        if scores:
            data["scores"] = scores
        
        latency_ms = kwargs.get("latency_ms")
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        
        response = self._request(
            "POST",
            f"/{self.entity}/{self.project}/evaluations/{evaluation_id}/results",
            data=data,
        )
        
        return ToolResult.from_data({
            "result_id": response.get("id"),
            "logged": True,
        })
    
    # Dataset operations
    def _list_datasets(
        self,
        **kwargs,
    ) -> ToolResult:
        """List datasets."""
        limit = kwargs.get("limit", 100)
        
        params = {
            "entity": self.entity,
            "project": self.project,
            "limit": min(limit, 1000),
        }
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/datasets",
            params=params,
        )
        
        datasets = response if isinstance(response, list) else response.get("datasets", [])
        
        return ToolResult.from_data({
            "datasets": datasets,
            "count": len(datasets),
        })
    
    def _get_dataset(
        self,
        dataset_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get dataset details."""
        if not dataset_id:
            return ToolResult.from_error("dataset_id is required")
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/datasets/{dataset_id}",
        )
        
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
        
        data = {
            "name": name,
        }
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        rows = kwargs.get("rows", [])
        if rows:
            data["rows"] = rows
        
        response = self._request(
            "POST",
            f"/{self.entity}/{self.project}/datasets",
            data=data,
        )
        
        return ToolResult.from_data({
            "dataset_id": response.get("id"),
            "created": True,
        })
    
    def _add_dataset_rows(
        self,
        dataset_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add rows to a dataset."""
        if not dataset_id:
            return ToolResult.from_error("dataset_id is required")
        
        rows = kwargs.get("rows", [])
        if not rows:
            return ToolResult.from_error("rows is required")
        
        data = {"rows": rows}
        
        response = self._request(
            "POST",
            f"/{self.entity}/{self.project}/datasets/{dataset_id}/rows",
            data=data,
        )
        
        return ToolResult.from_data({
            "dataset_id": dataset_id,
            "rows_added": len(rows),
        })
    
    # Model operations
    def _list_models(
        self,
        **kwargs,
    ) -> ToolResult:
        """List models."""
        limit = kwargs.get("limit", 100)
        
        params = {
            "entity": self.entity,
            "project": self.project,
            "limit": min(limit, 1000),
        }
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/models",
            params=params,
        )
        
        models = response if isinstance(response, list) else response.get("models", [])
        
        return ToolResult.from_data({
            "models": models,
            "count": len(models),
        })
    
    def _get_model(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get model details."""
        model_id = kwargs.get("model_id")
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/models/{model_id}",
        )
        
        return ToolResult.from_data({
            "model": response,
        })
    
    def _log_model(
        self,
        **kwargs,
    ) -> ToolResult:
        """Log a model."""
        name = kwargs.get("name")
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {
            "name": name,
        }
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        model_type = kwargs.get("model_type")
        if model_type:
            data["model_type"] = model_type
        
        metadata = kwargs.get("metadata")
        if metadata:
            data["metadata"] = metadata
        
        response = self._request(
            "POST",
            f"/{self.entity}/{self.project}/models",
            data=data,
        )
        
        return ToolResult.from_data({
            "model_id": response.get("id"),
            "logged": True,
        })
    
    # Ops operations
    def _list_ops(
        self,
        **kwargs,
    ) -> ToolResult:
        """List ops (traced functions)."""
        limit = kwargs.get("limit", 100)
        
        params = {
            "entity": self.entity,
            "project": self.project,
            "limit": min(limit, 1000),
        }
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/ops",
            params=params,
        )
        
        ops = response if isinstance(response, list) else response.get("ops", [])
        
        return ToolResult.from_data({
            "ops": ops,
            "count": len(ops),
        })
    
    def _get_op(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get op (traced function) details."""
        op_id = kwargs.get("op_id")
        op_name = kwargs.get("op_name")
        
        if not op_id and not op_name:
            return ToolResult.from_error("op_id or op_name is required")
        
        endpoint = f"/{self.entity}/{self.project}/ops/"
        endpoint += op_id if op_id else f"by-name/{op_name}"
        
        response = self._request("GET", endpoint)
        
        return ToolResult.from_data({
            "op": response,
        })
    
    # Analytics
    def _get_call_stats(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get call statistics."""
        params = {
            "entity": self.entity,
            "project": self.project,
        }
        
        start_time = kwargs.get("start_time")
        if start_time:
            params["start_time"] = start_time
        
        end_time = kwargs.get("end_time")
        if end_time:
            params["end_time"] = end_time
        
        op_name = kwargs.get("op_name")
        if op_name:
            params["op_name"] = op_name
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/analytics/calls",
            params=params,
        )
        
        return ToolResult.from_data({
            "stats": response,
        })
    
    def _get_cost_summary(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get cost summary."""
        params = {
            "entity": self.entity,
            "project": self.project,
        }
        
        start_time = kwargs.get("start_time")
        if start_time:
            params["start_time"] = start_time
        
        end_time = kwargs.get("end_time")
        if end_time:
            params["end_time"] = end_time
        
        group_by = kwargs.get("group_by", "model")
        params["group_by"] = group_by
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/analytics/cost",
            params=params,
        )
        
        return ToolResult.from_data({
            "cost_summary": response,
        })
    
    def _get_latency_stats(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get latency statistics."""
        params = {
            "entity": self.entity,
            "project": self.project,
        }
        
        start_time = kwargs.get("start_time")
        if start_time:
            params["start_time"] = start_time
        
        end_time = kwargs.get("end_time")
        if end_time:
            params["end_time"] = end_time
        
        op_name = kwargs.get("op_name")
        if op_name:
            params["op_name"] = op_name
        
        percentiles = kwargs.get("percentiles", [50, 90, 95, 99])
        params["percentiles"] = ",".join(map(str, percentiles))
        
        response = self._request(
            "GET",
            f"/{self.entity}/{self.project}/analytics/latency",
            params=params,
        )
        
        return ToolResult.from_data({
            "latency_stats": response,
        })
    
    # Project operations
    def _list_projects(
        self,
        **kwargs,
    ) -> ToolResult:
        """List projects."""
        if not self.entity:
            return ToolResult.from_error("entity is required")
        
        query = """
        query ListProjects($entity: String!) {
            entity(name: $entity) {
                projects(first: 100) {
                    edges {
                        node {
                            id
                            name
                            description
                            createdAt
                            updatedAt
                        }
                    }
                }
            }
        }
        """
        
        result = self._graphql(query, {"entity": self.entity})
        
        entity = result.get("entity", {})
        edges = entity.get("projects", {}).get("edges", [])
        projects = [edge.get("node", {}) for edge in edges]
        
        return ToolResult.from_data({
            "projects": projects,
            "count": len(projects),
        })
    
    def _get_project(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get project details."""
        project = kwargs.get("project", self.project)
        
        if not self.entity:
            return ToolResult.from_error("entity is required")
        
        query = """
        query GetProject($entity: String!, $project: String!) {
            project(entityName: $entity, name: $project) {
                id
                name
                description
                createdAt
                updatedAt
                runCount
            }
        }
        """
        
        result = self._graphql(query, {"entity": self.entity, "project": project})
        
        return ToolResult.from_data({
            "project": result.get("project"),
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
                            "list_calls", "get_call", "log_call",
                            "list_traces", "get_trace",
                            "list_evaluations", "get_evaluation",
                            "create_evaluation", "log_evaluation_result",
                            "list_datasets", "get_dataset",
                            "create_dataset", "add_dataset_rows",
                            "list_models", "get_model", "log_model",
                            "list_ops", "get_op",
                            "get_call_stats", "get_cost_summary", "get_latency_stats",
                            "list_projects", "get_project",
                        ],
                    },
                    "call_id": {"type": "string"},
                    "trace_id": {"type": "string"},
                    "evaluation_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                },
                "required": ["action"],
            },
        }
