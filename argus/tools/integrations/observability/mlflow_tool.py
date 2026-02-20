"""
MLflow Tool for ARGUS.

ML experiment tracking and model registry.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class MLflowTool(BaseTool):
    """
    MLflow - ML lifecycle management.
    
    Features:
    - Experiment tracking
    - Model registry
    - Model versioning
    - Run management
    - Artifact storage
    - Metrics logging
    
    Example:
        >>> tool = MLflowTool()
        >>> result = tool(action="list_experiments")
        >>> result = tool(action="create_run", experiment_id="1")
    """
    
    name = "mlflow"
    description = "ML experiment tracking and model registry"
    category = ToolCategory.OBSERVABILITY
    version = "1.0.0"
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.tracking_uri = (tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")).rstrip("/")
        self.api_key = api_key or os.getenv("MLFLOW_API_KEY")
        
        self._session = None
        
        logger.debug(f"MLflow tool initialized (uri={self.tracking_uri})")
    
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
        """Make API request to MLflow."""
        session = self._get_session()
        url = f"{self.tracking_uri}/api/2.0/mlflow{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            json=data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("message", error_data.get("error_code", f"HTTP {response.status_code}"))
                raise Exception(str(message))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        if response.status_code == 204:
            return {}
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_experiments",
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute MLflow operations."""
        actions = {
            # Experiment operations
            "list_experiments": self._list_experiments,
            "get_experiment": self._get_experiment,
            "create_experiment": self._create_experiment,
            "delete_experiment": self._delete_experiment,
            "update_experiment": self._update_experiment,
            "search_experiments": self._search_experiments,
            
            # Run operations
            "list_runs": self._list_runs,
            "get_run": self._get_run,
            "create_run": self._create_run,
            "update_run": self._update_run,
            "delete_run": self._delete_run,
            "search_runs": self._search_runs,
            
            # Metrics and params
            "log_metric": self._log_metric,
            "log_metrics": self._log_metrics,
            "log_param": self._log_param,
            "log_params": self._log_params,
            "get_metric_history": self._get_metric_history,
            
            # Tags
            "set_tag": self._set_tag,
            "delete_tag": self._delete_tag,
            
            # Model registry
            "list_registered_models": self._list_registered_models,
            "get_registered_model": self._get_registered_model,
            "create_registered_model": self._create_registered_model,
            "delete_registered_model": self._delete_registered_model,
            "update_registered_model": self._update_registered_model,
            "search_registered_models": self._search_registered_models,
            
            # Model versions
            "list_model_versions": self._list_model_versions,
            "get_model_version": self._get_model_version,
            "create_model_version": self._create_model_version,
            "update_model_version": self._update_model_version,
            "delete_model_version": self._delete_model_version,
            "transition_model_stage": self._transition_model_stage,
            
            # Artifacts
            "list_artifacts": self._list_artifacts,
            "log_artifact": self._log_artifact,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                experiment_id=experiment_id,
                run_id=run_id,
                model_name=model_name,
                version=version,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"MLflow error: {e}")
            return ToolResult.from_error(f"MLflow error: {e}")
    
    # Experiment operations
    def _list_experiments(
        self,
        **kwargs,
    ) -> ToolResult:
        """List experiments."""
        max_results = kwargs.get("max_results", 100)
        
        params = {"max_results": min(max_results, 1000)}
        
        view_type = kwargs.get("view_type", "ACTIVE_ONLY")
        params["view_type"] = view_type
        
        response = self._request("GET", "/experiments/list", params=params)
        
        experiments = response.get("experiments", [])
        
        return ToolResult.from_data({
            "experiments": experiments,
            "count": len(experiments),
        })
    
    def _get_experiment(
        self,
        experiment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get experiment details."""
        if not experiment_id:
            name = kwargs.get("name")
            if name:
                response = self._request(
                    "GET",
                    "/experiments/get-by-name",
                    params={"experiment_name": name},
                )
            else:
                return ToolResult.from_error("experiment_id or name is required")
        else:
            response = self._request(
                "GET",
                "/experiments/get",
                params={"experiment_id": experiment_id},
            )
        
        return ToolResult.from_data({
            "experiment": response.get("experiment"),
        })
    
    def _create_experiment(
        self,
        **kwargs,
    ) -> ToolResult:
        """Create an experiment."""
        name = kwargs.get("name")
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {"name": name}
        
        artifact_location = kwargs.get("artifact_location")
        if artifact_location:
            data["artifact_location"] = artifact_location
        
        tags = kwargs.get("tags", [])
        if tags:
            data["tags"] = tags
        
        response = self._request("POST", "/experiments/create", data=data)
        
        return ToolResult.from_data({
            "experiment_id": response.get("experiment_id"),
            "created": True,
        })
    
    def _delete_experiment(
        self,
        experiment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete an experiment."""
        if not experiment_id:
            return ToolResult.from_error("experiment_id is required")
        
        self._request("POST", "/experiments/delete", data={"experiment_id": experiment_id})
        
        return ToolResult.from_data({
            "experiment_id": experiment_id,
            "deleted": True,
        })
    
    def _update_experiment(
        self,
        experiment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update an experiment."""
        if not experiment_id:
            return ToolResult.from_error("experiment_id is required")
        
        data = {"experiment_id": experiment_id}
        
        new_name = kwargs.get("new_name")
        if new_name:
            data["new_name"] = new_name
        
        self._request("POST", "/experiments/update", data=data)
        
        return ToolResult.from_data({
            "experiment_id": experiment_id,
            "updated": True,
        })
    
    def _search_experiments(
        self,
        **kwargs,
    ) -> ToolResult:
        """Search experiments."""
        max_results = kwargs.get("max_results", 100)
        
        data = {"max_results": min(max_results, 1000)}
        
        filter_string = kwargs.get("filter_string")
        if filter_string:
            data["filter"] = filter_string
        
        order_by = kwargs.get("order_by", [])
        if order_by:
            data["order_by"] = order_by
        
        view_type = kwargs.get("view_type", "ACTIVE_ONLY")
        data["view_type"] = view_type
        
        response = self._request("POST", "/experiments/search", data=data)
        
        experiments = response.get("experiments", [])
        
        return ToolResult.from_data({
            "experiments": experiments,
            "count": len(experiments),
            "next_page_token": response.get("next_page_token"),
        })
    
    # Run operations
    def _list_runs(
        self,
        experiment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List runs."""
        experiment_ids = kwargs.get("experiment_ids", [])
        if experiment_id:
            experiment_ids.append(experiment_id)
        
        if not experiment_ids:
            return ToolResult.from_error("experiment_id or experiment_ids is required")
        
        max_results = kwargs.get("max_results", 100)
        
        data = {
            "experiment_ids": experiment_ids,
            "max_results": min(max_results, 1000),
        }
        
        response = self._request("POST", "/runs/search", data=data)
        
        runs = response.get("runs", [])
        
        return ToolResult.from_data({
            "runs": runs,
            "count": len(runs),
        })
    
    def _get_run(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get run details."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        response = self._request("GET", "/runs/get", params={"run_id": run_id})
        
        return ToolResult.from_data({
            "run": response.get("run"),
        })
    
    def _create_run(
        self,
        experiment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a run."""
        if not experiment_id:
            return ToolResult.from_error("experiment_id is required")
        
        data = {"experiment_id": experiment_id}
        
        run_name = kwargs.get("run_name")
        if run_name:
            data["run_name"] = run_name
        
        start_time = kwargs.get("start_time")
        if start_time:
            data["start_time"] = start_time
        
        tags = kwargs.get("tags", [])
        if tags:
            data["tags"] = tags
        
        response = self._request("POST", "/runs/create", data=data)
        
        run = response.get("run", {})
        
        return ToolResult.from_data({
            "run_id": run.get("info", {}).get("run_id"),
            "run": run,
            "created": True,
        })
    
    def _update_run(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a run."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        data = {"run_id": run_id}
        
        status = kwargs.get("status")
        if status:
            data["status"] = status
        
        end_time = kwargs.get("end_time")
        if end_time:
            data["end_time"] = end_time
        
        run_name = kwargs.get("run_name")
        if run_name:
            data["run_name"] = run_name
        
        self._request("POST", "/runs/update", data=data)
        
        return ToolResult.from_data({
            "run_id": run_id,
            "updated": True,
        })
    
    def _delete_run(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a run."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        self._request("POST", "/runs/delete", data={"run_id": run_id})
        
        return ToolResult.from_data({
            "run_id": run_id,
            "deleted": True,
        })
    
    def _search_runs(
        self,
        experiment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Search runs."""
        experiment_ids = kwargs.get("experiment_ids", [])
        if experiment_id:
            experiment_ids.append(experiment_id)
        
        if not experiment_ids:
            return ToolResult.from_error("experiment_id or experiment_ids is required")
        
        max_results = kwargs.get("max_results", 100)
        
        data = {
            "experiment_ids": experiment_ids,
            "max_results": min(max_results, 1000),
        }
        
        filter_string = kwargs.get("filter_string")
        if filter_string:
            data["filter"] = filter_string
        
        run_view_type = kwargs.get("run_view_type", "ACTIVE_ONLY")
        data["run_view_type"] = run_view_type
        
        order_by = kwargs.get("order_by", [])
        if order_by:
            data["order_by"] = order_by
        
        response = self._request("POST", "/runs/search", data=data)
        
        runs = response.get("runs", [])
        
        return ToolResult.from_data({
            "runs": runs,
            "count": len(runs),
            "next_page_token": response.get("next_page_token"),
        })
    
    # Metrics and params
    def _log_metric(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log a metric."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        key = kwargs.get("key")
        value = kwargs.get("value")
        
        if not key:
            return ToolResult.from_error("key is required")
        if value is None:
            return ToolResult.from_error("value is required")
        
        import time
        timestamp = kwargs.get("timestamp", int(time.time() * 1000))
        step = kwargs.get("step", 0)
        
        data = {
            "run_id": run_id,
            "key": key,
            "value": value,
            "timestamp": timestamp,
            "step": step,
        }
        
        self._request("POST", "/runs/log-metric", data=data)
        
        return ToolResult.from_data({
            "run_id": run_id,
            "key": key,
            "logged": True,
        })
    
    def _log_metrics(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log multiple metrics."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        metrics = kwargs.get("metrics", [])
        if not metrics:
            return ToolResult.from_error("metrics is required")
        
        import time
        timestamp = kwargs.get("timestamp", int(time.time() * 1000))
        
        data = {
            "run_id": run_id,
            "metrics": [
                {
                    "key": m.get("key"),
                    "value": m.get("value"),
                    "timestamp": m.get("timestamp", timestamp),
                    "step": m.get("step", 0),
                }
                for m in metrics
            ],
        }
        
        self._request("POST", "/runs/log-batch", data=data)
        
        return ToolResult.from_data({
            "run_id": run_id,
            "metrics_logged": len(metrics),
        })
    
    def _log_param(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log a parameter."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        key = kwargs.get("key")
        value = kwargs.get("value")
        
        if not key:
            return ToolResult.from_error("key is required")
        if value is None:
            return ToolResult.from_error("value is required")
        
        data = {
            "run_id": run_id,
            "key": key,
            "value": str(value),
        }
        
        self._request("POST", "/runs/log-parameter", data=data)
        
        return ToolResult.from_data({
            "run_id": run_id,
            "key": key,
            "logged": True,
        })
    
    def _log_params(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log multiple parameters."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        params = kwargs.get("params", [])
        if not params:
            return ToolResult.from_error("params is required")
        
        data = {
            "run_id": run_id,
            "params": [
                {"key": p.get("key"), "value": str(p.get("value"))}
                for p in params
            ],
        }
        
        self._request("POST", "/runs/log-batch", data=data)
        
        return ToolResult.from_data({
            "run_id": run_id,
            "params_logged": len(params),
        })
    
    def _get_metric_history(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get metric history."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        metric_key = kwargs.get("metric_key")
        if not metric_key:
            return ToolResult.from_error("metric_key is required")
        
        params = {
            "run_id": run_id,
            "metric_key": metric_key,
        }
        
        response = self._request("GET", "/metrics/get-history", params=params)
        
        return ToolResult.from_data({
            "metrics": response.get("metrics", []),
        })
    
    # Tags
    def _set_tag(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Set a tag on a run."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        key = kwargs.get("key")
        value = kwargs.get("value")
        
        if not key:
            return ToolResult.from_error("key is required")
        if value is None:
            return ToolResult.from_error("value is required")
        
        data = {
            "run_id": run_id,
            "key": key,
            "value": str(value),
        }
        
        self._request("POST", "/runs/set-tag", data=data)
        
        return ToolResult.from_data({
            "run_id": run_id,
            "key": key,
            "set": True,
        })
    
    def _delete_tag(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a tag from a run."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        key = kwargs.get("key")
        if not key:
            return ToolResult.from_error("key is required")
        
        data = {
            "run_id": run_id,
            "key": key,
        }
        
        self._request("POST", "/runs/delete-tag", data=data)
        
        return ToolResult.from_data({
            "run_id": run_id,
            "key": key,
            "deleted": True,
        })
    
    # Model registry
    def _list_registered_models(
        self,
        **kwargs,
    ) -> ToolResult:
        """List registered models."""
        max_results = kwargs.get("max_results", 100)
        
        params = {"max_results": min(max_results, 1000)}
        
        response = self._request("GET", "/registered-models/list", params=params)
        
        models = response.get("registered_models", [])
        
        return ToolResult.from_data({
            "models": models,
            "count": len(models),
            "next_page_token": response.get("next_page_token"),
        })
    
    def _get_registered_model(
        self,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get registered model details."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        
        response = self._request(
            "GET",
            "/registered-models/get",
            params={"name": model_name},
        )
        
        return ToolResult.from_data({
            "model": response.get("registered_model"),
        })
    
    def _create_registered_model(
        self,
        **kwargs,
    ) -> ToolResult:
        """Create a registered model."""
        name = kwargs.get("name")
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {"name": name}
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        tags = kwargs.get("tags", [])
        if tags:
            data["tags"] = tags
        
        response = self._request("POST", "/registered-models/create", data=data)
        
        return ToolResult.from_data({
            "model": response.get("registered_model"),
            "created": True,
        })
    
    def _delete_registered_model(
        self,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a registered model."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        
        self._request(
            "DELETE",
            "/registered-models/delete",
            data={"name": model_name},
        )
        
        return ToolResult.from_data({
            "model_name": model_name,
            "deleted": True,
        })
    
    def _update_registered_model(
        self,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a registered model."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        
        data = {"name": model_name}
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        self._request("PATCH", "/registered-models/update", data=data)
        
        return ToolResult.from_data({
            "model_name": model_name,
            "updated": True,
        })
    
    def _search_registered_models(
        self,
        **kwargs,
    ) -> ToolResult:
        """Search registered models."""
        max_results = kwargs.get("max_results", 100)
        
        data = {"max_results": min(max_results, 1000)}
        
        filter_string = kwargs.get("filter_string")
        if filter_string:
            data["filter"] = filter_string
        
        order_by = kwargs.get("order_by", [])
        if order_by:
            data["order_by"] = order_by
        
        response = self._request("POST", "/registered-models/search", data=data)
        
        models = response.get("registered_models", [])
        
        return ToolResult.from_data({
            "models": models,
            "count": len(models),
            "next_page_token": response.get("next_page_token"),
        })
    
    # Model versions
    def _list_model_versions(
        self,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List model versions."""
        max_results = kwargs.get("max_results", 100)
        
        params = {"max_results": min(max_results, 1000)}
        
        filter_string = kwargs.get("filter_string")
        if not filter_string and model_name:
            filter_string = f"name='{model_name}'"
        
        if filter_string:
            params["filter"] = filter_string
        
        response = self._request("GET", "/model-versions/search", params=params)
        
        versions = response.get("model_versions", [])
        
        return ToolResult.from_data({
            "versions": versions,
            "count": len(versions),
            "next_page_token": response.get("next_page_token"),
        })
    
    def _get_model_version(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get model version details."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        if not version:
            return ToolResult.from_error("version is required")
        
        response = self._request(
            "GET",
            "/model-versions/get",
            params={"name": model_name, "version": version},
        )
        
        return ToolResult.from_data({
            "model_version": response.get("model_version"),
        })
    
    def _create_model_version(
        self,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a model version."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        
        source = kwargs.get("source")
        if not source:
            return ToolResult.from_error("source is required")
        
        data = {
            "name": model_name,
            "source": source,
        }
        
        run_id = kwargs.get("run_id")
        if run_id:
            data["run_id"] = run_id
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        tags = kwargs.get("tags", [])
        if tags:
            data["tags"] = tags
        
        response = self._request("POST", "/model-versions/create", data=data)
        
        return ToolResult.from_data({
            "model_version": response.get("model_version"),
            "created": True,
        })
    
    def _update_model_version(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a model version."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        if not version:
            return ToolResult.from_error("version is required")
        
        data = {
            "name": model_name,
            "version": version,
        }
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        self._request("PATCH", "/model-versions/update", data=data)
        
        return ToolResult.from_data({
            "model_name": model_name,
            "version": version,
            "updated": True,
        })
    
    def _delete_model_version(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a model version."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        if not version:
            return ToolResult.from_error("version is required")
        
        self._request(
            "DELETE",
            "/model-versions/delete",
            data={"name": model_name, "version": version},
        )
        
        return ToolResult.from_data({
            "model_name": model_name,
            "version": version,
            "deleted": True,
        })
    
    def _transition_model_stage(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Transition model version to a stage."""
        if not model_name:
            return ToolResult.from_error("model_name is required")
        if not version:
            return ToolResult.from_error("version is required")
        
        stage = kwargs.get("stage")
        if not stage:
            return ToolResult.from_error("stage is required (None, Staging, Production, Archived)")
        
        data = {
            "name": model_name,
            "version": version,
            "stage": stage,
        }
        
        archive_existing = kwargs.get("archive_existing_versions", False)
        data["archive_existing_versions"] = archive_existing
        
        response = self._request("POST", "/model-versions/transition-stage", data=data)
        
        return ToolResult.from_data({
            "model_version": response.get("model_version"),
            "transitioned": True,
        })
    
    # Artifacts
    def _list_artifacts(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List artifacts for a run."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        params = {"run_id": run_id}
        
        path = kwargs.get("path")
        if path:
            params["path"] = path
        
        response = self._request("GET", "/artifacts/list", params=params)
        
        return ToolResult.from_data({
            "root_uri": response.get("root_uri"),
            "files": response.get("files", []),
        })
    
    def _log_artifact(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log an artifact for a run."""
        if not run_id:
            return ToolResult.from_error("run_id is required")
        
        local_path = kwargs.get("local_path")
        if not local_path:
            return ToolResult.from_error("local_path is required")
        
        artifact_path = kwargs.get("artifact_path", "")
        
        import os
        if not os.path.exists(local_path):
            return ToolResult.from_error(f"File not found: {local_path}")
        
        with open(local_path, "rb") as f:
            artifact_data = f.read()
        
        session = self._get_session()
        url = f"{self.tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{run_id}/{artifact_path}"
        
        filename = os.path.basename(local_path)
        files = {"file": (filename, artifact_data)}
        
        original_content_type = session.headers.pop("Content-Type", None)
        
        try:
            response = session.post(url, files=files)
            response.raise_for_status()
        finally:
            if original_content_type:
                session.headers["Content-Type"] = original_content_type
        
        return ToolResult.from_data({
            "run_id": run_id,
            "artifact_path": artifact_path,
            "filename": filename,
            "logged": True,
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
                            "list_experiments", "get_experiment", "create_experiment",
                            "delete_experiment", "update_experiment", "search_experiments",
                            "list_runs", "get_run", "create_run", "update_run",
                            "delete_run", "search_runs",
                            "log_metric", "log_metrics", "log_param", "log_params",
                            "get_metric_history",
                            "set_tag", "delete_tag",
                            "list_registered_models", "get_registered_model",
                            "create_registered_model", "delete_registered_model",
                            "update_registered_model", "search_registered_models",
                            "list_model_versions", "get_model_version",
                            "create_model_version", "update_model_version",
                            "delete_model_version", "transition_model_stage",
                            "list_artifacts", "log_artifact",
                        ],
                    },
                    "experiment_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "model_name": {"type": "string"},
                    "version": {"type": "string"},
                },
                "required": ["action"],
            },
        }
