"""
Arize Tool for ARGUS.

ML observability and model monitoring platform.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class ArizeTool(BaseTool):
    """
    Arize - ML observability and model monitoring.
    
    Features:
    - Model performance monitoring
    - Data drift detection
    - Prediction logging
    - Feature importance analysis
    - Alert management
    
    Example:
        >>> tool = ArizeTool()
        >>> result = tool(action="log_prediction", model_id="my-model", prediction_id="123")
        >>> result = tool(action="get_model_metrics", model_id="my-model")
    """
    
    name = "arize"
    description = "ML observability and model monitoring"
    category = ToolCategory.OBSERVABILITY
    version = "1.0.0"
    
    API_BASE = "https://api.arize.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        space_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("ARIZE_API_KEY")
        self.space_key = space_key or os.getenv("ARIZE_SPACE_KEY")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Arize API key provided")
        
        logger.debug("Arize tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Make API request to Arize."""
        session = self._get_session()
        url = f"{self.API_BASE}{endpoint}"
        
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
        
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_models",
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        prediction_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Arize operations."""
        actions = {
            # Model operations
            "list_models": self._list_models,
            "get_model": self._get_model,
            "get_model_metrics": self._get_model_metrics,
            "get_model_performance": self._get_model_performance,
            
            # Prediction logging
            "log_prediction": self._log_prediction,
            "log_actual": self._log_actual,
            "log_batch": self._log_batch,
            
            # Drift detection
            "get_drift_metrics": self._get_drift_metrics,
            "get_feature_drift": self._get_feature_drift,
            
            # Data quality
            "get_data_quality": self._get_data_quality,
            "get_missing_values": self._get_missing_values,
            
            # Alerts
            "list_alerts": self._list_alerts,
            "create_alert": self._create_alert,
            "delete_alert": self._delete_alert,
            
            # Monitors
            "list_monitors": self._list_monitors,
            "get_monitor": self._get_monitor,
            
            # Feature importance
            "get_feature_importance": self._get_feature_importance,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("Arize API key not configured")
            
            return actions[action](
                model_id=model_id,
                model_version=model_version,
                prediction_id=prediction_id,
                start_time=start_time,
                end_time=end_time,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Arize error: {e}")
            return ToolResult.from_error(f"Arize error: {e}")
    
    # Model operations
    def _list_models(self, **kwargs) -> ToolResult:
        """List models."""
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        
        response = self._request("GET", "/models", params=params if params else None)
        
        models = response.get("models", [])
        
        return ToolResult.from_data({
            "models": models,
            "count": len(models),
        })
    
    def _get_model(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get model details."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        
        response = self._request(
            "GET",
            f"/models/{model_id}",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "model": response,
        })
    
    def _get_model_metrics(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get model metrics."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        if model_version:
            params["model_version"] = model_version
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = self._request(
            "GET",
            f"/models/{model_id}/metrics",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "model_id": model_id,
            "metrics": response.get("metrics", {}),
        })
    
    def _get_model_performance(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get model performance metrics."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        if model_version:
            params["model_version"] = model_version
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        metric_type = kwargs.get("metric_type", "accuracy")
        params["metric_type"] = metric_type
        
        response = self._request(
            "GET",
            f"/models/{model_id}/performance",
            params=params,
        )
        
        return ToolResult.from_data({
            "model_id": model_id,
            "performance": response.get("performance", {}),
            "metric_type": metric_type,
        })
    
    # Prediction logging
    def _log_prediction(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        prediction_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log a prediction."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        if not prediction_id:
            return ToolResult.from_error("prediction_id is required")
        
        features = kwargs.get("features", {})
        prediction = kwargs.get("prediction")
        prediction_label = kwargs.get("prediction_label")
        prediction_score = kwargs.get("prediction_score")
        
        if prediction is None and prediction_label is None:
            return ToolResult.from_error("prediction or prediction_label is required")
        
        data = {
            "model_id": model_id,
            "prediction_id": prediction_id,
            "features": features,
            "timestamp": kwargs.get("timestamp", datetime.utcnow().isoformat()),
        }
        
        if model_version:
            data["model_version"] = model_version
        if prediction is not None:
            data["prediction"] = prediction
        if prediction_label is not None:
            data["prediction_label"] = prediction_label
        if prediction_score is not None:
            data["prediction_score"] = prediction_score
        
        if self.space_key:
            data["space_key"] = self.space_key
        
        response = self._request("POST", "/predictions", data=data)
        
        return ToolResult.from_data({
            "prediction_id": prediction_id,
            "logged": True,
            "response": response,
        })
    
    def _log_actual(
        self,
        model_id: Optional[str] = None,
        prediction_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log actual/ground truth value."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        if not prediction_id:
            return ToolResult.from_error("prediction_id is required")
        
        actual = kwargs.get("actual")
        actual_label = kwargs.get("actual_label")
        
        if actual is None and actual_label is None:
            return ToolResult.from_error("actual or actual_label is required")
        
        data = {
            "model_id": model_id,
            "prediction_id": prediction_id,
        }
        
        if actual is not None:
            data["actual"] = actual
        if actual_label is not None:
            data["actual_label"] = actual_label
        
        if self.space_key:
            data["space_key"] = self.space_key
        
        response = self._request("POST", "/actuals", data=data)
        
        return ToolResult.from_data({
            "prediction_id": prediction_id,
            "actual_logged": True,
            "response": response,
        })
    
    def _log_batch(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Log batch of predictions."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        predictions = kwargs.get("predictions", [])
        if not predictions:
            return ToolResult.from_error("predictions list is required")
        
        data = {
            "model_id": model_id,
            "predictions": predictions,
        }
        
        if self.space_key:
            data["space_key"] = self.space_key
        
        response = self._request("POST", "/predictions/batch", data=data)
        
        return ToolResult.from_data({
            "batch_size": len(predictions),
            "logged": True,
            "response": response,
        })
    
    # Drift detection
    def _get_drift_metrics(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get drift metrics."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = self._request(
            "GET",
            f"/models/{model_id}/drift",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "model_id": model_id,
            "drift_metrics": response.get("drift", {}),
        })
    
    def _get_feature_drift(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get per-feature drift metrics."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        feature_name = kwargs.get("feature_name")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        if feature_name:
            params["feature_name"] = feature_name
        
        response = self._request(
            "GET",
            f"/models/{model_id}/drift/features",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "model_id": model_id,
            "feature_drift": response.get("features", []),
        })
    
    # Data quality
    def _get_data_quality(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get data quality metrics."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        
        response = self._request(
            "GET",
            f"/models/{model_id}/data-quality",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "model_id": model_id,
            "data_quality": response,
        })
    
    def _get_missing_values(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get missing value statistics."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        
        response = self._request(
            "GET",
            f"/models/{model_id}/missing-values",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "model_id": model_id,
            "missing_values": response.get("missing", {}),
        })
    
    # Alerts
    def _list_alerts(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List alerts."""
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        if model_id:
            params["model_id"] = model_id
        
        response = self._request("GET", "/alerts", params=params if params else None)
        
        return ToolResult.from_data({
            "alerts": response.get("alerts", []),
            "count": len(response.get("alerts", [])),
        })
    
    def _create_alert(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create an alert."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        alert_name = kwargs.get("name")
        if not alert_name:
            return ToolResult.from_error("name is required")
        
        metric = kwargs.get("metric")
        threshold = kwargs.get("threshold")
        
        if not metric or threshold is None:
            return ToolResult.from_error("metric and threshold are required")
        
        data = {
            "model_id": model_id,
            "name": alert_name,
            "metric": metric,
            "threshold": threshold,
            "comparison": kwargs.get("comparison", "greater_than"),
        }
        
        if self.space_key:
            data["space_key"] = self.space_key
        
        response = self._request("POST", "/alerts", data=data)
        
        return ToolResult.from_data({
            "alert_id": response.get("id"),
            "created": True,
        })
    
    def _delete_alert(
        self,
        **kwargs,
    ) -> ToolResult:
        """Delete an alert."""
        alert_id = kwargs.get("alert_id")
        if not alert_id:
            return ToolResult.from_error("alert_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        
        self._request("DELETE", f"/alerts/{alert_id}", params=params if params else None)
        
        return ToolResult.from_data({
            "alert_id": alert_id,
            "deleted": True,
        })
    
    # Monitors
    def _list_monitors(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List monitors."""
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        if model_id:
            params["model_id"] = model_id
        
        response = self._request("GET", "/monitors", params=params if params else None)
        
        return ToolResult.from_data({
            "monitors": response.get("monitors", []),
            "count": len(response.get("monitors", [])),
        })
    
    def _get_monitor(
        self,
        **kwargs,
    ) -> ToolResult:
        """Get monitor details."""
        monitor_id = kwargs.get("monitor_id")
        if not monitor_id:
            return ToolResult.from_error("monitor_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        
        response = self._request(
            "GET",
            f"/monitors/{monitor_id}",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "monitor": response,
        })
    
    # Feature importance
    def _get_feature_importance(
        self,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get feature importance."""
        if not model_id:
            return ToolResult.from_error("model_id is required")
        
        params = {}
        if self.space_key:
            params["space_key"] = self.space_key
        
        response = self._request(
            "GET",
            f"/models/{model_id}/feature-importance",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "model_id": model_id,
            "feature_importance": response.get("features", []),
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
                            "list_models", "get_model", "get_model_metrics", "get_model_performance",
                            "log_prediction", "log_actual", "log_batch",
                            "get_drift_metrics", "get_feature_drift",
                            "get_data_quality", "get_missing_values",
                            "list_alerts", "create_alert", "delete_alert",
                            "list_monitors", "get_monitor",
                            "get_feature_importance",
                        ],
                    },
                    "model_id": {"type": "string"},
                    "model_version": {"type": "string"},
                    "prediction_id": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                },
                "required": ["action"],
            },
        }
