"""
n8n Tool for ARGUS.

Workflow automation platform integration.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class N8nTool(BaseTool):
    """
    n8n - Workflow automation platform.
    
    Features:
    - Workflow management
    - Execution management
    - Credential management
    - Webhook management
    - Tag management
    
    Example:
        >>> tool = N8nTool()
        >>> result = tool(action="list_workflows")
        >>> result = tool(action="execute_workflow", workflow_id="123")
    """
    
    name = "n8n"
    description = "Workflow automation platform"
    category = ToolCategory.AUTOMATION
    version = "1.0.0"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("N8N_API_KEY")
        self.base_url = (base_url or os.getenv("N8N_URL", "http://localhost:5678")).rstrip("/")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No n8n API key provided")
        
        logger.debug(f"n8n tool initialized (url={self.base_url})")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "X-N8N-API-KEY": self.api_key,
                "Content-Type": "application/json",
            })
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict | list:
        """Make API request to n8n."""
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
        action: str = "list_workflows",
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        credential_id: Optional[str] = None,
        tag_id: Optional[str] = None,
        name: Optional[str] = None,
        active: Optional[bool] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute n8n operations."""
        actions = {
            # Workflow operations
            "list_workflows": self._list_workflows,
            "get_workflow": self._get_workflow,
            "create_workflow": self._create_workflow,
            "update_workflow": self._update_workflow,
            "delete_workflow": self._delete_workflow,
            "activate_workflow": self._activate_workflow,
            "deactivate_workflow": self._deactivate_workflow,
            "execute_workflow": self._execute_workflow,
            
            # Execution operations
            "list_executions": self._list_executions,
            "get_execution": self._get_execution,
            "delete_execution": self._delete_execution,
            
            # Credential operations
            "list_credentials": self._list_credentials,
            "get_credential": self._get_credential,
            "create_credential": self._create_credential,
            "delete_credential": self._delete_credential,
            
            # Tag operations
            "list_tags": self._list_tags,
            "get_tag": self._get_tag,
            "create_tag": self._create_tag,
            "update_tag": self._update_tag,
            "delete_tag": self._delete_tag,
            
            # Webhook testing
            "test_webhook": self._test_webhook,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("n8n API key not configured")
            
            return actions[action](
                workflow_id=workflow_id,
                execution_id=execution_id,
                credential_id=credential_id,
                tag_id=tag_id,
                name=name,
                active=active,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"n8n error: {e}")
            return ToolResult.from_error(f"n8n error: {e}")
    
    # Workflow operations
    def _list_workflows(
        self,
        active: Optional[bool] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List workflows."""
        params = {"limit": min(limit, 250)}
        
        if active is not None:
            params["active"] = str(active).lower()
        
        tags = kwargs.get("tags")
        if tags:
            params["tags"] = ",".join(tags) if isinstance(tags, list) else tags
        
        response = self._request("GET", "/workflows", params=params)
        
        workflows = []
        data = response.get("data", []) if isinstance(response, dict) else response
        
        for w in data:
            workflows.append({
                "id": w.get("id"),
                "name": w.get("name"),
                "active": w.get("active"),
                "createdAt": w.get("createdAt"),
                "updatedAt": w.get("updatedAt"),
                "tags": [t.get("name") for t in w.get("tags", [])],
            })
        
        return ToolResult.from_data({
            "workflows": workflows,
            "count": len(workflows),
        })
    
    def _get_workflow(
        self,
        workflow_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get workflow details."""
        if not workflow_id:
            return ToolResult.from_error("workflow_id is required")
        
        response = self._request("GET", f"/workflows/{workflow_id}")
        
        return ToolResult.from_data({
            "workflow": {
                "id": response.get("id"),
                "name": response.get("name"),
                "active": response.get("active"),
                "nodes": response.get("nodes", []),
                "connections": response.get("connections", {}),
                "settings": response.get("settings", {}),
                "createdAt": response.get("createdAt"),
                "updatedAt": response.get("updatedAt"),
            }
        })
    
    def _create_workflow(
        self,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a workflow."""
        if not name:
            return ToolResult.from_error("name is required")
        
        nodes = kwargs.get("nodes", [
            {
                "id": "start",
                "name": "Start",
                "type": "n8n-nodes-base.start",
                "typeVersion": 1,
                "position": [250, 300],
            }
        ])
        
        connections = kwargs.get("connections", {})
        settings = kwargs.get("settings", {})
        
        data = {
            "name": name,
            "nodes": nodes,
            "connections": connections,
            "settings": settings,
        }
        
        response = self._request("POST", "/workflows", data=data)
        
        return ToolResult.from_data({
            "workflow_id": response.get("id"),
            "name": response.get("name"),
            "created": True,
        })
    
    def _update_workflow(
        self,
        workflow_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a workflow."""
        if not workflow_id:
            return ToolResult.from_error("workflow_id is required")
        
        # Get existing workflow
        existing = self._request("GET", f"/workflows/{workflow_id}")
        
        data = {
            "name": name or existing.get("name"),
            "nodes": kwargs.get("nodes", existing.get("nodes", [])),
            "connections": kwargs.get("connections", existing.get("connections", {})),
            "settings": kwargs.get("settings", existing.get("settings", {})),
        }
        
        response = self._request("PUT", f"/workflows/{workflow_id}", data=data)
        
        return ToolResult.from_data({
            "workflow_id": response.get("id"),
            "updated": True,
        })
    
    def _delete_workflow(
        self,
        workflow_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a workflow."""
        if not workflow_id:
            return ToolResult.from_error("workflow_id is required")
        
        self._request("DELETE", f"/workflows/{workflow_id}")
        
        return ToolResult.from_data({
            "workflow_id": workflow_id,
            "deleted": True,
        })
    
    def _activate_workflow(
        self,
        workflow_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Activate a workflow."""
        if not workflow_id:
            return ToolResult.from_error("workflow_id is required")
        
        response = self._request("POST", f"/workflows/{workflow_id}/activate")
        
        return ToolResult.from_data({
            "workflow_id": response.get("id"),
            "active": response.get("active"),
            "activated": True,
        })
    
    def _deactivate_workflow(
        self,
        workflow_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Deactivate a workflow."""
        if not workflow_id:
            return ToolResult.from_error("workflow_id is required")
        
        response = self._request("POST", f"/workflows/{workflow_id}/deactivate")
        
        return ToolResult.from_data({
            "workflow_id": response.get("id"),
            "active": response.get("active"),
            "deactivated": True,
        })
    
    def _execute_workflow(
        self,
        workflow_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute a workflow."""
        if not workflow_id:
            return ToolResult.from_error("workflow_id is required")
        
        data = {}
        
        input_data = kwargs.get("data")
        if input_data:
            data["data"] = input_data
        
        response = self._request(
            "POST",
            f"/workflows/{workflow_id}/execute",
            data=data if data else None,
        )
        
        return ToolResult.from_data({
            "execution_id": response.get("executionId"),
            "data": response.get("data"),
            "executed": True,
        })
    
    # Execution operations
    def _list_executions(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List executions."""
        params = {"limit": min(limit, 250)}
        
        if workflow_id:
            params["workflowId"] = workflow_id
        
        status = kwargs.get("status")
        if status:
            params["status"] = status
        
        response = self._request("GET", "/executions", params=params)
        
        executions = []
        data = response.get("data", []) if isinstance(response, dict) else response
        
        for e in data:
            executions.append({
                "id": e.get("id"),
                "workflowId": e.get("workflowId"),
                "finished": e.get("finished"),
                "mode": e.get("mode"),
                "status": e.get("status"),
                "startedAt": e.get("startedAt"),
                "stoppedAt": e.get("stoppedAt"),
            })
        
        return ToolResult.from_data({
            "executions": executions,
            "count": len(executions),
        })
    
    def _get_execution(
        self,
        execution_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get execution details."""
        if not execution_id:
            return ToolResult.from_error("execution_id is required")
        
        include_data = kwargs.get("include_data", True)
        params = {"includeData": str(include_data).lower()}
        
        response = self._request("GET", f"/executions/{execution_id}", params=params)
        
        return ToolResult.from_data({
            "execution": {
                "id": response.get("id"),
                "workflowId": response.get("workflowId"),
                "finished": response.get("finished"),
                "mode": response.get("mode"),
                "status": response.get("status"),
                "data": response.get("data"),
                "startedAt": response.get("startedAt"),
                "stoppedAt": response.get("stoppedAt"),
            }
        })
    
    def _delete_execution(
        self,
        execution_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete an execution."""
        if not execution_id:
            return ToolResult.from_error("execution_id is required")
        
        self._request("DELETE", f"/executions/{execution_id}")
        
        return ToolResult.from_data({
            "execution_id": execution_id,
            "deleted": True,
        })
    
    # Credential operations
    def _list_credentials(
        self,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List credentials."""
        params = {"limit": min(limit, 250)}
        
        response = self._request("GET", "/credentials", params=params)
        
        credentials = []
        data = response.get("data", []) if isinstance(response, dict) else response
        
        for c in data:
            credentials.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "type": c.get("type"),
                "createdAt": c.get("createdAt"),
                "updatedAt": c.get("updatedAt"),
            })
        
        return ToolResult.from_data({
            "credentials": credentials,
            "count": len(credentials),
        })
    
    def _get_credential(
        self,
        credential_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get credential details (without sensitive data)."""
        if not credential_id:
            return ToolResult.from_error("credential_id is required")
        
        response = self._request("GET", f"/credentials/{credential_id}")
        
        return ToolResult.from_data({
            "credential": {
                "id": response.get("id"),
                "name": response.get("name"),
                "type": response.get("type"),
                "createdAt": response.get("createdAt"),
                "updatedAt": response.get("updatedAt"),
            }
        })
    
    def _create_credential(
        self,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a credential."""
        if not name:
            return ToolResult.from_error("name is required")
        
        credential_type = kwargs.get("type")
        if not credential_type:
            return ToolResult.from_error("type is required")
        
        credential_data = kwargs.get("data", {})
        
        data = {
            "name": name,
            "type": credential_type,
            "data": credential_data,
        }
        
        response = self._request("POST", "/credentials", data=data)
        
        return ToolResult.from_data({
            "credential_id": response.get("id"),
            "name": response.get("name"),
            "created": True,
        })
    
    def _delete_credential(
        self,
        credential_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a credential."""
        if not credential_id:
            return ToolResult.from_error("credential_id is required")
        
        self._request("DELETE", f"/credentials/{credential_id}")
        
        return ToolResult.from_data({
            "credential_id": credential_id,
            "deleted": True,
        })
    
    # Tag operations
    def _list_tags(
        self,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List tags."""
        params = {"limit": min(limit, 250)}
        
        response = self._request("GET", "/tags", params=params)
        
        tags = []
        data = response.get("data", []) if isinstance(response, dict) else response
        
        for t in data:
            tags.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "createdAt": t.get("createdAt"),
                "updatedAt": t.get("updatedAt"),
            })
        
        return ToolResult.from_data({
            "tags": tags,
            "count": len(tags),
        })
    
    def _get_tag(
        self,
        tag_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get tag details."""
        if not tag_id:
            return ToolResult.from_error("tag_id is required")
        
        response = self._request("GET", f"/tags/{tag_id}")
        
        return ToolResult.from_data({
            "tag": {
                "id": response.get("id"),
                "name": response.get("name"),
                "createdAt": response.get("createdAt"),
                "updatedAt": response.get("updatedAt"),
            }
        })
    
    def _create_tag(
        self,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a tag."""
        if not name:
            return ToolResult.from_error("name is required")
        
        response = self._request("POST", "/tags", data={"name": name})
        
        return ToolResult.from_data({
            "tag_id": response.get("id"),
            "name": response.get("name"),
            "created": True,
        })
    
    def _update_tag(
        self,
        tag_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a tag."""
        if not tag_id:
            return ToolResult.from_error("tag_id is required")
        if not name:
            return ToolResult.from_error("name is required")
        
        response = self._request("PUT", f"/tags/{tag_id}", data={"name": name})
        
        return ToolResult.from_data({
            "tag_id": response.get("id"),
            "name": response.get("name"),
            "updated": True,
        })
    
    def _delete_tag(
        self,
        tag_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a tag."""
        if not tag_id:
            return ToolResult.from_error("tag_id is required")
        
        self._request("DELETE", f"/tags/{tag_id}")
        
        return ToolResult.from_data({
            "tag_id": tag_id,
            "deleted": True,
        })
    
    # Webhook testing
    def _test_webhook(
        self,
        workflow_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Test a webhook trigger."""
        if not workflow_id:
            return ToolResult.from_error("workflow_id is required")
        
        webhook_path = kwargs.get("webhook_path", "")
        method = kwargs.get("method", "GET")
        body = kwargs.get("body")
        
        session = self._get_session()
        url = f"{self.base_url}/webhook-test/{workflow_id}/{webhook_path}".rstrip("/")
        
        response = session.request(
            method=method,
            url=url,
            json=body,
        )
        
        return ToolResult.from_data({
            "status_code": response.status_code,
            "response": response.json() if response.text else None,
            "tested": True,
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
                            "list_workflows", "get_workflow", "create_workflow",
                            "update_workflow", "delete_workflow",
                            "activate_workflow", "deactivate_workflow", "execute_workflow",
                            "list_executions", "get_execution", "delete_execution",
                            "list_credentials", "get_credential", "create_credential", "delete_credential",
                            "list_tags", "get_tag", "create_tag", "update_tag", "delete_tag",
                            "test_webhook",
                        ],
                    },
                    "workflow_id": {"type": "string"},
                    "execution_id": {"type": "string"},
                    "credential_id": {"type": "string"},
                    "tag_id": {"type": "string"},
                    "name": {"type": "string"},
                    "active": {"type": "boolean"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
