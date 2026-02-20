"""
Daytona Tool for ARGUS.

Development environment management.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class DaytonaTool(BaseTool):
    """
    Daytona - Development environment management.
    
    Features:
    - Workspace management
    - Project management
    - Git provider integration
    - Environment configuration
    - SSH access management
    
    Example:
        >>> tool = DaytonaTool()
        >>> result = tool(action="list_workspaces")
        >>> result = tool(action="create_workspace", name="my-project")
    """
    
    name = "daytona"
    description = "Development environment management platform"
    category = ToolCategory.DEVELOPMENT
    version = "1.0.0"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("DAYTONA_API_KEY")
        self.base_url = (base_url or os.getenv("DAYTONA_URL", "http://localhost:3986")).rstrip("/")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Daytona API key provided")
        
        logger.debug(f"Daytona tool initialized (url={self.base_url})")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            headers = {
                "Content-Type": "application/json",
            }
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
        """Make API request to Daytona."""
        session = self._get_session()
        url = f"{self.base_url}{endpoint}"
        
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
        action: str = "list_workspaces",
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        target_id: Optional[str] = None,
        name: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Daytona operations."""
        actions = {
            # Workspace operations
            "list_workspaces": self._list_workspaces,
            "get_workspace": self._get_workspace,
            "create_workspace": self._create_workspace,
            "delete_workspace": self._delete_workspace,
            "start_workspace": self._start_workspace,
            "stop_workspace": self._stop_workspace,
            
            # Project operations
            "list_projects": self._list_projects,
            "get_project": self._get_project,
            "create_project": self._create_project,
            "delete_project": self._delete_project,
            
            # Git provider operations
            "list_providers": self._list_providers,
            "get_provider": self._get_provider,
            "add_provider": self._add_provider,
            "remove_provider": self._remove_provider,
            
            # Target operations
            "list_targets": self._list_targets,
            "get_target": self._get_target,
            "set_target": self._set_target,
            "remove_target": self._remove_target,
            
            # Server info
            "server_info": self._server_info,
            
            # SSH operations
            "get_ssh_key": self._get_ssh_key,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                workspace_id=workspace_id,
                project_id=project_id,
                provider_id=provider_id,
                target_id=target_id,
                name=name,
                repository=repository,
                branch=branch,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Daytona error: {e}")
            return ToolResult.from_error(f"Daytona error: {e}")
    
    # Workspace operations
    def _list_workspaces(self, **kwargs) -> ToolResult:
        """List workspaces."""
        response = self._request("GET", "/workspace")
        
        workspaces = []
        items = response if isinstance(response, list) else []
        
        for w in items:
            workspaces.append({
                "id": w.get("id"),
                "name": w.get("name"),
                "target": w.get("target"),
                "projects": [p.get("name") for p in w.get("projects", [])],
                "state": w.get("state"),
            })
        
        return ToolResult.from_data({
            "workspaces": workspaces,
            "count": len(workspaces),
        })
    
    def _get_workspace(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get workspace details."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        
        response = self._request("GET", f"/workspace/{workspace_id}")
        
        return ToolResult.from_data({
            "workspace": {
                "id": response.get("id"),
                "name": response.get("name"),
                "target": response.get("target"),
                "projects": response.get("projects", []),
                "state": response.get("state"),
                "info": response.get("info"),
            }
        })
    
    def _create_workspace(
        self,
        name: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        target_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a workspace."""
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {"name": name}
        
        if repository:
            project = {"repository": {"url": repository}}
            if branch:
                project["repository"]["branch"] = branch
            data["projects"] = [project]
        
        if target_id:
            data["target"] = target_id
        
        response = self._request("POST", "/workspace", data=data)
        
        return ToolResult.from_data({
            "workspace_id": response.get("id"),
            "name": response.get("name"),
            "created": True,
        })
    
    def _delete_workspace(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a workspace."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        
        force = kwargs.get("force", False)
        params = {"force": "true"} if force else None
        
        self._request("DELETE", f"/workspace/{workspace_id}", params=params)
        
        return ToolResult.from_data({
            "workspace_id": workspace_id,
            "deleted": True,
        })
    
    def _start_workspace(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Start a workspace."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        
        self._request("POST", f"/workspace/{workspace_id}/start")
        
        return ToolResult.from_data({
            "workspace_id": workspace_id,
            "started": True,
        })
    
    def _stop_workspace(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Stop a workspace."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        
        self._request("POST", f"/workspace/{workspace_id}/stop")
        
        return ToolResult.from_data({
            "workspace_id": workspace_id,
            "stopped": True,
        })
    
    # Project operations
    def _list_projects(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List projects in workspace."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        
        response = self._request("GET", f"/workspace/{workspace_id}")
        
        projects = response.get("projects", [])
        
        return ToolResult.from_data({
            "workspace_id": workspace_id,
            "projects": projects,
            "count": len(projects),
        })
    
    def _get_project(
        self,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get project details."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        response = self._request("GET", f"/workspace/{workspace_id}/{project_id}")
        
        return ToolResult.from_data({
            "project": response,
        })
    
    def _create_project(
        self,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a project in workspace."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        if not repository:
            return ToolResult.from_error("repository is required")
        
        data = {
            "repository": {"url": repository},
        }
        
        if name:
            data["name"] = name
        if branch:
            data["repository"]["branch"] = branch
        
        response = self._request("POST", f"/workspace/{workspace_id}/project", data=data)
        
        return ToolResult.from_data({
            "project": response,
            "created": True,
        })
    
    def _delete_project(
        self,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a project from workspace."""
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        self._request("DELETE", f"/workspace/{workspace_id}/{project_id}")
        
        return ToolResult.from_data({
            "workspace_id": workspace_id,
            "project_id": project_id,
            "deleted": True,
        })
    
    # Git provider operations
    def _list_providers(self, **kwargs) -> ToolResult:
        """List Git providers."""
        response = self._request("GET", "/gitprovider")
        
        providers = []
        items = response if isinstance(response, list) else []
        
        for p in items:
            providers.append({
                "id": p.get("id"),
                "providerId": p.get("providerId"),
                "username": p.get("username"),
                "baseApiUrl": p.get("baseApiUrl"),
            })
        
        return ToolResult.from_data({
            "providers": providers,
            "count": len(providers),
        })
    
    def _get_provider(
        self,
        provider_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get Git provider details."""
        if not provider_id:
            return ToolResult.from_error("provider_id is required")
        
        response = self._request("GET", f"/gitprovider/{provider_id}")
        
        return ToolResult.from_data({
            "provider": response,
        })
    
    def _add_provider(
        self,
        provider_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add a Git provider."""
        if not provider_id:
            return ToolResult.from_error("provider_id is required")
        
        token = kwargs.get("token")
        if not token:
            return ToolResult.from_error("token is required")
        
        data = {
            "providerId": provider_id,
            "token": token,
        }
        
        username = kwargs.get("username")
        if username:
            data["username"] = username
        
        base_url = kwargs.get("base_url")
        if base_url:
            data["baseApiUrl"] = base_url
        
        response = self._request("PUT", "/gitprovider", data=data)
        
        return ToolResult.from_data({
            "provider": response,
            "added": True,
        })
    
    def _remove_provider(
        self,
        provider_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Remove a Git provider."""
        if not provider_id:
            return ToolResult.from_error("provider_id is required")
        
        self._request("DELETE", f"/gitprovider/{provider_id}")
        
        return ToolResult.from_data({
            "provider_id": provider_id,
            "removed": True,
        })
    
    # Target operations
    def _list_targets(self, **kwargs) -> ToolResult:
        """List available targets."""
        response = self._request("GET", "/target")
        
        targets = []
        items = response if isinstance(response, list) else []
        
        for t in items:
            targets.append({
                "name": t.get("name"),
                "providerInfo": t.get("providerInfo"),
                "isDefault": t.get("isDefault", False),
            })
        
        return ToolResult.from_data({
            "targets": targets,
            "count": len(targets),
        })
    
    def _get_target(
        self,
        target_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get target details."""
        if not target_id:
            return ToolResult.from_error("target_id is required")
        
        response = self._request("GET", f"/target/{target_id}")
        
        return ToolResult.from_data({
            "target": response,
        })
    
    def _set_target(
        self,
        target_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Set a target."""
        if not target_id:
            return ToolResult.from_error("target_id is required")
        
        provider = kwargs.get("provider", "docker-provider")
        
        data = {
            "name": target_id,
            "providerInfo": {
                "name": provider,
            },
        }
        
        options = kwargs.get("options")
        if options:
            data["providerInfo"]["options"] = options
        
        response = self._request("PUT", "/target", data=data)
        
        return ToolResult.from_data({
            "target": response,
            "set": True,
        })
    
    def _remove_target(
        self,
        target_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Remove a target."""
        if not target_id:
            return ToolResult.from_error("target_id is required")
        
        self._request("DELETE", f"/target/{target_id}")
        
        return ToolResult.from_data({
            "target_id": target_id,
            "removed": True,
        })
    
    # Server info
    def _server_info(self, **kwargs) -> ToolResult:
        """Get server information."""
        response = self._request("GET", "/server")
        
        return ToolResult.from_data({
            "server": {
                "id": response.get("id"),
                "frps": response.get("frps"),
                "apiPort": response.get("apiPort"),
                "headscalePort": response.get("headscalePort"),
            }
        })
    
    # SSH operations
    def _get_ssh_key(self, **kwargs) -> ToolResult:
        """Get SSH public key."""
        response = self._request("GET", "/server/publickey")
        
        return ToolResult.from_data({
            "public_key": response.get("publicKey") if isinstance(response, dict) else response,
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
                            "list_workspaces", "get_workspace", "create_workspace",
                            "delete_workspace", "start_workspace", "stop_workspace",
                            "list_projects", "get_project", "create_project", "delete_project",
                            "list_providers", "get_provider", "add_provider", "remove_provider",
                            "list_targets", "get_target", "set_target", "remove_target",
                            "server_info", "get_ssh_key",
                        ],
                    },
                    "workspace_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "provider_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "name": {"type": "string"},
                    "repository": {"type": "string"},
                    "branch": {"type": "string"},
                },
                "required": ["action"],
            },
        }
