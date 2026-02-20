"""
Postman Tool for ARGUS.

API testing and documentation management.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class PostmanTool(BaseTool):
    """
    Postman - API testing and documentation.
    
    Features:
    - Collection management
    - Environment management
    - Monitor management
    - Mock server management
    - API documentation
    
    Example:
        >>> tool = PostmanTool()
        >>> result = tool(action="list_collections")
        >>> result = tool(action="run_collection", collection_id="123")
    """
    
    name = "postman"
    description = "API testing and documentation platform"
    category = ToolCategory.DEVELOPMENT
    version = "1.0.0"
    
    API_BASE = "https://api.getpostman.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("POSTMAN_API_KEY")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Postman API key provided")
        
        logger.debug("Postman tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "X-API-Key": self.api_key,
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
        """Make API request to Postman."""
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
                error = error_data.get("error", {})
                message = error.get("message", f"HTTP {response.status_code}")
                raise Exception(message)
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_collections",
        collection_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        monitor_id: Optional[str] = None,
        mock_id: Optional[str] = None,
        api_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Postman operations."""
        actions = {
            # Collection operations
            "list_collections": self._list_collections,
            "get_collection": self._get_collection,
            "create_collection": self._create_collection,
            "update_collection": self._update_collection,
            "delete_collection": self._delete_collection,
            "fork_collection": self._fork_collection,
            
            # Environment operations
            "list_environments": self._list_environments,
            "get_environment": self._get_environment,
            "create_environment": self._create_environment,
            "update_environment": self._update_environment,
            "delete_environment": self._delete_environment,
            
            # Workspace operations
            "list_workspaces": self._list_workspaces,
            "get_workspace": self._get_workspace,
            "create_workspace": self._create_workspace,
            
            # Monitor operations
            "list_monitors": self._list_monitors,
            "get_monitor": self._get_monitor,
            "run_monitor": self._run_monitor,
            
            # Mock operations
            "list_mocks": self._list_mocks,
            "get_mock": self._get_mock,
            "create_mock": self._create_mock,
            
            # API operations
            "list_apis": self._list_apis,
            "get_api": self._get_api,
            "create_api": self._create_api,
            
            # User
            "me": self._me,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("Postman API key not configured")
            
            return actions[action](
                collection_id=collection_id,
                environment_id=environment_id,
                workspace_id=workspace_id,
                monitor_id=monitor_id,
                mock_id=mock_id,
                api_id=api_id,
                name=name,
                description=description,
                schema=schema,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Postman error: {e}")
            return ToolResult.from_error(f"Postman error: {e}")
    
    # Collection operations
    def _list_collections(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List collections."""
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request("GET", "/collections", params=params if params else None)
        
        collections = [
            {
                "id": c["id"],
                "uid": c.get("uid"),
                "name": c["name"],
                "owner": c.get("owner"),
                "fork": c.get("fork"),
            }
            for c in response.get("collections", [])
        ]
        
        return ToolResult.from_data({
            "collections": collections,
            "count": len(collections),
        })
    
    def _get_collection(
        self,
        collection_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get collection details."""
        if not collection_id:
            return ToolResult.from_error("collection_id is required")
        
        response = self._request("GET", f"/collections/{collection_id}")
        
        collection = response.get("collection", {})
        info = collection.get("info", {})
        
        return ToolResult.from_data({
            "collection": {
                "id": info.get("_postman_id"),
                "name": info.get("name"),
                "description": info.get("description"),
                "schema": info.get("schema"),
                "item_count": len(collection.get("item", [])),
                "variable_count": len(collection.get("variable", [])),
            }
        })
    
    def _create_collection(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a collection."""
        if not name:
            return ToolResult.from_error("name is required")
        
        collection_data = {
            "info": {
                "name": name,
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            },
            "item": [],
        }
        
        if description:
            collection_data["info"]["description"] = description
        
        data = {"collection": collection_data}
        
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request(
            "POST",
            "/collections",
            data=data,
            params=params if params else None,
        )
        
        coll = response.get("collection", {})
        
        return ToolResult.from_data({
            "collection_id": coll.get("id"),
            "uid": coll.get("uid"),
            "name": coll.get("name"),
            "created": True,
        })
    
    def _update_collection(
        self,
        collection_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a collection."""
        if not collection_id:
            return ToolResult.from_error("collection_id is required")
        
        # Get existing collection
        existing = self._request("GET", f"/collections/{collection_id}")
        collection = existing.get("collection", {})
        
        if name:
            collection["info"]["name"] = name
        if description is not None:
            collection["info"]["description"] = description
        
        response = self._request(
            "PUT",
            f"/collections/{collection_id}",
            data={"collection": collection},
        )
        
        return ToolResult.from_data({
            "collection_id": response.get("collection", {}).get("id"),
            "updated": True,
        })
    
    def _delete_collection(
        self,
        collection_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a collection."""
        if not collection_id:
            return ToolResult.from_error("collection_id is required")
        
        self._request("DELETE", f"/collections/{collection_id}")
        
        return ToolResult.from_data({
            "collection_id": collection_id,
            "deleted": True,
        })
    
    def _fork_collection(
        self,
        collection_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Fork a collection."""
        if not collection_id:
            return ToolResult.from_error("collection_id is required")
        if not workspace_id:
            return ToolResult.from_error("workspace_id is required")
        
        label = name or "Fork"
        
        response = self._request(
            "POST",
            f"/collections/fork/{collection_id}",
            data={"label": label},
            params={"workspace": workspace_id},
        )
        
        return ToolResult.from_data({
            "forked_collection": response.get("collection", {}),
            "forked": True,
        })
    
    # Environment operations
    def _list_environments(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List environments."""
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request("GET", "/environments", params=params if params else None)
        
        environments = [
            {
                "id": e["id"],
                "uid": e.get("uid"),
                "name": e["name"],
                "owner": e.get("owner"),
            }
            for e in response.get("environments", [])
        ]
        
        return ToolResult.from_data({
            "environments": environments,
            "count": len(environments),
        })
    
    def _get_environment(
        self,
        environment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get environment details."""
        if not environment_id:
            return ToolResult.from_error("environment_id is required")
        
        response = self._request("GET", f"/environments/{environment_id}")
        
        env = response.get("environment", {})
        
        return ToolResult.from_data({
            "environment": {
                "id": env.get("id"),
                "name": env.get("name"),
                "values": env.get("values", []),
            }
        })
    
    def _create_environment(
        self,
        name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create an environment."""
        if not name:
            return ToolResult.from_error("name is required")
        
        values = kwargs.get("values", [])
        
        data = {
            "environment": {
                "name": name,
                "values": values,
            }
        }
        
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request(
            "POST",
            "/environments",
            data=data,
            params=params if params else None,
        )
        
        env = response.get("environment", {})
        
        return ToolResult.from_data({
            "environment_id": env.get("id"),
            "uid": env.get("uid"),
            "name": env.get("name"),
            "created": True,
        })
    
    def _update_environment(
        self,
        environment_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update an environment."""
        if not environment_id:
            return ToolResult.from_error("environment_id is required")
        
        data = {"environment": {}}
        
        if name:
            data["environment"]["name"] = name
        
        values = kwargs.get("values")
        if values:
            data["environment"]["values"] = values
        
        if not data["environment"]:
            return ToolResult.from_error("No fields to update")
        
        response = self._request(
            "PUT",
            f"/environments/{environment_id}",
            data=data,
        )
        
        return ToolResult.from_data({
            "environment_id": response.get("environment", {}).get("id"),
            "updated": True,
        })
    
    def _delete_environment(
        self,
        environment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete an environment."""
        if not environment_id:
            return ToolResult.from_error("environment_id is required")
        
        self._request("DELETE", f"/environments/{environment_id}")
        
        return ToolResult.from_data({
            "environment_id": environment_id,
            "deleted": True,
        })
    
    # Workspace operations
    def _list_workspaces(self, **kwargs) -> ToolResult:
        """List workspaces."""
        workspace_type = kwargs.get("type")
        
        params = {}
        if workspace_type:
            params["type"] = workspace_type
        
        response = self._request("GET", "/workspaces", params=params if params else None)
        
        workspaces = [
            {
                "id": w["id"],
                "name": w["name"],
                "type": w.get("type"),
                "visibility": w.get("visibility"),
            }
            for w in response.get("workspaces", [])
        ]
        
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
        
        response = self._request("GET", f"/workspaces/{workspace_id}")
        
        ws = response.get("workspace", {})
        
        return ToolResult.from_data({
            "workspace": {
                "id": ws.get("id"),
                "name": ws.get("name"),
                "type": ws.get("type"),
                "description": ws.get("description"),
                "collections": ws.get("collections", []),
                "environments": ws.get("environments", []),
                "mocks": ws.get("mocks", []),
                "monitors": ws.get("monitors", []),
            }
        })
    
    def _create_workspace(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a workspace."""
        if not name:
            return ToolResult.from_error("name is required")
        
        workspace_type = kwargs.get("type", "personal")
        
        data = {
            "workspace": {
                "name": name,
                "type": workspace_type,
            }
        }
        
        if description:
            data["workspace"]["description"] = description
        
        response = self._request("POST", "/workspaces", data=data)
        
        ws = response.get("workspace", {})
        
        return ToolResult.from_data({
            "workspace_id": ws.get("id"),
            "name": ws.get("name"),
            "created": True,
        })
    
    # Monitor operations
    def _list_monitors(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List monitors."""
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request("GET", "/monitors", params=params if params else None)
        
        monitors = [
            {
                "id": m["id"],
                "uid": m.get("uid"),
                "name": m["name"],
                "owner": m.get("owner"),
            }
            for m in response.get("monitors", [])
        ]
        
        return ToolResult.from_data({
            "monitors": monitors,
            "count": len(monitors),
        })
    
    def _get_monitor(
        self,
        monitor_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get monitor details."""
        if not monitor_id:
            return ToolResult.from_error("monitor_id is required")
        
        response = self._request("GET", f"/monitors/{monitor_id}")
        
        monitor = response.get("monitor", {})
        
        return ToolResult.from_data({
            "monitor": {
                "id": monitor.get("id"),
                "name": monitor.get("name"),
                "collection": monitor.get("collection"),
                "environment": monitor.get("environment"),
                "schedule": monitor.get("schedule"),
                "lastRun": monitor.get("lastRun"),
            }
        })
    
    def _run_monitor(
        self,
        monitor_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Run a monitor."""
        if not monitor_id:
            return ToolResult.from_error("monitor_id is required")
        
        response = self._request("POST", f"/monitors/{monitor_id}/run")
        
        run = response.get("run", {})
        
        return ToolResult.from_data({
            "run_id": run.get("id"),
            "status": run.get("status"),
            "stats": run.get("stats"),
            "executed": True,
        })
    
    # Mock operations
    def _list_mocks(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List mock servers."""
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request("GET", "/mocks", params=params if params else None)
        
        mocks = [
            {
                "id": m["id"],
                "uid": m.get("uid"),
                "name": m.get("name"),
                "collection": m.get("collection"),
                "mockUrl": m.get("mockUrl"),
            }
            for m in response.get("mocks", [])
        ]
        
        return ToolResult.from_data({
            "mocks": mocks,
            "count": len(mocks),
        })
    
    def _get_mock(
        self,
        mock_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get mock server details."""
        if not mock_id:
            return ToolResult.from_error("mock_id is required")
        
        response = self._request("GET", f"/mocks/{mock_id}")
        
        mock = response.get("mock", {})
        
        return ToolResult.from_data({
            "mock": {
                "id": mock.get("id"),
                "name": mock.get("name"),
                "collection": mock.get("collection"),
                "environment": mock.get("environment"),
                "mockUrl": mock.get("mockUrl"),
                "isPublic": mock.get("isPublic"),
            }
        })
    
    def _create_mock(
        self,
        collection_id: Optional[str] = None,
        name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a mock server."""
        if not collection_id:
            return ToolResult.from_error("collection_id is required")
        
        data = {
            "mock": {
                "collection": collection_id,
            }
        }
        
        if name:
            data["mock"]["name"] = name
        
        environment_id = kwargs.get("environment_id")
        if environment_id:
            data["mock"]["environment"] = environment_id
        
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request(
            "POST",
            "/mocks",
            data=data,
            params=params if params else None,
        )
        
        mock = response.get("mock", {})
        
        return ToolResult.from_data({
            "mock_id": mock.get("id"),
            "mockUrl": mock.get("mockUrl"),
            "created": True,
        })
    
    # API operations
    def _list_apis(
        self,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List APIs."""
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request("GET", "/apis", params=params if params else None)
        
        apis = [
            {
                "id": a["id"],
                "name": a["name"],
                "summary": a.get("summary"),
                "createdBy": a.get("createdBy"),
            }
            for a in response.get("apis", [])
        ]
        
        return ToolResult.from_data({
            "apis": apis,
            "count": len(apis),
        })
    
    def _get_api(
        self,
        api_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get API details."""
        if not api_id:
            return ToolResult.from_error("api_id is required")
        
        response = self._request("GET", f"/apis/{api_id}")
        
        api = response.get("api", {})
        
        return ToolResult.from_data({
            "api": {
                "id": api.get("id"),
                "name": api.get("name"),
                "summary": api.get("summary"),
                "description": api.get("description"),
                "createdAt": api.get("createdAt"),
                "updatedAt": api.get("updatedAt"),
            }
        })
    
    def _create_api(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create an API."""
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {
            "api": {
                "name": name,
            }
        }
        
        if description:
            data["api"]["summary"] = description
        
        params = {}
        if workspace_id:
            params["workspace"] = workspace_id
        
        response = self._request(
            "POST",
            "/apis",
            data=data,
            params=params if params else None,
        )
        
        api = response.get("api", {})
        
        return ToolResult.from_data({
            "api_id": api.get("id"),
            "name": api.get("name"),
            "created": True,
        })
    
    # User
    def _me(self, **kwargs) -> ToolResult:
        """Get current user."""
        response = self._request("GET", "/me")
        
        user = response.get("user", {})
        
        return ToolResult.from_data({
            "user": {
                "id": user.get("id"),
                "username": user.get("username"),
                "email": user.get("email"),
                "fullName": user.get("fullName"),
            }
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
                            "list_collections", "get_collection", "create_collection",
                            "update_collection", "delete_collection", "fork_collection",
                            "list_environments", "get_environment", "create_environment",
                            "update_environment", "delete_environment",
                            "list_workspaces", "get_workspace", "create_workspace",
                            "list_monitors", "get_monitor", "run_monitor",
                            "list_mocks", "get_mock", "create_mock",
                            "list_apis", "get_api", "create_api",
                            "me",
                        ],
                    },
                    "collection_id": {"type": "string"},
                    "environment_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "monitor_id": {"type": "string"},
                    "mock_id": {"type": "string"},
                    "api_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["action"],
            },
        }
