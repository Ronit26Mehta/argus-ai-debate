"""
Asana Tool for ARGUS.

Project management and task tracking integration.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class AsanaTool(BaseTool):
    """
    Asana - Project management and task tracking.
    
    Features:
    - Task CRUD operations
    - Project management
    - Workspace management
    - Team assignments
    - Section organization
    - Tag management
    - Search functionality
    
    Example:
        >>> tool = AsanaTool()
        >>> result = tool(action="list_tasks", project_gid="123456")
        >>> result = tool(action="create_task", project_gid="123456", name="New Task")
    """
    
    name = "asana"
    description = "Project management and task tracking with Asana"
    category = ToolCategory.PRODUCTIVITY
    version = "1.0.0"
    
    BASE_URL = "https://app.asana.com/api/1.0"
    
    def __init__(
        self,
        access_token: Optional[str] = None,
        workspace_gid: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.access_token = access_token or os.getenv("ASANA_ACCESS_TOKEN")
        self.workspace_gid = workspace_gid or os.getenv("ASANA_WORKSPACE_GID")
        
        self._session = None
        
        if not self.access_token:
            logger.warning("No Asana access token provided")
        
        logger.debug("Asana tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
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
        """Make API request to Asana."""
        session = self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            json={"data": data} if data else None,
            params=params,
        )
        
        if response.status_code >= 400:
            error_data = response.json().get("errors", [{}])[0]
            raise Exception(error_data.get("message", f"HTTP {response.status_code}"))
        
        return response.json()
    
    def execute(
        self,
        action: str = "list_workspaces",
        workspace_gid: Optional[str] = None,
        project_gid: Optional[str] = None,
        task_gid: Optional[str] = None,
        section_gid: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        due_on: Optional[str] = None,
        assignee: Optional[str] = None,
        completed: Optional[bool] = None,
        tags: Optional[list] = None,
        query: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Asana operations."""
        actions = {
            # Workspace operations
            "list_workspaces": self._list_workspaces,
            "get_workspace": self._get_workspace,
            
            # Project operations
            "list_projects": self._list_projects,
            "get_project": self._get_project,
            "create_project": self._create_project,
            "update_project": self._update_project,
            "delete_project": self._delete_project,
            
            # Task operations
            "list_tasks": self._list_tasks,
            "get_task": self._get_task,
            "create_task": self._create_task,
            "update_task": self._update_task,
            "delete_task": self._delete_task,
            "complete_task": self._complete_task,
            
            # Section operations
            "list_sections": self._list_sections,
            "create_section": self._create_section,
            "add_task_to_section": self._add_task_to_section,
            
            # Tag operations
            "list_tags": self._list_tags,
            "create_tag": self._create_tag,
            "add_tag_to_task": self._add_tag_to_task,
            
            # Search
            "search": self._search,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.access_token:
                return ToolResult.from_error("Asana access token not configured")
            
            ws_gid = workspace_gid or self.workspace_gid
            
            return actions[action](
                workspace_gid=ws_gid,
                project_gid=project_gid,
                task_gid=task_gid,
                section_gid=section_gid,
                name=name,
                notes=notes,
                due_on=due_on,
                assignee=assignee,
                completed=completed,
                tags=tags,
                query=query,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Asana error: {e}")
            return ToolResult.from_error(f"Asana error: {e}")
    
    # Workspace operations
    def _list_workspaces(self, limit: int = 100, **kwargs) -> ToolResult:
        """List all accessible workspaces."""
        response = self._request("GET", "/workspaces", params={"limit": limit})
        
        workspaces = [
            {
                "gid": w["gid"],
                "name": w["name"],
            }
            for w in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "workspaces": workspaces,
            "count": len(workspaces),
        })
    
    def _get_workspace(self, workspace_gid: Optional[str] = None, **kwargs) -> ToolResult:
        """Get workspace details."""
        if not workspace_gid:
            return ToolResult.from_error("workspace_gid is required")
        
        response = self._request("GET", f"/workspaces/{workspace_gid}")
        
        return ToolResult.from_data({"workspace": response.get("data")})
    
    # Project operations
    def _list_projects(
        self,
        workspace_gid: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List projects in workspace."""
        if not workspace_gid:
            return ToolResult.from_error("workspace_gid is required")
        
        response = self._request(
            "GET",
            "/projects",
            params={"workspace": workspace_gid, "limit": limit},
        )
        
        projects = [
            {
                "gid": p["gid"],
                "name": p["name"],
            }
            for p in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "workspace_gid": workspace_gid,
            "projects": projects,
            "count": len(projects),
        })
    
    def _get_project(self, project_gid: Optional[str] = None, **kwargs) -> ToolResult:
        """Get project details."""
        if not project_gid:
            return ToolResult.from_error("project_gid is required")
        
        response = self._request("GET", f"/projects/{project_gid}")
        
        return ToolResult.from_data({"project": response.get("data")})
    
    def _create_project(
        self,
        workspace_gid: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new project."""
        if not workspace_gid:
            return ToolResult.from_error("workspace_gid is required")
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {
            "workspace": workspace_gid,
            "name": name,
        }
        if notes:
            data["notes"] = notes
        
        response = self._request("POST", "/projects", data=data)
        
        return ToolResult.from_data({
            "project": response.get("data"),
            "created": True,
        })
    
    def _update_project(
        self,
        project_gid: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a project."""
        if not project_gid:
            return ToolResult.from_error("project_gid is required")
        
        data = {}
        if name:
            data["name"] = name
        if notes:
            data["notes"] = notes
        
        response = self._request("PUT", f"/projects/{project_gid}", data=data)
        
        return ToolResult.from_data({
            "project": response.get("data"),
            "updated": True,
        })
    
    def _delete_project(self, project_gid: Optional[str] = None, **kwargs) -> ToolResult:
        """Delete a project."""
        if not project_gid:
            return ToolResult.from_error("project_gid is required")
        
        self._request("DELETE", f"/projects/{project_gid}")
        
        return ToolResult.from_data({
            "project_gid": project_gid,
            "deleted": True,
        })
    
    # Task operations
    def _list_tasks(
        self,
        project_gid: Optional[str] = None,
        section_gid: Optional[str] = None,
        assignee: Optional[str] = None,
        completed: Optional[bool] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List tasks in a project or section."""
        params = {"limit": limit}
        
        if section_gid:
            params["section"] = section_gid
        elif project_gid:
            params["project"] = project_gid
        else:
            return ToolResult.from_error("project_gid or section_gid is required")
        
        if completed is not None:
            params["completed_since"] = "now" if not completed else None
        
        response = self._request("GET", "/tasks", params=params)
        
        tasks = [
            {
                "gid": t["gid"],
                "name": t["name"],
                "completed": t.get("completed", False),
            }
            for t in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "tasks": tasks,
            "count": len(tasks),
        })
    
    def _get_task(self, task_gid: Optional[str] = None, **kwargs) -> ToolResult:
        """Get task details."""
        if not task_gid:
            return ToolResult.from_error("task_gid is required")
        
        response = self._request("GET", f"/tasks/{task_gid}")
        
        return ToolResult.from_data({"task": response.get("data")})
    
    def _create_task(
        self,
        project_gid: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        due_on: Optional[str] = None,
        assignee: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new task."""
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {"name": name}
        if project_gid:
            data["projects"] = [project_gid]
        if notes:
            data["notes"] = notes
        if due_on:
            data["due_on"] = due_on
        if assignee:
            data["assignee"] = assignee
        
        response = self._request("POST", "/tasks", data=data)
        
        return ToolResult.from_data({
            "task": response.get("data"),
            "created": True,
        })
    
    def _update_task(
        self,
        task_gid: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        due_on: Optional[str] = None,
        assignee: Optional[str] = None,
        completed: Optional[bool] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a task."""
        if not task_gid:
            return ToolResult.from_error("task_gid is required")
        
        data = {}
        if name:
            data["name"] = name
        if notes:
            data["notes"] = notes
        if due_on:
            data["due_on"] = due_on
        if assignee:
            data["assignee"] = assignee
        if completed is not None:
            data["completed"] = completed
        
        response = self._request("PUT", f"/tasks/{task_gid}", data=data)
        
        return ToolResult.from_data({
            "task": response.get("data"),
            "updated": True,
        })
    
    def _delete_task(self, task_gid: Optional[str] = None, **kwargs) -> ToolResult:
        """Delete a task."""
        if not task_gid:
            return ToolResult.from_error("task_gid is required")
        
        self._request("DELETE", f"/tasks/{task_gid}")
        
        return ToolResult.from_data({
            "task_gid": task_gid,
            "deleted": True,
        })
    
    def _complete_task(self, task_gid: Optional[str] = None, **kwargs) -> ToolResult:
        """Mark a task as complete."""
        if not task_gid:
            return ToolResult.from_error("task_gid is required")
        
        response = self._request(
            "PUT",
            f"/tasks/{task_gid}",
            data={"completed": True},
        )
        
        return ToolResult.from_data({
            "task": response.get("data"),
            "completed": True,
        })
    
    # Section operations
    def _list_sections(
        self,
        project_gid: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List sections in a project."""
        if not project_gid:
            return ToolResult.from_error("project_gid is required")
        
        response = self._request(
            "GET",
            f"/projects/{project_gid}/sections",
            params={"limit": limit},
        )
        
        sections = [
            {
                "gid": s["gid"],
                "name": s["name"],
            }
            for s in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "project_gid": project_gid,
            "sections": sections,
            "count": len(sections),
        })
    
    def _create_section(
        self,
        project_gid: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a section in a project."""
        if not project_gid:
            return ToolResult.from_error("project_gid is required")
        if not name:
            return ToolResult.from_error("name is required")
        
        response = self._request(
            "POST",
            f"/projects/{project_gid}/sections",
            data={"name": name},
        )
        
        return ToolResult.from_data({
            "section": response.get("data"),
            "created": True,
        })
    
    def _add_task_to_section(
        self,
        section_gid: Optional[str] = None,
        task_gid: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add a task to a section."""
        if not section_gid:
            return ToolResult.from_error("section_gid is required")
        if not task_gid:
            return ToolResult.from_error("task_gid is required")
        
        self._request(
            "POST",
            f"/sections/{section_gid}/addTask",
            data={"task": task_gid},
        )
        
        return ToolResult.from_data({
            "section_gid": section_gid,
            "task_gid": task_gid,
            "added": True,
        })
    
    # Tag operations
    def _list_tags(
        self,
        workspace_gid: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List tags in workspace."""
        if not workspace_gid:
            return ToolResult.from_error("workspace_gid is required")
        
        response = self._request(
            "GET",
            "/tags",
            params={"workspace": workspace_gid, "limit": limit},
        )
        
        tags = [
            {
                "gid": t["gid"],
                "name": t["name"],
            }
            for t in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "workspace_gid": workspace_gid,
            "tags": tags,
            "count": len(tags),
        })
    
    def _create_tag(
        self,
        workspace_gid: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a tag."""
        if not workspace_gid:
            return ToolResult.from_error("workspace_gid is required")
        if not name:
            return ToolResult.from_error("name is required")
        
        response = self._request(
            "POST",
            "/tags",
            data={"workspace": workspace_gid, "name": name},
        )
        
        return ToolResult.from_data({
            "tag": response.get("data"),
            "created": True,
        })
    
    def _add_tag_to_task(
        self,
        task_gid: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add a tag to a task."""
        if not task_gid:
            return ToolResult.from_error("task_gid is required")
        
        tag_gid = kwargs.get("tag_gid")
        if not tag_gid:
            return ToolResult.from_error("tag_gid is required")
        
        self._request(
            "POST",
            f"/tasks/{task_gid}/addTag",
            data={"tag": tag_gid},
        )
        
        return ToolResult.from_data({
            "task_gid": task_gid,
            "tag_gid": tag_gid,
            "added": True,
        })
    
    # Search
    def _search(
        self,
        workspace_gid: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Search tasks in workspace."""
        if not workspace_gid:
            return ToolResult.from_error("workspace_gid is required")
        if not query:
            return ToolResult.from_error("query is required")
        
        response = self._request(
            "GET",
            f"/workspaces/{workspace_gid}/tasks/search",
            params={"text": query, "limit": limit},
        )
        
        tasks = [
            {
                "gid": t["gid"],
                "name": t["name"],
            }
            for t in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "query": query,
            "tasks": tasks,
            "count": len(tasks),
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
                            "list_workspaces", "get_workspace",
                            "list_projects", "get_project", "create_project",
                            "update_project", "delete_project",
                            "list_tasks", "get_task", "create_task",
                            "update_task", "delete_task", "complete_task",
                            "list_sections", "create_section", "add_task_to_section",
                            "list_tags", "create_tag", "add_tag_to_task",
                            "search",
                        ],
                    },
                    "workspace_gid": {"type": "string"},
                    "project_gid": {"type": "string"},
                    "task_gid": {"type": "string"},
                    "section_gid": {"type": "string"},
                    "name": {"type": "string"},
                    "notes": {"type": "string"},
                    "due_on": {"type": "string", "format": "date"},
                    "assignee": {"type": "string"},
                    "completed": {"type": "boolean"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
