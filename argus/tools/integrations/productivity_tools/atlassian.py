"""
Atlassian Tools for ARGUS.

Jira and Confluence integrations for project management and documentation.
"""

from __future__ import annotations

import os
import base64
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class JiraTool(BaseTool):
    """
    Jira - Issue and project tracking.
    
    Features:
    - Issue CRUD operations
    - Project management
    - Sprint management
    - JQL search
    - Transitions and workflows
    - Comments and attachments
    
    Example:
        >>> tool = JiraTool(domain="company.atlassian.net")
        >>> result = tool(action="create_issue", project_key="PROJ", summary="Bug fix")
        >>> result = tool(action="search", jql="project=PROJ AND status=Open")
    """
    
    name = "jira"
    description = "Issue and project tracking with Jira"
    category = ToolCategory.PRODUCTIVITY
    version = "1.0.0"
    
    def __init__(
        self,
        domain: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.domain = domain or os.getenv("JIRA_DOMAIN")
        self.email = email or os.getenv("JIRA_EMAIL")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN")
        
        self._session = None
        
        if not self.domain:
            logger.warning("No Jira domain provided")
        
        logger.debug(f"Jira tool initialized (domain={self.domain})")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            
            auth_str = f"{self.email}:{self.api_token}"
            auth_bytes = base64.b64encode(auth_str.encode()).decode()
            
            self._session.headers.update({
                "Authorization": f"Basic {auth_bytes}",
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
        """Make API request to Jira."""
        session = self._get_session()
        url = f"https://{self.domain}/rest/api/3{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            json=data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                errors = error_data.get("errorMessages", [])
                if not errors:
                    errors = [str(error_data.get("errors", {}))]
                raise Exception("; ".join(errors) or f"HTTP {response.status_code}")
            except ValueError:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        if response.status_code == 204:
            return {}
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_projects",
        project_key: Optional[str] = None,
        issue_key: Optional[str] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        issue_type: str = "Task",
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
        labels: Optional[list] = None,
        jql: Optional[str] = None,
        transition_id: Optional[str] = None,
        comment: Optional[str] = None,
        sprint_id: Optional[int] = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Jira operations."""
        actions = {
            # Project operations
            "list_projects": self._list_projects,
            "get_project": self._get_project,
            
            # Issue operations
            "list_issues": self._list_issues,
            "get_issue": self._get_issue,
            "create_issue": self._create_issue,
            "update_issue": self._update_issue,
            "delete_issue": self._delete_issue,
            "assign_issue": self._assign_issue,
            
            # Transitions
            "get_transitions": self._get_transitions,
            "transition_issue": self._transition_issue,
            
            # Comments
            "get_comments": self._get_comments,
            "add_comment": self._add_comment,
            
            # Search
            "search": self._search,
            
            # Sprint operations
            "list_sprints": self._list_sprints,
            "add_to_sprint": self._add_to_sprint,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.domain or not self.api_token:
                return ToolResult.from_error("Jira credentials not configured")
            
            return actions[action](
                project_key=project_key,
                issue_key=issue_key,
                summary=summary,
                description=description,
                issue_type=issue_type,
                priority=priority,
                assignee=assignee,
                labels=labels,
                jql=jql,
                transition_id=transition_id,
                comment=comment,
                sprint_id=sprint_id,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Jira error: {e}")
            return ToolResult.from_error(f"Jira error: {e}")
    
    # Project operations
    def _list_projects(self, limit: int = 50, **kwargs) -> ToolResult:
        """List all projects."""
        response = self._request(
            "GET",
            "/project/search",
            params={"maxResults": limit},
        )
        
        projects = [
            {
                "key": p["key"],
                "name": p["name"],
                "id": p["id"],
                "style": p.get("style"),
            }
            for p in response.get("values", [])
        ]
        
        return ToolResult.from_data({
            "projects": projects,
            "count": len(projects),
        })
    
    def _get_project(
        self,
        project_key: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get project details."""
        if not project_key:
            return ToolResult.from_error("project_key is required")
        
        response = self._request("GET", f"/project/{project_key}")
        
        return ToolResult.from_data({"project": response})
    
    # Issue operations
    def _list_issues(
        self,
        project_key: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """List issues in a project."""
        if not project_key:
            return ToolResult.from_error("project_key is required")
        
        jql = f"project = {project_key} ORDER BY created DESC"
        return self._search(jql=jql, limit=limit, **kwargs)
    
    def _get_issue(
        self,
        issue_key: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get issue details."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        
        response = self._request("GET", f"/issue/{issue_key}")
        
        fields = response.get("fields", {})
        issue = {
            "key": response.get("key"),
            "id": response.get("id"),
            "summary": fields.get("summary"),
            "description": fields.get("description"),
            "status": fields.get("status", {}).get("name"),
            "priority": fields.get("priority", {}).get("name"),
            "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
            "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
            "created": fields.get("created"),
            "updated": fields.get("updated"),
            "labels": fields.get("labels", []),
        }
        
        return ToolResult.from_data({"issue": issue})
    
    def _create_issue(
        self,
        project_key: Optional[str] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        issue_type: str = "Task",
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
        labels: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new issue."""
        if not project_key:
            return ToolResult.from_error("project_key is required")
        if not summary:
            return ToolResult.from_error("summary is required")
        
        fields = {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        
        if description:
            fields["description"] = {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    }
                ]
            }
        
        if priority:
            fields["priority"] = {"name": priority}
        
        if assignee:
            fields["assignee"] = {"accountId": assignee}
        
        if labels:
            fields["labels"] = labels
        
        response = self._request("POST", "/issue", data={"fields": fields})
        
        return ToolResult.from_data({
            "issue_key": response.get("key"),
            "issue_id": response.get("id"),
            "created": True,
        })
    
    def _update_issue(
        self,
        issue_key: Optional[str] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        labels: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Update an issue."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        
        fields = {}
        
        if summary:
            fields["summary"] = summary
        
        if description:
            fields["description"] = {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    }
                ]
            }
        
        if priority:
            fields["priority"] = {"name": priority}
        
        if labels:
            fields["labels"] = labels
        
        if not fields:
            return ToolResult.from_error("No fields to update")
        
        self._request("PUT", f"/issue/{issue_key}", data={"fields": fields})
        
        return ToolResult.from_data({
            "issue_key": issue_key,
            "updated": True,
        })
    
    def _delete_issue(
        self,
        issue_key: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete an issue."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        
        self._request("DELETE", f"/issue/{issue_key}")
        
        return ToolResult.from_data({
            "issue_key": issue_key,
            "deleted": True,
        })
    
    def _assign_issue(
        self,
        issue_key: Optional[str] = None,
        assignee: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Assign an issue to a user."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        
        data = {"accountId": assignee} if assignee else {"accountId": None}
        
        self._request("PUT", f"/issue/{issue_key}/assignee", data=data)
        
        return ToolResult.from_data({
            "issue_key": issue_key,
            "assignee": assignee,
            "assigned": True,
        })
    
    # Transitions
    def _get_transitions(
        self,
        issue_key: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get available transitions for an issue."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        
        response = self._request("GET", f"/issue/{issue_key}/transitions")
        
        transitions = [
            {
                "id": t["id"],
                "name": t["name"],
                "to": t.get("to", {}).get("name"),
            }
            for t in response.get("transitions", [])
        ]
        
        return ToolResult.from_data({
            "issue_key": issue_key,
            "transitions": transitions,
        })
    
    def _transition_issue(
        self,
        issue_key: Optional[str] = None,
        transition_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Transition an issue to a new status."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        if not transition_id:
            return ToolResult.from_error("transition_id is required")
        
        self._request(
            "POST",
            f"/issue/{issue_key}/transitions",
            data={"transition": {"id": transition_id}},
        )
        
        return ToolResult.from_data({
            "issue_key": issue_key,
            "transition_id": transition_id,
            "transitioned": True,
        })
    
    # Comments
    def _get_comments(
        self,
        issue_key: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """Get comments on an issue."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        
        response = self._request(
            "GET",
            f"/issue/{issue_key}/comment",
            params={"maxResults": limit},
        )
        
        comments = []
        for c in response.get("comments", []):
            body = c.get("body", {})
            text = ""
            if isinstance(body, dict):
                content = body.get("content", [])
                for block in content:
                    for item in block.get("content", []):
                        if item.get("type") == "text":
                            text += item.get("text", "")
            
            comments.append({
                "id": c["id"],
                "author": c.get("author", {}).get("displayName"),
                "body": text,
                "created": c.get("created"),
            })
        
        return ToolResult.from_data({
            "issue_key": issue_key,
            "comments": comments,
            "count": len(comments),
        })
    
    def _add_comment(
        self,
        issue_key: Optional[str] = None,
        comment: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add a comment to an issue."""
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        if not comment:
            return ToolResult.from_error("comment is required")
        
        data = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": comment}]
                    }
                ]
            }
        }
        
        response = self._request("POST", f"/issue/{issue_key}/comment", data=data)
        
        return ToolResult.from_data({
            "issue_key": issue_key,
            "comment_id": response.get("id"),
            "added": True,
        })
    
    # Search
    def _search(
        self,
        jql: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """Search issues using JQL."""
        if not jql:
            return ToolResult.from_error("jql is required")
        
        response = self._request(
            "POST",
            "/search",
            data={
                "jql": jql,
                "maxResults": limit,
                "fields": ["summary", "status", "priority", "assignee", "created"],
            },
        )
        
        issues = []
        for issue in response.get("issues", []):
            fields = issue.get("fields", {})
            issues.append({
                "key": issue["key"],
                "summary": fields.get("summary"),
                "status": fields.get("status", {}).get("name"),
                "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
                "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
                "created": fields.get("created"),
            })
        
        return ToolResult.from_data({
            "jql": jql,
            "issues": issues,
            "total": response.get("total", 0),
            "count": len(issues),
        })
    
    # Sprint operations
    def _list_sprints(
        self,
        **kwargs,
    ) -> ToolResult:
        """List sprints (requires board_id)."""
        board_id = kwargs.get("board_id")
        if not board_id:
            return ToolResult.from_error("board_id is required")
        
        session = self._get_session()
        url = f"https://{self.domain}/rest/agile/1.0/board/{board_id}/sprint"
        
        response = session.get(url, params={"maxResults": 50})
        
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        
        sprints = [
            {
                "id": s["id"],
                "name": s["name"],
                "state": s.get("state"),
                "startDate": s.get("startDate"),
                "endDate": s.get("endDate"),
            }
            for s in data.get("values", [])
        ]
        
        return ToolResult.from_data({
            "board_id": board_id,
            "sprints": sprints,
            "count": len(sprints),
        })
    
    def _add_to_sprint(
        self,
        sprint_id: Optional[int] = None,
        issue_key: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add issue to a sprint."""
        if not sprint_id:
            return ToolResult.from_error("sprint_id is required")
        if not issue_key:
            return ToolResult.from_error("issue_key is required")
        
        session = self._get_session()
        url = f"https://{self.domain}/rest/agile/1.0/sprint/{sprint_id}/issue"
        
        response = session.post(url, json={"issues": [issue_key]})
        
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}")
        
        return ToolResult.from_data({
            "sprint_id": sprint_id,
            "issue_key": issue_key,
            "added": True,
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
                            "list_projects", "get_project",
                            "list_issues", "get_issue", "create_issue",
                            "update_issue", "delete_issue", "assign_issue",
                            "get_transitions", "transition_issue",
                            "get_comments", "add_comment",
                            "search",
                            "list_sprints", "add_to_sprint",
                        ],
                    },
                    "project_key": {"type": "string"},
                    "issue_key": {"type": "string"},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "issue_type": {"type": "string", "default": "Task"},
                    "priority": {"type": "string"},
                    "assignee": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "jql": {"type": "string"},
                    "transition_id": {"type": "string"},
                    "comment": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["action"],
            },
        }


class ConfluenceTool(BaseTool):
    """
    Confluence - Documentation and knowledge management.
    
    Features:
    - Page CRUD operations
    - Space management
    - Content search
    - Page hierarchy
    - Comments and attachments
    
    Example:
        >>> tool = ConfluenceTool(domain="company.atlassian.net")
        >>> result = tool(action="create_page", space_key="DEV", title="API Docs")
        >>> result = tool(action="search", query="installation guide")
    """
    
    name = "confluence"
    description = "Documentation and knowledge management with Confluence"
    category = ToolCategory.PRODUCTIVITY
    version = "1.0.0"
    
    def __init__(
        self,
        domain: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.domain = domain or os.getenv("CONFLUENCE_DOMAIN", os.getenv("JIRA_DOMAIN"))
        self.email = email or os.getenv("CONFLUENCE_EMAIL", os.getenv("JIRA_EMAIL"))
        self.api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN", os.getenv("JIRA_API_TOKEN"))
        
        self._session = None
        
        logger.debug(f"Confluence tool initialized (domain={self.domain})")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            
            auth_str = f"{self.email}:{self.api_token}"
            auth_bytes = base64.b64encode(auth_str.encode()).decode()
            
            self._session.headers.update({
                "Authorization": f"Basic {auth_bytes}",
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
        """Make API request to Confluence."""
        session = self._get_session()
        url = f"https://{self.domain}/wiki/api/v2{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            json=data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise Exception(error_data.get("message", f"HTTP {response.status_code}"))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        if response.status_code == 204:
            return {}
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_spaces",
        space_key: Optional[str] = None,
        space_id: Optional[str] = None,
        page_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 25,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Confluence operations."""
        actions = {
            # Space operations
            "list_spaces": self._list_spaces,
            "get_space": self._get_space,
            
            # Page operations
            "list_pages": self._list_pages,
            "get_page": self._get_page,
            "create_page": self._create_page,
            "update_page": self._update_page,
            "delete_page": self._delete_page,
            
            # Search
            "search": self._search,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.domain or not self.api_token:
                return ToolResult.from_error("Confluence credentials not configured")
            
            return actions[action](
                space_key=space_key,
                space_id=space_id,
                page_id=page_id,
                parent_id=parent_id,
                title=title,
                content=content,
                query=query,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Confluence error: {e}")
            return ToolResult.from_error(f"Confluence error: {e}")
    
    # Space operations
    def _list_spaces(self, limit: int = 25, **kwargs) -> ToolResult:
        """List all spaces."""
        response = self._request(
            "GET",
            "/spaces",
            params={"limit": limit},
        )
        
        spaces = [
            {
                "id": s["id"],
                "key": s["key"],
                "name": s["name"],
                "type": s.get("type"),
            }
            for s in response.get("results", [])
        ]
        
        return ToolResult.from_data({
            "spaces": spaces,
            "count": len(spaces),
        })
    
    def _get_space(
        self,
        space_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get space details."""
        if not space_id:
            return ToolResult.from_error("space_id is required")
        
        response = self._request("GET", f"/spaces/{space_id}")
        
        return ToolResult.from_data({"space": response})
    
    # Page operations
    def _list_pages(
        self,
        space_id: Optional[str] = None,
        limit: int = 25,
        **kwargs,
    ) -> ToolResult:
        """List pages in a space."""
        params = {"limit": limit}
        if space_id:
            params["space-id"] = space_id
        
        response = self._request("GET", "/pages", params=params)
        
        pages = [
            {
                "id": p["id"],
                "title": p["title"],
                "status": p.get("status"),
                "spaceId": p.get("spaceId"),
            }
            for p in response.get("results", [])
        ]
        
        return ToolResult.from_data({
            "pages": pages,
            "count": len(pages),
        })
    
    def _get_page(
        self,
        page_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get page details with content."""
        if not page_id:
            return ToolResult.from_error("page_id is required")
        
        response = self._request(
            "GET",
            f"/pages/{page_id}",
            params={"body-format": "storage"},
        )
        
        return ToolResult.from_data({
            "page": {
                "id": response.get("id"),
                "title": response.get("title"),
                "status": response.get("status"),
                "spaceId": response.get("spaceId"),
                "version": response.get("version", {}).get("number"),
                "body": response.get("body", {}).get("storage", {}).get("value"),
            }
        })
    
    def _create_page(
        self,
        space_id: Optional[str] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new page."""
        if not space_id:
            return ToolResult.from_error("space_id is required")
        if not title:
            return ToolResult.from_error("title is required")
        
        data = {
            "spaceId": space_id,
            "status": "current",
            "title": title,
            "body": {
                "representation": "storage",
                "value": content or "",
            },
        }
        
        if parent_id:
            data["parentId"] = parent_id
        
        response = self._request("POST", "/pages", data=data)
        
        return ToolResult.from_data({
            "page_id": response.get("id"),
            "title": response.get("title"),
            "created": True,
        })
    
    def _update_page(
        self,
        page_id: Optional[str] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a page."""
        if not page_id:
            return ToolResult.from_error("page_id is required")
        
        # Get current page to get version
        current = self._request("GET", f"/pages/{page_id}")
        current_version = current.get("version", {}).get("number", 1)
        
        data = {
            "id": page_id,
            "status": "current",
            "version": {"number": current_version + 1},
        }
        
        if title:
            data["title"] = title
        else:
            data["title"] = current.get("title")
        
        if content is not None:
            data["body"] = {
                "representation": "storage",
                "value": content,
            }
        
        response = self._request("PUT", f"/pages/{page_id}", data=data)
        
        return ToolResult.from_data({
            "page_id": response.get("id"),
            "version": response.get("version", {}).get("number"),
            "updated": True,
        })
    
    def _delete_page(
        self,
        page_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a page."""
        if not page_id:
            return ToolResult.from_error("page_id is required")
        
        self._request("DELETE", f"/pages/{page_id}")
        
        return ToolResult.from_data({
            "page_id": page_id,
            "deleted": True,
        })
    
    # Search
    def _search(
        self,
        query: Optional[str] = None,
        limit: int = 25,
        **kwargs,
    ) -> ToolResult:
        """Search content."""
        if not query:
            return ToolResult.from_error("query is required")
        
        # Use v1 API for search (v2 doesn't have full CQL support yet)
        session = self._get_session()
        url = f"https://{self.domain}/wiki/rest/api/content/search"
        
        response = session.get(
            url,
            params={"cql": f'text ~ "{query}"', "limit": limit},
        )
        
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        
        results = [
            {
                "id": r["id"],
                "title": r["title"],
                "type": r["type"],
                "space": r.get("space", {}).get("key"),
            }
            for r in data.get("results", [])
        ]
        
        return ToolResult.from_data({
            "query": query,
            "results": results,
            "count": len(results),
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
                            "list_spaces", "get_space",
                            "list_pages", "get_page", "create_page",
                            "update_page", "delete_page",
                            "search",
                        ],
                    },
                    "space_key": {"type": "string"},
                    "space_id": {"type": "string"},
                    "page_id": {"type": "string"},
                    "parent_id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 25},
                },
                "required": ["action"],
            },
        }
