"""
GitLab Tool for ARGUS.

Git repository and CI/CD management integration.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from urllib.parse import quote_plus

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class GitLabTool(BaseTool):
    """
    GitLab - Git repository and CI/CD management.
    
    Features:
    - Repository management
    - Issue tracking
    - Merge request management
    - CI/CD pipeline management
    - User and group management
    
    Example:
        >>> tool = GitLabTool()
        >>> result = tool(action="list_projects")
        >>> result = tool(action="create_issue", project_id="123", title="Bug report")
    """
    
    name = "gitlab"
    description = "Git repository and CI/CD management"
    category = ToolCategory.DEVELOPMENT
    version = "1.0.0"
    
    def __init__(
        self,
        access_token: Optional[str] = None,
        base_url: str = "https://gitlab.com",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.access_token = access_token or os.getenv("GITLAB_ACCESS_TOKEN")
        self.base_url = (base_url or os.getenv("GITLAB_URL", "https://gitlab.com")).rstrip("/")
        
        self._session = None
        
        if not self.access_token:
            logger.warning("No GitLab access token provided")
        
        logger.debug(f"GitLab tool initialized (url={self.base_url})")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "PRIVATE-TOKEN": self.access_token,
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
        """Make API request to GitLab."""
        session = self._get_session()
        url = f"{self.base_url}/api/v4{endpoint}"
        
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
    
    def _encode_project_path(self, project_id: str) -> str:
        """Encode project path for API use."""
        if not project_id.isdigit():
            return quote_plus(project_id)
        return project_id
    
    def execute(
        self,
        action: str = "list_projects",
        project_id: Optional[str] = None,
        issue_iid: Optional[int] = None,
        mr_iid: Optional[int] = None,
        pipeline_id: Optional[int] = None,
        job_id: Optional[int] = None,
        branch: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[list] = None,
        assignee_ids: Optional[list] = None,
        source_branch: Optional[str] = None,
        target_branch: Optional[str] = None,
        query: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute GitLab operations."""
        actions = {
            # Project operations
            "list_projects": self._list_projects,
            "get_project": self._get_project,
            "create_project": self._create_project,
            "delete_project": self._delete_project,
            
            # Issue operations
            "list_issues": self._list_issues,
            "get_issue": self._get_issue,
            "create_issue": self._create_issue,
            "update_issue": self._update_issue,
            "close_issue": self._close_issue,
            
            # Merge request operations
            "list_merge_requests": self._list_merge_requests,
            "get_merge_request": self._get_merge_request,
            "create_merge_request": self._create_merge_request,
            "merge_merge_request": self._merge_merge_request,
            "close_merge_request": self._close_merge_request,
            
            # Pipeline operations
            "list_pipelines": self._list_pipelines,
            "get_pipeline": self._get_pipeline,
            "create_pipeline": self._create_pipeline,
            "cancel_pipeline": self._cancel_pipeline,
            "retry_pipeline": self._retry_pipeline,
            
            # Job operations
            "list_jobs": self._list_jobs,
            "get_job": self._get_job,
            "retry_job": self._retry_job,
            "cancel_job": self._cancel_job,
            
            # Branch operations
            "list_branches": self._list_branches,
            "get_branch": self._get_branch,
            "create_branch": self._create_branch,
            "delete_branch": self._delete_branch,
            
            # Search
            "search": self._search,
            
            # User
            "me": self._me,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.access_token:
                return ToolResult.from_error("GitLab access token not configured")
            
            return actions[action](
                project_id=project_id,
                issue_iid=issue_iid,
                mr_iid=mr_iid,
                pipeline_id=pipeline_id,
                job_id=job_id,
                branch=branch,
                title=title,
                description=description,
                labels=labels,
                assignee_ids=assignee_ids,
                source_branch=source_branch,
                target_branch=target_branch,
                query=query,
                state=state,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"GitLab error: {e}")
            return ToolResult.from_error(f"GitLab error: {e}")
    
    # Project operations
    def _list_projects(
        self,
        query: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List projects."""
        params = {"per_page": min(limit, 100)}
        
        if query:
            params["search"] = query
        
        response = self._request("GET", "/projects", params=params)
        
        projects = [
            {
                "id": p["id"],
                "name": p["name"],
                "path_with_namespace": p["path_with_namespace"],
                "description": p.get("description"),
                "web_url": p.get("web_url"),
                "default_branch": p.get("default_branch"),
            }
            for p in response
        ]
        
        return ToolResult.from_data({
            "projects": projects,
            "count": len(projects),
        })
    
    def _get_project(
        self,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get project details."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("GET", f"/projects/{pid}")
        
        return ToolResult.from_data({
            "project": {
                "id": response.get("id"),
                "name": response.get("name"),
                "path_with_namespace": response.get("path_with_namespace"),
                "description": response.get("description"),
                "web_url": response.get("web_url"),
                "default_branch": response.get("default_branch"),
                "visibility": response.get("visibility"),
                "created_at": response.get("created_at"),
            }
        })
    
    def _create_project(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a project."""
        if not title:
            return ToolResult.from_error("title (name) is required")
        
        data = {"name": title}
        
        if description:
            data["description"] = description
        
        visibility = kwargs.get("visibility", "private")
        data["visibility"] = visibility
        
        response = self._request("POST", "/projects", data=data)
        
        return ToolResult.from_data({
            "project_id": response.get("id"),
            "name": response.get("name"),
            "web_url": response.get("web_url"),
            "created": True,
        })
    
    def _delete_project(
        self,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a project."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        pid = self._encode_project_path(project_id)
        self._request("DELETE", f"/projects/{pid}")
        
        return ToolResult.from_data({
            "project_id": project_id,
            "deleted": True,
        })
    
    # Issue operations
    def _list_issues(
        self,
        project_id: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[list] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List issues."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        pid = self._encode_project_path(project_id)
        params = {"per_page": min(limit, 100)}
        
        if state:
            params["state"] = state
        if labels:
            params["labels"] = ",".join(labels)
        
        response = self._request("GET", f"/projects/{pid}/issues", params=params)
        
        issues = [
            {
                "iid": i["iid"],
                "title": i["title"],
                "state": i["state"],
                "author": i.get("author", {}).get("username"),
                "assignees": [a.get("username") for a in i.get("assignees", [])],
                "labels": i.get("labels", []),
                "web_url": i.get("web_url"),
            }
            for i in response
        ]
        
        return ToolResult.from_data({
            "project_id": project_id,
            "issues": issues,
            "count": len(issues),
        })
    
    def _get_issue(
        self,
        project_id: Optional[str] = None,
        issue_iid: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Get issue details."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not issue_iid:
            return ToolResult.from_error("issue_iid is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("GET", f"/projects/{pid}/issues/{issue_iid}")
        
        return ToolResult.from_data({
            "issue": {
                "iid": response.get("iid"),
                "title": response.get("title"),
                "description": response.get("description"),
                "state": response.get("state"),
                "author": response.get("author", {}).get("username"),
                "assignees": [a.get("username") for a in response.get("assignees", [])],
                "labels": response.get("labels", []),
                "created_at": response.get("created_at"),
                "web_url": response.get("web_url"),
            }
        })
    
    def _create_issue(
        self,
        project_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[list] = None,
        assignee_ids: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Create an issue."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not title:
            return ToolResult.from_error("title is required")
        
        pid = self._encode_project_path(project_id)
        data = {"title": title}
        
        if description:
            data["description"] = description
        if labels:
            data["labels"] = ",".join(labels)
        if assignee_ids:
            data["assignee_ids"] = assignee_ids
        
        response = self._request("POST", f"/projects/{pid}/issues", data=data)
        
        return ToolResult.from_data({
            "issue_iid": response.get("iid"),
            "title": response.get("title"),
            "web_url": response.get("web_url"),
            "created": True,
        })
    
    def _update_issue(
        self,
        project_id: Optional[str] = None,
        issue_iid: Optional[int] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Update an issue."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not issue_iid:
            return ToolResult.from_error("issue_iid is required")
        
        pid = self._encode_project_path(project_id)
        data = {}
        
        if title:
            data["title"] = title
        if description is not None:
            data["description"] = description
        if labels:
            data["labels"] = ",".join(labels)
        
        if not data:
            return ToolResult.from_error("No fields to update")
        
        response = self._request("PUT", f"/projects/{pid}/issues/{issue_iid}", data=data)
        
        return ToolResult.from_data({
            "issue_iid": response.get("iid"),
            "updated": True,
        })
    
    def _close_issue(
        self,
        project_id: Optional[str] = None,
        issue_iid: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Close an issue."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not issue_iid:
            return ToolResult.from_error("issue_iid is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request(
            "PUT",
            f"/projects/{pid}/issues/{issue_iid}",
            data={"state_event": "close"},
        )
        
        return ToolResult.from_data({
            "issue_iid": response.get("iid"),
            "state": response.get("state"),
            "closed": True,
        })
    
    # Merge request operations
    def _list_merge_requests(
        self,
        project_id: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List merge requests."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        pid = self._encode_project_path(project_id)
        params = {"per_page": min(limit, 100)}
        
        if state:
            params["state"] = state
        
        response = self._request("GET", f"/projects/{pid}/merge_requests", params=params)
        
        mrs = [
            {
                "iid": mr["iid"],
                "title": mr["title"],
                "state": mr["state"],
                "source_branch": mr.get("source_branch"),
                "target_branch": mr.get("target_branch"),
                "author": mr.get("author", {}).get("username"),
                "web_url": mr.get("web_url"),
            }
            for mr in response
        ]
        
        return ToolResult.from_data({
            "project_id": project_id,
            "merge_requests": mrs,
            "count": len(mrs),
        })
    
    def _get_merge_request(
        self,
        project_id: Optional[str] = None,
        mr_iid: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Get merge request details."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not mr_iid:
            return ToolResult.from_error("mr_iid is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("GET", f"/projects/{pid}/merge_requests/{mr_iid}")
        
        return ToolResult.from_data({
            "merge_request": {
                "iid": response.get("iid"),
                "title": response.get("title"),
                "description": response.get("description"),
                "state": response.get("state"),
                "source_branch": response.get("source_branch"),
                "target_branch": response.get("target_branch"),
                "author": response.get("author", {}).get("username"),
                "merge_status": response.get("merge_status"),
                "web_url": response.get("web_url"),
            }
        })
    
    def _create_merge_request(
        self,
        project_id: Optional[str] = None,
        source_branch: Optional[str] = None,
        target_branch: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a merge request."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not source_branch:
            return ToolResult.from_error("source_branch is required")
        if not target_branch:
            return ToolResult.from_error("target_branch is required")
        if not title:
            return ToolResult.from_error("title is required")
        
        pid = self._encode_project_path(project_id)
        data = {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "title": title,
        }
        
        if description:
            data["description"] = description
        
        response = self._request("POST", f"/projects/{pid}/merge_requests", data=data)
        
        return ToolResult.from_data({
            "mr_iid": response.get("iid"),
            "title": response.get("title"),
            "web_url": response.get("web_url"),
            "created": True,
        })
    
    def _merge_merge_request(
        self,
        project_id: Optional[str] = None,
        mr_iid: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Merge a merge request."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not mr_iid:
            return ToolResult.from_error("mr_iid is required")
        
        pid = self._encode_project_path(project_id)
        
        data = {}
        squash = kwargs.get("squash")
        if squash:
            data["squash"] = True
        
        response = self._request(
            "PUT",
            f"/projects/{pid}/merge_requests/{mr_iid}/merge",
            data=data if data else None,
        )
        
        return ToolResult.from_data({
            "mr_iid": response.get("iid"),
            "state": response.get("state"),
            "merged": True,
        })
    
    def _close_merge_request(
        self,
        project_id: Optional[str] = None,
        mr_iid: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Close a merge request."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not mr_iid:
            return ToolResult.from_error("mr_iid is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request(
            "PUT",
            f"/projects/{pid}/merge_requests/{mr_iid}",
            data={"state_event": "close"},
        )
        
        return ToolResult.from_data({
            "mr_iid": response.get("iid"),
            "state": response.get("state"),
            "closed": True,
        })
    
    # Pipeline operations
    def _list_pipelines(
        self,
        project_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List pipelines."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request(
            "GET",
            f"/projects/{pid}/pipelines",
            params={"per_page": min(limit, 100)},
        )
        
        pipelines = [
            {
                "id": p["id"],
                "status": p["status"],
                "ref": p.get("ref"),
                "sha": p.get("sha", "")[:8],
                "web_url": p.get("web_url"),
            }
            for p in response
        ]
        
        return ToolResult.from_data({
            "project_id": project_id,
            "pipelines": pipelines,
            "count": len(pipelines),
        })
    
    def _get_pipeline(
        self,
        project_id: Optional[str] = None,
        pipeline_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Get pipeline details."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not pipeline_id:
            return ToolResult.from_error("pipeline_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("GET", f"/projects/{pid}/pipelines/{pipeline_id}")
        
        return ToolResult.from_data({
            "pipeline": {
                "id": response.get("id"),
                "status": response.get("status"),
                "ref": response.get("ref"),
                "sha": response.get("sha"),
                "duration": response.get("duration"),
                "created_at": response.get("created_at"),
                "web_url": response.get("web_url"),
            }
        })
    
    def _create_pipeline(
        self,
        project_id: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a pipeline."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not branch:
            return ToolResult.from_error("branch (ref) is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request(
            "POST",
            f"/projects/{pid}/pipeline",
            data={"ref": branch},
        )
        
        return ToolResult.from_data({
            "pipeline_id": response.get("id"),
            "status": response.get("status"),
            "web_url": response.get("web_url"),
            "created": True,
        })
    
    def _cancel_pipeline(
        self,
        project_id: Optional[str] = None,
        pipeline_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Cancel a pipeline."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not pipeline_id:
            return ToolResult.from_error("pipeline_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("POST", f"/projects/{pid}/pipelines/{pipeline_id}/cancel")
        
        return ToolResult.from_data({
            "pipeline_id": response.get("id"),
            "status": response.get("status"),
            "canceled": True,
        })
    
    def _retry_pipeline(
        self,
        project_id: Optional[str] = None,
        pipeline_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Retry a pipeline."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not pipeline_id:
            return ToolResult.from_error("pipeline_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("POST", f"/projects/{pid}/pipelines/{pipeline_id}/retry")
        
        return ToolResult.from_data({
            "pipeline_id": response.get("id"),
            "status": response.get("status"),
            "retried": True,
        })
    
    # Job operations
    def _list_jobs(
        self,
        project_id: Optional[str] = None,
        pipeline_id: Optional[int] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List jobs."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not pipeline_id:
            return ToolResult.from_error("pipeline_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request(
            "GET",
            f"/projects/{pid}/pipelines/{pipeline_id}/jobs",
            params={"per_page": min(limit, 100)},
        )
        
        jobs = [
            {
                "id": j["id"],
                "name": j["name"],
                "stage": j.get("stage"),
                "status": j["status"],
                "duration": j.get("duration"),
            }
            for j in response
        ]
        
        return ToolResult.from_data({
            "pipeline_id": pipeline_id,
            "jobs": jobs,
            "count": len(jobs),
        })
    
    def _get_job(
        self,
        project_id: Optional[str] = None,
        job_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Get job details."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not job_id:
            return ToolResult.from_error("job_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("GET", f"/projects/{pid}/jobs/{job_id}")
        
        return ToolResult.from_data({
            "job": {
                "id": response.get("id"),
                "name": response.get("name"),
                "stage": response.get("stage"),
                "status": response.get("status"),
                "duration": response.get("duration"),
                "started_at": response.get("started_at"),
                "finished_at": response.get("finished_at"),
                "web_url": response.get("web_url"),
            }
        })
    
    def _retry_job(
        self,
        project_id: Optional[str] = None,
        job_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Retry a job."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not job_id:
            return ToolResult.from_error("job_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("POST", f"/projects/{pid}/jobs/{job_id}/retry")
        
        return ToolResult.from_data({
            "job_id": response.get("id"),
            "status": response.get("status"),
            "retried": True,
        })
    
    def _cancel_job(
        self,
        project_id: Optional[str] = None,
        job_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Cancel a job."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not job_id:
            return ToolResult.from_error("job_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request("POST", f"/projects/{pid}/jobs/{job_id}/cancel")
        
        return ToolResult.from_data({
            "job_id": response.get("id"),
            "status": response.get("status"),
            "canceled": True,
        })
    
    # Branch operations
    def _list_branches(
        self,
        project_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List branches."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        
        pid = self._encode_project_path(project_id)
        response = self._request(
            "GET",
            f"/projects/{pid}/repository/branches",
            params={"per_page": min(limit, 100)},
        )
        
        branches = [
            {
                "name": b["name"],
                "default": b.get("default", False),
                "protected": b.get("protected", False),
                "commit_sha": b.get("commit", {}).get("short_id"),
            }
            for b in response
        ]
        
        return ToolResult.from_data({
            "project_id": project_id,
            "branches": branches,
            "count": len(branches),
        })
    
    def _get_branch(
        self,
        project_id: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get branch details."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not branch:
            return ToolResult.from_error("branch is required")
        
        pid = self._encode_project_path(project_id)
        branch_encoded = quote_plus(branch)
        response = self._request("GET", f"/projects/{pid}/repository/branches/{branch_encoded}")
        
        return ToolResult.from_data({
            "branch": {
                "name": response.get("name"),
                "default": response.get("default", False),
                "protected": response.get("protected", False),
                "commit": response.get("commit"),
            }
        })
    
    def _create_branch(
        self,
        project_id: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a branch."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not branch:
            return ToolResult.from_error("branch is required")
        
        ref = kwargs.get("ref", "main")
        
        pid = self._encode_project_path(project_id)
        response = self._request(
            "POST",
            f"/projects/{pid}/repository/branches",
            data={"branch": branch, "ref": ref},
        )
        
        return ToolResult.from_data({
            "branch": response.get("name"),
            "created": True,
        })
    
    def _delete_branch(
        self,
        project_id: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a branch."""
        if not project_id:
            return ToolResult.from_error("project_id is required")
        if not branch:
            return ToolResult.from_error("branch is required")
        
        pid = self._encode_project_path(project_id)
        branch_encoded = quote_plus(branch)
        self._request("DELETE", f"/projects/{pid}/repository/branches/{branch_encoded}")
        
        return ToolResult.from_data({
            "branch": branch,
            "deleted": True,
        })
    
    # Search
    def _search(
        self,
        query: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Search across GitLab."""
        if not query:
            return ToolResult.from_error("query is required")
        
        scope = kwargs.get("scope", "projects")
        
        response = self._request(
            "GET",
            "/search",
            params={"scope": scope, "search": query},
        )
        
        return ToolResult.from_data({
            "query": query,
            "scope": scope,
            "results": response if isinstance(response, list) else [],
            "count": len(response) if isinstance(response, list) else 0,
        })
    
    # User
    def _me(self, **kwargs) -> ToolResult:
        """Get current user."""
        response = self._request("GET", "/user")
        
        return ToolResult.from_data({
            "user": {
                "id": response.get("id"),
                "username": response.get("username"),
                "name": response.get("name"),
                "email": response.get("email"),
                "state": response.get("state"),
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
                            "list_projects", "get_project", "create_project", "delete_project",
                            "list_issues", "get_issue", "create_issue", "update_issue", "close_issue",
                            "list_merge_requests", "get_merge_request", "create_merge_request",
                            "merge_merge_request", "close_merge_request",
                            "list_pipelines", "get_pipeline", "create_pipeline",
                            "cancel_pipeline", "retry_pipeline",
                            "list_jobs", "get_job", "retry_job", "cancel_job",
                            "list_branches", "get_branch", "create_branch", "delete_branch",
                            "search", "me",
                        ],
                    },
                    "project_id": {"type": "string"},
                    "issue_iid": {"type": "integer"},
                    "mr_iid": {"type": "integer"},
                    "pipeline_id": {"type": "integer"},
                    "job_id": {"type": "integer"},
                    "branch": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "source_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "query": {"type": "string"},
                    "state": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
