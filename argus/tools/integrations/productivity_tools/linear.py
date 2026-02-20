"""
Linear Tool for ARGUS.

Issue tracking and project management for engineering teams.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class LinearTool(BaseTool):
    """
    Linear - Issue tracking for engineering teams.
    
    Features:
    - Issue CRUD operations
    - Project management
    - Team management
    - Cycle/sprint tracking
    - Label management
    - GraphQL-based API
    
    Example:
        >>> tool = LinearTool()
        >>> result = tool(action="create_issue", team_id="TEAM123", title="Bug fix")
        >>> result = tool(action="list_issues", team_id="TEAM123")
    """
    
    name = "linear"
    description = "Issue tracking and project management for engineering teams"
    category = ToolCategory.PRODUCTIVITY
    version = "1.0.0"
    
    GRAPHQL_URL = "https://api.linear.app/graphql"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("LINEAR_API_KEY")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Linear API key provided")
        
        logger.debug("Linear tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            })
        return self._session
    
    def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute GraphQL query."""
        session = self._get_session()
        
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        response = session.post(self.GRAPHQL_URL, json=payload)
        
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        data = response.json()
        
        if "errors" in data:
            errors = [e.get("message", str(e)) for e in data["errors"]]
            raise Exception("; ".join(errors))
        
        return data.get("data", {})
    
    def execute(
        self,
        action: str = "list_teams",
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        issue_id: Optional[str] = None,
        cycle_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[int] = None,
        state_id: Optional[str] = None,
        assignee_id: Optional[str] = None,
        label_ids: Optional[list] = None,
        query: Optional[str] = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Linear operations."""
        actions = {
            # Team operations
            "list_teams": self._list_teams,
            "get_team": self._get_team,
            
            # Issue operations
            "list_issues": self._list_issues,
            "get_issue": self._get_issue,
            "create_issue": self._create_issue,
            "update_issue": self._update_issue,
            "delete_issue": self._delete_issue,
            
            # Project operations
            "list_projects": self._list_projects,
            "get_project": self._get_project,
            "create_project": self._create_project,
            
            # Cycle operations
            "list_cycles": self._list_cycles,
            "get_active_cycle": self._get_active_cycle,
            
            # State operations
            "list_states": self._list_states,
            
            # Label operations
            "list_labels": self._list_labels,
            "create_label": self._create_label,
            
            # Search
            "search": self._search,
            
            # User operations
            "list_users": self._list_users,
            "me": self._me,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("Linear API key not configured")
            
            return actions[action](
                team_id=team_id,
                project_id=project_id,
                issue_id=issue_id,
                cycle_id=cycle_id,
                title=title,
                description=description,
                priority=priority,
                state_id=state_id,
                assignee_id=assignee_id,
                label_ids=label_ids,
                query=query,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Linear error: {e}")
            return ToolResult.from_error(f"Linear error: {e}")
    
    # Team operations
    def _list_teams(self, limit: int = 50, **kwargs) -> ToolResult:
        """List all teams."""
        query = """
        query Teams($first: Int) {
            teams(first: $first) {
                nodes {
                    id
                    name
                    key
                    description
                }
            }
        }
        """
        
        data = self._graphql(query, {"first": limit})
        
        teams = [
            {
                "id": t["id"],
                "name": t["name"],
                "key": t["key"],
                "description": t.get("description"),
            }
            for t in data.get("teams", {}).get("nodes", [])
        ]
        
        return ToolResult.from_data({
            "teams": teams,
            "count": len(teams),
        })
    
    def _get_team(
        self,
        team_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get team details."""
        if not team_id:
            return ToolResult.from_error("team_id is required")
        
        query = """
        query Team($id: String!) {
            team(id: $id) {
                id
                name
                key
                description
                issueCount
                members {
                    nodes {
                        id
                        name
                        email
                    }
                }
            }
        }
        """
        
        data = self._graphql(query, {"id": team_id})
        
        return ToolResult.from_data({"team": data.get("team")})
    
    # Issue operations
    def _list_issues(
        self,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        cycle_id: Optional[str] = None,
        assignee_id: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """List issues."""
        filters = []
        
        if team_id:
            filters.append(f'team: {{ id: {{ eq: "{team_id}" }} }}')
        if project_id:
            filters.append(f'project: {{ id: {{ eq: "{project_id}" }} }}')
        if cycle_id:
            filters.append(f'cycle: {{ id: {{ eq: "{cycle_id}" }} }}')
        if assignee_id:
            filters.append(f'assignee: {{ id: {{ eq: "{assignee_id}" }} }}')
        
        filter_str = ", ".join(filters) if filters else ""
        filter_clause = f"filter: {{ {filter_str} }}" if filter_str else ""
        
        query = f"""
        query Issues($first: Int) {{
            issues(first: $first {", " + filter_clause if filter_clause else ""}) {{
                nodes {{
                    id
                    identifier
                    title
                    description
                    priority
                    state {{
                        id
                        name
                    }}
                    assignee {{
                        id
                        name
                    }}
                    createdAt
                    updatedAt
                }}
            }}
        }}
        """
        
        data = self._graphql(query, {"first": limit})
        
        issues = [
            {
                "id": i["id"],
                "identifier": i["identifier"],
                "title": i["title"],
                "description": i.get("description"),
                "priority": i.get("priority"),
                "state": i.get("state", {}).get("name"),
                "assignee": i.get("assignee", {}).get("name") if i.get("assignee") else None,
                "createdAt": i.get("createdAt"),
            }
            for i in data.get("issues", {}).get("nodes", [])
        ]
        
        return ToolResult.from_data({
            "issues": issues,
            "count": len(issues),
        })
    
    def _get_issue(
        self,
        issue_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get issue details."""
        if not issue_id:
            return ToolResult.from_error("issue_id is required")
        
        query = """
        query Issue($id: String!) {
            issue(id: $id) {
                id
                identifier
                title
                description
                priority
                estimate
                state {
                    id
                    name
                }
                assignee {
                    id
                    name
                    email
                }
                team {
                    id
                    name
                }
                project {
                    id
                    name
                }
                cycle {
                    id
                    name
                }
                labels {
                    nodes {
                        id
                        name
                        color
                    }
                }
                comments {
                    nodes {
                        id
                        body
                        user {
                            name
                        }
                        createdAt
                    }
                }
                createdAt
                updatedAt
            }
        }
        """
        
        data = self._graphql(query, {"id": issue_id})
        
        return ToolResult.from_data({"issue": data.get("issue")})
    
    def _create_issue(
        self,
        team_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[int] = None,
        state_id: Optional[str] = None,
        assignee_id: Optional[str] = None,
        project_id: Optional[str] = None,
        cycle_id: Optional[str] = None,
        label_ids: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new issue."""
        if not team_id:
            return ToolResult.from_error("team_id is required")
        if not title:
            return ToolResult.from_error("title is required")
        
        mutation = """
        mutation CreateIssue($input: IssueCreateInput!) {
            issueCreate(input: $input) {
                success
                issue {
                    id
                    identifier
                    title
                    url
                }
            }
        }
        """
        
        input_data = {
            "teamId": team_id,
            "title": title,
        }
        
        if description:
            input_data["description"] = description
        if priority is not None:
            input_data["priority"] = priority
        if state_id:
            input_data["stateId"] = state_id
        if assignee_id:
            input_data["assigneeId"] = assignee_id
        if project_id:
            input_data["projectId"] = project_id
        if cycle_id:
            input_data["cycleId"] = cycle_id
        if label_ids:
            input_data["labelIds"] = label_ids
        
        data = self._graphql(mutation, {"input": input_data})
        
        result = data.get("issueCreate", {})
        
        return ToolResult.from_data({
            "success": result.get("success"),
            "issue": result.get("issue"),
            "created": True,
        })
    
    def _update_issue(
        self,
        issue_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[int] = None,
        state_id: Optional[str] = None,
        assignee_id: Optional[str] = None,
        label_ids: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Update an issue."""
        if not issue_id:
            return ToolResult.from_error("issue_id is required")
        
        mutation = """
        mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
            issueUpdate(id: $id, input: $input) {
                success
                issue {
                    id
                    identifier
                    title
                }
            }
        }
        """
        
        input_data = {}
        
        if title:
            input_data["title"] = title
        if description is not None:
            input_data["description"] = description
        if priority is not None:
            input_data["priority"] = priority
        if state_id:
            input_data["stateId"] = state_id
        if assignee_id:
            input_data["assigneeId"] = assignee_id
        if label_ids:
            input_data["labelIds"] = label_ids
        
        if not input_data:
            return ToolResult.from_error("No fields to update")
        
        data = self._graphql(mutation, {"id": issue_id, "input": input_data})
        
        result = data.get("issueUpdate", {})
        
        return ToolResult.from_data({
            "success": result.get("success"),
            "issue": result.get("issue"),
            "updated": True,
        })
    
    def _delete_issue(
        self,
        issue_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete (archive) an issue."""
        if not issue_id:
            return ToolResult.from_error("issue_id is required")
        
        mutation = """
        mutation ArchiveIssue($id: String!) {
            issueArchive(id: $id) {
                success
            }
        }
        """
        
        data = self._graphql(mutation, {"id": issue_id})
        
        return ToolResult.from_data({
            "issue_id": issue_id,
            "success": data.get("issueArchive", {}).get("success"),
            "archived": True,
        })
    
    # Project operations
    def _list_projects(
        self,
        team_id: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """List projects."""
        filter_clause = ""
        if team_id:
            filter_clause = f', filter: {{ accessibleTeams: {{ id: {{ eq: "{team_id}" }} }} }}'
        
        query = f"""
        query Projects($first: Int) {{
            projects(first: $first{filter_clause}) {{
                nodes {{
                    id
                    name
                    description
                    state
                    progress
                    targetDate
                }}
            }}
        }}
        """
        
        data = self._graphql(query, {"first": limit})
        
        projects = [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p.get("description"),
                "state": p.get("state"),
                "progress": p.get("progress"),
            }
            for p in data.get("projects", {}).get("nodes", [])
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
        
        query = """
        query Project($id: String!) {
            project(id: $id) {
                id
                name
                description
                state
                progress
                targetDate
                startDate
                issues {
                    nodes {
                        id
                        identifier
                        title
                    }
                }
                teams {
                    nodes {
                        id
                        name
                    }
                }
            }
        }
        """
        
        data = self._graphql(query, {"id": project_id})
        
        return ToolResult.from_data({"project": data.get("project")})
    
    def _create_project(
        self,
        team_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new project."""
        if not title:
            return ToolResult.from_error("title is required")
        
        mutation = """
        mutation CreateProject($input: ProjectCreateInput!) {
            projectCreate(input: $input) {
                success
                project {
                    id
                    name
                }
            }
        }
        """
        
        input_data = {"name": title}
        
        if description:
            input_data["description"] = description
        if team_id:
            input_data["teamIds"] = [team_id]
        
        data = self._graphql(mutation, {"input": input_data})
        
        result = data.get("projectCreate", {})
        
        return ToolResult.from_data({
            "success": result.get("success"),
            "project": result.get("project"),
            "created": True,
        })
    
    # Cycle operations
    def _list_cycles(
        self,
        team_id: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """List cycles."""
        filter_clause = ""
        if team_id:
            filter_clause = f', filter: {{ team: {{ id: {{ eq: "{team_id}" }} }} }}'
        
        query = f"""
        query Cycles($first: Int) {{
            cycles(first: $first{filter_clause}) {{
                nodes {{
                    id
                    name
                    number
                    startsAt
                    endsAt
                    progress
                    issueCountScope
                }}
            }}
        }}
        """
        
        data = self._graphql(query, {"first": limit})
        
        cycles = [
            {
                "id": c["id"],
                "name": c.get("name"),
                "number": c.get("number"),
                "startsAt": c.get("startsAt"),
                "endsAt": c.get("endsAt"),
                "progress": c.get("progress"),
            }
            for c in data.get("cycles", {}).get("nodes", [])
        ]
        
        return ToolResult.from_data({
            "cycles": cycles,
            "count": len(cycles),
        })
    
    def _get_active_cycle(
        self,
        team_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get the active cycle for a team."""
        if not team_id:
            return ToolResult.from_error("team_id is required")
        
        query = """
        query Team($id: String!) {
            team(id: $id) {
                activeCycle {
                    id
                    name
                    number
                    startsAt
                    endsAt
                    progress
                }
            }
        }
        """
        
        data = self._graphql(query, {"id": team_id})
        
        return ToolResult.from_data({
            "team_id": team_id,
            "cycle": data.get("team", {}).get("activeCycle"),
        })
    
    # State operations
    def _list_states(
        self,
        team_id: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """List workflow states."""
        filter_clause = ""
        if team_id:
            filter_clause = f', filter: {{ team: {{ id: {{ eq: "{team_id}" }} }} }}'
        
        query = f"""
        query States($first: Int) {{
            workflowStates(first: $first{filter_clause}) {{
                nodes {{
                    id
                    name
                    color
                    type
                    position
                }}
            }}
        }}
        """
        
        data = self._graphql(query, {"first": limit})
        
        states = [
            {
                "id": s["id"],
                "name": s["name"],
                "color": s.get("color"),
                "type": s.get("type"),
            }
            for s in data.get("workflowStates", {}).get("nodes", [])
        ]
        
        return ToolResult.from_data({
            "states": states,
            "count": len(states),
        })
    
    # Label operations
    def _list_labels(
        self,
        team_id: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """List labels."""
        filter_clause = ""
        if team_id:
            filter_clause = f', filter: {{ team: {{ id: {{ eq: "{team_id}" }} }} }}'
        
        query = f"""
        query Labels($first: Int) {{
            issueLabels(first: $first{filter_clause}) {{
                nodes {{
                    id
                    name
                    color
                    description
                }}
            }}
        }}
        """
        
        data = self._graphql(query, {"first": limit})
        
        labels = [
            {
                "id": l["id"],
                "name": l["name"],
                "color": l.get("color"),
                "description": l.get("description"),
            }
            for l in data.get("issueLabels", {}).get("nodes", [])
        ]
        
        return ToolResult.from_data({
            "labels": labels,
            "count": len(labels),
        })
    
    def _create_label(
        self,
        team_id: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a label."""
        if not team_id:
            return ToolResult.from_error("team_id is required")
        if not title:
            return ToolResult.from_error("title (name) is required")
        
        mutation = """
        mutation CreateLabel($input: IssueLabelCreateInput!) {
            issueLabelCreate(input: $input) {
                success
                issueLabel {
                    id
                    name
                    color
                }
            }
        }
        """
        
        input_data = {
            "teamId": team_id,
            "name": title,
        }
        
        color = kwargs.get("color")
        if color:
            input_data["color"] = color
        
        data = self._graphql(mutation, {"input": input_data})
        
        result = data.get("issueLabelCreate", {})
        
        return ToolResult.from_data({
            "success": result.get("success"),
            "label": result.get("issueLabel"),
            "created": True,
        })
    
    # Search
    def _search(
        self,
        query: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> ToolResult:
        """Search issues."""
        if not query:
            return ToolResult.from_error("query is required")
        
        gql_query = """
        query SearchIssues($query: String!, $first: Int) {
            searchIssues(query: $query, first: $first) {
                nodes {
                    id
                    identifier
                    title
                    state {
                        name
                    }
                    assignee {
                        name
                    }
                }
            }
        }
        """
        
        data = self._graphql(gql_query, {"query": query, "first": limit})
        
        issues = [
            {
                "id": i["id"],
                "identifier": i["identifier"],
                "title": i["title"],
                "state": i.get("state", {}).get("name"),
                "assignee": i.get("assignee", {}).get("name") if i.get("assignee") else None,
            }
            for i in data.get("searchIssues", {}).get("nodes", [])
        ]
        
        return ToolResult.from_data({
            "query": query,
            "issues": issues,
            "count": len(issues),
        })
    
    # User operations
    def _list_users(self, limit: int = 50, **kwargs) -> ToolResult:
        """List users."""
        query = """
        query Users($first: Int) {
            users(first: $first) {
                nodes {
                    id
                    name
                    email
                    active
                }
            }
        }
        """
        
        data = self._graphql(query, {"first": limit})
        
        users = [
            {
                "id": u["id"],
                "name": u["name"],
                "email": u.get("email"),
                "active": u.get("active"),
            }
            for u in data.get("users", {}).get("nodes", [])
        ]
        
        return ToolResult.from_data({
            "users": users,
            "count": len(users),
        })
    
    def _me(self, **kwargs) -> ToolResult:
        """Get current user."""
        query = """
        query Me {
            viewer {
                id
                name
                email
                teams {
                    nodes {
                        id
                        name
                    }
                }
            }
        }
        """
        
        data = self._graphql(query)
        
        return ToolResult.from_data({"user": data.get("viewer")})
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_teams", "get_team",
                            "list_issues", "get_issue", "create_issue",
                            "update_issue", "delete_issue",
                            "list_projects", "get_project", "create_project",
                            "list_cycles", "get_active_cycle",
                            "list_states",
                            "list_labels", "create_label",
                            "search",
                            "list_users", "me",
                        ],
                    },
                    "team_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "issue_id": {"type": "string"},
                    "cycle_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 0, "maximum": 4},
                    "state_id": {"type": "string"},
                    "assignee_id": {"type": "string"},
                    "label_ids": {"type": "array", "items": {"type": "string"}},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["action"],
            },
        }
