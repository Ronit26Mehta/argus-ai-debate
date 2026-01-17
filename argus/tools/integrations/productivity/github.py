"""
GitHub Tool for ARGUS.

Interact with GitHub repositories.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class GitHubTool(BaseTool):
    """
    GitHub API tool.
    
    Example:
        >>> tool = GitHubTool(token="ghp_...")
        >>> result = tool(action="search_repos", query="machine learning")
    """
    
    name = "github"
    description = "Search repositories, get issues, read files, and interact with GitHub."
    category = ToolCategory.EXTERNAL_API
    
    def __init__(
        self,
        token: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.token = token or os.getenv("GITHUB_TOKEN")
    
    def execute(
        self,
        action: str = "search_repos",
        query: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Perform GitHub operation.
        
        Args:
            action: 'search_repos', 'get_repo', 'list_issues', 'get_file', 'list_prs'
            query: Search query
            owner: Repository owner
            repo: Repository name
            path: File path in repository
            
        Returns:
            ToolResult with GitHub data
        """
        try:
            from github import Github, GithubException
        except ImportError:
            return ToolResult.from_error(
                "PyGithub not installed. Run: pip install PyGithub"
            )
        
        try:
            g = Github(self.token) if self.token else Github()
            
            if action == "search_repos":
                if not query:
                    return ToolResult.from_error("Query required for search")
                repos = g.search_repositories(query, sort="stars")
                results = [
                    {
                        "name": r.full_name,
                        "url": r.html_url,
                        "description": r.description,
                        "stars": r.stargazers_count,
                        "language": r.language,
                    }
                    for r in list(repos)[:10]
                ]
                return ToolResult.from_data({"query": query, "results": results})
            
            elif action == "get_repo":
                if not owner or not repo:
                    return ToolResult.from_error("owner and repo required")
                r = g.get_repo(f"{owner}/{repo}")
                return ToolResult.from_data({
                    "name": r.full_name,
                    "description": r.description,
                    "url": r.html_url,
                    "stars": r.stargazers_count,
                    "forks": r.forks_count,
                    "language": r.language,
                    "topics": r.get_topics(),
                })
            
            elif action == "list_issues":
                if not owner or not repo:
                    return ToolResult.from_error("owner and repo required")
                r = g.get_repo(f"{owner}/{repo}")
                issues = [
                    {
                        "number": i.number,
                        "title": i.title,
                        "state": i.state,
                        "url": i.html_url,
                        "labels": [l.name for l in i.labels],
                    }
                    for i in list(r.get_issues(state="open"))[:20]
                ]
                return ToolResult.from_data({"repo": f"{owner}/{repo}", "issues": issues})
            
            elif action == "get_file":
                if not owner or not repo or not path:
                    return ToolResult.from_error("owner, repo, and path required")
                r = g.get_repo(f"{owner}/{repo}")
                content = r.get_contents(path)
                if isinstance(content, list):
                    return ToolResult.from_error("Path is a directory")
                return ToolResult.from_data({
                    "path": path,
                    "content": content.decoded_content.decode("utf-8")[:10000],
                    "size": content.size,
                })
            
            else:
                return ToolResult.from_error(f"Unknown action: {action}")
                
        except GithubException as e:
            return ToolResult.from_error(f"GitHub API error: {e}")
        except Exception as e:
            logger.error(f"GitHub error: {e}")
            return ToolResult.from_error(f"GitHub error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search_repos", "get_repo", "list_issues", "get_file"]},
                    "query": {"type": "string"},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["action"],
            },
        }
