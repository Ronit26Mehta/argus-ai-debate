"""
Brave Search Tool for ARGUS.

Free, privacy-focused web search.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class BraveTool(BaseTool):
    """
    Brave Search tool.
    
    Example:
        >>> tool = BraveTool(api_key="BSA...")
        >>> result = tool(query="latest tech news")
    """
    
    name = "brave_search"
    description = "Search the web using Brave Search. Privacy-focused with free tier."
    category = ToolCategory.SEARCH
    
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
    
    def execute(
        self,
        query: str = "",
        count: int = 10,
        freshness: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Search with Brave.
        
        Args:
            query: Search query
            count: Number of results
            freshness: Time filter (pd=past_day, pw=past_week, pm=past_month)
            
        Returns:
            ToolResult with search results
        """
        try:
            import requests
        except ImportError:
            return ToolResult.from_error("requests not installed")
        
        if not self.api_key:
            return ToolResult.from_error("BRAVE_API_KEY not set")
        
        if not query:
            return ToolResult.from_error("Query is required")
        
        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            }
            params = {"q": query, "count": count}
            if freshness:
                params["freshness"] = freshness
            
            response = requests.get(self.BASE_URL, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for r in data.get("web", {}).get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                })
            
            return ToolResult.from_data({
                "query": query,
                "results": results,
                "count": len(results),
            })
            
        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return ToolResult.from_error(f"Brave search failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "default": 10},
                    "freshness": {"type": "string", "enum": ["pd", "pw", "pm", "py"]},
                },
                "required": ["query"],
            },
        }
