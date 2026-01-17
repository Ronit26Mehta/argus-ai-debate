"""
DuckDuckGo Search Tool for ARGUS.

Free, privacy-focused web search.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class DuckDuckGoTool(BaseTool):
    """
    DuckDuckGo search tool - free, no API key required.
    
    Example:
        >>> tool = DuckDuckGoTool()
        >>> result = tool(query="machine learning trends 2024")
        >>> print(result.data["results"])
    """
    
    name = "duckduckgo_search"
    description = "Search the web using DuckDuckGo. Returns URLs, titles, and snippets. Free and privacy-focused."
    category = ToolCategory.SEARCH
    
    def __init__(
        self,
        max_results: int = 10,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch
    
    def execute(
        self,
        query: str = "",
        max_results: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Search DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            ToolResult with search results
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult.from_error(
                "duckduckgo-search not installed. Run: pip install duckduckgo-search"
            )
        
        if not query:
            return ToolResult.from_error("Query is required")
        
        num_results = max_results or self.max_results
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=num_results,
                    region=self.region,
                    safesearch=self.safesearch,
                ))
            
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]
            
            return ToolResult.from_data({
                "query": query,
                "results": formatted,
                "count": len(formatted),
            })
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return ToolResult.from_error(f"Search failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        }
