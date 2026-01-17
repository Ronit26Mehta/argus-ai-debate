"""
Tavily Search Tool for ARGUS.

AI-optimized search with 1000 free searches/month.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class TavilyTool(BaseTool):
    """
    Tavily AI-optimized search tool.
    
    Example:
        >>> tool = TavilyTool(api_key="tvly-...")
        >>> result = tool(query="latest AI research 2024", search_depth="advanced")
    """
    
    name = "tavily_search"
    description = "AI-optimized web search with answers. Returns content, images, and direct answers. 1000 free/month."
    category = ToolCategory.SEARCH
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "basic",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.search_depth = search_depth
    
    def execute(
        self,
        query: str = "",
        search_depth: Optional[str] = None,
        include_answer: bool = True,
        include_images: bool = False,
        max_results: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Search with Tavily.
        
        Args:
            query: Search query
            search_depth: 'basic' or 'advanced'
            include_answer: Include AI-generated answer
            include_images: Include image results
            max_results: Max results
            
        Returns:
            ToolResult with search results
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            return ToolResult.from_error(
                "tavily-python not installed. Run: pip install tavily-python"
            )
        
        if not self.api_key:
            return ToolResult.from_error("TAVILY_API_KEY not set")
        
        if not query:
            return ToolResult.from_error("Query is required")
        
        try:
            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query=query,
                search_depth=search_depth or self.search_depth,
                include_answer=include_answer,
                include_images=include_images,
                max_results=max_results,
            )
            
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                }
                for r in response.get("results", [])
            ]
            
            return ToolResult.from_data({
                "query": query,
                "answer": response.get("answer"),
                "results": results,
                "images": response.get("images", []),
                "count": len(results),
            })
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return ToolResult.from_error(f"Tavily search failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "search_depth": {"type": "string", "enum": ["basic", "advanced"]},
                    "include_answer": {"type": "boolean", "default": True},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        }
