"""
Exa Search Tool for ARGUS.

Neural search with 1000 free searches/month.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class ExaTool(BaseTool):
    """
    Exa neural search tool.
    
    Example:
        >>> tool = ExaTool(api_key="exa-...")
        >>> result = tool(query="best machine learning papers", num_results=5)
    """
    
    name = "exa_search"
    description = "Neural web search optimized for AI. Returns URLs, titles, authors, and published dates."
    category = ToolCategory.SEARCH
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.getenv("EXA_API_KEY")
    
    def execute(
        self,
        query: str = "",
        num_results: int = 10,
        use_autoprompt: bool = True,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Search with Exa.
        
        Args:
            query: Search query
            num_results: Number of results
            use_autoprompt: Optimize query for neural search
            include_domains: Only include these domains
            exclude_domains: Exclude these domains
            
        Returns:
            ToolResult with search results
        """
        try:
            from exa_py import Exa
        except ImportError:
            return ToolResult.from_error(
                "exa-py not installed. Run: pip install exa-py"
            )
        
        if not self.api_key:
            return ToolResult.from_error("EXA_API_KEY not set")
        
        if not query:
            return ToolResult.from_error("Query is required")
        
        try:
            exa = Exa(api_key=self.api_key)
            
            search_kwargs = {
                "query": query,
                "num_results": num_results,
                "use_autoprompt": use_autoprompt,
            }
            if include_domains:
                search_kwargs["include_domains"] = include_domains
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
            
            response = exa.search(**search_kwargs)
            
            results = [
                {
                    "title": r.title,
                    "url": r.url,
                    "author": getattr(r, 'author', None),
                    "published_date": getattr(r, 'published_date', None),
                    "score": getattr(r, 'score', None),
                }
                for r in response.results
            ]
            
            return ToolResult.from_data({
                "query": query,
                "results": results,
                "count": len(results),
            })
            
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return ToolResult.from_error(f"Exa search failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 10},
                    "use_autoprompt": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
        }
