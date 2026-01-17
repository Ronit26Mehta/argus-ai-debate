"""
ArXiv Tool for ARGUS.

Free access to scientific papers.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class ArxivTool(BaseTool):
    """
    ArXiv paper search tool.
    
    Example:
        >>> tool = ArxivTool()
        >>> result = tool(query="transformer neural networks")
    """
    
    name = "arxiv"
    description = "Search ArXiv for scientific papers. Returns titles, authors, abstracts, and PDF links."
    category = ToolCategory.SEARCH
    
    def __init__(
        self,
        max_results: int = 10,
        sort_by: str = "relevance",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.max_results = max_results
        self.sort_by = sort_by
    
    def execute(
        self,
        query: str = "",
        max_results: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Search ArXiv for papers.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            ToolResult with paper results
        """
        try:
            import arxiv
        except ImportError:
            return ToolResult.from_error(
                "arxiv not installed. Run: pip install arxiv"
            )
        
        if not query:
            return ToolResult.from_error("Query is required")
        
        num_results = max_results or self.max_results
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=num_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            
            results = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:5]],
                    "summary": paper.summary[:500],
                    "published": paper.published.isoformat() if paper.published else None,
                    "pdf_url": paper.pdf_url,
                    "entry_id": paper.entry_id,
                    "categories": paper.categories,
                })
            
            return ToolResult.from_data({
                "query": query,
                "results": results,
                "count": len(results),
            })
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return ToolResult.from_error(f"ArXiv search failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for papers"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        }
