"""
Wikipedia Tool for ARGUS.

Free access to Wikipedia articles.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class WikipediaTool(BaseTool):
    """
    Wikipedia search and article retrieval tool.
    
    Example:
        >>> tool = WikipediaTool()
        >>> result = tool(query="Artificial Intelligence", action="search")
    """
    
    name = "wikipedia"
    description = "Search Wikipedia and retrieve article content. Use action='search' to find articles, action='page' to get full content."
    category = ToolCategory.SEARCH
    
    def __init__(
        self,
        language: str = "en",
        sentences: int = 5,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.language = language
        self.sentences = sentences
    
    def execute(
        self,
        query: str = "",
        action: str = "search",
        sentences: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Search or retrieve Wikipedia content.
        
        Args:
            query: Search term or page title
            action: 'search' for search, 'page' for full page, 'summary' for summary
            sentences: Number of sentences for summary
            
        Returns:
            ToolResult with Wikipedia content
        """
        try:
            import wikipedia
            wikipedia.set_lang(self.language)
        except ImportError:
            return ToolResult.from_error(
                "wikipedia not installed. Run: pip install wikipedia"
            )
        
        if not query:
            return ToolResult.from_error("Query is required")
        
        try:
            if action == "search":
                results = wikipedia.search(query, results=10)
                return ToolResult.from_data({
                    "query": query,
                    "results": results,
                    "count": len(results),
                })
            
            elif action == "summary":
                num_sentences = sentences or self.sentences
                summary = wikipedia.summary(query, sentences=num_sentences)
                return ToolResult.from_data({
                    "title": query,
                    "summary": summary,
                    "sentences": num_sentences,
                })
            
            elif action == "page":
                page = wikipedia.page(query)
                return ToolResult.from_data({
                    "title": page.title,
                    "url": page.url,
                    "content": page.content[:5000],  # Truncate for safety
                    "summary": page.summary,
                    "categories": page.categories[:10],
                    "links": page.links[:20],
                })
            
            else:
                return ToolResult.from_error(f"Unknown action: {action}")
                
        except wikipedia.DisambiguationError as e:
            return ToolResult.from_data({
                "query": query,
                "disambiguation": True,
                "options": e.options[:10],
            })
        except wikipedia.PageError:
            return ToolResult.from_error(f"Page not found: {query}")
        except Exception as e:
            logger.error(f"Wikipedia error: {e}")
            return ToolResult.from_error(f"Wikipedia error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term or page title"},
                    "action": {
                        "type": "string",
                        "enum": ["search", "summary", "page"],
                        "description": "Action to perform",
                        "default": "search",
                    },
                    "sentences": {"type": "integer", "description": "Summary sentences"},
                },
                "required": ["query"],
            },
        }
