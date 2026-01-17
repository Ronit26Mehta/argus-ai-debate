"""
Jina Reader Tool for ARGUS.

Convert URLs to LLM-friendly markdown.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class JinaReaderTool(BaseTool):
    """
    Jina Reader - convert web pages to clean markdown.
    
    Example:
        >>> tool = JinaReaderTool()
        >>> result = tool(url="https://example.com")
    """
    
    name = "jina_reader"
    description = "Convert web pages to LLM-friendly clean markdown. 1M free tokens."
    category = ToolCategory.DATA
    
    BASE_URL = "https://r.jina.ai"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        import os
        self.api_key = api_key or os.getenv("JINA_API_KEY")
    
    def execute(
        self,
        url: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """
        Convert URL to markdown.
        
        Args:
            url: URL to convert
            
        Returns:
            ToolResult with markdown content
        """
        try:
            import requests
        except ImportError:
            return ToolResult.from_error("requests not installed")
        
        if not url:
            return ToolResult.from_error("URL is required")
        
        try:
            headers = {"Accept": "text/markdown"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.BASE_URL}/{url}",
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            
            content = response.text[:15000]  # Truncate
            
            return ToolResult.from_data({
                "url": url,
                "content": content,
                "length": len(content),
            })
            
        except Exception as e:
            logger.error(f"Jina Reader failed: {e}")
            return ToolResult.from_error(f"Jina Reader failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to convert to markdown"},
                },
                "required": ["url"],
            },
        }
