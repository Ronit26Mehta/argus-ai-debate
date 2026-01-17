"""
Web Scraper Tool for ARGUS.

Extract content from web pages.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class WebScraperTool(BaseTool):
    """
    Web scraping tool using BeautifulSoup.
    
    Example:
        >>> tool = WebScraperTool()
        >>> result = tool(url="https://example.com", extract="text")
    """
    
    name = "web_scraper"
    description = "Extract content from web pages. Can extract text, links, or specific elements."
    category = ToolCategory.DATA
    
    def __init__(
        self,
        timeout: float = 30.0,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.timeout = timeout
    
    def execute(
        self,
        url: str = "",
        extract: str = "text",
        selector: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Scrape web page.
        
        Args:
            url: URL to scrape
            extract: What to extract ('text', 'links', 'html', 'selector')
            selector: CSS selector if extract='selector'
            
        Returns:
            ToolResult with extracted content
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return ToolResult.from_error(
                "Required packages not installed. Run: pip install requests beautifulsoup4"
            )
        
        if not url:
            return ToolResult.from_error("URL is required")
        
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "ARGUS-Agent/1.0"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            if extract == "text":
                text = soup.get_text(separator="\n", strip=True)
                content = text[:10000]  # Truncate
                
            elif extract == "links":
                links = []
                for a in soup.find_all("a", href=True):
                    links.append({
                        "text": a.get_text(strip=True),
                        "href": a["href"],
                    })
                content = links[:100]  # Limit
                
            elif extract == "html":
                content = str(soup)[:10000]
                
            elif extract == "selector" and selector:
                elements = soup.select(selector)
                content = [el.get_text(strip=True) for el in elements[:50]]
                
            else:
                content = soup.get_text(separator="\n", strip=True)[:5000]
            
            return ToolResult.from_data({
                "url": url,
                "extract_type": extract,
                "content": content,
                "title": soup.title.string if soup.title else None,
            })
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return ToolResult.from_error(f"Scraping failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "extract": {"type": "string", "enum": ["text", "links", "html", "selector"]},
                    "selector": {"type": "string", "description": "CSS selector"},
                },
                "required": ["url"],
            },
        }
