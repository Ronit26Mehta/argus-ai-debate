"""Web tools for ARGUS."""

from argus.tools.integrations.web.requests_tool import RequestsTool
from argus.tools.integrations.web.scraper import WebScraperTool
from argus.tools.integrations.web.jina_reader import JinaReaderTool
from argus.tools.integrations.web.youtube import YouTubeTool

__all__ = [
    "RequestsTool",
    "WebScraperTool",
    "JinaReaderTool",
    "YouTubeTool",
]
