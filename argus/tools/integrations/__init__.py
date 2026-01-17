"""
ARGUS Tool Integrations.

Comprehensive collection of pre-built tools for agents.
"""

from argus.tools.integrations.search import (
    DuckDuckGoTool,
    WikipediaTool,
    ArxivTool,
    TavilyTool,
    BraveTool,
    ExaTool,
)
from argus.tools.integrations.web import (
    RequestsTool,
    WebScraperTool,
    JinaReaderTool,
    YouTubeTool,
)
from argus.tools.integrations.productivity import (
    FileSystemTool,
    PythonReplTool,
    ShellTool,
    GitHubTool,
    JsonTool,
)
from argus.tools.integrations.database import (
    SqlTool,
    PandasTool,
)
from argus.tools.integrations.finance import (
    YahooFinanceTool,
    WeatherTool,
)

__all__ = [
    # Search
    "DuckDuckGoTool",
    "WikipediaTool", 
    "ArxivTool",
    "TavilyTool",
    "BraveTool",
    "ExaTool",
    # Web
    "RequestsTool",
    "WebScraperTool",
    "JinaReaderTool",
    "YouTubeTool",
    # Productivity
    "FileSystemTool",
    "PythonReplTool",
    "ShellTool",
    "GitHubTool",
    "JsonTool",
    # Database
    "SqlTool",
    "PandasTool",
    # Finance
    "YahooFinanceTool",
    "WeatherTool",
]


def list_all_tools() -> list[str]:
    """List all available integration tools."""
    return __all__


def get_all_tools():
    """Get instances of all tools."""
    return [
        DuckDuckGoTool(),
        WikipediaTool(),
        ArxivTool(),
        TavilyTool(),
        BraveTool(),
        ExaTool(),
        RequestsTool(),
        WebScraperTool(),
        JinaReaderTool(),
        YouTubeTool(),
        FileSystemTool(),
        PythonReplTool(),
        ShellTool(),
        GitHubTool(),
        JsonTool(),
        SqlTool(),
        PandasTool(),
        YahooFinanceTool(),
        WeatherTool(),
    ]
