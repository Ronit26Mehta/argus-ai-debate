"""
ARGUS Tool Integrations.

Comprehensive collection of pre-built tools for agents.
Includes 50+ integrations across multiple categories.
"""

# ==============================================================================
# Search Tools
# ==============================================================================
from argus.tools.integrations.search import (
    DuckDuckGoTool,
    WikipediaTool,
    ArxivTool,
    TavilyTool,
    BraveTool,
    ExaTool,
)

# ==============================================================================
# Web Tools
# ==============================================================================
from argus.tools.integrations.web import (
    RequestsTool,
    WebScraperTool,
    JinaReaderTool,
    YouTubeTool,
)

# ==============================================================================
# Productivity Tools (Core)
# ==============================================================================
from argus.tools.integrations.productivity import (
    FileSystemTool,
    PythonReplTool,
    ShellTool,
    GitHubTool,
    JsonTool,
)

# ==============================================================================
# Database Tools
# ==============================================================================
from argus.tools.integrations.database import (
    SqlTool,
    PandasTool,
)

# ==============================================================================
# Finance Tools
# ==============================================================================
from argus.tools.integrations.finance import (
    YahooFinanceTool,
    WeatherTool,
)

# ==============================================================================
# AI Agent Tools
# ==============================================================================
from argus.tools.integrations.ai_agents import (
    AgentMailTool,
    AgentOpsTool,
    GoodMemTool,
    FreeplayTool,
)

# ==============================================================================
# Cloud Tools
# ==============================================================================
from argus.tools.integrations.cloud import (
    BigQueryTool,
    PubSubTool,
    CloudTraceTool,
    VertexAISearchTool,
    VertexAIRAGTool,
)

# ==============================================================================
# Vector Database Tools
# ==============================================================================
from argus.tools.integrations.vectordb import (
    ChromaTool,
    PineconeTool,
    QdrantTool,
    MongoDBTool,
)

# ==============================================================================
# Productivity Tools (Extended)
# ==============================================================================
from argus.tools.integrations.productivity_tools import (
    AsanaTool,
    JiraTool,
    ConfluenceTool,
    LinearTool,
    NotionTool,
)

# ==============================================================================
# Communication & Payment Tools
# ==============================================================================
from argus.tools.integrations.communication import (
    MailgunTool,
    StripeTool,
    PayPalTool,
)

# ==============================================================================
# DevOps Tools
# ==============================================================================
from argus.tools.integrations.devops import (
    GitLabTool,
    PostmanTool,
    DaytonaTool,
    N8nTool,
)

# ==============================================================================
# Media & AI Tools
# ==============================================================================
from argus.tools.integrations.media_ai import (
    ElevenLabsTool,
    CartesiaTool,
    HuggingFaceTool,
)

# ==============================================================================
# Observability Tools
# ==============================================================================
from argus.tools.integrations.observability import (
    ArizeTool,
    PhoenixTool,
    MonocleTool,
    MLflowTool,
    WandBWeaveTool,
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
    # Productivity (Core)
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
    # AI Agents
    "AgentMailTool",
    "AgentOpsTool",
    "GoodMemTool",
    "FreeplayTool",
    # Cloud
    "BigQueryTool",
    "PubSubTool",
    "CloudTraceTool",
    "VertexAISearchTool",
    "VertexAIRAGTool",
    # Vector DB
    "ChromaTool",
    "PineconeTool",
    "QdrantTool",
    "MongoDBTool",
    # Productivity (Extended)
    "AsanaTool",
    "JiraTool",
    "ConfluenceTool",
    "LinearTool",
    "NotionTool",
    # Communication
    "MailgunTool",
    "StripeTool",
    "PayPalTool",
    # DevOps
    "GitLabTool",
    "PostmanTool",
    "DaytonaTool",
    "N8nTool",
    # Media & AI
    "ElevenLabsTool",
    "CartesiaTool",
    "HuggingFaceTool",
    # Observability
    "ArizeTool",
    "PhoenixTool",
    "MonocleTool",
    "MLflowTool",
    "WandBWeaveTool",
]


# Tool categories for organization
TOOL_CATEGORIES = {
    "search": [
        DuckDuckGoTool, WikipediaTool, ArxivTool,
        TavilyTool, BraveTool, ExaTool,
    ],
    "web": [
        RequestsTool, WebScraperTool, JinaReaderTool, YouTubeTool,
    ],
    "productivity": [
        FileSystemTool, PythonReplTool, ShellTool, GitHubTool, JsonTool,
    ],
    "database": [
        SqlTool, PandasTool,
    ],
    "finance": [
        YahooFinanceTool, WeatherTool,
    ],
    "ai_agents": [
        AgentMailTool, AgentOpsTool, GoodMemTool, FreeplayTool,
    ],
    "cloud": [
        BigQueryTool, PubSubTool, CloudTraceTool,
        VertexAISearchTool, VertexAIRAGTool,
    ],
    "vectordb": [
        ChromaTool, PineconeTool, QdrantTool, MongoDBTool,
    ],
    "productivity_extended": [
        AsanaTool, JiraTool, ConfluenceTool, LinearTool, NotionTool,
    ],
    "communication": [
        MailgunTool, StripeTool, PayPalTool,
    ],
    "devops": [
        GitLabTool, PostmanTool, DaytonaTool, N8nTool,
    ],
    "media_ai": [
        ElevenLabsTool, CartesiaTool, HuggingFaceTool,
    ],
    "observability": [
        ArizeTool, PhoenixTool, MonocleTool, MLflowTool, WandBWeaveTool,
    ],
}


def list_all_tools() -> list[str]:
    """List all available integration tools."""
    return __all__


def list_tool_categories() -> list[str]:
    """List available tool categories."""
    return list(TOOL_CATEGORIES.keys())


def get_tools_by_category(category: str) -> list:
    """Get tool classes for a specific category."""
    if category not in TOOL_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(TOOL_CATEGORIES.keys())}")
    return TOOL_CATEGORIES[category]


def get_all_tools():
    """Get instances of all tools."""
    all_tools = []
    for category_tools in TOOL_CATEGORIES.values():
        for tool_class in category_tools:
            all_tools.append(tool_class())
    return all_tools


def get_tool_count() -> int:
    """Get total count of available tools."""
    return len(__all__)
