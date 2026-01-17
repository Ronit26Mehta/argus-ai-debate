"""
ARGUS Tools Framework.

Provides an extensible framework for creating and registering custom tools
that agents can use during debate and evidence gathering.

Features:
    - User-defined tool creation via BaseTool subclassing
    - Automatic tool registration and discovery
    - Result caching for expensive operations
    - Guardrails for content filtering and policy enforcement
    - 19+ pre-built tool integrations (search, web, productivity, database, finance)

Example:
    >>> from argus.tools import BaseTool, ToolRegistry, ToolExecutor
    >>> 
    >>> # Create a custom tool
    >>> class MySearchTool(BaseTool):
    ...     name = "web_search"
    ...     description = "Search the web for information"
    ...     
    ...     def execute(self, query: str, **kwargs) -> ToolResult:
    ...         # Implement search logic
    ...         return ToolResult(success=True, data={"results": [...]})
    >>> 
    >>> # Register and use
    >>> registry = ToolRegistry()
    >>> registry.register(MySearchTool())
    >>> 
    >>> executor = ToolExecutor(registry)
    >>> result = executor.run("web_search", query="climate change effects")
    
    >>> # Or use pre-built tools
    >>> from argus.tools.integrations import DuckDuckGoTool, WikipediaTool
    >>> search = DuckDuckGoTool()
    >>> result = search(query="AI research")
"""

from argus.tools.base import (
    BaseTool,
    ToolResult,
    ToolConfig,
    ToolCategory,
)
from argus.tools.registry import (
    ToolRegistry,
    get_default_registry,
    register_tool,
    get_tool,
    list_tools,
)
from argus.tools.executor import (
    ToolExecutor,
    ExecutorConfig,
)
from argus.tools.cache import (
    ResultCache,
    CacheConfig,
    cached_tool,
)
from argus.tools.guardrails import (
    Guardrail,
    GuardrailConfig,
    ContentFilter,
    PolicyEnforcer,
    GuardrailResult,
)

# Import integrations for convenience
from argus.tools import integrations

__all__ = [
    # Base
    "BaseTool",
    "ToolResult",
    "ToolConfig",
    "ToolCategory",
    # Registry
    "ToolRegistry",
    "get_default_registry",
    "register_tool",
    "get_tool",
    "list_tools",
    # Executor
    "ToolExecutor",
    "ExecutorConfig",
    # Cache
    "ResultCache",
    "CacheConfig",
    "cached_tool",
    # Guardrails
    "Guardrail",
    "GuardrailConfig",
    "ContentFilter",
    "PolicyEnforcer",
    "GuardrailResult",
    # Integrations module
    "integrations",
]

