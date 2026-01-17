"""
ARGUS MCP Integration Module.

Model Context Protocol server and client for exposing ARGUS
capabilities and connecting to external MCP servers.

Example:
    >>> from argus.mcp import ArgusServer, MCPClient
    >>> 
    >>> # Create and run an MCP server
    >>> server = ArgusServer(name="my-argus")
    >>> @server.tool()
    ... def search(query: str) -> str:
    ...     return f"Results for: {query}"
    >>> 
    >>> # Connect to external MCP server
    >>> client = MCPClient()
    >>> client.connect("python", "-m", "external_server")
    >>> tools = client.list_tools()
"""

from argus.mcp.config import (
    MCPServerConfig,
    MCPClientConfig,
    TransportType,
    MCPToolSchema,
    MCPResourceSchema,
    get_default_server_config,
    get_default_client_config,
)

from argus.mcp.server import (
    ArgusServer,
    MCPRequest,
    MCPResponse,
)

from argus.mcp.client import (
    MCPClient,
    MCPToolInfo,
    MCPResourceInfo,
)

from argus.mcp.tools import (
    ToolAdapter,
    ToolSchemaGenerator,
    MCPToolWrapper,
    ToolRegistry,
)

from argus.mcp.resources import (
    BaseResourceAdapter,
    CDAGResource,
    EvidenceResource,
    ProvenanceResource,
    ConfigResource,
    ResourceRegistry,
)

__all__ = [
    # Config
    "MCPServerConfig",
    "MCPClientConfig",
    "TransportType",
    "MCPToolSchema",
    "MCPResourceSchema",
    "get_default_server_config",
    "get_default_client_config",
    # Server
    "ArgusServer",
    "MCPRequest",
    "MCPResponse",
    # Client
    "MCPClient",
    "MCPToolInfo",
    "MCPResourceInfo",
    # Tools
    "ToolAdapter",
    "ToolSchemaGenerator",
    "MCPToolWrapper",
    "ToolRegistry",
    # Resources
    "BaseResourceAdapter",
    "CDAGResource",
    "EvidenceResource",
    "ProvenanceResource",
    "ConfigResource",
    "ResourceRegistry",
]
