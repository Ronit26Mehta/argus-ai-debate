"""
ARGUS MCP Server Configuration.

Configuration for Model Context Protocol server and client.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Any, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TransportType(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"     # Standard I/O
    HTTP = "http"       # HTTP/REST
    SSE = "sse"         # Server-Sent Events


class MCPServerConfig(BaseModel):
    """Configuration for MCP server."""
    name: str = Field(default="argus-mcp-server", description="Server name")
    version: str = Field(default="1.0.0", description="Server version")
    transport: TransportType = Field(default=TransportType.STDIO)
    host: str = Field(default="localhost")
    port: int = Field(default=8080, ge=1, le=65535)
    enable_tools: bool = Field(default=True)
    enable_resources: bool = Field(default=True)
    enable_prompts: bool = Field(default=False)
    auto_register_tools: bool = Field(default=True)
    timeout: float = Field(default=30.0, ge=1.0)


class MCPClientConfig(BaseModel):
    """Configuration for MCP client."""
    server_url: Optional[str] = Field(default=None)
    transport: TransportType = Field(default=TransportType.STDIO)
    timeout: float = Field(default=30.0, ge=1.0)
    retry_count: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.1)


class MCPToolSchema(BaseModel):
    """Schema for an MCP tool."""
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[dict[str, Any]] = None


class MCPResourceSchema(BaseModel):
    """Schema for an MCP resource."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "application/json"


def get_default_server_config() -> MCPServerConfig:
    return MCPServerConfig()


def get_default_client_config() -> MCPClientConfig:
    return MCPClientConfig()
