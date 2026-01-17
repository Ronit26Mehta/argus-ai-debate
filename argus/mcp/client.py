"""
ARGUS MCP Client.

Connect to external MCP servers and invoke their tools/resources.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Any, List, Dict

from argus.mcp.config import MCPClientConfig, TransportType

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """Information about a remote MCP tool."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPResourceInfo:
    """Information about a remote MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: str


class MCPClient:
    """Client for connecting to MCP servers.
    
    Example:
        >>> client = MCPClient()
        >>> client.connect("python", "-m", "some_mcp_server")
        >>> tools = client.list_tools()
        >>> result = client.call_tool("add", a=1, b=2)
    """
    
    def __init__(self, config: Optional[MCPClientConfig] = None):
        self.config = config or MCPClientConfig()
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._connected = False
        self._server_info: Dict[str, Any] = {}
        self._tools: Dict[str, MCPToolInfo] = {}
        self._resources: Dict[str, MCPResourceInfo] = {}
    
    def connect(self, *command: str) -> bool:
        """Connect to an MCP server via stdio."""
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1
            )
            response = self._send_request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
            if response and "result" in response:
                self._server_info = response["result"]
                self._connected = True
                self._send_request("notifications/initialized", {})
                logger.info(f"Connected to MCP server: {self._server_info.get('serverInfo', {}).get('name', 'unknown')}")
                return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
        return False
    
    def _send_request(self, method: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Send a JSON-RPC request."""
        if not self._process:
            return None
        self._request_id += 1
        request = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            if "notifications" not in method:
                response_line = self._process.stdout.readline()
                if response_line:
                    return json.loads(response_line)
        except Exception as e:
            logger.error(f"Request failed: {e}")
        return None
    
    def list_tools(self) -> List[MCPToolInfo]:
        """List available tools from the server."""
        response = self._send_request("tools/list", {})
        if response and "result" in response:
            tools = response["result"].get("tools", [])
            self._tools = {
                t["name"]: MCPToolInfo(name=t["name"], description=t.get("description", ""),
                                       input_schema=t.get("inputSchema", {}))
                for t in tools
            }
            return list(self._tools.values())
        return []
    
    def call_tool(self, name: str, **arguments: Any) -> Any:
        """Call a tool on the remote server."""
        response = self._send_request("tools/call", {"name": name, "arguments": arguments})
        if response and "result" in response:
            content = response["result"].get("content", [])
            if content and content[0].get("type") == "text":
                try:
                    return json.loads(content[0]["text"])
                except json.JSONDecodeError:
                    return content[0]["text"]
        if response and "error" in response:
            raise RuntimeError(response["error"].get("message", "Unknown error"))
        return None
    
    def list_resources(self) -> List[MCPResourceInfo]:
        """List available resources from the server."""
        response = self._send_request("resources/list", {})
        if response and "result" in response:
            resources = response["result"].get("resources", [])
            self._resources = {
                r["uri"]: MCPResourceInfo(uri=r["uri"], name=r["name"],
                                          description=r.get("description", ""),
                                          mime_type=r.get("mimeType", "application/json"))
                for r in resources
            }
            return list(self._resources.values())
        return []
    
    def read_resource(self, uri: str) -> Any:
        """Read a resource from the server."""
        response = self._send_request("resources/read", {"uri": uri})
        if response and "result" in response:
            contents = response["result"].get("contents", [])
            if contents:
                text = contents[0].get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return None
    
    def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
        self._connected = False
        logger.info("Disconnected from MCP server")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def __enter__(self) -> "MCPClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.disconnect()
