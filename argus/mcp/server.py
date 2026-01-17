"""
ARGUS MCP Server.

Build MCP-compliant servers exposing ARGUS tools and resources.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable, Dict, List, TYPE_CHECKING

from argus.mcp.config import MCPServerConfig, TransportType, MCPToolSchema, MCPResourceSchema

if TYPE_CHECKING:
    from argus.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """MCP JSON-RPC request."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """MCP JSON-RPC response."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[dict[str, Any]] = None
    
    def to_dict(self) -> dict[str, Any]:
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


class ArgusServer:
    """MCP Server for ARGUS.
    
    Exposes ARGUS tools and resources via Model Context Protocol.
    
    Example:
        >>> server = ArgusServer(name="my-argus-server")
        >>> 
        >>> @server.tool()
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> 
        >>> @server.resource("config://settings")
        ... def get_settings() -> dict:
        ...     return {"debug": False}
        >>> 
        >>> server.run()
    """
    
    def __init__(self, config: Optional[MCPServerConfig] = None, name: str = "argus-server"):
        self.config = config or MCPServerConfig(name=name)
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, MCPToolSchema] = {}
        self._resources: Dict[str, Callable] = {}
        self._resource_schemas: Dict[str, MCPResourceSchema] = {}
        self._running = False
        logger.debug(f"ArgusServer '{self.config.name}' initialized")
    
    def tool(self, name: Optional[str] = None, description: str = "") -> Callable:
        """Decorator to register a function as an MCP tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
            self._tools[tool_name] = func
            schema = self._generate_tool_schema(func, tool_name, tool_desc)
            self._tool_schemas[tool_name] = schema
            logger.debug(f"Registered tool: {tool_name}")
            return func
        return decorator
    
    def resource(self, uri: str, name: Optional[str] = None, description: str = "") -> Callable:
        """Decorator to register a function as an MCP resource."""
        def decorator(func: Callable) -> Callable:
            res_name = name or func.__name__
            res_desc = description or func.__doc__ or f"Resource: {res_name}"
            self._resources[uri] = func
            self._resource_schemas[uri] = MCPResourceSchema(
                uri=uri, name=res_name, description=res_desc
            )
            logger.debug(f"Registered resource: {uri}")
            return func
        return decorator
    
    def register_argus_tool(self, tool: "BaseTool") -> None:
        """Register an existing ARGUS tool."""
        def wrapper(**kwargs: Any) -> Any:
            result = tool.execute(**kwargs)
            return result.data if result.success else {"error": result.error}
        self._tools[tool.name] = wrapper
        self._tool_schemas[tool.name] = MCPToolSchema(
            name=tool.name, description=tool.description,
            input_schema=tool.get_schema() if hasattr(tool, 'get_schema') else {}
        )
        logger.debug(f"Registered ARGUS tool: {tool.name}")
    
    def _generate_tool_schema(self, func: Callable, name: str, description: str) -> MCPToolSchema:
        """Generate JSON schema from function signature."""
        import inspect
        sig = inspect.signature(func)
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            ptype = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    ptype = "integer"
                elif param.annotation == float:
                    ptype = "number"
                elif param.annotation == bool:
                    ptype = "boolean"
            properties[pname] = {"type": ptype}
            if param.default == inspect.Parameter.empty:
                required.append(pname)
        return MCPToolSchema(
            name=name, description=description,
            input_schema={"type": "object", "properties": properties, "required": required}
        )
    
    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request."""
        try:
            if request.method == "initialize":
                return self._handle_initialize(request)
            elif request.method == "tools/list":
                return self._handle_tools_list(request)
            elif request.method == "tools/call":
                return self._handle_tools_call(request)
            elif request.method == "resources/list":
                return self._handle_resources_list(request)
            elif request.method == "resources/read":
                return self._handle_resources_read(request)
            else:
                return MCPResponse(id=request.id, error={"code": -32601, "message": f"Unknown method: {request.method}"})
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return MCPResponse(id=request.id, error={"code": -32603, "message": str(e)})
    
    def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        return MCPResponse(id=request.id, result={
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {} if self.config.enable_tools else None,
                "resources": {} if self.config.enable_resources else None,
            },
            "serverInfo": {"name": self.config.name, "version": self.config.version}
        })
    
    def _handle_tools_list(self, request: MCPRequest) -> MCPResponse:
        tools = [{"name": s.name, "description": s.description, "inputSchema": s.input_schema}
                 for s in self._tool_schemas.values()]
        return MCPResponse(id=request.id, result={"tools": tools})
    
    def _handle_tools_call(self, request: MCPRequest) -> MCPResponse:
        name = request.params.get("name")
        args = request.params.get("arguments", {})
        if name not in self._tools:
            return MCPResponse(id=request.id, error={"code": -32602, "message": f"Unknown tool: {name}"})
        try:
            result = self._tools[name](**args)
            return MCPResponse(id=request.id, result={"content": [{"type": "text", "text": json.dumps(result)}]})
        except Exception as e:
            return MCPResponse(id=request.id, error={"code": -32000, "message": str(e)})
    
    def _handle_resources_list(self, request: MCPRequest) -> MCPResponse:
        resources = [{"uri": s.uri, "name": s.name, "description": s.description, "mimeType": s.mime_type}
                     for s in self._resource_schemas.values()]
        return MCPResponse(id=request.id, result={"resources": resources})
    
    def _handle_resources_read(self, request: MCPRequest) -> MCPResponse:
        uri = request.params.get("uri")
        if uri not in self._resources:
            return MCPResponse(id=request.id, error={"code": -32602, "message": f"Unknown resource: {uri}"})
        try:
            result = self._resources[uri]()
            return MCPResponse(id=request.id, result={"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(result)}]})
        except Exception as e:
            return MCPResponse(id=request.id, error={"code": -32000, "message": str(e)})
    
    def run(self) -> None:
        """Run the MCP server (STDIO transport)."""
        logger.info(f"Starting MCP server: {self.config.name}")
        self._running = True
        if self.config.transport == TransportType.STDIO:
            self._run_stdio()
    
    def _run_stdio(self) -> None:
        """Run with stdio transport."""
        while self._running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                request_data = json.loads(line)
                request = MCPRequest(**request_data)
                response = self.handle_request(request)
                print(json.dumps(response.to_dict()), flush=True)
            except json.JSONDecodeError:
                continue
            except KeyboardInterrupt:
                break
    
    def stop(self) -> None:
        """Stop the server."""
        self._running = False
