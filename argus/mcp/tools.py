"""
ARGUS MCP Tool Adapters.

Convert between ARGUS tools and MCP tool formats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict, TYPE_CHECKING

from argus.mcp.config import MCPToolSchema

if TYPE_CHECKING:
    from argus.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolAdapter:
    """Convert ARGUS BaseTool to MCP tool format."""
    
    @staticmethod
    def to_mcp_schema(tool: "BaseTool") -> MCPToolSchema:
        """Convert ARGUS tool to MCP schema."""
        schema = tool.get_schema() if hasattr(tool, 'get_schema') else {}
        return MCPToolSchema(
            name=tool.name,
            description=tool.description,
            input_schema=schema,
        )
    
    @staticmethod
    def create_mcp_handler(tool: "BaseTool"):
        """Create MCP-compatible handler from ARGUS tool."""
        def handler(**kwargs: Any) -> Dict[str, Any]:
            try:
                result = tool.execute(**kwargs)
                if result.success:
                    return {"success": True, "data": result.data}
                return {"success": False, "error": result.error}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return handler


class ToolSchemaGenerator:
    """Generate JSON schemas for MCP tools."""
    
    @staticmethod
    def from_function(func, name: Optional[str] = None, description: Optional[str] = None) -> MCPToolSchema:
        """Generate schema from Python function."""
        import inspect
        sig = inspect.signature(func)
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            ptype = "string"
            if param.annotation != inspect.Parameter.empty:
                ann = param.annotation
                if ann == int:
                    ptype = "integer"
                elif ann == float:
                    ptype = "number"
                elif ann == bool:
                    ptype = "boolean"
                elif ann == list:
                    ptype = "array"
                elif ann == dict:
                    ptype = "object"
            properties[pname] = {"type": ptype}
            if param.default == inspect.Parameter.empty:
                required.append(pname)
        return MCPToolSchema(
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            input_schema={"type": "object", "properties": properties, "required": required},
        )


class MCPToolWrapper:
    """Wrap an MCP tool as an ARGUS-compatible tool."""
    
    def __init__(self, name: str, description: str, client: Any, input_schema: Optional[dict] = None):
        self.name = name
        self.description = description
        self._client = client
        self._input_schema = input_schema or {}
    
    def execute(self, **kwargs: Any) -> "ToolResult":
        """Execute the remote MCP tool."""
        from argus.tools.base import ToolResult
        try:
            result = self._client.call_tool(self.name, **kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def get_schema(self) -> dict[str, Any]:
        return self._input_schema
    
    def __call__(self, **kwargs: Any) -> "ToolResult":
        return self.execute(**kwargs)


class ToolRegistry:
    """Registry for managing tool conversions."""
    
    def __init__(self):
        self._argus_tools: Dict[str, "BaseTool"] = {}
        self._mcp_wrappers: Dict[str, MCPToolWrapper] = {}
    
    def register_argus_tool(self, tool: "BaseTool") -> MCPToolSchema:
        """Register ARGUS tool and return MCP schema."""
        self._argus_tools[tool.name] = tool
        return ToolAdapter.to_mcp_schema(tool)
    
    def register_mcp_tool(self, name: str, description: str, client: Any, schema: dict = None) -> MCPToolWrapper:
        """Register external MCP tool as ARGUS tool."""
        wrapper = MCPToolWrapper(name, description, client, schema)
        self._mcp_wrappers[name] = wrapper
        return wrapper
    
    def get_argus_tool(self, name: str) -> Optional["BaseTool"]:
        return self._argus_tools.get(name)
    
    def get_mcp_wrapper(self, name: str) -> Optional[MCPToolWrapper]:
        return self._mcp_wrappers.get(name)
