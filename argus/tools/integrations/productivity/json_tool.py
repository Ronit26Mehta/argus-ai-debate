"""
JSON Tool for ARGUS.

Parse, validate, and manipulate JSON data.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class JsonTool(BaseTool):
    """
    JSON manipulation tool.
    
    Example:
        >>> tool = JsonTool()
        >>> result = tool(action="parse", data='{"key": "value"}')
    """
    
    name = "json_tool"
    description = "Parse, validate, format, and query JSON data."
    category = ToolCategory.UTILITY
    
    def __init__(self, config: Optional[ToolConfig] = None):
        super().__init__(config)
    
    def execute(
        self,
        action: str = "parse",
        data: str = "",
        path: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Perform JSON operation.
        
        Args:
            action: 'parse', 'validate', 'format', 'get', 'set'
            data: JSON string
            path: JSONPath-like path (e.g., 'data.items[0].name')
            value: Value for set operations
            
        Returns:
            ToolResult with result
        """
        if action == "validate":
            try:
                json.loads(data)
                return ToolResult.from_data({"valid": True})
            except json.JSONDecodeError as e:
                return ToolResult.from_data({"valid": False, "error": str(e)})
        
        elif action == "parse":
            try:
                parsed = json.loads(data)
                return ToolResult.from_data({"parsed": parsed})
            except json.JSONDecodeError as e:
                return ToolResult.from_error(f"Invalid JSON: {e}")
        
        elif action == "format":
            try:
                parsed = json.loads(data)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                return ToolResult.from_data({"formatted": formatted})
            except json.JSONDecodeError as e:
                return ToolResult.from_error(f"Invalid JSON: {e}")
        
        elif action == "get":
            if not path:
                return ToolResult.from_error("Path required for get")
            try:
                parsed = json.loads(data)
                result = self._get_path(parsed, path)
                return ToolResult.from_data({"path": path, "value": result})
            except Exception as e:
                return ToolResult.from_error(f"Get error: {e}")
        
        elif action == "stringify":
            try:
                obj = json.loads(data) if isinstance(data, str) else data
                result = json.dumps(obj, ensure_ascii=False)
                return ToolResult.from_data({"string": result})
            except Exception as e:
                return ToolResult.from_error(f"Stringify error: {e}")
        
        else:
            return ToolResult.from_error(f"Unknown action: {action}")
    
    def _get_path(self, data: Any, path: str) -> Any:
        """Get value at path."""
        parts = path.replace("[", ".").replace("]", "").split(".")
        current = data
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = current[part]
        return current
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["parse", "validate", "format", "get", "stringify"]},
                    "data": {"type": "string", "description": "JSON string"},
                    "path": {"type": "string", "description": "Path for get operations"},
                },
                "required": ["action", "data"],
            },
        }
