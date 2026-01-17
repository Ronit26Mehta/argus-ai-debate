"""
File System Tool for ARGUS.

Read, write, and manage files.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class FileSystemTool(BaseTool):
    """
    File system operations tool.
    
    Example:
        >>> tool = FileSystemTool(base_dir="/safe/directory")
        >>> result = tool(action="read", path="file.txt")
    """
    
    name = "filesystem"
    description = "Read, write, list, and manage files and directories."
    category = ToolCategory.UTILITY
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
        allow_write: bool = True,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.allow_write = allow_write
    
    def _safe_path(self, path: str) -> Path:
        """Ensure path is within base directory."""
        full_path = (self.base_dir / path).resolve()
        if not str(full_path).startswith(str(self.base_dir.resolve())):
            raise ValueError("Path escapes base directory")
        return full_path
    
    def execute(
        self,
        action: str = "list",
        path: str = ".",
        content: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Perform file system operation.
        
        Args:
            action: 'read', 'write', 'list', 'exists', 'delete', 'mkdir'
            path: File or directory path
            content: Content for write operations
            
        Returns:
            ToolResult with operation result
        """
        try:
            safe_path = self._safe_path(path)
            
            if action == "list":
                if not safe_path.exists():
                    return ToolResult.from_error(f"Path not found: {path}")
                if safe_path.is_file():
                    return ToolResult.from_data({"path": path, "type": "file", "size": safe_path.stat().st_size})
                items = [
                    {"name": p.name, "type": "dir" if p.is_dir() else "file", "size": p.stat().st_size if p.is_file() else 0}
                    for p in sorted(safe_path.iterdir())[:100]
                ]
                return ToolResult.from_data({"path": path, "items": items, "count": len(items)})
            
            elif action == "read":
                if not safe_path.exists():
                    return ToolResult.from_error(f"File not found: {path}")
                content = safe_path.read_text(encoding="utf-8")[:50000]  # Limit
                return ToolResult.from_data({"path": path, "content": content, "size": len(content)})
            
            elif action == "write":
                if not self.allow_write:
                    return ToolResult.from_error("Write operations disabled")
                if content is None:
                    return ToolResult.from_error("Content required for write")
                safe_path.parent.mkdir(parents=True, exist_ok=True)
                safe_path.write_text(content, encoding="utf-8")
                return ToolResult.from_data({"path": path, "written": len(content)})
            
            elif action == "exists":
                return ToolResult.from_data({"path": path, "exists": safe_path.exists(), "is_file": safe_path.is_file() if safe_path.exists() else False})
            
            elif action == "mkdir":
                if not self.allow_write:
                    return ToolResult.from_error("Write operations disabled")
                safe_path.mkdir(parents=True, exist_ok=True)
                return ToolResult.from_data({"path": path, "created": True})
            
            elif action == "delete":
                if not self.allow_write:
                    return ToolResult.from_error("Write operations disabled")
                if safe_path.is_file():
                    safe_path.unlink()
                return ToolResult.from_data({"path": path, "deleted": True})
            
            else:
                return ToolResult.from_error(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Filesystem error: {e}")
            return ToolResult.from_error(f"Filesystem error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "read", "write", "exists", "mkdir", "delete"]},
                    "path": {"type": "string", "description": "File or directory path"},
                    "content": {"type": "string", "description": "Content for write"},
                },
                "required": ["action", "path"],
            },
        }
