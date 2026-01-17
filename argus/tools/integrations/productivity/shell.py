"""
Shell Tool for ARGUS.

Execute shell commands.
"""

from __future__ import annotations

import subprocess
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class ShellTool(BaseTool):
    """
    Shell command execution tool.
    
    Example:
        >>> tool = ShellTool()
        >>> result = tool(command="ls -la")
    """
    
    name = "shell"
    description = "Execute shell commands. Use with caution."
    category = ToolCategory.UTILITY
    
    # Commands that are always blocked
    BLOCKED_COMMANDS = ["rm -rf", "mkfs", "dd if=", ":(){", "chmod 777"]
    
    def __init__(
        self,
        allowed_commands: Optional[list[str]] = None,
        timeout: float = 30.0,
        cwd: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.allowed_commands = allowed_commands  # If set, whitelist mode
        self.timeout = timeout
        self.cwd = cwd
    
    def _is_safe(self, command: str) -> bool:
        """Check if command is safe to execute."""
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in command:
                return False
        return True
    
    def execute(
        self,
        command: str = "",
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute shell command.
        
        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds
            
        Returns:
            ToolResult with output
        """
        if not command:
            return ToolResult.from_error("Command is required")
        
        # Safety check
        if not self._is_safe(command):
            return ToolResult.from_error("Command contains blocked patterns")
        
        # Whitelist check
        if self.allowed_commands:
            cmd_base = command.split()[0] if command else ""
            if cmd_base not in self.allowed_commands:
                return ToolResult.from_error(f"Command not in allowed list: {cmd_base}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                cwd=self.cwd,
            )
            
            return ToolResult.from_data({
                "command": command,
                "stdout": result.stdout[:5000] if result.stdout else "",
                "stderr": result.stderr[:2000] if result.stderr else "",
                "return_code": result.returncode,
            })
            
        except subprocess.TimeoutExpired:
            return ToolResult.from_error(f"Command timed out after {timeout or self.timeout}s")
        except Exception as e:
            logger.error(f"Shell error: {e}")
            return ToolResult.from_error(f"Shell error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds"},
                },
                "required": ["command"],
            },
        }
