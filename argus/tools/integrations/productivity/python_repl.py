"""
Python REPL Tool for ARGUS.

Execute Python code safely.
"""

from __future__ import annotations

import sys
import logging
from io import StringIO
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class PythonReplTool(BaseTool):
    """
    Python code execution tool.
    
    Example:
        >>> tool = PythonReplTool()
        >>> result = tool(code="print(2 + 2)")
    """
    
    name = "python_repl"
    description = "Execute Python code and return the output. Useful for calculations, data processing, and testing."
    category = ToolCategory.UTILITY
    
    def __init__(
        self,
        globals_dict: Optional[dict] = None,
        timeout: float = 30.0,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.globals_dict = globals_dict or {"__builtins__": __builtins__}
        self.timeout = timeout
        self.local_vars = {}
    
    def execute(
        self,
        code: str = "",
        reset: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            reset: Reset local variables
            
        Returns:
            ToolResult with output
        """
        if not code:
            return ToolResult.from_error("Code is required")
        
        if reset:
            self.local_vars = {}
        
        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_out = StringIO()
        sys.stderr = captured_err = StringIO()
        
        result_value = None
        error = None
        
        try:
            # Try as expression first
            try:
                result_value = eval(code, self.globals_dict, self.local_vars)
            except SyntaxError:
                # Execute as statement
                exec(code, self.globals_dict, self.local_vars)
                result_value = None
                
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error(f"Python execution error: {error}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        stdout = captured_out.getvalue()
        stderr = captured_err.getvalue()
        
        if error:
            return ToolResult.from_error(error)
        
        return ToolResult.from_data({
            "output": stdout[:5000] if stdout else None,
            "result": str(result_value)[:2000] if result_value is not None else None,
            "stderr": stderr[:1000] if stderr else None,
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "reset": {"type": "boolean", "description": "Reset local variables", "default": False},
                },
                "required": ["code"],
            },
        }
