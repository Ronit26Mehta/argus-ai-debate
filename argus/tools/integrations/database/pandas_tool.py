"""
Pandas DataFrame Tool for ARGUS.

Analyze and manipulate data with Pandas.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class PandasTool(BaseTool):
    """
    Pandas DataFrame analysis tool.
    
    Example:
        >>> tool = PandasTool()
        >>> result = tool(action="read_csv", path="data.csv")
        >>> result = tool(action="describe")
    """
    
    name = "pandas"
    description = "Load, analyze, and manipulate data with Pandas DataFrames."
    category = ToolCategory.DATA
    
    def __init__(self, config: Optional[ToolConfig] = None):
        super().__init__(config)
        self._df = None
    
    def execute(
        self,
        action: str = "info",
        path: Optional[str] = None,
        query: Optional[str] = None,
        columns: Optional[list[str]] = None,
        n: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Perform Pandas operation.
        
        Args:
            action: 'read_csv', 'read_json', 'head', 'tail', 'describe', 'info', 'query', 'columns'
            path: File path for read operations
            query: Query string for filtering
            columns: Columns to select
            n: Number of rows for head/tail
            
        Returns:
            ToolResult with data
        """
        try:
            import pandas as pd
        except ImportError:
            return ToolResult.from_error("pandas not installed")
        
        try:
            if action == "read_csv":
                if not path:
                    return ToolResult.from_error("Path required")
                self._df = pd.read_csv(path, **kwargs)
                return ToolResult.from_data({
                    "shape": list(self._df.shape),
                    "columns": list(self._df.columns),
                    "dtypes": {str(k): str(v) for k, v in self._df.dtypes.items()},
                })
            
            elif action == "read_json":
                if not path:
                    return ToolResult.from_error("Path required")
                self._df = pd.read_json(path, **kwargs)
                return ToolResult.from_data({
                    "shape": list(self._df.shape),
                    "columns": list(self._df.columns),
                })
            
            elif action == "head":
                if self._df is None:
                    return ToolResult.from_error("No DataFrame loaded")
                df = self._df[columns] if columns else self._df
                return ToolResult.from_data({"rows": df.head(n).to_dict("records")})
            
            elif action == "tail":
                if self._df is None:
                    return ToolResult.from_error("No DataFrame loaded")
                df = self._df[columns] if columns else self._df
                return ToolResult.from_data({"rows": df.tail(n).to_dict("records")})
            
            elif action == "describe":
                if self._df is None:
                    return ToolResult.from_error("No DataFrame loaded")
                stats = self._df.describe(include="all").to_dict()
                return ToolResult.from_data({"statistics": stats})
            
            elif action == "info":
                if self._df is None:
                    return ToolResult.from_error("No DataFrame loaded")
                return ToolResult.from_data({
                    "shape": list(self._df.shape),
                    "columns": list(self._df.columns),
                    "dtypes": {str(k): str(v) for k, v in self._df.dtypes.items()},
                    "memory_usage": self._df.memory_usage(deep=True).sum(),
                    "null_counts": self._df.isnull().sum().to_dict(),
                })
            
            elif action == "query":
                if self._df is None:
                    return ToolResult.from_error("No DataFrame loaded")
                if not query:
                    return ToolResult.from_error("Query required")
                result = self._df.query(query)
                return ToolResult.from_data({
                    "rows": result.head(100).to_dict("records"),
                    "count": len(result),
                })
            
            elif action == "columns":
                if self._df is None:
                    return ToolResult.from_error("No DataFrame loaded")
                return ToolResult.from_data({"columns": list(self._df.columns)})
            
            else:
                return ToolResult.from_error(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Pandas error: {e}")
            return ToolResult.from_error(f"Pandas error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read_csv", "read_json", "head", "tail", "describe", "info", "query", "columns"]},
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "n": {"type": "integer", "default": 10},
                },
                "required": ["action"],
            },
        }
