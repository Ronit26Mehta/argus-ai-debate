"""
SQL Database Tool for ARGUS.

Execute SQL queries on databases.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class SqlTool(BaseTool):
    """
    SQL database query tool.
    
    Example:
        >>> tool = SqlTool(connection_string="sqlite:///mydb.db")
        >>> result = tool(query="SELECT * FROM users LIMIT 10")
    """
    
    name = "sql_database"
    description = "Execute SQL queries on databases. Supports SQLite, PostgreSQL, MySQL."
    category = ToolCategory.DATA
    
    # Read-only by default
    ALLOWED_STATEMENTS = ["SELECT", "SHOW", "DESCRIBE", "EXPLAIN"]
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        read_only: bool = True,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.connection_string = connection_string
        self.read_only = read_only
        self._engine = None
    
    def _get_engine(self):
        """Lazy load SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
            except ImportError:
                raise ImportError("sqlalchemy not installed. Run: pip install sqlalchemy")
            if not self.connection_string:
                raise ValueError("connection_string required")
            self._engine = create_engine(self.connection_string)
        return self._engine
    
    def _is_read_query(self, query: str) -> bool:
        """Check if query is read-only."""
        query_upper = query.strip().upper()
        return any(query_upper.startswith(stmt) for stmt in self.ALLOWED_STATEMENTS)
    
    def execute(
        self,
        query: str = "",
        params: Optional[dict] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            limit: Maximum rows to return
            
        Returns:
            ToolResult with query results
        """
        try:
            from sqlalchemy import text
            import pandas as pd
        except ImportError:
            return ToolResult.from_error("sqlalchemy and pandas required")
        
        if not query:
            return ToolResult.from_error("Query is required")
        
        # Safety check for read-only mode
        if self.read_only and not self._is_read_query(query):
            return ToolResult.from_error("Only SELECT queries allowed in read-only mode")
        
        try:
            engine = self._get_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchmany(limit))
                    if not df.empty:
                        df.columns = result.keys()
                    
                    return ToolResult.from_data({
                        "query": query,
                        "columns": list(df.columns),
                        "rows": df.to_dict("records"),
                        "row_count": len(df),
                    })
                else:
                    return ToolResult.from_data({
                        "query": query,
                        "affected_rows": result.rowcount,
                    })
                    
        except Exception as e:
            logger.error(f"SQL error: {e}")
            return ToolResult.from_error(f"SQL error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"},
                    "params": {"type": "object", "description": "Query parameters"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["query"],
            },
        }
