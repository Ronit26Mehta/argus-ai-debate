"""
BigQuery Tool for ARGUS.

Connect with BigQuery to retrieve data, perform analysis, and execute queries.
Supports both direct queries and agent analytics capabilities.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a BigQuery query."""
    query_id: str
    rows: list[dict[str, Any]]
    schema: list[dict[str, str]]
    total_rows: int
    total_bytes_processed: int
    cache_hit: bool
    execution_time_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "rows": self.rows[:100],  # Limit displayed rows
            "schema": self.schema,
            "total_rows": self.total_rows,
            "total_bytes_processed": self.total_bytes_processed,
            "cache_hit": self.cache_hit,
            "execution_time_ms": self.execution_time_ms,
        }


class BigQueryTool(BaseTool):
    """
    BigQuery - Data analysis and querying tool.
    
    Features:
    - Execute SQL queries
    - Retrieve and analyze data
    - Schema inspection
    - Table management
    - Agent analytics plugin capabilities
    
    Example:
        >>> tool = BigQueryTool(project_id="my-project")
        >>> result = tool(action="query", sql="SELECT * FROM dataset.table LIMIT 10")
        >>> result = tool(action="list_tables", dataset="my_dataset")
    """
    
    name = "bigquery"
    description = "Connect with BigQuery to retrieve data and perform analysis"
    category = ToolCategory.DATA
    version = "1.0.0"
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: str = "US",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.location = location
        self._client = None
        
        logger.debug(f"BigQuery initialized for project {self.project_id}")
    
    def _get_client(self):
        """Lazy-load BigQuery client."""
        if self._client is None:
            try:
                from google.cloud import bigquery
                self._client = bigquery.Client(project=self.project_id)
            except ImportError:
                raise ImportError("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
        return self._client
    
    def execute(
        self,
        action: str = "query",
        sql: Optional[str] = None,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
        limit: int = 100,
        dry_run: bool = False,
        parameters: Optional[dict] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute BigQuery operations.
        
        Args:
            action: Operation to perform
            sql: SQL query to execute
            dataset: Dataset name
            table: Table name
            limit: Max rows to return
            dry_run: Validate query without executing
            parameters: Query parameters
            
        Returns:
            ToolResult with operation result
        """
        actions = {
            "query": self._execute_query,
            "list_datasets": self._list_datasets,
            "list_tables": self._list_tables,
            "get_schema": self._get_schema,
            "preview": self._preview_table,
            "create_dataset": self._create_dataset,
            "delete_dataset": self._delete_dataset,
            "analyze_table": self._analyze_table,
            "estimate_cost": self._estimate_cost,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                sql=sql,
                dataset=dataset,
                table=table,
                limit=limit,
                dry_run=dry_run,
                parameters=parameters or {},
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"BigQuery error: {e}")
            return ToolResult.from_error(f"BigQuery error: {e}")
    
    def _execute_query(
        self,
        sql: Optional[str] = None,
        limit: int = 100,
        dry_run: bool = False,
        parameters: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute a SQL query."""
        if not sql:
            return ToolResult.from_error("sql query is required")
        
        from google.cloud import bigquery
        
        client = self._get_client()
        
        job_config = bigquery.QueryJobConfig(
            dry_run=dry_run,
            use_query_cache=True,
        )
        
        # Add parameters if provided
        if parameters:
            query_params = []
            for name, value in parameters.items():
                if isinstance(value, str):
                    param = bigquery.ScalarQueryParameter(name, "STRING", value)
                elif isinstance(value, int):
                    param = bigquery.ScalarQueryParameter(name, "INT64", value)
                elif isinstance(value, float):
                    param = bigquery.ScalarQueryParameter(name, "FLOAT64", value)
                elif isinstance(value, bool):
                    param = bigquery.ScalarQueryParameter(name, "BOOL", value)
                else:
                    param = bigquery.ScalarQueryParameter(name, "STRING", str(value))
                query_params.append(param)
            job_config.query_parameters = query_params
        
        start_time = datetime.utcnow()
        query_job = client.query(sql, job_config=job_config)
        
        if dry_run:
            return ToolResult.from_data({
                "total_bytes_processed": query_job.total_bytes_processed,
                "estimated_cost_usd": query_job.total_bytes_processed / (1024 ** 4) * 5,
                "valid": True,
            })
        
        results = query_job.result()
        end_time = datetime.utcnow()
        
        rows = []
        schema = []
        
        # Extract schema
        for field in results.schema:
            schema.append({
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
            })
        
        # Extract rows
        for row in results:
            row_dict = {}
            for key, value in row.items():
                if hasattr(value, 'isoformat'):
                    row_dict[key] = value.isoformat()
                else:
                    row_dict[key] = value
            rows.append(row_dict)
            if len(rows) >= limit:
                break
        
        query_result = QueryResult(
            query_id=query_job.job_id,
            rows=rows,
            schema=schema,
            total_rows=results.total_rows or len(rows),
            total_bytes_processed=query_job.total_bytes_processed or 0,
            cache_hit=query_job.cache_hit or False,
            execution_time_ms=(end_time - start_time).total_seconds() * 1000,
        )
        
        return ToolResult.from_data(query_result.to_dict())
    
    def _list_datasets(self, **kwargs) -> ToolResult:
        """List all datasets in the project."""
        client = self._get_client()
        
        datasets = []
        for dataset in client.list_datasets():
            datasets.append({
                "dataset_id": dataset.dataset_id,
                "full_id": dataset.full_dataset_id,
                "location": dataset.location,
            })
        
        return ToolResult.from_data({
            "project": self.project_id,
            "datasets": datasets,
            "count": len(datasets),
        })
    
    def _list_tables(
        self,
        dataset: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List tables in a dataset."""
        if not dataset:
            return ToolResult.from_error("dataset is required")
        
        client = self._get_client()
        
        tables = []
        for table in client.list_tables(dataset):
            tables.append({
                "table_id": table.table_id,
                "full_id": f"{table.project}.{table.dataset_id}.{table.table_id}",
                "type": table.table_type,
            })
        
        return ToolResult.from_data({
            "dataset": dataset,
            "tables": tables,
            "count": len(tables),
        })
    
    def _get_schema(
        self,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get table schema."""
        if not dataset or not table:
            return ToolResult.from_error("dataset and table are required")
        
        client = self._get_client()
        
        table_ref = client.get_table(f"{dataset}.{table}")
        
        schema = []
        for field in table_ref.schema:
            schema.append({
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description,
            })
        
        return ToolResult.from_data({
            "table": f"{dataset}.{table}",
            "schema": schema,
            "num_rows": table_ref.num_rows,
            "num_bytes": table_ref.num_bytes,
            "created": table_ref.created.isoformat() if table_ref.created else None,
            "modified": table_ref.modified.isoformat() if table_ref.modified else None,
        })
    
    def _preview_table(
        self,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
        limit: int = 10,
        **kwargs,
    ) -> ToolResult:
        """Preview table data."""
        if not dataset or not table:
            return ToolResult.from_error("dataset and table are required")
        
        sql = f"SELECT * FROM `{dataset}.{table}` LIMIT {limit}"
        return self._execute_query(sql=sql, limit=limit)
    
    def _create_dataset(
        self,
        dataset: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new dataset."""
        if not dataset:
            return ToolResult.from_error("dataset name is required")
        
        from google.cloud import bigquery
        
        client = self._get_client()
        
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{dataset}")
        dataset_ref.location = self.location
        if description:
            dataset_ref.description = description
        
        created = client.create_dataset(dataset_ref)
        
        return ToolResult.from_data({
            "dataset_id": created.dataset_id,
            "location": created.location,
            "created": True,
        })
    
    def _delete_dataset(
        self,
        dataset: Optional[str] = None,
        delete_contents: bool = False,
        **kwargs,
    ) -> ToolResult:
        """Delete a dataset."""
        if not dataset:
            return ToolResult.from_error("dataset name is required")
        
        client = self._get_client()
        
        client.delete_dataset(dataset, delete_contents=delete_contents)
        
        return ToolResult.from_data({
            "dataset": dataset,
            "deleted": True,
        })
    
    def _analyze_table(
        self,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Analyze table statistics."""
        if not dataset or not table:
            return ToolResult.from_error("dataset and table are required")
        
        client = self._get_client()
        
        table_ref = client.get_table(f"{dataset}.{table}")
        
        # Get column statistics
        stats_sql = f"""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT *) as unique_rows
        FROM `{dataset}.{table}`
        """
        
        try:
            stats_job = client.query(stats_sql)
            stats_result = list(stats_job.result())[0]
            
            return ToolResult.from_data({
                "table": f"{dataset}.{table}",
                "num_rows": table_ref.num_rows,
                "num_bytes": table_ref.num_bytes,
                "num_columns": len(table_ref.schema),
                "estimated_unique_rows": dict(stats_result).get("unique_rows", 0),
                "size_mb": (table_ref.num_bytes or 0) / (1024 * 1024),
            })
        except Exception as e:
            return ToolResult.from_data({
                "table": f"{dataset}.{table}",
                "num_rows": table_ref.num_rows,
                "num_bytes": table_ref.num_bytes,
                "num_columns": len(table_ref.schema),
                "size_mb": (table_ref.num_bytes or 0) / (1024 * 1024),
            })
    
    def _estimate_cost(
        self,
        sql: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Estimate query cost."""
        return self._execute_query(sql=sql, dry_run=True)
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "list_datasets", "list_tables", "get_schema",
                                "preview", "create_dataset", "delete_dataset",
                                "analyze_table", "estimate_cost"],
                    },
                    "sql": {"type": "string", "description": "SQL query to execute"},
                    "dataset": {"type": "string"},
                    "table": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                    "dry_run": {"type": "boolean", "default": False},
                    "parameters": {"type": "object"},
                },
                "required": ["action"],
            },
        }
