"""
Tool Executor for ARGUS.

Executes registered tools with proper error handling, timeouts,
retries, and integration with caching and guardrails.

Example:
    >>> executor = ToolExecutor(registry)
    >>> result = executor.run("web_search", query="climate change")
    >>> 
    >>> # Or execute multiple tools
    >>> results = executor.run_batch([
    ...     ("search", {"query": "topic A"}),
    ...     ("search", {"query": "topic B"}),
    ... ])
"""

from __future__ import annotations

import logging
import time
import concurrent.futures
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field

from argus.tools.base import ToolResult

if TYPE_CHECKING:
    from argus.tools.registry import ToolRegistry
    from argus.tools.cache import ResultCache
    from argus.tools.guardrails import Guardrail

logger = logging.getLogger(__name__)


class ExecutorConfig(BaseModel):
    """Configuration for tool executor.
    
    Attributes:
        timeout: Default timeout for tool execution
        max_concurrent: Maximum concurrent tool executions
        enable_caching: Whether to use result caching
        enable_guardrails: Whether to apply guardrails
    """
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Default timeout in seconds",
    )
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent tool executions",
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching",
    )
    enable_guardrails: bool = Field(
        default=True,
        description="Enable guardrails",
    )


@dataclass
class ExecutionRecord:
    """Record of a tool execution for auditing."""
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult
    started_at: datetime
    completed_at: datetime
    cached: bool = False
    guardrail_blocked: bool = False


class ToolExecutor:
    """Executes tools with caching, guardrails, and error handling.
    
    The executor provides a unified interface for running tools
    registered in a ToolRegistry, with support for:
    
    - Result caching to avoid redundant expensive operations
    - Guardrails for content filtering and policy enforcement
    - Timeouts and retries for reliability
    - Batch execution for parallel operations
    - Execution history for auditing
    
    Example:
        >>> from argus.tools import ToolRegistry, ToolExecutor
        >>> 
        >>> registry = ToolRegistry()
        >>> registry.register(MyTool())
        >>> 
        >>> executor = ToolExecutor(registry)
        >>> result = executor.run("my_tool", arg1="value1")
        >>> 
        >>> if result.success:
        ...     print(result.data)
    """
    
    def __init__(
        self,
        registry: "ToolRegistry",
        config: Optional[ExecutorConfig] = None,
        cache: Optional["ResultCache"] = None,
        guardrails: Optional[list["Guardrail"]] = None,
    ):
        """Initialize the executor.
        
        Args:
            registry: Tool registry to use
            config: Executor configuration
            cache: Optional result cache
            guardrails: Optional list of guardrails to apply
        """
        self.registry = registry
        self.config = config or ExecutorConfig()
        self.cache = cache
        self.guardrails = guardrails or []
        self._execution_history: list[ExecutionRecord] = []
        self._history_limit = 1000
        
        logger.info(f"Initialized ToolExecutor with {len(registry)} tools")
    
    def run(
        self,
        tool_name: str,
        timeout: Optional[float] = None,
        skip_cache: bool = False,
        skip_guardrails: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            timeout: Override timeout (uses config default if None)
            skip_cache: Skip cache lookup
            skip_guardrails: Skip guardrail checks
            **kwargs: Arguments to pass to the tool
            
        Returns:
            ToolResult from execution
        """
        started_at = datetime.utcnow()
        
        # Get tool
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult.from_error(f"Tool not found: {tool_name}")
        
        # Check cache first
        if (
            self.config.enable_caching
            and self.cache is not None
            and not skip_cache
            and tool.config.cache_ttl > 0
        ):
            cached_result = self.cache.get(tool_name, kwargs)
            if cached_result is not None:
                cached_result.cached = True
                self._record_execution(
                    tool_name, kwargs, cached_result,
                    started_at, datetime.utcnow(), cached=True
                )
                return cached_result
        
        # Apply guardrails
        if self.config.enable_guardrails and not skip_guardrails:
            for guardrail in self.guardrails:
                check = guardrail.check(tool_name, kwargs)
                if not check.allowed:
                    result = ToolResult.from_error(
                        f"Blocked by guardrail: {check.reason}"
                    )
                    self._record_execution(
                        tool_name, kwargs, result,
                        started_at, datetime.utcnow(),
                        guardrail_blocked=True
                    )
                    return result
        
        # Execute with timeout
        effective_timeout = timeout or self.config.timeout
        try:
            result = self._execute_with_timeout(tool, kwargs, effective_timeout)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            result = ToolResult.from_error(str(e))
        
        # Cache result if successful
        if (
            result.success
            and self.config.enable_caching
            and self.cache is not None
            and tool.config.cache_ttl > 0
        ):
            self.cache.set(tool_name, kwargs, result, tool.config.cache_ttl)
        
        # Record execution
        self._record_execution(
            tool_name, kwargs, result,
            started_at, datetime.utcnow()
        )
        
        return result
    
    def _execute_with_timeout(
        self,
        tool: Any,
        kwargs: dict[str, Any],
        timeout: float,
    ) -> ToolResult:
        """Execute tool with timeout using thread pool.
        
        Args:
            tool: Tool instance
            kwargs: Tool arguments
            timeout: Timeout in seconds
            
        Returns:
            ToolResult
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                return ToolResult.from_error(
                    f"Tool execution timed out after {timeout}s"
                )
    
    def run_batch(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> list[ToolResult]:
        """Execute multiple tools in parallel.
        
        Args:
            calls: List of (tool_name, kwargs) tuples
            timeout: Timeout per call
            
        Returns:
            List of ToolResults in same order as calls
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_concurrent
        ) as executor:
            futures = [
                executor.submit(self.run, name, timeout, **kwargs)
                for name, kwargs in calls
            ]
            return [f.result() for f in futures]
    
    def _record_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        started_at: datetime,
        completed_at: datetime,
        cached: bool = False,
        guardrail_blocked: bool = False,
    ) -> None:
        """Record an execution in history."""
        record = ExecutionRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            started_at=started_at,
            completed_at=completed_at,
            cached=cached,
            guardrail_blocked=guardrail_blocked,
        )
        
        self._execution_history.append(record)
        
        # Trim history if needed
        if len(self._execution_history) > self._history_limit:
            self._execution_history = self._execution_history[-self._history_limit:]
    
    def get_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[ExecutionRecord]:
        """Get execution history.
        
        Args:
            tool_name: Filter by tool name (None = all)
            limit: Maximum records to return
            
        Returns:
            List of ExecutionRecords
        """
        history = self._execution_history
        if tool_name:
            history = [r for r in history if r.tool_name == tool_name]
        return history[-limit:]
    
    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics.
        
        Returns:
            Dict with execution counts, cache hits, etc.
        """
        total = len(self._execution_history)
        cached = sum(1 for r in self._execution_history if r.cached)
        blocked = sum(1 for r in self._execution_history if r.guardrail_blocked)
        successful = sum(1 for r in self._execution_history if r.result.success)
        
        return {
            "total_executions": total,
            "cache_hits": cached,
            "guardrail_blocks": blocked,
            "successful": successful,
            "failed": total - successful - blocked,
            "cache_hit_rate": cached / total if total > 0 else 0,
            "success_rate": successful / (total - blocked) if (total - blocked) > 0 else 0,
            "registry_stats": self.registry.get_stats(),
        }
    
    def add_guardrail(self, guardrail: "Guardrail") -> None:
        """Add a guardrail to the executor.
        
        Args:
            guardrail: Guardrail to add
        """
        self.guardrails.append(guardrail)
        logger.info(f"Added guardrail: {guardrail.__class__.__name__}")
    
    def set_cache(self, cache: "ResultCache") -> None:
        """Set the result cache.
        
        Args:
            cache: ResultCache instance
        """
        self.cache = cache
        logger.info("Set result cache")
