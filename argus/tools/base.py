"""
Base Tool Classes for ARGUS.

Defines the abstract base class and data structures for creating
custom tools that integrate with the ARGUS agent framework.

Users can create their own tools by subclassing BaseTool and implementing
the execute() method. Tools are registered with a ToolRegistry and
executed via the ToolExecutor.

Example:
    >>> class CalculatorTool(BaseTool):
    ...     name = "calculator"
    ...     description = "Perform mathematical calculations"
    ...     category = ToolCategory.UTILITY
    ...     
    ...     def execute(self, expression: str, **kwargs) -> ToolResult:
    ...         try:
    ...             result = eval(expression)  # Note: Use safe eval in production
    ...             return ToolResult(success=True, data={"result": result})
    ...         except Exception as e:
    ...             return ToolResult(success=False, error=str(e))
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools for organization and filtering.
    
    Users can extend this by using custom string values.
    """
    SEARCH = "search"
    DATA = "data"
    ANALYSIS = "analysis"
    UTILITY = "utility"
    EXTERNAL_API = "external_api"
    SIMULATION = "simulation"
    CUSTOM = "custom"


class ToolConfig(BaseModel):
    """Configuration for a tool.
    
    Attributes:
        timeout: Maximum execution time in seconds
        retries: Number of retry attempts on failure
        cache_ttl: Cache time-to-live in seconds (0 = no caching)
        require_confirmation: Whether to require user confirmation before execution
    """
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Maximum execution time in seconds",
    )
    retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Number of retry attempts",
    )
    cache_ttl: int = Field(
        default=0,
        ge=0,
        description="Cache TTL in seconds (0 = no caching)",
    )
    require_confirmation: bool = Field(
        default=False,
        description="Require user confirmation before execution",
    )


@dataclass
class ToolResult:
    """Result from a tool execution.
    
    Attributes:
        success: Whether the execution was successful
        data: Result data (structure depends on tool)
        error: Error message if failed
        metadata: Additional metadata about the execution
        execution_time_ms: Time taken to execute in milliseconds
        cached: Whether result was from cache
    """
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "cached": self.cached,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_error(cls, error: str) -> "ToolResult":
        """Create a failed result from an error message."""
        return cls(success=False, error=error)
    
    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "ToolResult":
        """Create a successful result from data."""
        return cls(success=True, data=data)


class BaseTool(ABC):
    """Abstract base class for all ARGUS tools.
    
    Subclass this to create custom tools that agents can use.
    Each tool must define a name, description, and implement the execute() method.
    
    Class Attributes:
        name: Unique identifier for the tool (lowercase, no spaces)
        description: Human-readable description of what the tool does
        category: Category for organization
        version: Tool version string
        
    Example:
        >>> class WeatherTool(BaseTool):
        ...     name = "weather"
        ...     description = "Get current weather for a location"
        ...     category = ToolCategory.EXTERNAL_API
        ...     
        ...     def execute(self, location: str, **kwargs) -> ToolResult:
        ...         # Call weather API
        ...         return ToolResult(success=True, data={"temp": 72, "unit": "F"})
    """
    
    # Class attributes that subclasses should override
    name: str = "base_tool"
    description: str = "Base tool description"
    category: ToolCategory = ToolCategory.CUSTOM
    version: str = "1.0.0"
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize the tool.
        
        Args:
            config: Optional tool configuration
        """
        self.config = config or ToolConfig()
        self._call_count = 0
        self._total_execution_time = 0.0
        logger.debug(f"Initialized tool: {self.name}")
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given arguments.
        
        This method must be implemented by all tool subclasses.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult containing success status and data/error
        """
        pass
    
    def validate_input(self, **kwargs: Any) -> Optional[str]:
        """Validate input arguments before execution.
        
        Override this method to add input validation.
        
        Args:
            **kwargs: Input arguments to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        return None
    
    def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the tool (callable interface).
        
        This wraps execute() with timing and error handling.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            ToolResult from execution
        """
        # Validate input
        validation_error = self.validate_input(**kwargs)
        if validation_error:
            return ToolResult.from_error(f"Validation error: {validation_error}")
        
        # Execute with timing
        start = time.perf_counter()
        try:
            result = self.execute(**kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.execution_time_ms = elapsed_ms
            
            # Update stats
            self._call_count += 1
            self._total_execution_time += elapsed_ms
            
            return result
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Tool {self.name} execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
    
    def get_schema(self) -> dict[str, Any]:
        """Get JSON schema describing the tool's parameters.
        
        Override this to provide parameter documentation.
        
        Returns:
            JSON schema dict
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "parameters": {},
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics for this tool.
        
        Returns:
            Dict with call count, total time, average time
        """
        avg_time = (
            self._total_execution_time / self._call_count
            if self._call_count > 0
            else 0
        )
        return {
            "name": self.name,
            "call_count": self._call_count,
            "total_execution_time_ms": self._total_execution_time,
            "average_execution_time_ms": avg_time,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# Built-in Example Tools (for reference)
# =============================================================================

class EchoTool(BaseTool):
    """Simple echo tool for testing.
    
    Returns the input message back as output.
    """
    name = "echo"
    description = "Echo back the input message (for testing)"
    category = ToolCategory.UTILITY
    
    def execute(self, message: str = "", **kwargs: Any) -> ToolResult:
        """Echo the message back.
        
        Args:
            message: Message to echo
            
        Returns:
            ToolResult with echoed message
        """
        return ToolResult(
            success=True,
            data={"message": message, "kwargs": kwargs},
        )
    
    def validate_input(self, **kwargs: Any) -> Optional[str]:
        if "message" not in kwargs:
            return "Missing required argument: message"
        return None


class CalculatorTool(BaseTool):
    """Safe calculator tool for mathematical expressions.
    
    Supports basic arithmetic operations.
    """
    name = "calculator"
    description = "Perform safe mathematical calculations"
    category = ToolCategory.UTILITY
    
    # Safe operations whitelist
    ALLOWED_NAMES = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    }
    
    def execute(
        self,
        expression: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Evaluate a mathematical expression safely.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            ToolResult with calculation result
        """
        try:
            # Simple safe eval using ast
            import ast
            import operator
            
            # Define allowed operators
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }
            
            def safe_eval(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    return operators[type(node.op)](left, right)
                elif isinstance(node, ast.UnaryOp):
                    operand = safe_eval(node.operand)
                    return operators[type(node.op)](operand)
                elif isinstance(node, ast.Call):
                    func_name = node.func.id if isinstance(node.func, ast.Name) else None
                    if func_name in self.ALLOWED_NAMES:
                        args = [safe_eval(arg) for arg in node.args]
                        return self.ALLOWED_NAMES[func_name](*args)
                    raise ValueError(f"Function not allowed: {func_name}")
                else:
                    raise ValueError(f"Unsupported expression type: {type(node)}")
            
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree.body)
            
            return ToolResult(
                success=True,
                data={"expression": expression, "result": result},
            )
            
        except Exception as e:
            return ToolResult.from_error(f"Calculation error: {e}")
    
    def validate_input(self, **kwargs: Any) -> Optional[str]:
        if "expression" not in kwargs or not kwargs["expression"]:
            return "Missing required argument: expression"
        return None


# Factory function for creating tools dynamically
def create_tool(
    name: str,
    description: str,
    execute_fn: Callable[..., ToolResult],
    category: ToolCategory = ToolCategory.CUSTOM,
    config: Optional[ToolConfig] = None,
) -> BaseTool:
    """Factory function to create a tool from a function.
    
    Allows creating tools without subclassing.
    
    Args:
        name: Tool name
        description: Tool description
        execute_fn: Function that implements the tool logic
        category: Tool category
        config: Optional configuration
        
    Returns:
        BaseTool instance
        
    Example:
        >>> def my_search(query: str, **kwargs) -> ToolResult:
        ...     return ToolResult(success=True, data={"results": []})
        >>> 
        >>> tool = create_tool("my_search", "Custom search", my_search)
    """
    class DynamicTool(BaseTool):
        pass
    
    DynamicTool.name = name
    DynamicTool.description = description
    DynamicTool.category = category
    DynamicTool.execute = lambda self, **kw: execute_fn(**kw)
    
    return DynamicTool(config)
