"""
Tool Registry for ARGUS.

Provides centralized management of tools, allowing registration,
discovery, and retrieval of tools by name or category.

Example:
    >>> registry = ToolRegistry()
    >>> registry.register(MyCustomTool())
    >>> tool = registry.get("my_tool")
    >>> tools = registry.list_by_category(ToolCategory.SEARCH)
"""

from __future__ import annotations

import logging
from typing import Optional, Type, Any
from threading import RLock

from argus.tools.base import BaseTool, ToolCategory

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for ARGUS tools.
    
    Manages tool registration, discovery, and lifecycle.
    Thread-safe for concurrent access.
    
    Example:
        >>> registry = ToolRegistry()
        >>> 
        >>> # Register a tool instance
        >>> registry.register(MySearchTool())
        >>> 
        >>> # Or register a tool class (lazily instantiated)
        >>> registry.register_class(MySearchTool)
        >>> 
        >>> # Get and use a tool
        >>> tool = registry.get("my_search")
        >>> result = tool(query="test")
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._tools: dict[str, BaseTool] = {}
        self._tool_classes: dict[str, Type[BaseTool]] = {}
        self._lock = RLock()
        logger.debug("Initialized ToolRegistry")
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool with same name already registered
        """
        with self._lock:
            if tool.name in self._tools:
                logger.warning(f"Overwriting existing tool: {tool.name}")
            
            self._tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name} ({tool.category.value})")
    
    def register_class(
        self,
        tool_class: Type[BaseTool],
        **init_kwargs: Any,
    ) -> None:
        """Register a tool class for lazy instantiation.
        
        The tool will be instantiated on first access.
        
        Args:
            tool_class: Tool class to register
            **init_kwargs: Arguments to pass when instantiating
        """
        with self._lock:
            # Create instance
            tool = tool_class(**init_kwargs)
            self.register(tool)
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                logger.info(f"Unregistered tool: {name}")
                return True
            return False
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        with self._lock:
            return self._tools.get(name)
    
    def get_or_raise(self, name: str) -> BaseTool:
        """Get a tool by name, raising if not found.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance
            
        Raises:
            KeyError: If tool not found
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Tool name
            
        Returns:
            True if registered
        """
        with self._lock:
            return name in self._tools
    
    def list_all(self) -> list[str]:
        """Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        with self._lock:
            return list(self._tools.keys())
    
    def list_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """Get all tools in a category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of tools in the category
        """
        with self._lock:
            return [
                tool for tool in self._tools.values()
                if tool.category == category
            ]
    
    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all registered tools.
        
        Returns:
            List of tool schemas
        """
        with self._lock:
            return [tool.get_schema() for tool in self._tools.values()]
    
    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics for all tools.
        
        Returns:
            Dict with per-tool stats
        """
        with self._lock:
            return {
                "tool_count": len(self._tools),
                "tools": {
                    name: tool.get_stats()
                    for name, tool in self._tools.items()
                },
            }
    
    def clear(self) -> None:
        """Unregister all tools."""
        with self._lock:
            self._tools.clear()
            logger.info("Cleared all tools from registry")
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        with self._lock:
            return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return self.has(name)
    
    def __iter__(self):
        """Iterate over tools."""
        with self._lock:
            return iter(self._tools.values())


# =============================================================================
# Global Default Registry
# =============================================================================

_default_registry: Optional[ToolRegistry] = None
_registry_lock = RLock()


def get_default_registry() -> ToolRegistry:
    """Get the global default tool registry.
    
    Creates the registry on first access with built-in tools.
    
    Returns:
        Default ToolRegistry instance
    """
    global _default_registry
    
    with _registry_lock:
        if _default_registry is None:
            _default_registry = ToolRegistry()
            
            # Register built-in tools
            from argus.tools.base import EchoTool, CalculatorTool
            _default_registry.register(EchoTool())
            _default_registry.register(CalculatorTool())
            
            logger.info("Created default registry with built-in tools")
        
        return _default_registry


def register_tool(tool: BaseTool) -> None:
    """Register a tool with the default registry.
    
    Convenience function for quick registration.
    
    Args:
        tool: Tool to register
    """
    get_default_registry().register(tool)


def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool from the default registry.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance or None
    """
    return get_default_registry().get(name)


def list_tools() -> list[str]:
    """List all tools in the default registry.
    
    Returns:
        List of tool names
    """
    return get_default_registry().list_all()
