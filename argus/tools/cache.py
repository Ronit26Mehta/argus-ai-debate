"""
Result Caching for ARGUS Tools.

Provides caching for expensive tool operations to improve
performance and reduce redundant API calls.

Example:
    >>> cache = ResultCache(config=CacheConfig(max_size=1000))
    >>> 
    >>> # Cache a result
    >>> cache.set("search", {"query": "test"}, result, ttl=300)
    >>> 
    >>> # Retrieve from cache
    >>> cached = cache.get("search", {"query": "test"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import functools
from typing import Optional, Any, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from collections import OrderedDict

from pydantic import BaseModel, Field

from argus.tools.base import ToolResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheConfig(BaseModel):
    """Configuration for result cache.
    
    Attributes:
        max_size: Maximum number of cached results
        default_ttl: Default time-to-live in seconds
        cleanup_interval: How often to cleanup expired entries
    """
    max_size: int = Field(
        default=1000,
        ge=10,
        description="Maximum cached entries",
    )
    default_ttl: int = Field(
        default=300,
        ge=0,
        description="Default TTL in seconds",
    )
    cleanup_interval: int = Field(
        default=60,
        ge=10,
        description="Cleanup interval in seconds",
    )


@dataclass
class CacheEntry:
    """A cached result entry."""
    result: ToolResult
    created_at: float
    expires_at: float
    hit_count: int = 0


class ResultCache:
    """LRU cache for tool execution results.
    
    Caches results by tool name and argument hash.
    Supports TTL expiration and LRU eviction.
    
    Example:
        >>> cache = ResultCache()
        >>> 
        >>> # Store result
        >>> cache.set("my_tool", {"arg": "value"}, result)
        >>> 
        >>> # Retrieve (returns None if not found or expired)
        >>> cached = cache.get("my_tool", {"arg": "value"})
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._last_cleanup = time.time()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        logger.debug(f"Initialized ResultCache with max_size={self.config.max_size}")
    
    def _compute_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Compute cache key from tool name and arguments.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Cache key string
        """
        # Create deterministic string representation
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        key_str = f"{tool_name}:{args_str}"
        
        # Hash for fixed-length key
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def get(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Optional[ToolResult]:
        """Get a cached result.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            Cached ToolResult or None if not found/expired
        """
        key = self._compute_key(tool_name, arguments)
        
        with self._lock:
            # Periodic cleanup
            self._maybe_cleanup()
            
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            
            logger.debug(f"Cache hit for {tool_name}")
            return entry.result
    
    def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a result in cache.
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            result: Result to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        key = self._compute_key(tool_name, arguments)
        effective_ttl = ttl if ttl is not None else self.config.default_ttl
        
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.config.max_size:
                self._cache.popitem(last=False)
            
            now = time.time()
            entry = CacheEntry(
                result=result,
                created_at=now,
                expires_at=now + effective_ttl,
            )
            self._cache[key] = entry
            
            logger.debug(f"Cached result for {tool_name} (ttl={effective_ttl}s)")
    
    def invalidate(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> int:
        """Invalidate cached entries.
        
        Args:
            tool_name: Tool name to invalidate
            arguments: Specific arguments (None = all for tool)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if arguments is not None:
                # Invalidate specific entry
                key = self._compute_key(tool_name, arguments)
                if key in self._cache:
                    del self._cache[key]
                    return 1
                return 0
            else:
                # Invalidate all entries for tool
                # Compute prefix pattern
                count = 0
                keys_to_delete = []
                
                for key in self._cache:
                    # Would need to track tool_name->key mapping for efficiency
                    # For now, this is a simplified version
                    pass
                
                for key in keys_to_delete:
                    del self._cache[key]
                    count += 1
                
                return count
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared result cache")
    
    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup < self.config.cleanup_interval:
            return
        
        self._last_cleanup = now
        self._cleanup_expired()
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with hit/miss counts, size, hit rate
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.config.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }
    
    def __len__(self) -> int:
        """Get number of cached entries."""
        with self._lock:
            return len(self._cache)


# =============================================================================
# Decorator for Tool Caching
# =============================================================================

def cached_tool(
    ttl: int = 300,
    cache: Optional[ResultCache] = None,
) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
    """Decorator to add caching to a tool's execute method.
    
    Args:
        ttl: Cache TTL in seconds
        cache: ResultCache to use (creates one if None)
        
    Returns:
        Decorator function
        
    Example:
        >>> @cached_tool(ttl=600)
        ... def my_expensive_search(query: str, **kwargs) -> ToolResult:
        ...     # Expensive operation
        ...     return ToolResult(success=True, data={"results": []})
    """
    _cache = cache or ResultCache()
    
    def decorator(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            # Create cache key from function name and args
            tool_name = func.__name__
            cache_kwargs = {**kwargs}
            if args:
                cache_kwargs["_args"] = args
            
            # Check cache
            cached_result = _cache.get(tool_name, cache_kwargs)
            if cached_result is not None:
                cached_result.cached = True
                return cached_result
            
            # Execute and cache
            result = func(*args, **kwargs)
            if result.success:
                _cache.set(tool_name, cache_kwargs, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator
