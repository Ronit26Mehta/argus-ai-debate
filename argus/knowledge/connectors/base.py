"""
Base Connector Classes for ARGUS.

Defines the abstract base class for creating custom connectors
that fetch data from external sources (APIs, databases, web, etc.).

Users can extend the framework by subclassing BaseConnector.

Example:
    >>> class ArxivConnector(BaseConnector):
    ...     name = "arxiv"
    ...     description = "Fetch papers from arXiv"
    ...     
    ...     def fetch(self, query: str, **kwargs) -> ConnectorResult:
    ...         # Implementation here
    ...         return ConnectorResult(success=True, documents=[...])
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from argus.core.models import Document

logger = logging.getLogger(__name__)


class ConnectorConfig(BaseModel):
    """Configuration for a connector.
    
    Attributes:
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        rate_limit: Maximum requests per minute
        cache_ttl: Result cache TTL in seconds
    """
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Request timeout in seconds",
    )
    retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Number of retry attempts",
    )
    rate_limit: int = Field(
        default=60,
        ge=1,
        description="Max requests per minute",
    )
    cache_ttl: int = Field(
        default=300,
        ge=0,
        description="Cache TTL in seconds",
    )


@dataclass
class ConnectorResult:
    """Result from a connector fetch operation.
    
    Attributes:
        success: Whether the operation succeeded
        documents: List of fetched documents
        error: Error message if failed
        metadata: Additional result metadata
    """
    success: bool
    documents: list["Document"] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fetch_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "document_count": len(self.documents),
            "error": self.error,
            "metadata": self.metadata,
            "fetch_time_ms": self.fetch_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_error(cls, error: str) -> "ConnectorResult":
        """Create a failed result."""
        return cls(success=False, error=error)


class BaseConnector(ABC):
    """Abstract base class for external data connectors.
    
    Subclass this to create connectors for external data sources
    like APIs, databases, web scraping, etc.
    
    Class Attributes:
        name: Unique identifier for the connector
        description: Human-readable description
        
    Example:
        >>> class GithubConnector(BaseConnector):
        ...     name = "github"
        ...     description = "Fetch from GitHub repositories"
        ...     
        ...     def fetch(self, query: str, **kwargs) -> ConnectorResult:
        ...         # Fetch repos matching query
        ...         return ConnectorResult(success=True, documents=[...])
    """
    
    name: str = "base_connector"
    description: str = "Base connector"
    version: str = "1.0.0"
    
    def __init__(self, config: Optional[ConnectorConfig] = None):
        """Initialize the connector.
        
        Args:
            config: Connector configuration
        """
        self.config = config or ConnectorConfig()
        self._fetch_count = 0
        self._total_documents = 0
        self._last_request_time = 0.0
        logger.debug(f"Initialized connector: {self.name}")
    
    @abstractmethod
    def fetch(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> ConnectorResult:
        """Fetch documents matching the query.
        
        This method must be implemented by all connector subclasses.
        
        Args:
            query: Search query
            max_results: Maximum documents to return
            **kwargs: Additional connector-specific arguments
            
        Returns:
            ConnectorResult with fetched documents
        """
        pass
    
    def test_connection(self) -> bool:
        """Test if the connector can connect to its data source.
        
        Override this to implement connection testing.
        
        Returns:
            True if connection successful
        """
        return True
    
    def validate_config(self) -> Optional[str]:
        """Validate connector configuration.
        
        Override to add configuration validation.
        
        Returns:
            Error message if invalid, None if valid
        """
        return None
    
    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit."""
        if self.config.rate_limit <= 0:
            return
        
        min_interval = 60.0 / self.config.rate_limit
        elapsed = time.time() - self._last_request_time
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def __call__(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> ConnectorResult:
        """Fetch documents (callable interface).
        
        Wraps fetch() with rate limiting and timing.
        
        Args:
            query: Search query
            max_results: Maximum results
            **kwargs: Additional arguments
            
        Returns:
            ConnectorResult
        """
        # Rate limit
        self._rate_limit_wait()
        
        # Execute with timing
        start = time.perf_counter()
        try:
            result = self.fetch(query, max_results, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.fetch_time_ms = elapsed_ms
            
            # Update stats
            self._fetch_count += 1
            self._total_documents += len(result.documents)
            
            return result
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Connector {self.name} fetch failed: {e}")
            return ConnectorResult(
                success=False,
                error=str(e),
                fetch_time_ms=elapsed_ms,
            )
    
    def get_schema(self) -> dict[str, Any]:
        """Get connector schema/metadata.
        
        Returns:
            Dict with connector info
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "config": self.config.model_dump(),
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get connector statistics.
        
        Returns:
            Dict with fetch counts, etc.
        """
        return {
            "name": self.name,
            "fetch_count": self._fetch_count,
            "total_documents": self._total_documents,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ConnectorRegistry:
    """Registry for managing connectors.
    
    Provides centralized management of connector instances.
    
    Example:
        >>> registry = ConnectorRegistry()
        >>> registry.register(MyConnector())
        >>> connector = registry.get("my_connector")
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._connectors: dict[str, BaseConnector] = {}
        self._lock = RLock()
        logger.debug("Initialized ConnectorRegistry")
    
    def register(self, connector: BaseConnector) -> None:
        """Register a connector.
        
        Args:
            connector: Connector instance to register
        """
        with self._lock:
            if connector.name in self._connectors:
                logger.warning(f"Overwriting connector: {connector.name}")
            
            # Validate config
            error = connector.validate_config()
            if error:
                raise ValueError(f"Invalid connector config: {error}")
            
            self._connectors[connector.name] = connector
            logger.info(f"Registered connector: {connector.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a connector.
        
        Args:
            name: Connector name
            
        Returns:
            True if removed
        """
        with self._lock:
            if name in self._connectors:
                del self._connectors[name]
                return True
            return False
    
    def get(self, name: str) -> Optional[BaseConnector]:
        """Get a connector by name.
        
        Args:
            name: Connector name
            
        Returns:
            Connector or None
        """
        with self._lock:
            return self._connectors.get(name)
    
    def list_all(self) -> list[str]:
        """Get all registered connector names.
        
        Returns:
            List of connector names
        """
        with self._lock:
            return list(self._connectors.keys())
    
    def get_all(self) -> list[BaseConnector]:
        """Get all registered connectors.
        
        Returns:
            List of connectors
        """
        with self._lock:
            return list(self._connectors.values())
    
    def fetch_from_all(
        self,
        query: str,
        max_results_per_connector: int = 5,
    ) -> dict[str, ConnectorResult]:
        """Fetch from all registered connectors.
        
        Args:
            query: Search query
            max_results_per_connector: Max results per connector
            
        Returns:
            Dict mapping connector name to results
        """
        results = {}
        for name, connector in self._connectors.items():
            try:
                results[name] = connector(query, max_results_per_connector)
            except Exception as e:
                results[name] = ConnectorResult.from_error(str(e))
        return results
    
    def __len__(self) -> int:
        return len(self._connectors)
    
    def __contains__(self, name: str) -> bool:
        return name in self._connectors


# =============================================================================
# Global Default Registry
# =============================================================================

_default_registry: Optional[ConnectorRegistry] = None
_registry_lock = RLock()


def get_default_registry() -> ConnectorRegistry:
    """Get the default connector registry.
    
    Returns:
        ConnectorRegistry instance
    """
    global _default_registry
    
    with _registry_lock:
        if _default_registry is None:
            _default_registry = ConnectorRegistry()
        return _default_registry


def register_connector(connector: BaseConnector) -> None:
    """Register a connector with the default registry.
    
    Args:
        connector: Connector to register
    """
    get_default_registry().register(connector)
