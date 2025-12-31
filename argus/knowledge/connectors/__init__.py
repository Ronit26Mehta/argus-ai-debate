"""
ARGUS Connectors Framework.

Provides extensible connectors for external data sources.
Users can create custom connectors by subclassing BaseConnector.

Example:
    >>> from argus.knowledge.connectors import BaseConnector, ConnectorRegistry
    >>> 
    >>> # Create a custom connector
    >>> class MyAPIConnector(BaseConnector):
    ...     name = "my_api"
    ...     
    ...     def fetch(self, query: str, **kwargs):
    ...         # Fetch from API
    ...         return [Document(...)]
    >>> 
    >>> # Register and use
    >>> registry = ConnectorRegistry()
    >>> registry.register(MyAPIConnector())
"""

from argus.knowledge.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResult,
    ConnectorRegistry,
    get_default_registry,
    register_connector,
)
from argus.knowledge.connectors.web import (
    WebConnector,
    WebConnectorConfig,
    RobotsTxtParser,
    RobotsTxtRules,
)
from argus.knowledge.connectors.arxiv import (
    ArxivConnector,
    ArxivConnectorConfig,
)
from argus.knowledge.connectors.crossref import (
    CrossRefConnector,
    CrossRefConnectorConfig,
)

__all__ = [
    # Base
    "BaseConnector",
    "ConnectorConfig",
    "ConnectorResult",
    "ConnectorRegistry",
    "get_default_registry",
    "register_connector",
    # Web
    "WebConnector",
    "WebConnectorConfig",
    "RobotsTxtParser",
    "RobotsTxtRules",
    # arXiv
    "ArxivConnector",
    "ArxivConnectorConfig",
    # CrossRef
    "CrossRefConnector",
    "CrossRefConnectorConfig",
]

