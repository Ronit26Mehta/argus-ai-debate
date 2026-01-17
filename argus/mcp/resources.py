"""
ARGUS MCP Resource Adapters.

Expose ARGUS components as MCP resources.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, TYPE_CHECKING

from argus.mcp.config import MCPResourceSchema

if TYPE_CHECKING:
    from argus.cdag.graph import CDAG
    from argus.provenance.ledger import ProvenanceLedger

logger = logging.getLogger(__name__)


class BaseResourceAdapter(ABC):
    """Base class for MCP resource adapters."""
    
    @property
    @abstractmethod
    def uri(self) -> str:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    def description(self) -> str:
        return ""
    
    @property
    def mime_type(self) -> str:
        return "application/json"
    
    @abstractmethod
    def read(self) -> Any:
        pass
    
    def get_schema(self) -> MCPResourceSchema:
        return MCPResourceSchema(
            uri=self.uri, name=self.name,
            description=self.description, mime_type=self.mime_type
        )


class CDAGResource(BaseResourceAdapter):
    """Expose CDAG debate graph as MCP resource."""
    
    def __init__(self, graph: "CDAG", resource_id: str = "default"):
        self._graph = graph
        self._resource_id = resource_id
    
    @property
    def uri(self) -> str:
        return f"argus://cdag/{self._resource_id}"
    
    @property
    def name(self) -> str:
        return f"CDAG-{self._resource_id}"
    
    @property
    def description(self) -> str:
        return "Conceptual Debate Graph with propositions, evidence, and rebuttals"
    
    def read(self) -> Dict[str, Any]:
        """Read the CDAG structure."""
        nodes = []
        edges = []
        if hasattr(self._graph, 'nodes'):
            for node_id, node in self._graph.nodes.items():
                nodes.append({
                    "id": node_id,
                    "type": node.node_type.value if hasattr(node, 'node_type') else "unknown",
                    "text": getattr(node, 'text', str(node)),
                })
        if hasattr(self._graph, 'edges'):
            for edge in self._graph.edges:
                edges.append({
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.edge_type.value if hasattr(edge, 'edge_type') else "unknown",
                })
        return {"nodes": nodes, "edges": edges, "resource_id": self._resource_id}


class EvidenceResource(BaseResourceAdapter):
    """Expose evidence nodes from CDAG as resource."""
    
    def __init__(self, graph: "CDAG", resource_id: str = "default"):
        self._graph = graph
        self._resource_id = resource_id
    
    @property
    def uri(self) -> str:
        return f"argus://evidence/{self._resource_id}"
    
    @property
    def name(self) -> str:
        return f"Evidence-{self._resource_id}"
    
    @property
    def description(self) -> str:
        return "Evidence nodes from the debate graph"
    
    def read(self) -> Dict[str, Any]:
        evidence = []
        if hasattr(self._graph, 'get_evidence'):
            for e in self._graph.get_evidence():
                evidence.append({
                    "id": e.id,
                    "text": e.text,
                    "confidence": getattr(e, 'confidence', None),
                    "source": getattr(e, 'source', None),
                })
        return {"evidence": evidence, "count": len(evidence)}


class ProvenanceResource(BaseResourceAdapter):
    """Expose provenance ledger as MCP resource."""
    
    def __init__(self, ledger: "ProvenanceLedger"):
        self._ledger = ledger
    
    @property
    def uri(self) -> str:
        return "argus://provenance/ledger"
    
    @property
    def name(self) -> str:
        return "Provenance-Ledger"
    
    @property
    def description(self) -> str:
        return "PROV-O compatible audit log of system activities"
    
    def read(self) -> Dict[str, Any]:
        events = []
        if hasattr(self._ledger, 'query'):
            for event in self._ledger.query(limit=100):
                events.append(event.to_dict() if hasattr(event, 'to_dict') else str(event))
        return {"events": events, "count": len(events)}


class ConfigResource(BaseResourceAdapter):
    """Expose current ARGUS configuration as resource."""
    
    def __init__(self, config: Any = None):
        self._config = config
    
    @property
    def uri(self) -> str:
        return "argus://config/current"
    
    @property
    def name(self) -> str:
        return "Config-Current"
    
    @property
    def description(self) -> str:
        return "Current ARGUS system configuration"
    
    def read(self) -> Dict[str, Any]:
        if self._config is None:
            from argus.core.config import get_config
            self._config = get_config()
        if hasattr(self._config, 'model_dump'):
            return self._config.model_dump()
        elif hasattr(self._config, 'dict'):
            return self._config.dict()
        return {"config": str(self._config)}


class ResourceRegistry:
    """Registry for MCP resources."""
    
    def __init__(self):
        self._resources: Dict[str, BaseResourceAdapter] = {}
    
    def register(self, adapter: BaseResourceAdapter) -> None:
        self._resources[adapter.uri] = adapter
        logger.debug(f"Registered resource: {adapter.uri}")
    
    def get(self, uri: str) -> Optional[BaseResourceAdapter]:
        return self._resources.get(uri)
    
    def list_all(self) -> list[MCPResourceSchema]:
        return [r.get_schema() for r in self._resources.values()]
    
    def read(self, uri: str) -> Optional[Any]:
        adapter = self.get(uri)
        if adapter:
            return adapter.read()
        return None
