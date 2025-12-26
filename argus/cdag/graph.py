"""
C-DAG Graph Implementation.

The Conceptual Debate Graph (C-DAG) is the core data structure for ARGUS.
It maintains nodes (propositions, evidence, rebuttals) and edges (support, attack)
with operations for querying, modification, and influence propagation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Any, Iterator, Callable
from collections import defaultdict

import networkx as nx

from argus.cdag.nodes import (
    NodeBase,
    Proposition,
    Evidence,
    Rebuttal,
    Finding,
    Assumption,
    NodeStatus,
)
from argus.cdag.edges import (
    Edge,
    EdgeType,
    EdgePolarity,
    create_support_edge,
    create_attack_edge,
)

logger = logging.getLogger(__name__)


class CDAG:
    """
    Conceptual Debate Graph.
    
    A typed, directed multigraph for structured argumentation.
    Nodes represent claims, evidence, and rebuttals.
    Edges represent support, attack, and other relations.
    
    Example:
        >>> graph = CDAG()
        >>> 
        >>> # Add a proposition
        >>> prop = Proposition(text="Drug X is effective", prior=0.5)
        >>> graph.add_proposition(prop)
        >>> 
        >>> # Add supporting evidence
        >>> evidence = Evidence(
        ...     text="RCT showed 25% improvement",
        ...     polarity=1,
        ...     confidence=0.85,
        ... )
        >>> graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)
        >>> 
        >>> # Get posterior
        >>> posterior = graph.compute_posterior(prop.id)
        >>> print(f"Posterior: {posterior:.3f}")
    
    Attributes:
        name: Optional graph name
        created_at: Creation timestamp
        metadata: Additional graph metadata
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize a new C-DAG.
        
        Args:
            name: Optional graph name
            metadata: Additional metadata
        """
        self.name = name or "cdag"
        self.created_at = datetime.utcnow()
        self.metadata = metadata or {}
        
        # Internal graph using NetworkX
        self._graph = nx.DiGraph()
        
        # Node and edge storage
        self._nodes: dict[str, NodeBase] = {}
        self._edges: dict[str, Edge] = {}
        
        # Index structures for fast lookup
        self._propositions: set[str] = set()
        self._evidence: set[str] = set()
        self._rebuttals: set[str] = set()
        
        # Edge indexes
        self._incoming: dict[str, list[str]] = defaultdict(list)
        self._outgoing: dict[str, list[str]] = defaultdict(list)
        
        logger.debug(f"Created new C-DAG: {self.name}")
    
    # ==================== Node Operations ====================
    
    def add_node(self, node: NodeBase) -> str:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
            
        Returns:
            Node ID
        """
        if node.id in self._nodes:
            logger.warning(f"Node {node.id} already exists, updating")
        
        self._nodes[node.id] = node
        self._graph.add_node(node.id, node=node)
        
        # Update type indexes
        if isinstance(node, Proposition):
            self._propositions.add(node.id)
        elif isinstance(node, Evidence):
            self._evidence.add(node.id)
        elif isinstance(node, Rebuttal):
            self._rebuttals.add(node.id)
        
        logger.debug(f"Added node: {node.id} ({type(node).__name__})")
        return node.id
    
    def add_proposition(self, prop: Proposition) -> str:
        """
        Add a proposition to the graph.
        
        Args:
            prop: Proposition to add
            
        Returns:
            Proposition ID
        """
        return self.add_node(prop)
    
    def add_evidence(
        self,
        evidence: Evidence,
        target_id: str,
        edge_type: EdgeType = EdgeType.SUPPORTS,
        **edge_kwargs: Any,
    ) -> tuple[str, str]:
        """
        Add evidence linked to a target node.
        
        Args:
            evidence: Evidence node to add
            target_id: Target node ID (usually a proposition)
            edge_type: Type of edge (SUPPORTS or ATTACKS)
            **edge_kwargs: Additional edge attributes
            
        Returns:
            Tuple of (evidence_id, edge_id)
        """
        # Verify target exists
        if target_id not in self._nodes:
            raise ValueError(f"Target node {target_id} does not exist")
        
        # Add evidence node
        evidence_id = self.add_node(evidence)
        
        # Determine polarity from evidence or edge type
        if evidence.polarity > 0 or edge_type == EdgeType.SUPPORTS:
            polarity = EdgePolarity.POSITIVE
            edge_type = EdgeType.SUPPORTS
        elif evidence.polarity < 0 or edge_type == EdgeType.ATTACKS:
            polarity = EdgePolarity.NEGATIVE
            edge_type = EdgeType.ATTACKS
        else:
            polarity = EdgePolarity.NEUTRAL
        
        # Create edge
        edge = Edge(
            source_id=evidence_id,
            target_id=target_id,
            edge_type=edge_type,
            polarity=polarity,
            confidence=evidence.confidence,
            relevance=evidence.relevance,
            quality=evidence.quality,
            **edge_kwargs,
        )
        
        edge_id = self.add_edge(edge)
        
        return evidence_id, edge_id
    
    def add_rebuttal(
        self,
        rebuttal: Rebuttal,
        target_id: str,
        **edge_kwargs: Any,
    ) -> tuple[str, str]:
        """
        Add a rebuttal attacking another node.
        
        Args:
            rebuttal: Rebuttal node to add
            target_id: Target node ID
            **edge_kwargs: Additional edge attributes
            
        Returns:
            Tuple of (rebuttal_id, edge_id)
        """
        if target_id not in self._nodes:
            raise ValueError(f"Target node {target_id} does not exist")
        
        # Update rebuttal with target
        object.__setattr__(rebuttal, "target_id", target_id)
        
        # Add rebuttal node
        rebuttal_id = self.add_node(rebuttal)
        
        # Create attack edge
        edge = Edge(
            source_id=rebuttal_id,
            target_id=target_id,
            edge_type=EdgeType.REBUTS,
            polarity=EdgePolarity.NEGATIVE,
            confidence=rebuttal.confidence,
            weight=rebuttal.strength,
            **edge_kwargs,
        )
        
        edge_id = self.add_edge(edge)
        
        return rebuttal_id, edge_id
    
    def get_node(self, node_id: str) -> Optional[NodeBase]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node or None if not found
        """
        return self._nodes.get(node_id)
    
    def get_proposition(self, prop_id: str) -> Optional[Proposition]:
        """
        Get a proposition by ID.
        
        Args:
            prop_id: Proposition ID
            
        Returns:
            Proposition or None
        """
        node = self._nodes.get(prop_id)
        if isinstance(node, Proposition):
            return node
        return None
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and its connected edges.
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if node_id not in self._nodes:
            return False
        
        # Remove connected edges
        for edge_id in list(self._incoming.get(node_id, [])):
            self.remove_edge(edge_id)
        for edge_id in list(self._outgoing.get(node_id, [])):
            self.remove_edge(edge_id)
        
        # Remove from graph and storage
        self._graph.remove_node(node_id)
        del self._nodes[node_id]
        
        # Update indexes
        self._propositions.discard(node_id)
        self._evidence.discard(node_id)
        self._rebuttals.discard(node_id)
        
        logger.debug(f"Removed node: {node_id}")
        return True
    
    # ==================== Edge Operations ====================
    
    def add_edge(self, edge: Edge) -> str:
        """
        Add an edge to the graph.
        
        Args:
            edge: Edge to add
            
        Returns:
            Edge ID
        """
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} does not exist")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} does not exist")
        
        self._edges[edge.id] = edge
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.id,
            edge=edge,
        )
        
        # Update indexes
        self._incoming[edge.target_id].append(edge.id)
        self._outgoing[edge.source_id].append(edge.id)
        
        logger.debug(f"Added edge: {edge.id} ({edge.edge_type.value})")
        return edge.id
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """
        Get an edge by ID.
        
        Args:
            edge_id: Edge ID
            
        Returns:
            Edge or None
        """
        return self._edges.get(edge_id)
    
    def remove_edge(self, edge_id: str) -> bool:
        """
        Remove an edge.
        
        Args:
            edge_id: Edge ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if edge_id not in self._edges:
            return False
        
        edge = self._edges[edge_id]
        
        # Remove from graph
        if self._graph.has_edge(edge.source_id, edge.target_id):
            self._graph.remove_edge(edge.source_id, edge.target_id)
        
        # Update indexes
        if edge_id in self._incoming.get(edge.target_id, []):
            self._incoming[edge.target_id].remove(edge_id)
        if edge_id in self._outgoing.get(edge.source_id, []):
            self._outgoing[edge.source_id].remove(edge_id)
        
        del self._edges[edge_id]
        
        logger.debug(f"Removed edge: {edge_id}")
        return True
    
    # ==================== Query Operations ====================
    
    def get_incoming_edges(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        polarity: Optional[EdgePolarity] = None,
    ) -> list[Edge]:
        """
        Get edges pointing to a node.
        
        Args:
            node_id: Node ID
            edge_type: Filter by edge type
            polarity: Filter by polarity
            
        Returns:
            List of incoming edges
        """
        edge_ids = self._incoming.get(node_id, [])
        edges = [self._edges[eid] for eid in edge_ids if eid in self._edges]
        
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        if polarity:
            edges = [e for e in edges if e.polarity == polarity]
        
        return edges
    
    def get_outgoing_edges(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
    ) -> list[Edge]:
        """
        Get edges originating from a node.
        
        Args:
            node_id: Node ID
            edge_type: Filter by edge type
            
        Returns:
            List of outgoing edges
        """
        edge_ids = self._outgoing.get(node_id, [])
        edges = [self._edges[eid] for eid in edge_ids if eid in self._edges]
        
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        
        return edges
    
    def get_supporting_evidence(self, prop_id: str) -> list[Evidence]:
        """
        Get all evidence supporting a proposition.
        
        Args:
            prop_id: Proposition ID
            
        Returns:
            List of supporting evidence nodes
        """
        edges = self.get_incoming_edges(
            prop_id,
            edge_type=EdgeType.SUPPORTS,
            polarity=EdgePolarity.POSITIVE,
        )
        
        evidence = []
        for edge in edges:
            node = self._nodes.get(edge.source_id)
            if isinstance(node, Evidence):
                evidence.append(node)
        
        return evidence
    
    def get_attacking_evidence(self, prop_id: str) -> list[Evidence]:
        """
        Get all evidence attacking a proposition.
        
        Args:
            prop_id: Proposition ID
            
        Returns:
            List of attacking evidence nodes
        """
        edges = self.get_incoming_edges(
            prop_id,
            edge_type=EdgeType.ATTACKS,
            polarity=EdgePolarity.NEGATIVE,
        )
        
        evidence = []
        for edge in edges:
            node = self._nodes.get(edge.source_id)
            if isinstance(node, Evidence):
                evidence.append(node)
        
        return evidence
    
    def get_all_propositions(self) -> list[Proposition]:
        """
        Get all propositions in the graph.
        
        Returns:
            List of all propositions
        """
        return [
            self._nodes[pid]
            for pid in self._propositions
            if isinstance(self._nodes.get(pid), Proposition)
        ]
    
    def get_all_evidence(self) -> list[Evidence]:
        """
        Get all evidence nodes in the graph.
        
        Returns:
            List of all evidence nodes
        """
        return [
            self._nodes[eid]
            for eid in self._evidence
            if isinstance(self._nodes.get(eid), Evidence)
        ]
    
    # ==================== Path and Analysis ====================
    
    def get_support_path(
        self,
        prop_id: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """
        Get paths of support to a proposition.
        
        Args:
            prop_id: Proposition ID
            max_depth: Maximum path depth
            
        Returns:
            List of support paths (each path is list of node IDs)
        """
        paths: list[list[str]] = []
        
        def _trace_path(
            current_id: str,
            current_path: list[str],
            depth: int,
        ) -> None:
            if depth > max_depth:
                return
            
            edges = self.get_incoming_edges(
                current_id,
                polarity=EdgePolarity.POSITIVE,
            )
            
            for edge in edges:
                new_path = [edge.source_id] + current_path
                paths.append(new_path)
                _trace_path(edge.source_id, new_path, depth + 1)
        
        _trace_path(prop_id, [prop_id], 1)
        return paths
    
    def get_attack_path(
        self,
        prop_id: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """
        Get paths of attack to a proposition.
        
        Args:
            prop_id: Proposition ID
            max_depth: Maximum path depth
            
        Returns:
            List of attack paths
        """
        paths: list[list[str]] = []
        
        def _trace_path(
            current_id: str,
            current_path: list[str],
            depth: int,
        ) -> None:
            if depth > max_depth:
                return
            
            edges = self.get_incoming_edges(
                current_id,
                polarity=EdgePolarity.NEGATIVE,
            )
            
            for edge in edges:
                new_path = [edge.source_id] + current_path
                paths.append(new_path)
                _trace_path(edge.source_id, new_path, depth + 1)
        
        _trace_path(prop_id, [prop_id], 1)
        return paths
    
    # ==================== Scoring and Aggregation ====================
    
    def compute_support_score(self, prop_id: str) -> float:
        """
        Compute aggregate support score for a proposition.
        
        Sum of weighted support from all supporting edges.
        
        Args:
            prop_id: Proposition ID
            
        Returns:
            Aggregate support score
        """
        edges = self.get_incoming_edges(
            prop_id,
            polarity=EdgePolarity.POSITIVE,
        )
        
        total = 0.0
        for edge in edges:
            source = self._nodes.get(edge.source_id)
            if source:
                # Get source score (confidence for evidence)
                source_score = getattr(source, "confidence", 0.5)
                total += edge.effective_weight * source_score
        
        return total
    
    def compute_attack_score(self, prop_id: str) -> float:
        """
        Compute aggregate attack score for a proposition.
        
        Sum of weighted attack from all attacking edges.
        
        Args:
            prop_id: Proposition ID
            
        Returns:
            Aggregate attack score
        """
        edges = self.get_incoming_edges(
            prop_id,
            polarity=EdgePolarity.NEGATIVE,
        )
        
        total = 0.0
        for edge in edges:
            source = self._nodes.get(edge.source_id)
            if source:
                source_score = getattr(source, "confidence", 0.5)
                total += edge.effective_weight * source_score
        
        return total
    
    def compute_net_influence(self, prop_id: str) -> float:
        """
        Compute net influence (support - attack).
        
        Args:
            prop_id: Proposition ID
            
        Returns:
            Net influence score
        """
        return self.compute_support_score(prop_id) - self.compute_attack_score(prop_id)
    
    # ==================== Graph Properties ====================
    
    @property
    def num_nodes(self) -> int:
        """Total number of nodes."""
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        """Total number of edges."""
        return len(self._edges)
    
    @property
    def num_propositions(self) -> int:
        """Number of propositions."""
        return len(self._propositions)
    
    @property
    def num_evidence(self) -> int:
        """Number of evidence nodes."""
        return len(self._evidence)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CDAG(name='{self.name}', "
            f"nodes={self.num_nodes}, "
            f"edges={self.num_edges}, "
            f"propositions={self.num_propositions})"
        )
    
    def summary(self) -> dict[str, Any]:
        """
        Get graph summary statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_propositions": self.num_propositions,
            "num_evidence": self.num_evidence,
            "num_rebuttals": len(self._rebuttals),
        }
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Get underlying NetworkX graph.
        
        Returns:
            NetworkX DiGraph
        """
        return self._graph.copy()
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._graph.clear()
        self._nodes.clear()
        self._edges.clear()
        self._propositions.clear()
        self._evidence.clear()
        self._rebuttals.clear()
        self._incoming.clear()
        self._outgoing.clear()
        logger.debug(f"Cleared C-DAG: {self.name}")
