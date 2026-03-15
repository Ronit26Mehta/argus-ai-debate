"""
Proposition Decomposer — LLM-powered decomposition of complex propositions.
"""

from __future__ import annotations

import uuid
import re
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PropositionNode:
    """A node in the proposition decomposition tree."""
    node_id: str = field(default_factory=lambda: f"pnode_{uuid.uuid4().hex[:8]}")
    text: str = ""
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    relationship_to_parent: str = "contributing"  # necessary, sufficient, contributing, independent
    depth: int = 0
    is_leaf: bool = True
    posterior: Optional[float] = None
    debated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text[:150],
            "relationship": self.relationship_to_parent,
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "posterior": round(self.posterior, 4) if self.posterior is not None else None,
            "num_children": len(self.children_ids),
        }


class PropositionTree:
    """
    Hierarchical tree of sub-propositions.

    Example:
        >>> tree = PropositionTree("AI will exceed human intelligence by 2040")
        >>> tree.add_child(tree.root_id, "Progress in ML continues at current rate")
        >>> tree.add_child(tree.root_id, "Hardware capabilities scale sufficiently")
    """

    def __init__(self, root_text: str = ""):
        root = PropositionNode(text=root_text, depth=0, is_leaf=True)
        self.root_id = root.node_id
        self._nodes: dict[str, PropositionNode] = {root.node_id: root}

    @property
    def root(self) -> PropositionNode:
        return self._nodes[self.root_id]

    def add_child(
        self,
        parent_id: str,
        text: str,
        relationship: str = "contributing",
    ) -> str:
        """Add a child node and return its ID."""
        parent = self._nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent {parent_id} not found")

        child = PropositionNode(
            text=text,
            parent_id=parent_id,
            relationship_to_parent=relationship,
            depth=parent.depth + 1,
        )
        self._nodes[child.node_id] = child
        parent.children_ids.append(child.node_id)
        parent.is_leaf = False

        return child.node_id

    def get_node(self, node_id: str) -> Optional[PropositionNode]:
        return self._nodes.get(node_id)

    @property
    def leaves(self) -> list[PropositionNode]:
        return [n for n in self._nodes.values() if n.is_leaf]

    @property
    def all_nodes(self) -> list[PropositionNode]:
        return list(self._nodes.values())

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self._nodes.values()), default=0)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def get_children(self, node_id: str) -> list[PropositionNode]:
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "num_nodes": self.num_nodes,
            "max_depth": self.max_depth,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
        }


class PropositionDecomposer:
    """
    Decomposes complex propositions into sub-proposition trees.

    Uses LLM-powered analysis or heuristic decomposition to break
    compound propositions into atomic, debatable components.

    Example:
        >>> decomposer = PropositionDecomposer(max_depth=3)
        >>> tree = decomposer.decompose("Market will crash AND inflation will rise above 5%")
        >>> print(tree.num_nodes)  # 3 (root + 2 children)
    """

    # Compound patterns
    COMPOUND_PATTERNS = [
        (r'\band\b', 'contributing'),
        (r'\bbut\b', 'contributing'),
        (r'\bor\b', 'sufficient'),
        (r'\bif\b.*\bthen\b', 'necessary'),
        (r'\bbecause\b', 'necessary'),
        (r'\brequires?\b', 'necessary'),
        (r'\bimplies?\b', 'contributing'),
        (r'\bleads?\s+to\b', 'contributing'),
    ]

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_depth: int = 3,
        max_children: int = 5,
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.max_children = max_children

    def decompose(self, proposition: str) -> PropositionTree:
        """
        Decompose a proposition into a tree.

        Args:
            proposition: Complex proposition text

        Returns:
            PropositionTree with sub-propositions
        """
        tree = PropositionTree(proposition)
        self._decompose_recursive(tree, tree.root_id, 1)
        logger.info(
            f"Decomposed into tree: {tree.num_nodes} nodes, "
            f"depth {tree.max_depth}, {len(tree.leaves)} leaves"
        )
        return tree

    def _decompose_recursive(
        self,
        tree: PropositionTree,
        node_id: str,
        current_depth: int,
    ) -> None:
        """Recursively decompose a node into children."""
        if current_depth > self.max_depth:
            return

        node = tree.get_node(node_id)
        if not node:
            return

        # Try to split using compound patterns
        sub_props = self._split_proposition(node.text)

        if len(sub_props) <= 1:
            return  # Atomic — no further decomposition

        for sub_text, relationship in sub_props[:self.max_children]:
            child_id = tree.add_child(node_id, sub_text, relationship)
            self._decompose_recursive(tree, child_id, current_depth + 1)

    def _split_proposition(self, text: str) -> list[tuple[str, str]]:
        """Split a proposition using compound patterns."""
        results = []

        for pattern, relationship in self.COMPOUND_PATTERNS:
            parts = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                for part in parts:
                    clean = part.strip().strip('.,;:')
                    if len(clean) > 15:
                        results.append((clean, relationship))
                if results:
                    return results

        # Try splitting by semicolons
        if ';' in text:
            parts = text.split(';')
            for part in parts:
                clean = part.strip()
                if len(clean) > 15:
                    results.append((clean, "contributing"))

        return results
