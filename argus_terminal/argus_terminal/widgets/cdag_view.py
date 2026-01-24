"""
C-DAG View Widget - ASCII visualization of Conceptual Debate Graph.

Features:
- ASCII box-drawing graph
- Node display (Proposition, Evidence, Rebuttal)
- Edge visualization with polarity colors
- Scrollable view
"""

from textual.app import ComposeResult
from textual.containers import Vertical, ScrollableContainer
from textual.widgets import Static
from textual.reactive import reactive
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """C-DAG node types."""
    PROPOSITION = "proposition"
    EVIDENCE = "evidence"
    REBUTTAL = "rebuttal"
    FINDING = "finding"


class EdgeType(Enum):
    """C-DAG edge types."""
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    REBUTS = "rebuts"


@dataclass
class CDAGNode:
    """C-DAG node data."""
    id: str
    type: NodeType
    text: str
    confidence: float = 0.0
    polarity: int = 0  # 1=support, -1=attack, 0=neutral


@dataclass
class CDAGEdge:
    """C-DAG edge data."""
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0


class CDAGViewWidget(Vertical):
    """ASCII C-DAG graph visualization widget."""
    
    DEFAULT_CSS = """
    CDAGViewWidget {
        width: 100%;
        height: 100%;
        background: #0a0a0a;
        border: double #665200;
        padding: 1;
    }
    
    CDAGViewWidget #cdag-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    CDAGViewWidget #cdag-container {
        width: 100%;
        height: 1fr;
        background: #0a0a0a;
    }
    
    CDAGViewWidget #cdag-content {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    CDAGViewWidget .node-proposition {
        color: #ffcc33;
        text-style: bold;
    }
    
    CDAGViewWidget .node-evidence-support {
        color: #33ff33;
    }
    
    CDAGViewWidget .node-evidence-attack {
        color: #ff3333;
    }
    
    CDAGViewWidget .node-rebuttal {
        color: #b833ff;
    }
    
    CDAGViewWidget .edge-line {
        color: #4d3800;
    }
    
    CDAGViewWidget .edge-support {
        color: #33ff33;
    }
    
    CDAGViewWidget .edge-attack {
        color: #ff3333;
    }
    
    CDAGViewWidget #cdag-empty {
        width: 100%;
        height: 100%;
        content-align: center middle;
        color: #806000;
    }
    """
    
    # Reactive graph data
    nodes: reactive[list[CDAGNode]] = reactive(list, always_update=True)
    edges: reactive[list[CDAGEdge]] = reactive(list, always_update=True)
    
    def __init__(
        self,
        title: str = "◈ CONCEPTUAL DEBATE GRAPH",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.title = title
        self._graph_nodes: list[CDAGNode] = []
        self._graph_edges: list[CDAGEdge] = []
    
    def compose(self) -> ComposeResult:
        """Compose the C-DAG view layout."""
        yield Static(self.title, id="cdag-title")
        with ScrollableContainer(id="cdag-container"):
            yield Static(id="cdag-content")
    
    def on_mount(self) -> None:
        """Initialize the view."""
        self._render_graph()
    
    def _render_graph(self) -> None:
        """Render the C-DAG as ASCII art."""
        content = self.query_one("#cdag-content", Static)
        
        if not self._graph_nodes:
            content.update(
                "[dim]No graph data. Run a debate to see the C-DAG visualization.[/dim]"
            )
            return
        
        # Build ASCII representation
        lines = []
        
        # Find proposition nodes (roots)
        propositions = [n for n in self._graph_nodes if n.type == NodeType.PROPOSITION]
        
        for prop in propositions:
            lines.extend(self._render_proposition(prop))
            
            # Find evidence for this proposition
            evidence_nodes = [
                n for n in self._graph_nodes
                if n.type == NodeType.EVIDENCE
            ]
            
            for i, ev in enumerate(evidence_nodes):
                is_last = i == len(evidence_nodes) - 1
                lines.extend(self._render_evidence(ev, is_last))
                
                # Find rebuttals for this evidence
                rebuttals = [
                    n for n in self._graph_nodes
                    if n.type == NodeType.REBUTTAL
                ]
                
                for j, reb in enumerate(rebuttals[:2]):  # Limit rebuttals shown
                    reb_is_last = j == min(len(rebuttals), 2) - 1
                    lines.extend(
                        self._render_rebuttal(reb, is_last, reb_is_last)
                    )
        
        content.update("\n".join(lines))
    
    def _render_proposition(self, node: CDAGNode) -> list[str]:
        """Render a proposition node."""
        text = node.text[:60] + "..." if len(node.text) > 60 else node.text
        return [
            "[bold #ffcc33]╔══════════════════════════════════════════════════════════════╗[/]",
            f"[bold #ffcc33]║[/] [bold]◈ PROPOSITION[/]                                              [bold #ffcc33]║[/]",
            f"[bold #ffcc33]║[/] {text:<62} [bold #ffcc33]║[/]",
            f"[bold #ffcc33]║[/] Confidence: [bold]{node.confidence:.2%}[/]                                         [bold #ffcc33]║[/]",
            "[bold #ffcc33]╚══════════════════════════════════════════════════════════════╝[/]",
            "[#4d3800]                              │[/]",
        ]
    
    def _render_evidence(
        self, node: CDAGNode, is_last: bool
    ) -> list[str]:
        """Render an evidence node."""
        connector = "└" if is_last else "├"
        line_char = " " if is_last else "│"
        
        # Determine polarity color
        if node.polarity > 0:
            color = "#33ff33"
            icon = "+"
            polarity_text = "SUPPORTS"
        else:
            color = "#ff3333"
            icon = "-"
            polarity_text = "ATTACKS"
        
        text = node.text[:50] + "..." if len(node.text) > 50 else node.text
        
        return [
            f"[#4d3800]                              {connector}────[/][{color}]╭──────────────────────────────────────────────╮[/]",
            f"[#4d3800]                              {line_char}    [/][{color}]│ {icon} {polarity_text}: {text:<37} │[/]",
            f"[#4d3800]                              {line_char}    [/][{color}]│   Confidence: {node.confidence:.0%}                            │[/]",
            f"[#4d3800]                              {line_char}    [/][{color}]╰──────────────────────────────────────────────╯[/]",
        ]
    
    def _render_rebuttal(
        self, node: CDAGNode, parent_is_last: bool, is_last: bool
    ) -> list[str]:
        """Render a rebuttal node."""
        parent_line = " " if parent_is_last else "│"
        connector = "└" if is_last else "├"
        
        text = node.text[:40] + "..." if len(node.text) > 40 else node.text
        
        return [
            f"[#4d3800]                              {parent_line}         {connector}──[/][#b833ff]╭────────────────────────────────────╮[/]",
            f"[#4d3800]                              {parent_line}            [/][#b833ff]│ ⚔ REBUTTAL: {text:<22} │[/]",
            f"[#4d3800]                              {parent_line}            [/][#b833ff]╰────────────────────────────────────╯[/]",
        ]
    
    def set_graph(
        self, nodes: list[CDAGNode], edges: list[CDAGEdge] | None = None
    ) -> None:
        """Set the graph data."""
        self._graph_nodes = nodes
        self._graph_edges = edges or []
        self._render_graph()
    
    def add_node(self, node: CDAGNode) -> None:
        """Add a node to the graph."""
        self._graph_nodes.append(node)
        self._render_graph()
    
    def clear(self) -> None:
        """Clear the graph."""
        self._graph_nodes = []
        self._graph_edges = []
        self._render_graph()
    
    def set_demo_data(self) -> None:
        """Set demo data for display."""
        self._graph_nodes = [
            CDAGNode(
                "p1", NodeType.PROPOSITION,
                "The new treatment is effective for patients",
                0.72, 0
            ),
            CDAGNode(
                "e1", NodeType.EVIDENCE,
                "Phase 3 trial shows 35% symptom reduction",
                0.90, 1
            ),
            CDAGNode(
                "e2", NodeType.EVIDENCE,
                "15% of patients experienced side effects",
                0.75, -1
            ),
            CDAGNode(
                "r1", NodeType.REBUTTAL,
                "Side effects were mild and temporary",
                0.80, 1
            ),
        ]
        self._render_graph()
