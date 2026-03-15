"""
FRACTAL Visualization — dual-theme proposition tree plots.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

if TYPE_CHECKING:
    from argus.fractal.decomposer import PropositionTree

DARK_THEME = {
    "bg": "#0e1117", "paper": "#1a1f2e", "grid": "#2a3040",
    "text": "#e0e0e0", "title": "#00d4ff",
    "node_high": "#00ff88", "node_mid": "#ffbf00", "node_low": "#ff4466",
    "edge": "#4a5568", "root": "#00d4ff", "leaf": "#b388ff",
}

LIGHT_THEME = {
    "bg": "#ffffff", "paper": "#f8f9fa", "grid": "#e0e0e0",
    "text": "#1a1a1a", "title": "#0066cc",
    "node_high": "#00aa44", "node_mid": "#cc8800", "node_low": "#cc0033",
    "edge": "#aaaaaa", "root": "#0066cc", "leaf": "#6633cc",
}


def _get_theme(theme: str = "dark") -> dict:
    return LIGHT_THEME if theme == "light" else DARK_THEME


def plot_proposition_tree(
    tree: "PropositionTree",
    theme: str = "dark",
    height: int = 600,
) -> Any:
    """
    Plot proposition decomposition tree with posteriors.

    Args:
        tree: PropositionTree to visualize
        theme: 'dark' or 'light'
        height: Figure height

    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required: pip install plotly")

    t = _get_theme(theme)
    fig = go.Figure()

    if not tree or tree.num_nodes == 0:
        fig.add_annotation(
            text="Empty tree", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=t["text"]),
        )
        return fig

    # Layout nodes hierarchically
    positions: dict[str, tuple[float, float]] = {}
    _layout_tree(tree, tree.root_id, positions, 0, 0, 2.0)

    # Draw edges
    for node in tree.all_nodes:
        if node.parent_id and node.parent_id in positions and node.node_id in positions:
            px, py = positions[node.parent_id]
            cx, cy = positions[node.node_id]
            fig.add_trace(go.Scatter(
                x=[px, cx], y=[py, cy], mode="lines",
                line=dict(color=t["edge"], width=1.5, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

    # Draw nodes
    for node in tree.all_nodes:
        if node.node_id not in positions:
            continue
        x, y = positions[node.node_id]
        p = node.posterior if node.posterior is not None else 0.5

        if p > 0.6:
            color = t["node_high"]
        elif p > 0.4:
            color = t["node_mid"]
        else:
            color = t["node_low"]

        if node.node_id == tree.root_id:
            color = t["root"]
            size = 22
        elif node.is_leaf:
            color = t["leaf"] if node.posterior is None else color
            size = 16
        else:
            size = 18

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=2, color=t["bg"])),
            text=[f"P={p:.2f}"],
            textposition="top center",
            textfont=dict(size=9, color=t["text"]),
            name=node.text[:30],
            showlegend=False,
            hovertemplate=(
                f"<b>{node.text[:60]}</b><br>"
                f"Posterior: {p:.3f}<br>"
                f"Depth: {node.depth}<br>"
                f"Relationship: {node.relationship_to_parent}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"], size=12),
        title=dict(text="🌿 Proposition Decomposition Tree",
                   font=dict(size=18, color=t["title"])),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        height=height, showlegend=False,
    )
    return fig


def _layout_tree(
    tree: "PropositionTree",
    node_id: str,
    positions: dict[str, tuple[float, float]],
    depth: int,
    x_offset: float,
    x_spread: float,
) -> float:
    """Assign positions to tree nodes."""
    node = tree.get_node(node_id)
    if not node:
        return x_offset

    children = tree.get_children(node_id)
    if not children:
        positions[node_id] = (x_offset, depth)
        return x_offset + 1.0

    child_spread = x_spread / max(len(children), 1)
    current_x = x_offset
    child_positions = []

    for child in children:
        final_x = _layout_tree(tree, child.node_id, positions, depth + 1, current_x, child_spread)
        child_positions.append((current_x + final_x) / 2.0)
        current_x = final_x

    parent_x = sum(child_positions) / len(child_positions) if child_positions else x_offset
    positions[node_id] = (parent_x, depth)
    return current_x


def export_tree_html(
    tree: "PropositionTree",
    output_path: str = "proposition_tree.html",
    theme: str = "dark",
) -> str:
    """Export tree as standalone HTML."""
    fig = plot_proposition_tree(tree, theme=theme)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
