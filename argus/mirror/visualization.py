"""
MIRROR Visualization — dual-theme consequence graph plots.
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
    from argus.mirror.graph import ConsequenceGraph

DARK = {"bg": "#0e1117", "paper": "#1a1f2e", "text": "#e0e0e0", "title": "#00d4ff",
        "grid": "#2a3040", "root": "#00d4ff", "opp": "#00ff88", "risk": "#ff4466",
        "edge": "#4a5568", "anno_bg": "rgba(26,31,46,0.9)"}

LIGHT = {"bg": "#ffffff", "paper": "#f8f9fa", "text": "#1a1a1a", "title": "#0066cc",
         "grid": "#e0e0e0", "root": "#0066cc", "opp": "#00aa44", "risk": "#cc0033",
         "edge": "#aaaaaa", "anno_bg": "rgba(248,249,250,0.9)"}


def plot_consequence_graph(
    graph: "ConsequenceGraph",
    theme: str = "dark",
    height: int = 600,
) -> Any:
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required: pip install plotly")
    t = DARK if theme == "dark" else LIGHT
    fig = go.Figure()

    if not graph or graph.num_nodes == 0:
        fig.add_annotation(text="No consequences", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=t["text"]))
        fig.update_layout(paper_bgcolor=t["bg"], plot_bgcolor=t["paper"])
        return fig

    graph.compute_marginals()
    nodes = graph.all_nodes

    # Layout: root at left, consequences radiating right by category
    categories = graph.categories
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Root node
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=28, color=t["root"], symbol="diamond",
                    line=dict(width=3, color=t["bg"])),
        text=[f"Root P={graph.root_posterior:.2f}"],
        textposition="bottom center",
        textfont=dict(size=10, color=t["text"]),
        name="Root Verdict", showlegend=True,
        hovertemplate=f"<b>Root: {graph.proposition[:60]}</b><br>"
                      f"Posterior: {graph.root_posterior:.3f}<extra></extra>",
    ))

    # Consequence nodes
    for i, node in enumerate(nodes):
        cat_y = cat_idx.get(node.category, 0) * 1.5
        x = 2.0 + (i % 4) * 1.2
        y = cat_y + (i % 3 - 1) * 0.5

        marginal = graph._marginals.get(node.node_id, 0.5)
        color = t["opp"] if node.severity < 0.5 else t["risk"]
        size = 10 + marginal * 20

        # Edge from root
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y], mode="lines",
            line=dict(color=t["edge"], width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=size, color=color,
                        line=dict(width=1.5, color=t["bg"])),
            name=node.text[:25], showlegend=False,
            hovertemplate=(
                f"<b>{node.text[:60]}</b><br>"
                f"P(if true): {node.conditional_probability:.3f}<br>"
                f"P(if false): {node.inverse_probability:.3f}<br>"
                f"Marginal: {marginal:.3f}<br>"
                f"Category: {node.category}<br>"
                f"Severity: {node.severity:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"], size=12),
        title=dict(text="🪞 Consequence Inference Graph",
                   font=dict(size=18, color=t["title"])),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height, showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor=t["anno_bg"]),
    )
    return fig


def export_consequence_html(
    graph: "ConsequenceGraph",
    output_path: str = "consequence_graph.html",
    theme: str = "dark",
) -> str:
    fig = plot_consequence_graph(graph, theme=theme)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
