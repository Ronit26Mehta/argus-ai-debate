"""
War Room Visualization — Tournament Tree Plotly figures.

Provides Plotly-based visualisations for the HANNIBAL War Room interface:
    1. Tournament Tree bracket (horizontal/vertical layout)
    2. Force Posterior timeline
    3. Evidence quality heatmap
    4. Engagement margin bar chart
    5. CANNAE dominance matrix (multipolar only)

All figures use the ``plotly_dark`` template matching ARISTOTLE/AGORA styling.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from argus.hannibal.models import (
    ForceType,
    TournamentNode,
    TournamentNodeType,
)

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore

import logging

logger = logging.getLogger(__name__)

# ── Force colours ──────────────────────────────────────────────────

_FORCE_COLORS: dict[str, str] = {
    "proposition": "#2ECC71",
    "opposition": "#E74C3C",
    "faction_1": "#3498DB",
    "faction_2": "#E67E22",
    "faction_3": "#9B59B6",
}
_UNRESOLVED_COLOR = "#555555"
_DRAW_COLOR = "#FFD700"


class TournamentTreeViz:
    """Plotly-based Tournament Tree bracket visualisation."""

    def __init__(self, tree_state: dict[str, Any]):
        """Initialise from a serialised tree state dict.

        Args:
            tree_state: Output of ``TournamentTree.get_bracket_state()``.
        """
        self._state = tree_state
        self._nodes: dict[str, dict] = tree_state.get("nodes", {})
        self._root_id: str = tree_state.get("root_id", "")

    def build_figure(self) -> "go.Figure":
        """Build the Tournament Tree bracket Plotly figure.

        Uses a horizontal layout where the root is on the right
        and leaves (skirmishes) are on the left.
        """
        if go is None:
            raise ImportError("plotly is required for visualisation")

        fig = go.Figure()

        if not self._nodes:
            fig.update_layout(
                template="plotly_dark",
                title="Tournament Tree — No Data",
                height=400,
            )
            return fig

        # Assign positions (x = depth from root, y = stacked)
        positions = self._layout_positions()

        # Draw edges
        edge_x: list[Optional[float]] = []
        edge_y: list[Optional[float]] = []
        for nid, node in self._nodes.items():
            if nid not in positions:
                continue
            px, py = positions[nid]
            for child_id in node.get("child_ids", []):
                if child_id in positions:
                    cx, cy = positions[child_id]
                    edge_x.extend([px, cx, None])
                    edge_y.extend([py, cy, None])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="#2A3942", width=1.5),
            hoverinfo="none",
            showlegend=False,
        ))

        # Draw nodes grouped by type
        for node_type, label_prefix, marker_size in [
            (TournamentNodeType.CAMPAIGN_ROOT.value, "★", 24),
            (TournamentNodeType.THEATRE.value, "⚔", 18),
            (TournamentNodeType.ENGAGEMENT.value, "⚡", 14),
            (TournamentNodeType.SKIRMISH.value, "●", 10),
        ]:
            xs, ys, texts, colors, sizes = [], [], [], [], []
            for nid, node in self._nodes.items():
                if node.get("node_type") != node_type:
                    continue
                if nid not in positions:
                    continue
                px, py = positions[nid]
                xs.append(px)
                ys.append(py)

                winner = node.get("winner_force")
                is_resolved = node.get("is_resolved", False)
                color = _FORCE_COLORS.get(winner, _UNRESOLVED_COLOR) if winner else _UNRESOLVED_COLOR
                if is_resolved and not winner:
                    color = _DRAW_COLOR

                colors.append(color)
                sizes.append(marker_size)

                label = node.get("label", "")
                conf = node.get("confidence", 0)
                winner_abbr = ""
                if winner:
                    try:
                        winner_abbr = ForceType(winner).abbreviation
                    except ValueError:
                        winner_abbr = winner[:3].upper()
                hover = f"{label}<br>Winner: {winner_abbr}<br>Confidence: {conf:.2f}"
                texts.append(hover)

            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="markers",
                    marker=dict(
                        color=colors,
                        size=sizes,
                        line=dict(color="#E9EDEF", width=1),
                    ),
                    text=texts,
                    hoverinfo="text",
                    name=node_type.replace("_", " ").title(),
                ))

        fig.update_layout(
            template="plotly_dark",
            title="Tournament Tree — Campaign Bracket",
            height=450,
            showlegend=True,
            legend=dict(font=dict(size=9)),
            xaxis=dict(
                showgrid=False, zeroline=False,
                showticklabels=False, title="",
            ),
            yaxis=dict(
                showgrid=False, zeroline=False,
                showticklabels=False, title="",
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#0B141A",
            paper_bgcolor="#0B141A",
        )
        return fig

    def _layout_positions(self) -> dict[str, tuple[float, float]]:
        """Compute (x, y) positions for all nodes.

        Horizontal layout: root at x=max_depth, leaves at x=0.
        Y positions are evenly distributed per depth level.
        """
        if not self._root_id or self._root_id not in self._nodes:
            return {}

        # BFS to compute depth and assign y-positions
        depth_map: dict[str, int] = {}
        level_order: list[list[str]] = []

        queue: list[tuple[str, int]] = [(self._root_id, 0)]
        visited: set[str] = set()

        while queue:
            nid, depth = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            depth_map[nid] = depth

            while len(level_order) <= depth:
                level_order.append([])
            level_order[depth].append(nid)

            node = self._nodes.get(nid, {})
            for child_id in node.get("child_ids", []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        max_depth = max(depth_map.values()) if depth_map else 0
        positions: dict[str, tuple[float, float]] = {}

        for depth, node_ids in enumerate(level_order):
            x = max_depth - depth  # Root at right
            n = len(node_ids)
            for i, nid in enumerate(node_ids):
                y = (i - (n - 1) / 2.0) * 2.0  # Centered
                positions[nid] = (float(x), y)

        return positions


# ══════════════════════════════════════════════════════════════════════
# Standalone chart builders
# ══════════════════════════════════════════════════════════════════════

def build_force_posterior_timeline(
    posterior_history: dict[str, list[float]],
) -> "go.Figure":
    """Build a Force Posterior timeline chart."""
    if go is None:
        raise ImportError("plotly required")

    fig = go.Figure()

    for force_val, history in posterior_history.items():
        try:
            ft = ForceType(force_val)
            name = ft.display_name
            color = ft.color_hex
        except ValueError:
            name = force_val
            color = "#888888"

        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=history,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Force Posterior Timeline",
        height=280,
        xaxis_title="Event",
        yaxis_title="Posterior",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=10, t=35, b=30),
        showlegend=True,
        legend=dict(font=dict(size=9)),
        plot_bgcolor="#0B141A",
        paper_bgcolor="#0B141A",
    )
    return fig


def build_evidence_heatmap(
    evidence_items: list[dict[str, Any]],
) -> "go.Figure":
    """Build an evidence quality heatmap."""
    if go is None:
        raise ImportError("plotly required")

    labels: list[str] = []
    evid_q_scores: list[float] = []
    colors: list[str] = []

    for i, item in enumerate(evidence_items[:30]):
        labels.append(f"E{i+1}")
        evid_q_scores.append(item.get("evid_q", 0.5))
        force = item.get("force_type", "proposition")
        colors.append(_FORCE_COLORS.get(force, "#888888"))

    fig = go.Figure(go.Bar(
        x=labels,
        y=evid_q_scores,
        marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Evidence EVID-Q Scores",
        height=250,
        xaxis_title="Evidence Item",
        yaxis_title="EVID-Q",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=10, t=35, b=30),
        plot_bgcolor="#0B141A",
        paper_bgcolor="#0B141A",
    )
    return fig


def build_engagement_margin_chart(
    engagement_data: list[dict[str, Any]],
) -> "go.Figure":
    """Build engagement margin bar chart."""
    if go is None:
        raise ImportError("plotly required")

    labels: list[str] = []
    margins: list[float] = []
    colors: list[str] = []

    for eng in engagement_data:
        labels.append(eng.get("label", ""))
        margins.append(eng.get("margin", 0.0))
        winner = eng.get("winner_force", "proposition")
        colors.append(_FORCE_COLORS.get(winner, "#888888"))

    fig = go.Figure(go.Bar(
        x=labels,
        y=margins,
        marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Engagement Victory Margins",
        height=250,
        xaxis_title="Engagement",
        yaxis_title="Margin",
        margin=dict(l=40, r=10, t=35, b=30),
        plot_bgcolor="#0B141A",
        paper_bgcolor="#0B141A",
    )
    return fig


def build_cannae_matrix(
    dominance_matrix: dict[str, dict[str, float]],
    force_labels: list[str],
) -> "go.Figure":
    """Build CANNAE pairwise dominance heatmap."""
    if go is None:
        raise ImportError("plotly required")

    z: list[list[float]] = []
    for row_label in force_labels:
        row = dominance_matrix.get(row_label, {})
        z.append([row.get(col, 0.0) for col in force_labels])

    fig = go.Figure(go.Heatmap(
        z=z,
        x=force_labels,
        y=force_labels,
        colorscale="RdYlGn",
        zmin=0.0,
        zmax=1.0,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        template="plotly_dark",
        title="CANNAE Dominance Matrix",
        height=350,
        margin=dict(l=80, r=20, t=40, b=60),
        plot_bgcolor="#0B141A",
        paper_bgcolor="#0B141A",
    )
    return fig
