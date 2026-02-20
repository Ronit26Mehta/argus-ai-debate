"""
ARGUS Debate Visualization Module.

Comprehensive visualization tools for multi-agent debates,
argument flows, and reasoning analysis.
"""

from __future__ import annotations

import math
import json
import logging
from datetime import datetime
from typing import Any, Optional, Union, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

# Check for visualization dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


# ---------------------------------------------------------------------------
# Color Theme
# ---------------------------------------------------------------------------

DEBATE_COLORS = {
    # Base theme
    "bg": "#0e1117",
    "paper": "#1a1f2e",
    "grid": "#2a3040",
    "text": "#e0e0e0",
    "text_muted": "#888888",
    
    # Accent colors
    "accent_primary": "#00d4ff",
    "accent_secondary": "#ff00d4",
    "accent_tertiary": "#ffbf00",
    
    # Argument types
    "claim": "#00d4ff",
    "support": "#00ff88",
    "attack": "#ff4466",
    "rebuttal": "#ff8800",
    "concession": "#b388ff",
    "neutral": "#888888",
    
    # Agent roles
    "proponent": "#00ff88",
    "opponent": "#ff4466",
    "moderator": "#ffbf00",
    "jury": "#b388ff",
    "specialist": "#00d4ff",
    
    # Confidence levels
    "confidence_high": "#00ff88",
    "confidence_medium": "#ffbf00",
    "confidence_low": "#ff4466",
    
    # Status
    "validated": "#00ff88",
    "challenged": "#ff4466",
    "pending": "#ffbf00",
    "resolved": "#b388ff",
}

DARK_LAYOUT = dict(
    paper_bgcolor=DEBATE_COLORS["bg"],
    plot_bgcolor=DEBATE_COLORS["paper"],
    font=dict(family="Inter, sans-serif", color=DEBATE_COLORS["text"], size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=DEBATE_COLORS["grid"], zerolinecolor=DEBATE_COLORS["grid"]),
    yaxis=dict(gridcolor=DEBATE_COLORS["grid"], zerolinecolor=DEBATE_COLORS["grid"]),
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ArgumentType(Enum):
    """Types of arguments in a debate."""
    CLAIM = "claim"
    SUPPORT = "support"
    ATTACK = "attack"
    REBUTTAL = "rebuttal"
    CONCESSION = "concession"
    QUESTION = "question"


class ArgumentStatus(Enum):
    """Status of an argument."""
    PENDING = "pending"
    VALIDATED = "validated"
    CHALLENGED = "challenged"
    REFUTED = "refuted"
    RESOLVED = "resolved"


@dataclass
class Argument:
    """Represents an argument in a debate."""
    id: str
    content: str
    arg_type: ArgumentType
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    status: ArgumentStatus = ArgumentStatus.PENDING
    parent_id: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    """Represents a round of debate."""
    round_number: int
    topic: str
    arguments: List[Argument] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    winner: Optional[str] = None
    summary: str = ""


@dataclass
class DebateSession:
    """Represents a complete debate session."""
    session_id: str
    topic: str
    agents: List[str]
    rounds: List[DebateRound] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    verdict: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_arguments(self) -> List[Argument]:
        """Get all arguments from all rounds."""
        args = []
        for round in self.rounds:
            args.extend(round.arguments)
        return args
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()


def _check_plotly() -> None:
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualizations. "
            "Install with: pip install plotly"
        )


def _check_networkx() -> None:
    """Check if NetworkX is available."""
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "NetworkX is required for graph layouts. "
            "Install with: pip install networkx"
        )


# ---------------------------------------------------------------------------
# Argument Flow Visualization
# ---------------------------------------------------------------------------

def plot_argument_flow(
    session: DebateSession,
    layout: str = "hierarchical",
    show_labels: bool = True,
    height: int = 700,
    width: int = 1000,
) -> "go.Figure":
    """
    Plot argument flow diagram showing relationships between arguments.
    
    Args:
        session: Debate session to visualize
        layout: Layout type ("hierarchical", "radial", "force")
        show_labels: Whether to show argument labels
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    _check_networkx()
    
    # Build graph
    G = nx.DiGraph()
    
    for arg in session.all_arguments:
        G.add_node(
            arg.id,
            content=arg.content[:50] + "..." if len(arg.content) > 50 else arg.content,
            agent=arg.agent_id,
            arg_type=arg.arg_type.value,
            confidence=arg.confidence,
            status=arg.status.value,
        )
        
        if arg.parent_id:
            edge_type = "support" if arg.arg_type in (ArgumentType.SUPPORT, ArgumentType.REBUTTAL) else "attack"
            G.add_edge(arg.parent_id, arg.id, edge_type=edge_type)
    
    # Compute layout
    if layout == "hierarchical":
        pos = _hierarchical_layout(G)
    elif layout == "radial":
        pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=100)
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_type = edge[2].get("edge_type", "support")
        color = DEBATE_COLORS["support"] if edge_type == "support" else DEBATE_COLORS["attack"]
        
        # Create arrow
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=2, color=color),
            hoverinfo="none",
            showlegend=False,
        ))
        
        # Add arrowhead
        arrow_length = 0.05
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx**2 + dy**2)
        
        if length > 0:
            dx /= length
            dy /= length
            
            edge_traces.append(go.Scatter(
                x=[x1 - arrow_length * (dx - dy * 0.5),
                   x1,
                   x1 - arrow_length * (dx + dy * 0.5)],
                y=[y1 - arrow_length * (dy + dx * 0.5),
                   y1,
                   y1 - arrow_length * (dy - dx * 0.5)],
                mode="lines",
                line=dict(width=2, color=color),
                fill="toself",
                fillcolor=color,
                hoverinfo="none",
                showlegend=False,
            ))
    
    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_colors = []
    node_sizes = []
    node_texts = []
    hover_texts = []
    
    for node in G.nodes():
        data = G.nodes[node]
        arg_type = data.get("arg_type", "claim")
        
        # Color by type
        color_key = arg_type if arg_type in DEBATE_COLORS else "claim"
        node_colors.append(DEBATE_COLORS[color_key])
        
        # Size by confidence
        confidence = data.get("confidence", 1.0)
        node_sizes.append(20 + confidence * 20)
        
        # Labels
        if show_labels:
            node_texts.append(data.get("content", "")[:20])
        else:
            node_texts.append("")
        
        hover_texts.append(
            f"<b>{data.get('agent', 'Unknown')}</b><br>"
            f"Type: {arg_type}<br>"
            f"Confidence: {confidence:.2f}<br>"
            f"Status: {data.get('status', 'pending')}<br>"
            f"<br>{data.get('content', '')}"
        )
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=node_texts,
        textposition="top center",
        textfont=dict(size=10, color=DEBATE_COLORS["text"]),
        hovertext=hover_texts,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color=DEBATE_COLORS["paper"]),
        ),
    )
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Argument Flow: {session.topic}",
                font=dict(size=16, color=DEBATE_COLORS["accent_primary"]),
            ),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=height,
            width=width,
            **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
        ),
    )
    
    # Add legend
    legend_items = [
        ("Claim", DEBATE_COLORS["claim"]),
        ("Support", DEBATE_COLORS["support"]),
        ("Attack", DEBATE_COLORS["attack"]),
        ("Rebuttal", DEBATE_COLORS["rebuttal"]),
    ]
    
    for i, (name, color) in enumerate(legend_items):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=name,
            showlegend=True,
        ))
    
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
        ),
    )
    
    return fig


def _hierarchical_layout(G: "nx.DiGraph") -> Dict[str, Tuple[float, float]]:
    """Compute hierarchical layout for directed graph."""
    # Find root nodes (no incoming edges)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    if not roots:
        roots = list(G.nodes())[:1]
    
    pos = {}
    levels = defaultdict(list)
    
    # BFS to assign levels
    visited = set()
    queue = [(root, 0) for root in roots]
    
    while queue:
        node, level = queue.pop(0)
        if node in visited:
            continue
        
        visited.add(node)
        levels[level].append(node)
        
        for child in G.successors(node):
            if child not in visited:
                queue.append((child, level + 1))
    
    # Position nodes
    max_level = max(levels.keys()) if levels else 0
    
    for level, nodes in levels.items():
        y = 1 - (level / max(max_level, 1))
        for i, node in enumerate(nodes):
            x = (i + 1) / (len(nodes) + 1)
            pos[node] = (x, y)
    
    return pos


# ---------------------------------------------------------------------------
# Timeline Visualization
# ---------------------------------------------------------------------------

def plot_debate_timeline(
    session: DebateSession,
    height: int = 500,
    width: int = 1000,
) -> "go.Figure":
    """
    Plot debate timeline showing arguments over time.
    
    Args:
        session: Debate session to visualize
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    # Prepare data
    data = []
    agents = list(set(session.agents))
    agent_y = {agent: i for i, agent in enumerate(agents)}
    
    start_time = session.start_time
    
    for arg in session.all_arguments:
        rel_time = (arg.timestamp - start_time).total_seconds()
        
        data.append({
            "time": rel_time,
            "agent": arg.agent_id,
            "y": agent_y.get(arg.agent_id, 0),
            "type": arg.arg_type.value,
            "content": arg.content[:100],
            "confidence": arg.confidence,
            "status": arg.status.value,
        })
    
    if not data:
        # Empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No arguments to display",
            **DARK_LAYOUT,
        )
        return fig
    
    # Create traces by type
    fig = go.Figure()
    
    for arg_type in ArgumentType:
        type_data = [d for d in data if d["type"] == arg_type.value]
        
        if not type_data:
            continue
        
        color = DEBATE_COLORS.get(arg_type.value, DEBATE_COLORS["neutral"])
        
        fig.add_trace(go.Scatter(
            x=[d["time"] for d in type_data],
            y=[d["y"] for d in type_data],
            mode="markers",
            name=arg_type.value.title(),
            marker=dict(
                size=[10 + d["confidence"] * 10 for d in type_data],
                color=color,
                line=dict(width=1, color=DEBATE_COLORS["paper"]),
            ),
            hovertext=[
                f"<b>{d['agent']}</b><br>"
                f"Type: {d['type']}<br>"
                f"Time: {d['time']:.1f}s<br>"
                f"Confidence: {d['confidence']:.2f}<br>"
                f"<br>{d['content']}"
                for d in type_data
            ],
            hoverinfo="text",
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Debate Timeline: {session.topic}",
            font=dict(size=16, color=DEBATE_COLORS["accent_primary"]),
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor=DEBATE_COLORS["grid"],
        ),
        yaxis=dict(
            tickvals=list(range(len(agents))),
            ticktext=agents,
            gridcolor=DEBATE_COLORS["grid"],
        ),
        height=height,
        width=width,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    
    return fig


# ---------------------------------------------------------------------------
# Agent Performance Visualization
# ---------------------------------------------------------------------------

def plot_agent_performance(
    session: DebateSession,
    height: int = 500,
    width: int = 800,
) -> "go.Figure":
    """
    Plot agent performance metrics.
    
    Args:
        session: Debate session to visualize
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    # Calculate metrics per agent
    agent_metrics = defaultdict(lambda: {
        "total_args": 0,
        "claims": 0,
        "supports": 0,
        "attacks": 0,
        "rebuttals": 0,
        "avg_confidence": [],
        "validated": 0,
        "challenged": 0,
    })
    
    for arg in session.all_arguments:
        metrics = agent_metrics[arg.agent_id]
        metrics["total_args"] += 1
        metrics[f"{arg.arg_type.value}s" if arg.arg_type.value != "claim" else "claims"] += 1
        metrics["avg_confidence"].append(arg.confidence)
        
        if arg.status == ArgumentStatus.VALIDATED:
            metrics["validated"] += 1
        elif arg.status == ArgumentStatus.CHALLENGED:
            metrics["challenged"] += 1
    
    # Compute averages
    for agent, metrics in agent_metrics.items():
        if metrics["avg_confidence"]:
            metrics["avg_confidence"] = sum(metrics["avg_confidence"]) / len(metrics["avg_confidence"])
        else:
            metrics["avg_confidence"] = 0
    
    agents = list(agent_metrics.keys())
    
    if not agents:
        fig = go.Figure()
        fig.update_layout(title="No agent data available", **DARK_LAYOUT)
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Arguments by Type",
            "Argument Status",
            "Average Confidence",
            "Total Arguments",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
    )
    
    # Arguments by type
    for arg_type in ["claims", "supports", "attacks", "rebuttals"]:
        color = DEBATE_COLORS.get(arg_type[:-1], DEBATE_COLORS["neutral"])
        fig.add_trace(
            go.Bar(
                name=arg_type.title(),
                x=agents,
                y=[agent_metrics[a][arg_type] for a in agents],
                marker_color=color,
            ),
            row=1, col=1,
        )
    
    # Argument status
    fig.add_trace(
        go.Bar(
            name="Validated",
            x=agents,
            y=[agent_metrics[a]["validated"] for a in agents],
            marker_color=DEBATE_COLORS["validated"],
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Challenged",
            x=agents,
            y=[agent_metrics[a]["challenged"] for a in agents],
            marker_color=DEBATE_COLORS["challenged"],
        ),
        row=1, col=2,
    )
    
    # Average confidence
    confidences = [agent_metrics[a]["avg_confidence"] for a in agents]
    colors = [
        DEBATE_COLORS["confidence_high"] if c >= 0.7
        else DEBATE_COLORS["confidence_medium"] if c >= 0.4
        else DEBATE_COLORS["confidence_low"]
        for c in confidences
    ]
    
    fig.add_trace(
        go.Bar(
            name="Confidence",
            x=agents,
            y=confidences,
            marker_color=colors,
            showlegend=False,
        ),
        row=2, col=1,
    )
    
    # Total arguments
    totals = [agent_metrics[a]["total_args"] for a in agents]
    
    fig.add_trace(
        go.Bar(
            name="Total",
            x=agents,
            y=totals,
            marker_color=DEBATE_COLORS["accent_primary"],
            showlegend=False,
        ),
        row=2, col=2,
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Agent Performance: {session.topic}",
            font=dict(size=16, color=DEBATE_COLORS["accent_primary"]),
        ),
        barmode="group",
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    
    return fig


# ---------------------------------------------------------------------------
# Confidence Evolution
# ---------------------------------------------------------------------------

def plot_confidence_evolution(
    session: DebateSession,
    height: int = 400,
    width: int = 800,
) -> "go.Figure":
    """
    Plot how confidence scores evolve over the debate.
    
    Args:
        session: Debate session to visualize
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    # Group arguments by agent and time
    agent_data = defaultdict(lambda: {"times": [], "confidences": []})
    start_time = session.start_time
    
    for arg in session.all_arguments:
        rel_time = (arg.timestamp - start_time).total_seconds()
        agent_data[arg.agent_id]["times"].append(rel_time)
        agent_data[arg.agent_id]["confidences"].append(arg.confidence)
    
    fig = go.Figure()
    
    colors = [
        DEBATE_COLORS["proponent"],
        DEBATE_COLORS["opponent"],
        DEBATE_COLORS["moderator"],
        DEBATE_COLORS["specialist"],
        DEBATE_COLORS["accent_primary"],
    ]
    
    for i, (agent, data) in enumerate(agent_data.items()):
        if not data["times"]:
            continue
        
        # Sort by time
        sorted_data = sorted(zip(data["times"], data["confidences"]))
        times, confidences = zip(*sorted_data)
        
        # Compute rolling average
        rolling_avg = []
        window = min(3, len(confidences))
        for j in range(len(confidences)):
            start = max(0, j - window + 1)
            rolling_avg.append(sum(confidences[start:j+1]) / (j - start + 1))
        
        color = colors[i % len(colors)]
        
        # Scatter for actual points
        fig.add_trace(go.Scatter(
            x=times,
            y=confidences,
            mode="markers",
            name=f"{agent} (actual)",
            marker=dict(size=8, color=color, opacity=0.5),
            showlegend=False,
        ))
        
        # Line for trend
        fig.add_trace(go.Scatter(
            x=times,
            y=rolling_avg,
            mode="lines",
            name=agent,
            line=dict(width=2, color=color),
        ))
    
    fig.update_layout(
        title=dict(
            text="Confidence Evolution Over Debate",
            font=dict(size=16, color=DEBATE_COLORS["accent_primary"]),
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor=DEBATE_COLORS["grid"],
        ),
        yaxis=dict(
            title="Confidence",
            range=[0, 1.1],
            gridcolor=DEBATE_COLORS["grid"],
        ),
        height=height,
        width=width,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    
    return fig


# ---------------------------------------------------------------------------
# Round Summary
# ---------------------------------------------------------------------------

def plot_round_summary(
    session: DebateSession,
    height: int = 400,
    width: int = 800,
) -> "go.Figure":
    """
    Plot summary of each debate round.
    
    Args:
        session: Debate session to visualize
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    rounds = session.rounds
    
    if not rounds:
        fig = go.Figure()
        fig.update_layout(title="No rounds to display", **DARK_LAYOUT)
        return fig
    
    # Compute metrics per round
    round_nums = []
    arg_counts = []
    avg_confidences = []
    support_ratios = []
    
    for rnd in rounds:
        round_nums.append(rnd.round_number)
        arg_counts.append(len(rnd.arguments))
        
        if rnd.arguments:
            confidences = [a.confidence for a in rnd.arguments]
            avg_confidences.append(sum(confidences) / len(confidences))
            
            supports = sum(1 for a in rnd.arguments if a.arg_type == ArgumentType.SUPPORT)
            attacks = sum(1 for a in rnd.arguments if a.arg_type == ArgumentType.ATTACK)
            total = supports + attacks
            support_ratios.append(supports / total if total > 0 else 0.5)
        else:
            avg_confidences.append(0)
            support_ratios.append(0.5)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Arguments per Round", "Average Confidence", "Support vs Attack Ratio"),
    )
    
    # Arguments per round
    fig.add_trace(
        go.Bar(
            x=round_nums,
            y=arg_counts,
            marker_color=DEBATE_COLORS["accent_primary"],
            name="Arguments",
        ),
        row=1, col=1,
    )
    
    # Average confidence
    fig.add_trace(
        go.Scatter(
            x=round_nums,
            y=avg_confidences,
            mode="lines+markers",
            marker=dict(size=10, color=DEBATE_COLORS["confidence_high"]),
            line=dict(width=2, color=DEBATE_COLORS["confidence_high"]),
            name="Confidence",
        ),
        row=1, col=2,
    )
    
    # Support ratio (stacked bar)
    fig.add_trace(
        go.Bar(
            x=round_nums,
            y=support_ratios,
            marker_color=DEBATE_COLORS["support"],
            name="Support",
        ),
        row=1, col=3,
    )
    fig.add_trace(
        go.Bar(
            x=round_nums,
            y=[1 - r for r in support_ratios],
            marker_color=DEBATE_COLORS["attack"],
            name="Attack",
        ),
        row=1, col=3,
    )
    
    fig.update_layout(
        title=dict(
            text="Round Summary",
            font=dict(size=16, color=DEBATE_COLORS["accent_primary"]),
        ),
        barmode="stack",
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    
    for i in range(1, 4):
        fig.update_xaxes(title_text="Round", row=1, col=i)
    
    return fig


# ---------------------------------------------------------------------------
# Heatmap Visualizations
# ---------------------------------------------------------------------------

def plot_interaction_heatmap(
    session: DebateSession,
    height: int = 500,
    width: int = 600,
) -> "go.Figure":
    """
    Plot heatmap of agent interactions.
    
    Args:
        session: Debate session to visualize
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    agents = list(set(session.agents))
    n = len(agents)
    
    # Initialize interaction matrix
    interactions = [[0] * n for _ in range(n)]
    
    # Build parent-child relationships
    arg_map = {arg.id: arg for arg in session.all_arguments}
    
    for arg in session.all_arguments:
        if arg.parent_id and arg.parent_id in arg_map:
            parent = arg_map[arg.parent_id]
            
            try:
                parent_idx = agents.index(parent.agent_id)
                child_idx = agents.index(arg.agent_id)
                interactions[parent_idx][child_idx] += 1
            except ValueError:
                pass
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=interactions,
        x=agents,
        y=agents,
        colorscale=[
            [0, DEBATE_COLORS["paper"]],
            [0.5, DEBATE_COLORS["accent_tertiary"]],
            [1, DEBATE_COLORS["accent_primary"]],
        ],
        hoverongaps=False,
        hovertemplate="From: %{y}<br>To: %{x}<br>Count: %{z}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(
            text="Agent Interaction Heatmap",
            font=dict(size=16, color=DEBATE_COLORS["accent_primary"]),
        ),
        xaxis=dict(title="Responding Agent", tickangle=45),
        yaxis=dict(title="Original Agent"),
        height=height,
        width=width,
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    
    return fig


def plot_argument_type_distribution(
    session: DebateSession,
    height: int = 400,
    width: int = 400,
) -> "go.Figure":
    """
    Plot distribution of argument types as pie chart.
    
    Args:
        session: Debate session to visualize
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    # Count argument types
    type_counts = defaultdict(int)
    
    for arg in session.all_arguments:
        type_counts[arg.arg_type.value] += 1
    
    if not type_counts:
        fig = go.Figure()
        fig.update_layout(title="No arguments to display", **DARK_LAYOUT)
        return fig
    
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = [DEBATE_COLORS.get(label, DEBATE_COLORS["neutral"]) for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=[l.title() for l in labels],
        values=values,
        marker=dict(colors=colors, line=dict(color=DEBATE_COLORS["paper"], width=2)),
        textinfo="label+percent",
        textfont=dict(color=DEBATE_COLORS["text"]),
        hole=0.4,
    )])
    
    fig.update_layout(
        title=dict(
            text="Argument Type Distribution",
            font=dict(size=16, color=DEBATE_COLORS["accent_primary"]),
        ),
        height=height,
        width=width,
        showlegend=False,
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    
    return fig


# ---------------------------------------------------------------------------
# Comprehensive Dashboard
# ---------------------------------------------------------------------------

def create_debate_dashboard(
    session: DebateSession,
    height: int = 1200,
    width: int = 1400,
) -> "go.Figure":
    """
    Create comprehensive debate dashboard with multiple visualizations.
    
    Args:
        session: Debate session to visualize
        height: Figure height
        width: Figure width
    
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            "Argument Flow", "", "",
            "Timeline", "Agent Performance", "Type Distribution",
            "Confidence Evolution", "Interaction Heatmap", "Round Summary",
        ),
        specs=[
            [{"colspan": 3}, None, None],
            [{"colspan": 1}, {"colspan": 1}, {"type": "pie"}],
            [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
        ],
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    
    # Add individual plots (simplified versions for dashboard)
    # Note: Full argument flow requires separate figure due to complexity
    
    # Timeline data (row 2, col 1)
    start_time = session.start_time
    for arg in session.all_arguments[:50]:  # Limit for performance
        rel_time = (arg.timestamp - start_time).total_seconds()
        color = DEBATE_COLORS.get(arg.arg_type.value, DEBATE_COLORS["neutral"])
        
        fig.add_trace(
            go.Scatter(
                x=[rel_time],
                y=[session.agents.index(arg.agent_id) if arg.agent_id in session.agents else 0],
                mode="markers",
                marker=dict(size=8, color=color),
                showlegend=False,
                hovertext=f"{arg.agent_id}: {arg.content[:30]}...",
                hoverinfo="text",
            ),
            row=2, col=1,
        )
    
    # Agent performance (row 2, col 2)
    agent_args = defaultdict(int)
    for arg in session.all_arguments:
        agent_args[arg.agent_id] += 1
    
    if agent_args:
        agents = list(agent_args.keys())
        counts = list(agent_args.values())
        
        fig.add_trace(
            go.Bar(
                x=agents,
                y=counts,
                marker_color=DEBATE_COLORS["accent_primary"],
                showlegend=False,
            ),
            row=2, col=2,
        )
    
    # Type distribution (row 2, col 3)
    type_counts = defaultdict(int)
    for arg in session.all_arguments:
        type_counts[arg.arg_type.value] += 1
    
    if type_counts:
        labels = [t.title() for t in type_counts.keys()]
        values = list(type_counts.values())
        colors = [DEBATE_COLORS.get(t, DEBATE_COLORS["neutral"]) for t in type_counts.keys()]
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                showlegend=False,
                textinfo="label+percent",
            ),
            row=2, col=3,
        )
    
    # Confidence evolution (row 3, col 1)
    confidences_by_time = []
    for arg in session.all_arguments:
        rel_time = (arg.timestamp - start_time).total_seconds()
        confidences_by_time.append((rel_time, arg.confidence))
    
    if confidences_by_time:
        confidences_by_time.sort()
        times, confidences = zip(*confidences_by_time)
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=confidences,
                mode="lines+markers",
                marker=dict(size=5, color=DEBATE_COLORS["confidence_high"]),
                line=dict(width=1, color=DEBATE_COLORS["confidence_high"]),
                showlegend=False,
            ),
            row=3, col=1,
        )
    
    # Interaction heatmap (row 3, col 2)
    agents_list = list(set(session.agents))
    n = len(agents_list)
    interactions = [[0] * n for _ in range(n)]
    
    arg_map = {arg.id: arg for arg in session.all_arguments}
    for arg in session.all_arguments:
        if arg.parent_id and arg.parent_id in arg_map:
            parent = arg_map[arg.parent_id]
            try:
                parent_idx = agents_list.index(parent.agent_id)
                child_idx = agents_list.index(arg.agent_id)
                interactions[parent_idx][child_idx] += 1
            except ValueError:
                pass
    
    fig.add_trace(
        go.Heatmap(
            z=interactions,
            x=agents_list,
            y=agents_list,
            colorscale=[[0, DEBATE_COLORS["paper"]], [1, DEBATE_COLORS["accent_primary"]]],
            showscale=False,
        ),
        row=3, col=2,
    )
    
    # Round summary (row 3, col 3)
    if session.rounds:
        round_nums = [r.round_number for r in session.rounds]
        arg_counts = [len(r.arguments) for r in session.rounds]
        
        fig.add_trace(
            go.Bar(
                x=round_nums,
                y=arg_counts,
                marker_color=DEBATE_COLORS["accent_tertiary"],
                showlegend=False,
            ),
            row=3, col=3,
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Debate Dashboard: {session.topic}",
            font=dict(size=20, color=DEBATE_COLORS["accent_primary"]),
            x=0.5,
        ),
        height=height,
        width=width,
        showlegend=False,
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    )
    
    # Add session info annotation
    duration = session.duration_seconds
    info_text = (
        f"Session: {session.session_id}<br>"
        f"Duration: {duration/60:.1f} min<br>"
        f"Agents: {len(session.agents)}<br>"
        f"Arguments: {len(session.all_arguments)}<br>"
        f"Rounds: {len(session.rounds)}"
    )
    
    fig.add_annotation(
        text=info_text,
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        showarrow=False,
        font=dict(size=10, color=DEBATE_COLORS["text"]),
        align="left",
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4,
    )
    
    return fig


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------

def export_debate_html(
    session: DebateSession,
    output_path: str,
    include_dashboard: bool = True,
) -> str:
    """
    Export debate visualization as interactive HTML.
    
    Args:
        session: Debate session to visualize
        output_path: Path for output HTML file
        include_dashboard: Whether to include full dashboard
    
    Returns:
        Path to created file
    """
    _check_plotly()
    
    if include_dashboard:
        fig = create_debate_dashboard(session)
    else:
        fig = plot_argument_flow(session)
    
    fig.write_html(output_path, include_plotlyjs=True)
    
    return output_path


def export_debate_png(
    session: DebateSession,
    output_path: str,
    width: int = 1400,
    height: int = 1000,
) -> str:
    """
    Export debate visualization as PNG image.
    
    Args:
        session: Debate session to visualize
        output_path: Path for output PNG file
        width: Image width
        height: Image height
    
    Returns:
        Path to created file
    """
    _check_plotly()
    
    fig = create_debate_dashboard(session, height=height, width=width)
    fig.write_image(output_path, width=width, height=height)
    
    return output_path


def generate_debate_report(
    session: DebateSession,
) -> Dict[str, Any]:
    """
    Generate a JSON report of debate statistics.
    
    Args:
        session: Debate session to analyze
    
    Returns:
        Dictionary with debate statistics
    """
    # Compute statistics
    agent_stats = defaultdict(lambda: {
        "total_arguments": 0,
        "claims": 0,
        "supports": 0,
        "attacks": 0,
        "rebuttals": 0,
        "avg_confidence": 0,
        "validated": 0,
        "challenged": 0,
    })
    
    type_counts = defaultdict(int)
    status_counts = defaultdict(int)
    confidences = []
    
    for arg in session.all_arguments:
        stats = agent_stats[arg.agent_id]
        stats["total_arguments"] += 1
        
        type_key = arg.arg_type.value + "s" if arg.arg_type.value != "claim" else "claims"
        if type_key in stats:
            stats[type_key] += 1
        
        type_counts[arg.arg_type.value] += 1
        status_counts[arg.status.value] += 1
        confidences.append(arg.confidence)
        
        if arg.status == ArgumentStatus.VALIDATED:
            stats["validated"] += 1
        elif arg.status == ArgumentStatus.CHALLENGED:
            stats["challenged"] += 1
    
    # Compute averages
    for agent, stats in agent_stats.items():
        agent_confidences = [
            a.confidence for a in session.all_arguments
            if a.agent_id == agent
        ]
        if agent_confidences:
            stats["avg_confidence"] = sum(agent_confidences) / len(agent_confidences)
    
    return {
        "session_id": session.session_id,
        "topic": session.topic,
        "duration_seconds": session.duration_seconds,
        "total_arguments": len(session.all_arguments),
        "total_rounds": len(session.rounds),
        "agents": list(agent_stats.keys()),
        "agent_statistics": dict(agent_stats),
        "argument_types": dict(type_counts),
        "argument_status": dict(status_counts),
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "verdict": session.verdict,
    }


__all__ = [
    # Data models
    "ArgumentType",
    "ArgumentStatus",
    "Argument",
    "DebateRound",
    "DebateSession",
    
    # Theme
    "DEBATE_COLORS",
    "DARK_LAYOUT",
    
    # Visualization functions
    "plot_argument_flow",
    "plot_debate_timeline",
    "plot_agent_performance",
    "plot_confidence_evolution",
    "plot_round_summary",
    "plot_interaction_heatmap",
    "plot_argument_type_distribution",
    "create_debate_dashboard",
    
    # Export functions
    "export_debate_html",
    "export_debate_png",
    "generate_debate_report",
]
