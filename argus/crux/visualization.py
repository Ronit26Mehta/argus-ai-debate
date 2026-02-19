"""
CRUX Protocol Visualizations.

Interactive and static visualizations for CRUX debates, including:
    - Debate flow DAG with Claim Bundles
    - Credibility evolution over sessions
    - BRP reconciliation visualization
    - DFS heatmap for agent-claim pairs
    - Auction results visualization
    - Comprehensive dashboard

All chart functions return plotly.graph_objects.Figure objects
that can be rendered in Streamlit, Jupyter, or exported as images.

Example:
    >>> from argus.crux.visualization import plot_crux_debate_flow
    >>> 
    >>> # After running a CRUX debate
    >>> fig = plot_crux_debate_flow(crux_result.session)
    >>> fig.show()
"""

from __future__ import annotations

import math
import json
from datetime import datetime
from typing import Any, Optional, Union, TYPE_CHECKING
from collections import defaultdict

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

if TYPE_CHECKING:
    from plotly.graph_objects import Figure as PlotlyFigure
    from argus.crux.orchestrator import CRUXSession, CRUXDebateResult
    from argus.crux.claim_bundle import ClaimBundle
    from argus.crux.auction import AuctionResult
    from argus.crux.brp import BRPSession
    from argus.crux.ledger import CredibilityLedger

# ---------------------------------------------------------------------------
# Theme constants (matching argus_viz style)
# ---------------------------------------------------------------------------

CRUX_COLORS = {
    "bg": "#0e1117",
    "paper": "#1a1f2e",
    "grid": "#2a3040",
    "text": "#e0e0e0",
    "accent_cyan": "#00d4ff",
    "accent_magenta": "#ff00d4",
    "accent_amber": "#ffbf00",
    "support_green": "#00ff88",
    "attack_red": "#ff4466",
    "neutral": "#888888",
    
    # CRUX-specific colors
    "claim_bundle": "#00d4ff",
    "claim_supports": "#00ff88",
    "claim_attacks": "#ff4466",
    "brp_merge": "#b388ff",
    "auction": "#ff8800",
    "challenger": "#ff00d4",
    "credibility_high": "#00ff88",
    "credibility_low": "#ff4466",
    "credibility_mid": "#ffbf00",
    "edr_checkpoint": "#9966ff",
    "unchallenged": "#ff6b6b",
}

DARK_LAYOUT = dict(
    paper_bgcolor=CRUX_COLORS["bg"],
    plot_bgcolor=CRUX_COLORS["paper"],
    font=dict(family="Inter, sans-serif", color=CRUX_COLORS["text"], size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=CRUX_COLORS["grid"], zerolinecolor=CRUX_COLORS["grid"]),
    yaxis=dict(gridcolor=CRUX_COLORS["grid"], zerolinecolor=CRUX_COLORS["grid"]),
)


def _check_plotly() -> None:
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for CRUX visualizations. "
            "Install with: pip install plotly"
        )


def _apply_dark_layout(fig: Any, title: str = "") -> Any:
    """Apply consistent dark styling to a figure."""
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, font=dict(size=18, color=CRUX_COLORS["accent_cyan"])),
    )
    return fig


def _get_polarity_color(polarity: int) -> str:
    """Get color based on claim polarity."""
    if polarity > 0:
        return CRUX_COLORS["claim_supports"]
    elif polarity < 0:
        return CRUX_COLORS["claim_attacks"]
    return CRUX_COLORS["neutral"]


def _get_credibility_color(credibility: float) -> str:
    """Get color based on credibility rating."""
    if credibility >= 0.7:
        return CRUX_COLORS["credibility_high"]
    elif credibility >= 0.4:
        return CRUX_COLORS["credibility_mid"]
    return CRUX_COLORS["credibility_low"]


# ============================================================================
# 1. CRUX Debate Flow DAG
# ============================================================================

def plot_crux_debate_flow(
    session: "CRUXSession",
    show_auctions: bool = True,
    show_brp: bool = True,
    interactive: bool = True,
) -> Any:
    """
    Plot the entire CRUX debate as an interactive flow DAG.
    
    Shows Claim Bundles, their relationships, auctions, and BRP merges.
    
    Args:
        session: CRUX session data
        show_auctions: Include auction nodes
        show_brp: Include BRP reconciliation nodes
        interactive: Enable interactive features
        
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    fig = go.Figure()
    
    if not session or not session.claim_bundles:
        fig.add_annotation(
            text="No Claim Bundles in session",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=CRUX_COLORS["neutral"]),
        )
        return _apply_dark_layout(fig, "CRUX Debate Flow")
    
    # Build graph layout
    nodes = []
    edges = []
    node_positions = {}
    
    # Group claim bundles by time/round
    sorted_cbs = sorted(session.claim_bundles, key=lambda cb: cb.issued_at)
    
    # Position nodes in a timeline layout
    y_offset = 0
    time_slots = defaultdict(list)
    
    for cb in sorted_cbs:
        # Group by minute for layout
        minute_slot = cb.issued_at.strftime("%Y-%m-%d_%H:%M")
        time_slots[minute_slot].append(cb)
    
    x_pos = 0
    for minute, cbs in time_slots.items():
        for i, cb in enumerate(cbs):
            y_pos = i * 1.5 - (len(cbs) - 1) * 0.75
            node_positions[cb.cb_id] = (x_pos, y_pos)
            
            polarity_color = _get_polarity_color(cb.polarity.value if hasattr(cb.polarity, 'value') else cb.polarity)
            
            nodes.append({
                "id": cb.cb_id,
                "type": "claim_bundle",
                "x": x_pos,
                "y": y_pos,
                "color": polarity_color,
                "size": 30 + cb.issuer_credibility * 20,
                "label": f"{cb.claim_text[:30]}...",
                "posterior": cb.posterior,
                "agent": cb.issuer_agent,
                "challenged": not cb.challenge_open,
            })
        x_pos += 2
    
    # Add auction nodes
    if show_auctions:
        for auction in session.auctions:
            if auction.cb_id in node_positions:
                cb_pos = node_positions[auction.cb_id]
                a_x = cb_pos[0] + 0.5
                a_y = cb_pos[1] - 0.8
                
                node_positions[auction.auction_id] = (a_x, a_y)
                
                color = (
                    CRUX_COLORS["unchallenged"] 
                    if auction.unchallenged 
                    else CRUX_COLORS["auction"]
                )
                
                nodes.append({
                    "id": auction.auction_id,
                    "type": "auction",
                    "x": a_x,
                    "y": a_y,
                    "color": color,
                    "size": 20,
                    "label": f"Auction ({auction.num_bids} bids)",
                    "winner": auction.winner,
                    "unchallenged": auction.unchallenged,
                })
                
                edges.append({
                    "from": auction.cb_id,
                    "to": auction.auction_id,
                    "type": "auction",
                })
    
    # Add BRP nodes
    if show_brp:
        for brp in session.brp_sessions:
            if brp.resolution:
                # Position BRP node between the reconciled bundles
                involved_cbs = brp.resolution.original_cb_ids
                if involved_cbs:
                    avg_x = sum(
                        node_positions.get(cb_id, (0, 0))[0] 
                        for cb_id in involved_cbs
                    ) / max(len(involved_cbs), 1) + 0.5
                    avg_y = sum(
                        node_positions.get(cb_id, (0, 0))[1] 
                        for cb_id in involved_cbs
                    ) / max(len(involved_cbs), 1) + 1
                else:
                    avg_x, avg_y = x_pos, 0
                
                node_positions[brp.resolution.resolution_id] = (avg_x, avg_y)
                
                nodes.append({
                    "id": brp.resolution.resolution_id,
                    "type": "brp",
                    "x": avg_x,
                    "y": avg_y,
                    "color": CRUX_COLORS["brp_merge"],
                    "size": 35,
                    "label": f"BRP Merge\n(P={brp.resolution.reconciled_posterior:.2f})",
                    "posterior": brp.resolution.reconciled_posterior,
                    "contributors": brp.resolution.contributor_agents,
                })
                
                # Add edges from original CBs to BRP
                for cb_id in involved_cbs:
                    if cb_id in node_positions:
                        edges.append({
                            "from": cb_id,
                            "to": brp.resolution.resolution_id,
                            "type": "brp_input",
                        })
    
    # Draw edges
    for edge in edges:
        if edge["from"] in node_positions and edge["to"] in node_positions:
            from_pos = node_positions[edge["from"]]
            to_pos = node_positions[edge["to"]]
            
            edge_color = (
                CRUX_COLORS["auction"] if edge["type"] == "auction"
                else CRUX_COLORS["brp_merge"] if "brp" in edge["type"]
                else CRUX_COLORS["claim_bundle"]
            )
            
            fig.add_trace(go.Scatter(
                x=[from_pos[0], to_pos[0]],
                y=[from_pos[1], to_pos[1]],
                mode="lines",
                line=dict(color=edge_color, width=2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ))
    
    # Draw nodes by type
    node_types = defaultdict(list)
    for node in nodes:
        node_types[node["type"]].append(node)
    
    # Claim Bundle nodes
    if "claim_bundle" in node_types:
        cb_nodes = node_types["claim_bundle"]
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in cb_nodes],
            y=[n["y"] for n in cb_nodes],
            mode="markers+text",
            marker=dict(
                size=[n["size"] for n in cb_nodes],
                color=[n["color"] for n in cb_nodes],
                line=dict(width=2, color=CRUX_COLORS["bg"]),
                symbol="circle",
            ),
            text=[f"P={n['posterior']:.2f}" for n in cb_nodes],
            textposition="top center",
            textfont=dict(size=10, color=CRUX_COLORS["text"]),
            name="Claim Bundles",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Posterior: %{customdata[1]:.3f}<br>"
                "Agent: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=[
                [n["label"], n["posterior"], n["agent"]] for n in cb_nodes
            ],
        ))
    
    # Auction nodes
    if show_auctions and "auction" in node_types:
        a_nodes = node_types["auction"]
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in a_nodes],
            y=[n["y"] for n in a_nodes],
            mode="markers",
            marker=dict(
                size=[n["size"] for n in a_nodes],
                color=[n["color"] for n in a_nodes],
                symbol="diamond",
                line=dict(width=2, color=CRUX_COLORS["bg"]),
            ),
            name="Auctions",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Winner: %{customdata[1]}<br>"
                "Unchallenged: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=[
                [n["label"], n["winner"] or "None", n["unchallenged"]] 
                for n in a_nodes
            ],
        ))
    
    # BRP nodes
    if show_brp and "brp" in node_types:
        brp_nodes = node_types["brp"]
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in brp_nodes],
            y=[n["y"] for n in brp_nodes],
            mode="markers+text",
            marker=dict(
                size=[n["size"] for n in brp_nodes],
                color=CRUX_COLORS["brp_merge"],
                symbol="star",
                line=dict(width=2, color=CRUX_COLORS["bg"]),
            ),
            text=[n["label"].split("\n")[0] for n in brp_nodes],
            textposition="top center",
            textfont=dict(size=10, color=CRUX_COLORS["text"]),
            name="BRP Merges",
            hovertemplate=(
                "<b>BRP Reconciliation</b><br>"
                "Posterior: %{customdata[0]:.3f}<br>"
                "Contributors: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=[
                [n["posterior"], ", ".join(n.get("contributors", []))] 
                for n in brp_nodes
            ],
        ))
    
    # Layout
    _apply_dark_layout(fig, "🔄 CRUX Debate Flow")
    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            title="Time →",
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        height=600,
    )
    
    return fig


# ============================================================================
# 2. Credibility Evolution
# ============================================================================

def plot_credibility_evolution(
    ledger: "CredibilityLedger",
    agent_ids: Optional[list[str]] = None,
    num_points: int = 50,
) -> Any:
    """
    Plot credibility rating evolution over time for agents.
    
    Args:
        ledger: Credibility ledger
        agent_ids: Agent IDs to plot (None for all)
        num_points: Maximum points per agent
        
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    fig = go.Figure()
    
    # Get all agents if not specified
    if agent_ids is None:
        agent_ids = ledger.get_all_agent_ids()
    
    if not agent_ids:
        fig.add_annotation(
            text="No credibility data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=CRUX_COLORS["neutral"]),
        )
        return _apply_dark_layout(fig, "Agent Credibility Evolution")
    
    # Color palette for agents
    colors = [
        CRUX_COLORS["accent_cyan"],
        CRUX_COLORS["accent_magenta"],
        CRUX_COLORS["accent_amber"],
        CRUX_COLORS["claim_supports"],
        CRUX_COLORS["claim_attacks"],
        CRUX_COLORS["brp_merge"],
    ]
    
    for i, agent_id in enumerate(agent_ids):
        history = ledger.get_credibility_history(agent_id)
        
        if not history:
            continue
        
        # Limit to num_points
        if len(history) > num_points:
            step = len(history) // num_points
            history = history[::step]
        
        timestamps = [h["timestamp"] for h in history]
        credibilities = [h["credibility"] for h in history]
        
        color = colors[i % len(colors)]
        
        # Main line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=credibilities,
            mode="lines+markers",
            name=agent_id.split("-")[-1] if "-" in agent_id else agent_id,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate=(
                f"<b>{agent_id}</b><br>"
                "Time: %{x}<br>"
                "Credibility: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        ))
    
    # Add reference lines
    fig.add_hline(
        y=0.3, line_dash="dash", line_color=CRUX_COLORS["credibility_low"],
        annotation_text="Suspension Floor (0.30)",
    )
    fig.add_hline(
        y=0.7, line_dash="dash", line_color=CRUX_COLORS["credibility_high"],
        annotation_text="High Trust (0.70)",
    )
    
    _apply_dark_layout(fig, "📊 Agent Credibility Evolution")
    fig.update_layout(
        xaxis_title="Session / Time",
        yaxis_title="Credibility Rating",
        yaxis=dict(range=[0, 1]),
        height=450,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
        ),
    )
    
    return fig


# ============================================================================
# 3. BRP Merge Visualization
# ============================================================================

def plot_brp_merge(
    brp_session: "BRPSession",
    show_distributions: bool = True,
) -> Any:
    """
    Visualize a Belief Reconciliation Protocol merge.
    
    Shows original claim posteriors and the reconciled result.
    
    Args:
        brp_session: BRP session data
        show_distributions: Show confidence distributions
        
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    if show_distributions:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Posterior Merge", "Confidence Distributions"],
            specs=[[{"type": "bar"}, {"type": "scatter"}]],
        )
    else:
        fig = go.Figure()
    
    if not brp_session or not brp_session.resolution:
        fig.add_annotation(
            text="No BRP resolution data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=CRUX_COLORS["neutral"]),
        )
        return _apply_dark_layout(fig, "BRP Reconciliation")
    
    resolution = brp_session.resolution
    
    # Prepare data for bar chart
    agents = resolution.contributor_agents or []
    weights = resolution.contributor_weights or [1.0] * len(agents)
    
    # If we have the original claim bundles
    original_posteriors = []
    original_labels = []
    
    # Try to reconstruct from session data
    if hasattr(brp_session, 'claim_bundles'):
        for cb in brp_session.claim_bundles:
            original_posteriors.append(cb.posterior)
            original_labels.append(cb.issuer_agent.split("-")[-1] if "-" in cb.issuer_agent else "CB")
    else:
        # Use placeholder data
        for i, agent in enumerate(agents):
            # Estimate from weights (rough approximation)
            original_posteriors.append(0.5)  # Placeholder
            original_labels.append(agent.split("-")[-1] if "-" in agent else f"Agent{i+1}")
    
    # Add reconciled posterior
    original_posteriors.append(resolution.reconciled_posterior)
    original_labels.append("Reconciled")
    
    # Colors
    bar_colors = []
    for i, label in enumerate(original_labels):
        if label == "Reconciled":
            bar_colors.append(CRUX_COLORS["brp_merge"])
        else:
            bar_colors.append(CRUX_COLORS["claim_bundle"])
    
    # Bar chart
    bar_trace = go.Bar(
        x=original_labels,
        y=original_posteriors,
        marker=dict(
            color=bar_colors,
            line=dict(width=2, color=CRUX_COLORS["bg"]),
        ),
        text=[f"{p:.2f}" for p in original_posteriors],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Posterior: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    )
    
    if show_distributions:
        fig.add_trace(bar_trace, row=1, col=1)
        
        # Distribution plot
        x_range = [i/100 for i in range(101)]
        
        # Plot reconciled distribution
        if resolution.reconciled_distribution:
            dist = resolution.reconciled_distribution
            if hasattr(dist, 'beta') and dist.beta:
                from scipy import stats
                y_vals = [
                    stats.beta.pdf(x, dist.beta.alpha, dist.beta.beta)
                    for x in x_range
                ]
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_vals,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(179, 136, 255, 0.3)",
                    line=dict(color=CRUX_COLORS["brp_merge"], width=2),
                    name="Reconciled Distribution",
                ), row=1, col=2)
        
        fig.update_xaxes(title_text="Agent/Result", row=1, col=1)
        fig.update_yaxes(title_text="Posterior", row=1, col=1)
        fig.update_xaxes(title_text="Probability", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=2)
    else:
        fig.add_trace(bar_trace)
        fig.update_xaxes(title_text="Agent/Result")
        fig.update_yaxes(title_text="Posterior Probability")
    
    _apply_dark_layout(fig, "🔀 BRP Belief Reconciliation")
    fig.update_layout(height=400, showlegend=True)
    
    return fig


# ============================================================================
# 4. DFS Heatmap
# ============================================================================

def plot_dfs_heatmap(
    dfs_scores: dict[str, dict[str, float]],
    title: str = "Dialectical Fitness Scores",
) -> Any:
    """
    Create heatmap of Dialectical Fitness Scores.
    
    Args:
        dfs_scores: Nested dict of {agent_id: {cb_id: score}}
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    fig = go.Figure()
    
    if not dfs_scores:
        fig.add_annotation(
            text="No DFS data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=CRUX_COLORS["neutral"]),
        )
        return _apply_dark_layout(fig, title)
    
    # Extract agents and claim bundles
    agents = list(dfs_scores.keys())
    cb_ids = set()
    for agent_scores in dfs_scores.values():
        cb_ids.update(agent_scores.keys())
    cb_ids = sorted(cb_ids)
    
    # Build matrix
    z_values = []
    for agent in agents:
        row = []
        for cb_id in cb_ids:
            score = dfs_scores.get(agent, {}).get(cb_id, 0)
            row.append(score)
        z_values.append(row)
    
    # Shorten labels
    agent_labels = [a.split("-")[-1] if "-" in a else a[:15] for a in agents]
    cb_labels = [cb[:10] + "..." if len(cb) > 10 else cb for cb in cb_ids]
    
    # Create heatmap
    fig.add_trace(go.Heatmap(
        z=z_values,
        x=cb_labels,
        y=agent_labels,
        colorscale=[
            [0, CRUX_COLORS["credibility_low"]],
            [0.5, CRUX_COLORS["credibility_mid"]],
            [1, CRUX_COLORS["credibility_high"]],
        ],
        colorbar=dict(
            title="DFS Score",
            titleside="right",
        ),
        hovertemplate=(
            "Agent: %{y}<br>"
            "Claim: %{x}<br>"
            "DFS: %{z:.3f}<br>"
            "<extra></extra>"
        ),
    ))
    
    _apply_dark_layout(fig, f"🎯 {title}")
    fig.update_layout(
        xaxis_title="Claim Bundles",
        yaxis_title="Agents",
        height=max(300, len(agents) * 40 + 100),
    )
    
    return fig


def plot_dfs_breakdown(
    dfs_result: Any,
) -> Any:
    """
    Plot breakdown of a single DFS score components.
    
    Args:
        dfs_result: DialecticalFitnessScore object
        
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    fig = go.Figure()
    
    components = [
        ("Domain Match", dfs_result.domain_match, CRUX_COLORS["claim_bundle"]),
        ("Adversarial Potential", dfs_result.adversarial_potential, CRUX_COLORS["challenger"]),
        ("Credibility Factor", dfs_result.credibility_factor, CRUX_COLORS["credibility_high"]),
        ("Recency Factor", dfs_result.recency_factor, CRUX_COLORS["accent_amber"]),
    ]
    
    labels = [c[0] for c in components]
    values = [c[1] for c in components]
    colors = [c[2] for c in components]
    
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors, line=dict(width=2, color=CRUX_COLORS["bg"])),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    
    # Total score annotation
    fig.add_annotation(
        text=f"Total DFS: {dfs_result.total_score:.3f}",
        xref="paper", yref="paper",
        x=0.5, y=1.1,
        showarrow=False,
        font=dict(size=16, color=CRUX_COLORS["accent_cyan"]),
    )
    
    _apply_dark_layout(fig, f"DFS Breakdown: {dfs_result.agent_id}")
    fig.update_layout(
        yaxis_title="Score Component",
        yaxis=dict(range=[0, 1]),
        height=350,
    )
    
    return fig


# ============================================================================
# 5. Auction Results
# ============================================================================

def plot_auction_results(
    auction_result: "AuctionResult",
) -> Any:
    """
    Visualize Challenger Auction results.
    
    Shows all bids, DFS scores, and winner.
    
    Args:
        auction_result: Auction result data
        
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Bids by DFS Score", "Bid Distribution"],
        specs=[[{"type": "bar"}, {"type": "pie"}]],
    )
    
    if not auction_result or not auction_result.all_bids:
        fig.add_annotation(
            text="No bids in auction" if auction_result else "No auction data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=CRUX_COLORS["neutral"]),
        )
        return _apply_dark_layout(fig, "Challenger Auction Results")
    
    # Sort bids by DFS score
    sorted_bids = sorted(
        auction_result.all_bids,
        key=lambda b: auction_result.dfs_scores.get(b.bidder_agent, b.estimated_dfs),
        reverse=True,
    )
    
    # Prepare data
    agents = [b.bidder_agent.split("-")[-1] if "-" in b.bidder_agent else b.bidder_agent[:10] for b in sorted_bids]
    scores = [auction_result.dfs_scores.get(b.bidder_agent, b.estimated_dfs) for b in sorted_bids]
    strategies = [b.adversarial_strategy for b in sorted_bids]
    
    # Colors - highlight winner
    colors = []
    winner_agent = auction_result.winner
    for bid in sorted_bids:
        if bid.bidder_agent == winner_agent:
            colors.append(CRUX_COLORS["challenger"])
        else:
            colors.append(CRUX_COLORS["claim_bundle"])
    
    # Bar chart of bids
    fig.add_trace(go.Bar(
        x=agents,
        y=scores,
        marker=dict(color=colors, line=dict(width=2, color=CRUX_COLORS["bg"])),
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "DFS: %{y:.3f}<br>"
            "Strategy: %{customdata}<br>"
            "<extra></extra>"
        ),
        customdata=strategies,
    ), row=1, col=1)
    
    # Pie chart of strategies
    strategy_counts = defaultdict(int)
    for bid in sorted_bids:
        strategy_counts[bid.adversarial_strategy] += 1
    
    fig.add_trace(go.Pie(
        labels=list(strategy_counts.keys()),
        values=list(strategy_counts.values()),
        marker=dict(
            colors=[CRUX_COLORS["accent_cyan"], CRUX_COLORS["accent_magenta"],
                    CRUX_COLORS["accent_amber"], CRUX_COLORS["claim_supports"]],
        ),
        hole=0.4,
        textinfo="label+percent",
    ), row=1, col=2)
    
    # Title with winner info
    title = "⚔️ Challenger Auction Results"
    if auction_result.unchallenged:
        title += " (UNCHALLENGED)"
    elif winner_agent:
        title += f" — Winner: {winner_agent.split('-')[-1] if '-' in winner_agent else winner_agent}"
    
    _apply_dark_layout(fig, title)
    fig.update_layout(
        height=400,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Bidder", row=1, col=1)
    fig.update_yaxes(title_text="DFS Score", row=1, col=1)
    
    return fig


def plot_auction_timeline(
    auctions: list["AuctionResult"],
) -> Any:
    """
    Plot timeline of multiple auctions in a session.
    
    Args:
        auctions: List of auction results
        
    Returns:
        Plotly Figure object
    """
    _check_plotly()
    
    fig = go.Figure()
    
    if not auctions:
        fig.add_annotation(
            text="No auctions to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=CRUX_COLORS["neutral"]),
        )
        return _apply_dark_layout(fig, "Auction Timeline")
    
    # Prepare data
    for i, auction in enumerate(auctions):
        y_pos = i
        
        # Auction bar
        color = (
            CRUX_COLORS["unchallenged"] 
            if auction.unchallenged 
            else CRUX_COLORS["auction"]
        )
        
        fig.add_trace(go.Scatter(
            x=[auction.started_at, auction.completed_at],
            y=[y_pos, y_pos],
            mode="lines+markers",
            line=dict(color=color, width=10),
            marker=dict(size=15, color=color, symbol="circle"),
            name=f"Auction {i+1}",
            hovertemplate=(
                f"<b>Auction {i+1}</b><br>"
                f"Winner: {auction.winner or 'None'}<br>"
                f"Bids: {auction.num_bids}<br>"
                f"Duration: {auction.duration_seconds:.1f}s<br>"
                "<extra></extra>"
            ),
        ))
    
    # Labels
    auction_labels = [
        f"A{i+1}: {a.winner.split('-')[-1] if a.winner and '-' in a.winner else a.winner or 'Unchallenged'}"
        for i, a in enumerate(auctions)
    ]
    
    _apply_dark_layout(fig, "📅 Auction Timeline")
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(auctions))),
            ticktext=auction_labels,
        ),
        xaxis_title="Time",
        height=max(300, len(auctions) * 50 + 100),
        showlegend=False,
    )
    
    return fig


# ============================================================================
# 6. CRUX Dashboard
# ============================================================================

def create_crux_dashboard(
    session: "CRUXSession",
    ledger: Optional["CredibilityLedger"] = None,
) -> Any:
    """
    Create comprehensive CRUX dashboard with multiple visualizations.
    
    Args:
        session: CRUX session data
        ledger: Optional credibility ledger
        
    Returns:
        Plotly Figure with subplots
    """
    _check_plotly()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Debate Flow Overview",
            "Posterior Distribution",
            "Agent Activity",
            "Session Statistics",
        ],
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    if not session:
        fig.add_annotation(
            text="No session data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=CRUX_COLORS["neutral"]),
        )
        return _apply_dark_layout(fig, "CRUX Dashboard")
    
    # 1. Debate flow (simplified scatter)
    if session.claim_bundles:
        x_vals = list(range(len(session.claim_bundles)))
        y_vals = [cb.posterior for cb in session.claim_bundles]
        colors = [_get_polarity_color(cb.polarity.value if hasattr(cb.polarity, 'value') else cb.polarity) for cb in session.claim_bundles]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+lines",
            marker=dict(size=15, color=colors, line=dict(width=2, color=CRUX_COLORS["bg"])),
            line=dict(color=CRUX_COLORS["neutral"], width=1, dash="dot"),
            name="Posteriors",
        ), row=1, col=1)
        
        fig.add_hline(y=0.5, line_dash="dash", line_color=CRUX_COLORS["neutral"], row=1, col=1)
    
    # 2. Posterior histogram
    if session.claim_bundles:
        posteriors = [cb.posterior for cb in session.claim_bundles]
        fig.add_trace(go.Histogram(
            x=posteriors,
            nbinsx=20,
            marker=dict(color=CRUX_COLORS["claim_bundle"]),
            name="Posterior Dist",
        ), row=1, col=2)
    
    # 3. Agent activity bar chart
    agent_counts = defaultdict(int)
    for cb in session.claim_bundles:
        agent_id = cb.issuer_agent.split("-")[-1] if "-" in cb.issuer_agent else cb.issuer_agent
        agent_counts[agent_id] += 1
    
    if agent_counts:
        fig.add_trace(go.Bar(
            x=list(agent_counts.keys()),
            y=list(agent_counts.values()),
            marker=dict(color=CRUX_COLORS["accent_cyan"]),
            name="Claims/Agent",
        ), row=2, col=1)
    
    # 4. Session statistics indicator
    stats = session.stats
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=stats.num_claim_bundles,
        title=dict(text="Total Claims"),
        delta=dict(reference=5, relative=True),
        domain=dict(x=[0.6, 1], y=[0, 0.4]),
    ), row=2, col=2)
    
    _apply_dark_layout(fig, f"📊 CRUX Dashboard: {session.session_id[:12]}...")
    fig.update_layout(
        height=700,
        showlegend=False,
    )
    
    # Update axes
    fig.update_xaxes(title_text="Claim Index", row=1, col=1)
    fig.update_yaxes(title_text="Posterior", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Posterior Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Agent", row=2, col=1)
    fig.update_yaxes(title_text="Claims Issued", row=2, col=1)
    
    return fig


# ============================================================================
# 7. Static Export
# ============================================================================

def export_debate_static(
    result: "CRUXDebateResult",
    output_path: str,
    format: str = "png",
    width: int = 1200,
    height: int = 800,
) -> str:
    """
    Export debate visualization as static image.
    
    Args:
        result: CRUX debate result
        output_path: Output file path (without extension)
        format: Image format (png, svg, pdf, jpeg)
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Path to exported file
    """
    _check_plotly()
    
    try:
        import kaleido
    except ImportError:
        raise ImportError(
            "Kaleido is required for static export. "
            "Install with: pip install kaleido"
        )
    
    if result.session:
        fig = create_crux_dashboard(result.session)
    else:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Verdict: {result.verdict}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=24, color=CRUX_COLORS["accent_cyan"]),
        )
        _apply_dark_layout(fig, "CRUX Debate Result")
    
    # Export
    full_path = f"{output_path}.{format}"
    fig.write_image(
        full_path,
        format=format,
        width=width,
        height=height,
        scale=2,
    )
    
    return full_path


def export_session_json(
    session: "CRUXSession",
    output_path: str,
) -> str:
    """
    Export session data as JSON for external visualization.
    
    Args:
        session: CRUX session
        output_path: Output file path
        
    Returns:
        Path to exported file
    """
    data = {
        "session_id": session.session_id,
        "proposition": session.proposition_text,
        "prior": session.prior,
        "state": session.state.value,
        "stats": session.stats.to_dict(),
        "claim_bundles": [cb.to_dict() for cb in session.claim_bundles],
        "auctions": [a.to_dict() for a in session.auctions],
        "started_at": session.started_at.isoformat(),
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        "duration_seconds": session.duration_seconds,
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    return output_path


# ============================================================================
# Utility Functions
# ============================================================================

def build_dfs_matrix(
    session: "CRUXSession",
    router: Any,
) -> dict[str, dict[str, float]]:
    """
    Build DFS matrix from session and router.
    
    Args:
        session: CRUX session
        router: Dialectical router
        
    Returns:
        DFS scores matrix
    """
    dfs_matrix = {}
    
    for cb in session.claim_bundles:
        for agent_id in session.cb_by_agent.keys():
            if agent_id not in dfs_matrix:
                dfs_matrix[agent_id] = {}
            
            # Get agent card and compute DFS
            card = router.registry.get_card(agent_id)
            if card:
                dfs = router.compute_dfs(card, cb)
                dfs_matrix[agent_id][cb.cb_id] = dfs.total_score
    
    return dfs_matrix
