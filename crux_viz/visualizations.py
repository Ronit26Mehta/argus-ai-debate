"""
CRUX-Viz Visualizations.

Wrappers and extensions around argus.crux.visualization,
providing additional charts for session analytics and protocol KPIs.

All functions return Plotly Figure objects.
"""

from __future__ import annotations

from typing import Any, Optional
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

# ---------------------------------------------------------------------------
# Re-export from argus.crux.visualization
# ---------------------------------------------------------------------------

try:
    from argus.crux.visualization import (
        plot_crux_debate_flow,
        plot_credibility_evolution,
        plot_brp_merge,
        plot_dfs_heatmap,
        plot_auction_results,
        create_crux_dashboard,
        export_debate_static,
    )
except ImportError:
    # Graceful fallback — visualization module not available
    def _missing(name: str):
        def _stub(*args, **kwargs):
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"{name} requires argus.crux.visualization",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="#888"),
                )
                fig.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                    font=dict(color="#e0e0e0"), height=300,
                )
                return fig
        return _stub

    plot_crux_debate_flow = _missing("plot_crux_debate_flow")
    plot_credibility_evolution = _missing("plot_credibility_evolution")
    plot_brp_merge = _missing("plot_brp_merge")
    plot_dfs_heatmap = _missing("plot_dfs_heatmap")
    plot_auction_results = _missing("plot_auction_results")
    create_crux_dashboard = _missing("create_crux_dashboard")
    export_debate_static = _missing("export_debate_static")


# ---------------------------------------------------------------------------
# Theme constants (matching CRUX-Viz magenta theme)
# ---------------------------------------------------------------------------

CRUX_VIZ_COLORS = {
    "bg": "#0e1117",
    "paper": "#1a1f2e",
    "grid": "#2a3040",
    "text": "#e0e0e0",
    "accent": "#ff00d4",
    "cyan": "#00d4ff",
    "green": "#00ff88",
    "red": "#ff4466",
    "amber": "#ffbf00",
    "purple": "#b388ff",
    "orange": "#ff8800",
}

_DARK_LAYOUT = dict(
    paper_bgcolor=CRUX_VIZ_COLORS["bg"],
    plot_bgcolor=CRUX_VIZ_COLORS["paper"],
    font=dict(family="Inter, sans-serif", color=CRUX_VIZ_COLORS["text"], size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=CRUX_VIZ_COLORS["grid"], zerolinecolor=CRUX_VIZ_COLORS["grid"]),
    yaxis=dict(gridcolor=CRUX_VIZ_COLORS["grid"], zerolinecolor=CRUX_VIZ_COLORS["grid"]),
)


def _dark(fig: Any, title: str = "", height: int = 400) -> Any:
    """Apply dark CRUX-Viz theme to figure."""
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=16, color=CRUX_VIZ_COLORS["accent"])),
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Posterior Evolution (from rounds data)
# ---------------------------------------------------------------------------

def plot_posterior_evolution(rounds: list[dict]) -> Any:
    """Plot posterior probability trajectory across debate rounds."""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    if not rounds:
        fig.add_annotation(text="No rounds data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=18, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "⚡ Posterior Evolution")

    round_nums = [r["round"] for r in rounds]
    posteriors_before = [r.get("posterior_before", 0.5) for r in rounds]
    posteriors_after = [r.get("posterior_after", 0.5) for r in rounds]

    # Before line
    fig.add_trace(go.Scatter(
        x=round_nums, y=posteriors_before,
        mode="lines+markers",
        name="Before Round",
        line=dict(color=CRUX_VIZ_COLORS["cyan"], width=1, dash="dot"),
        marker=dict(size=6, color=CRUX_VIZ_COLORS["cyan"]),
        hovertemplate="Round %{x}<br>Prior: %{y:.3f}<extra></extra>",
    ))

    # After line
    fig.add_trace(go.Scatter(
        x=round_nums, y=posteriors_after,
        mode="lines+markers",
        name="After Round",
        line=dict(color=CRUX_VIZ_COLORS["accent"], width=3),
        marker=dict(size=10, color=CRUX_VIZ_COLORS["accent"],
                    line=dict(width=2, color=CRUX_VIZ_COLORS["bg"])),
        hovertemplate="Round %{x}<br>Posterior: %{y:.3f}<extra></extra>",
    ))

    # Fill between
    fig.add_trace(go.Scatter(
        x=round_nums + round_nums[::-1],
        y=posteriors_after + posteriors_before[::-1],
        fill="toself",
        fillcolor="rgba(255, 0, 212, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Decision threshold line
    fig.add_hline(y=0.7, line_dash="dash", line_color=CRUX_VIZ_COLORS["green"],
                  annotation_text="Support threshold (0.70)")
    fig.add_hline(y=0.3, line_dash="dash", line_color=CRUX_VIZ_COLORS["red"],
                  annotation_text="Rejection threshold (0.30)")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#555")

    _dark(fig, "⚡ CRUX Posterior Evolution", height=420)
    fig.update_layout(
        xaxis_title="Debate Round",
        yaxis_title="Posterior Probability",
        yaxis=dict(range=[0, 1]),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Claim Bundle Timeline
# ---------------------------------------------------------------------------

def plot_claim_bundle_timeline(claim_bundles: list[dict]) -> Any:
    """Gantt-style timeline of Claim Bundles per agent."""
    if not PLOTLY_AVAILABLE or not claim_bundles:
        fig = go.Figure()
        if PLOTLY_AVAILABLE:
            fig.add_annotation(text="No Claim Bundles yet", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "📦 Claim Bundle Timeline") if PLOTLY_AVAILABLE else None

    fig = go.Figure()

    agents = list({cb.get("issuer_agent", "Unknown") for cb in claim_bundles})
    agent_y = {a: i for i, a in enumerate(agents)}

    color_map = {
        "1": CRUX_VIZ_COLORS["green"],
        "supports": CRUX_VIZ_COLORS["green"],
        "SUPPORTS": CRUX_VIZ_COLORS["green"],
        "-1": CRUX_VIZ_COLORS["red"],
        "attacks": CRUX_VIZ_COLORS["red"],
        "ATTACKS": CRUX_VIZ_COLORS["red"],
    }

    for i, cb in enumerate(claim_bundles):
        polarity_raw = str(cb.get("polarity", "0"))
        color = color_map.get(polarity_raw, CRUX_VIZ_COLORS["amber"])
        agent = cb.get("issuer_agent", "Unknown")
        y = agent_y.get(agent, 0)
        posterior = cb.get("posterior", 0.5)
        claim_text = cb.get("claim_text", "")[:50]

        fig.add_trace(go.Scatter(
            x=[i], y=[y],
            mode="markers",
            marker=dict(
                size=16 + posterior * 16,
                color=color,
                symbol="circle",
                line=dict(width=2, color=CRUX_VIZ_COLORS["bg"]),
                opacity=0.85,
            ),
            name=agent.split("-")[-1] if "-" in agent else agent,
            showlegend=(i == 0 or agent != claim_bundles[i-1].get("issuer_agent")),
            hovertemplate=(
                f"<b>{claim_text}</b><br>"
                f"Agent: {agent}<br>"
                f"Posterior: {posterior:.3f}<br>"
                f"Polarity: {polarity_raw}<extra></extra>"
            ),
        ))

    _dark(fig, "📦 Claim Bundle Timeline", height=380)
    fig.update_layout(
        xaxis_title="Claim Bundle Index",
        yaxis=dict(
            tickmode="array",
            tickvals=list(agent_y.values()),
            ticktext=[a.split("-")[-1] if "-" in a else a[:15] for a in agents],
        ),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Session Stats Radar
# ---------------------------------------------------------------------------

def plot_session_stats_radar(stats: dict) -> Any:
    """Spider/radar chart of CRUX session KPIs."""
    if not PLOTLY_AVAILABLE:
        return None

    categories = [
        "Claim Bundles", "Challenges", "BRP Sessions",
        "Auctions", "Credibility Updates", "Avg DFS×10",
    ]

    values = [
        min(stats.get("num_claim_bundles", 0), 20),
        min(stats.get("num_challenges", 0), 20),
        min(stats.get("num_brp_sessions", 0), 20),
        min(stats.get("num_auctions", 0), 20),
        min(stats.get("total_credibility_updates", 0), 20),
        min(stats.get("avg_dfs", 0.0) * 10, 20),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255, 0, 212, 0.15)",
        line=dict(color=CRUX_VIZ_COLORS["accent"], width=2),
        marker=dict(size=8, color=CRUX_VIZ_COLORS["accent"]),
        name="CRUX KPIs",
    ))

    _dark(fig, "🕸️ CRUX Session KPI Radar", height=420)
    fig.update_layout(
        polar=dict(
            bgcolor=CRUX_VIZ_COLORS["paper"],
            radialaxis=dict(
                visible=True,
                range=[0, 20],
                gridcolor=CRUX_VIZ_COLORS["grid"],
                linecolor=CRUX_VIZ_COLORS["grid"],
                tickcolor=CRUX_VIZ_COLORS["text"],
            ),
            angularaxis=dict(
                gridcolor=CRUX_VIZ_COLORS["grid"],
                linecolor=CRUX_VIZ_COLORS["grid"],
            ),
        ),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. EDR Checkpoints Trace
# ---------------------------------------------------------------------------

def plot_edr_checkpoints(checkpoints: list[dict]) -> Any:
    """Plot EDR checkpoint posterior trace."""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    if not checkpoints:
        fig.add_annotation(text="No EDR checkpoints (EDR disabled or not triggered)",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "📍 EDR Checkpoints")

    indices = list(range(len(checkpoints)))
    posteriors = [cp.get("posterior", 0.5) for cp in checkpoints]
    labels = [cp.get("checkpoint_id", f"CP-{i}")[:12] for i, cp in enumerate(checkpoints)]

    fig.add_trace(go.Scatter(
        x=indices, y=posteriors,
        mode="lines+markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=9, color=CRUX_VIZ_COLORS["purple"]),
        line=dict(color=CRUX_VIZ_COLORS["purple"], width=2),
        marker=dict(size=14, color=CRUX_VIZ_COLORS["purple"], symbol="diamond",
                    line=dict(width=2, color=CRUX_VIZ_COLORS["bg"])),
        name="EDR Checkpoint",
        hovertemplate="Checkpoint: %{text}<br>Posterior: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="#555")

    _dark(fig, "📍 EDR Checkpoint Posteriors", height=360)
    fig.update_layout(
        xaxis_title="Checkpoint Index",
        yaxis_title="Posterior",
        yaxis=dict(range=[0, 1]),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Evidence Polarity Donut
# ---------------------------------------------------------------------------

def plot_evidence_polarity_donut(rounds: list[dict]) -> Any:
    """Donut chart of evidence polarity distribution."""
    if not PLOTLY_AVAILABLE:
        return None

    total_support = sum(r.get("support_count", 0) for r in rounds)
    total_attack  = sum(r.get("attack_count", 0) for r in rounds)
    total_neutral = sum(
        r.get("total_evidence", 0) - r.get("support_count", 0) - r.get("attack_count", 0)
        for r in rounds
    )

    if total_support + total_attack + total_neutral == 0:
        fig = go.Figure()
        fig.add_annotation(text="No evidence yet", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "Evidence Polarity")

    fig = go.Figure(go.Pie(
        labels=["Support", "Attack", "Neutral"],
        values=[total_support, total_attack, max(total_neutral, 0)],
        hole=0.6,
        marker=dict(colors=[
            CRUX_VIZ_COLORS["green"],
            CRUX_VIZ_COLORS["red"],
            CRUX_VIZ_COLORS["amber"],
        ], line=dict(color=CRUX_VIZ_COLORS["bg"], width=3)),
        textinfo="label+percent",
        textfont=dict(size=12, color=CRUX_VIZ_COLORS["text"]),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))

    _dark(fig, "🎯 Evidence Polarity", height=380)
    fig.update_layout(
        annotations=[dict(
            text=f"{total_support + total_attack + total_neutral}<br>items",
            font=dict(size=20, color=CRUX_VIZ_COLORS["accent"]),
            showarrow=False,
        )],
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. BRP Summary Bar
# ---------------------------------------------------------------------------

def plot_brp_summary(brp_sessions: list[dict]) -> Any:
    """Bar chart showing BRP reconciliation outcomes."""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    if not brp_sessions:
        fig.add_annotation(text="No BRP sessions triggered",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "🔀 BRP Reconciliation Summary")

    labels = [f"BRP-{i+1}" for i in range(len(brp_sessions))]
    reconciled = [b.get("resolution", {}).get("reconciled_posterior", 0.5) for b in brp_sessions]
    deltas = [b.get("contradiction_delta", 0.0) for b in brp_sessions]

    fig.add_trace(go.Bar(
        x=labels, y=reconciled,
        name="Reconciled Posterior",
        marker=dict(color=CRUX_VIZ_COLORS["purple"],
                    line=dict(width=2, color=CRUX_VIZ_COLORS["bg"])),
        text=[f"{p:.2f}" for p in reconciled],
        textposition="outside",
        hovertemplate="%{x}<br>Reconciled: %{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=deltas,
        mode="lines+markers",
        name="Contradiction Δ",
        yaxis="y2",
        line=dict(color=CRUX_VIZ_COLORS["orange"], width=2),
        marker=dict(size=8, color=CRUX_VIZ_COLORS["orange"]),
        hovertemplate="%{x}<br>Δ: %{y:.3f}<extra></extra>",
    ))

    _dark(fig, "🔀 BRP Reconciliation Summary", height=380)
    fig.update_layout(
        xaxis_title="BRP Session",
        yaxis_title="Posterior",
        yaxis=dict(range=[0, 1.15]),
        yaxis2=dict(
            title="Contradiction Δ",
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False,
            color=CRUX_VIZ_COLORS["orange"],
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Credibility Snapshot Bar
# ---------------------------------------------------------------------------

def plot_credibility_snapshot(credibility_ledger: list[dict]) -> Any:
    """Bar chart of current agent credibility ratings."""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    if not credibility_ledger:
        fig.add_annotation(text="No credibility data",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "📊 Agent Credibility")

    agents = [e.get("agent_id", "?").split("-")[-1] for e in credibility_ledger]
    credibilities = [e.get("credibility", 0.5) for e in credibility_ledger]
    colors = [
        CRUX_VIZ_COLORS["green"] if c >= 0.7 else
        CRUX_VIZ_COLORS["amber"] if c >= 0.4 else
        CRUX_VIZ_COLORS["red"]
        for c in credibilities
    ]

    fig.add_trace(go.Bar(
        x=agents, y=credibilities,
        marker=dict(color=colors, line=dict(width=2, color=CRUX_VIZ_COLORS["bg"])),
        text=[f"{c:.2f}" for c in credibilities],
        textposition="outside",
        hovertemplate="Agent: %{x}<br>Credibility: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0.7, line_dash="dash", line_color=CRUX_VIZ_COLORS["green"],
                  annotation_text="High trust (0.70)")
    fig.add_hline(y=0.3, line_dash="dash", line_color=CRUX_VIZ_COLORS["red"],
                  annotation_text="Suspension floor (0.30)")

    _dark(fig, "📊 Agent Credibility Snapshot", height=360)
    fig.update_layout(
        yaxis_title="Credibility Rating",
        yaxis=dict(range=[0, 1.15]),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Auction Summary
# ---------------------------------------------------------------------------

def plot_auction_summary(auctions: list[dict]) -> Any:
    """Scatter chart of auctions showing bid count vs winning DFS."""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    if not auctions:
        fig.add_annotation(text="No auctions conducted",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "🏆 Challenger Auctions")

    x_vals = [i for i in range(len(auctions))]
    bid_counts = [a.get("num_bids", 0) for a in auctions]
    dfs_vals = [a.get("winning_dfs", 0.0) for a in auctions]
    unchallenged = [a.get("unchallenged", True) for a in auctions]
    colors_ = [CRUX_VIZ_COLORS["orange"] if not u else CRUX_VIZ_COLORS["red"]
               for u in unchallenged]
    labels = [
        f"Auction {i+1}<br>Bids: {b}<br>Winner DFS: {d:.2f}"
        for i, (b, d) in enumerate(zip(bid_counts, dfs_vals))
    ]

    fig.add_trace(go.Scatter(
        x=x_vals, y=dfs_vals,
        mode="markers",
        marker=dict(
            size=[max(12, b * 6) for b in bid_counts],
            color=colors_,
            line=dict(width=2, color=CRUX_VIZ_COLORS["bg"]),
            symbol=["circle" if not u else "x" for u in unchallenged],
        ),
        text=labels,
        hovertemplate="%{text}<extra></extra>",
        name="Auctions",
    ))

    _dark(fig, "🏆 Challenger Auction Results", height=380)
    fig.update_layout(
        xaxis_title="Auction Index",
        yaxis_title="Winning DFS Score",
        yaxis=dict(range=[0, 1.15]),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Full CRUX Dashboard
# ---------------------------------------------------------------------------

def create_crux_full_dashboard(result: dict) -> dict:
    """
    Create complete set of CRUX dashboard charts from a debate result dict.

    Returns:
        Dict mapping chart name → Plotly Figure.
    """
    rounds = result.get("rounds", [])
    claim_bundles = result.get("claim_bundles", [])
    brp_sessions = result.get("brp_sessions", [])
    auctions = result.get("auctions", [])
    credibility_ledger = result.get("credibility_ledger", [])
    crux_stats = result.get("crux_session_stats", {})
    edr_checkpoints = result.get("edr_checkpoints", [])

    # Try to get the live crux session for the protocol-level flow diagram
    crux_session_obj = result.get("_crux_session_obj")  # Set internally if available

    charts = {}

    charts["posterior_evolution"]  = plot_posterior_evolution(rounds)
    charts["evidence_polarity"]    = plot_evidence_polarity_donut(rounds)
    charts["cb_timeline"]          = plot_claim_bundle_timeline(claim_bundles)
    charts["session_radar"]        = plot_session_stats_radar(crux_stats)
    charts["brp_summary"]          = plot_brp_summary(brp_sessions)
    charts["credibility_snapshot"] = plot_credibility_snapshot(credibility_ledger)
    charts["auction_summary"]      = plot_auction_summary(auctions)
    charts["edr_checkpoints"]      = plot_edr_checkpoints(edr_checkpoints)

    # Try CRUX-native charts if session object is available
    if crux_session_obj:
        charts["crux_debate_flow"] = plot_crux_debate_flow(crux_session_obj)
    else:
        # Fallback synthetic flow from claim bundles list
        charts["crux_debate_flow"] = _synthetic_crux_flow(
            claim_bundles, brp_sessions, auctions
        )

    return charts


def _synthetic_crux_flow(
    claim_bundles: list[dict],
    brp_sessions: list[dict],
    auctions: list[dict],
) -> Any:
    """Build a synthetic CRUX flow chart from plain dicts when no session obj available."""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    if not claim_bundles:
        fig.add_annotation(text="Run a debate to see the CRUX flow",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=18, color=CRUX_VIZ_COLORS["text"]))
        return _dark(fig, "🔄 CRUX Debate Flow", height=500)

    polarity_color_map = {
        "1": CRUX_VIZ_COLORS["green"],
        "supports": CRUX_VIZ_COLORS["green"],
        "SUPPORTS": CRUX_VIZ_COLORS["green"],
        "-1": CRUX_VIZ_COLORS["red"],
        "attacks": CRUX_VIZ_COLORS["red"],
        "ATTACKS": CRUX_VIZ_COLORS["red"],
    }

    # Draw claim bundle nodes in a grid
    nodes_x, nodes_y, node_colors, node_sizes, node_texts, node_hover = [], [], [], [], [], []

    for i, cb in enumerate(claim_bundles):
        x = i % 8
        y = -(i // 8)
        polarity_raw = str(cb.get("polarity", "0"))
        color = polarity_color_map.get(polarity_raw, CRUX_VIZ_COLORS["amber"])
        posterior = cb.get("posterior", 0.5)
        agent = cb.get("issuer_agent", "")
        claim = cb.get("claim_text", "")[:40]

        nodes_x.append(x)
        nodes_y.append(y)
        node_colors.append(color)
        node_sizes.append(18 + posterior * 20)
        node_texts.append(f"P={posterior:.2f}")
        node_hover.append(
            f"<b>{claim}...</b><br>Agent: {agent}<br>Posterior: {posterior:.3f}"
        )

    fig.add_trace(go.Scatter(
        x=nodes_x, y=nodes_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol="circle",
            line=dict(width=2, color=CRUX_VIZ_COLORS["bg"]),
        ),
        text=node_texts,
        textposition="top center",
        textfont=dict(size=9, color=CRUX_VIZ_COLORS["text"]),
        name="Claim Bundles",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_hover,
    ))

    # Add BRP merge nodes
    if brp_sessions:
        for j, brp in enumerate(brp_sessions):
            res = brp.get("resolution", {})
            post = res.get("reconciled_posterior", 0.5)
            fig.add_trace(go.Scatter(
                x=[len(claim_bundles) // 2 + j],
                y=[1.5],
                mode="markers+text",
                marker=dict(
                    size=30,
                    color=CRUX_VIZ_COLORS["purple"],
                    symbol="star",
                    line=dict(width=2, color=CRUX_VIZ_COLORS["bg"]),
                ),
                text=[f"BRP-{j+1}<br>P={post:.2f}"],
                textposition="top center",
                textfont=dict(size=9, color=CRUX_VIZ_COLORS["purple"]),
                name="BRP Merge",
                hovertemplate=f"BRP Merge — Posterior: {post:.3f}<extra></extra>",
            ))

    _dark(fig, "🔄 CRUX Debate Flow (Synthetic)", height=500)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig
