"""
Visualisation sub-panels for the right pane.

Zone B (bottom row):
    - Belief Trajectory Line Chart
    - Evidence Quality Heatmap

Zone C (expandable drawer):
    - Argument Flow Sankey Diagram
    - Agent Influence Radar Chart
    - Calibration Reliability Diagram
    - Debate Timeline Event Plot

All panels share the dark theme (#1A1A2E background) and are built as
Plotly ``go.Figure`` objects rendered via ``st.plotly_chart``.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.express as px

from argus.aristotle.models import MonitorEvent, MonitorEventType

BG_COLOR = "#1A1A2E"
GRID_COLOR = "#333355"
TEXT_COLOR = "#CCCCCC"


# ── 8.1 Belief Trajectory Line Chart ──────────────────────────────────

def build_belief_trajectory(
    posteriors: list[float],
    round_labels: list[str] | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """
    Belief Trajectory: posterior probability on Y-axis, round on X-axis,
    with a shaded confidence band.

    Args:
        posteriors: Posterior values per round (index 0 = prior).
        round_labels: Optional x-axis labels.
        annotations: Optional annotation dicts for inflection points.
    """
    n = len(posteriors)
    x = list(range(n))
    labels = round_labels or [f"R{i}" if i > 0 else "Prior" for i in x]

    # Confidence band (±0.05 heuristic around posterior)
    upper = [min(p + 0.05, 1.0) for p in posteriors]
    lower = [max(p - 0.05, 0.0) for p in posteriors]

    fig = go.Figure()

    # band
    fig.add_trace(go.Scatter(
        x=x, y=upper,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lower,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,191,255,0.15)",
        showlegend=False, hoverinfo="skip",
    ))

    # main line
    fig.add_trace(go.Scatter(
        x=x, y=posteriors,
        mode="lines+markers",
        line=dict(color="#00BFFF", width=2.5),
        marker=dict(size=7, color="#00BFFF"),
        text=[f"{p:.1%}" for p in posteriors],
        hoverinfo="text",
        name="Posterior",
    ))

    # annotations
    if annotations:
        for ann in annotations:
            fig.add_annotation(
                x=ann.get("round", 0),
                y=ann.get("posterior", 0.5),
                text=ann.get("text", ""),
                showarrow=True,
                arrowhead=2,
                font=dict(color=TEXT_COLOR, size=9),
                arrowcolor=TEXT_COLOR,
            )

    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(
            tickvals=x, ticktext=labels,
            showgrid=True, gridcolor=GRID_COLOR,
            color=TEXT_COLOR,
        ),
        yaxis=dict(
            range=[0, 1], dtick=0.1,
            showgrid=True, gridcolor=GRID_COLOR,
            color=TEXT_COLOR,
            title="Posterior Probability",
        ),
        margin=dict(l=40, r=20, t=30, b=30),
        height=250,
        title=dict(text="Belief Trajectory", font=dict(color=TEXT_COLOR, size=12), x=0.5),
        legend=dict(font=dict(color=TEXT_COLOR)),
    )
    return fig


# ── 8.2 Evidence Quality Heatmap ──────────────────────────────────────

def build_evidence_heatmap(
    evidence_log: list[dict[str, Any]],
) -> go.Figure:
    """
    Heatmap: rows = agents, columns = evidence items, cell colour = EVID-Q.

    Args:
        evidence_log: List of dicts with 'agent_name' and 'evid_q' keys.
    """
    # Organise by agent
    agents: dict[str, list[float]] = {}
    for entry in evidence_log:
        name = entry.get("agent_name", "Unknown")
        agents.setdefault(name, []).append(entry.get("evid_q", 0.0))

    if not agents:
        return _empty_figure("Evidence Quality Heatmap", "No evidence yet")

    # Pad to rectangular
    max_len = max(len(v) for v in agents.values())
    agent_names = list(agents.keys())
    z = []
    for name in agent_names:
        row = agents[name] + [None] * (max_len - len(agents[name]))
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        y=agent_names,
        x=[f"E{i+1}" for i in range(max_len)],
        colorscale=[
            [0.0, "#FFEB3B"],   # low quality — yellow
            [0.5, "#FF9800"],   # medium — orange
            [1.0, "#B71C1C"],   # high quality — dark red
        ],
        zmin=0, zmax=1,
        colorbar=dict(
            title=dict(text="EVID-Q", font=dict(color=TEXT_COLOR)),
            tickfont=dict(color=TEXT_COLOR),
        ),
        hovertemplate="Agent: %{y}<br>Evidence: %{x}<br>EVID-Q: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(color=TEXT_COLOR, showgrid=False),
        yaxis=dict(color=TEXT_COLOR, showgrid=False, autorange="reversed"),
        margin=dict(l=100, r=20, t=30, b=30),
        height=250,
        title=dict(text="Evidence Quality Heatmap", font=dict(color=TEXT_COLOR, size=12), x=0.5),
    )
    return fig


# ── 8.3 Argument Flow Sankey ──────────────────────────────────────────

def build_sankey(
    events: list[MonitorEvent],
    evidence_log: list[dict[str, Any]],
) -> go.Figure:
    """
    Sankey diagram: Proposition → Evidence → Rebuttal → Verdict.
    Flow width represents cumulative weight.
    """
    labels = ["Proposition"]
    source: list[int] = []
    target: list[int] = []
    values: list[float] = []
    colors: list[str] = []

    agent_idx: dict[str, int] = {}
    for entry in evidence_log:
        name = entry.get("agent_name", "Unknown")
        if name not in agent_idx:
            agent_idx[name] = len(labels)
            labels.append(name)
        idx = agent_idx[name]
        source.append(0)
        target.append(idx)
        values.append(max(entry.get("evid_q", 0.3), 0.1))
        colors.append(
            "rgba(46,204,113,0.5)" if entry.get("polarity", 1) >= 0
            else "rgba(231,76,60,0.5)"
        )

    # Verdict node
    verdict_idx = len(labels)
    labels.append("Verdict")
    for idx in agent_idx.values():
        source.append(idx)
        target.append(verdict_idx)
        values.append(0.5)
        colors.append("rgba(255,215,0,0.5)")

    if not source:
        return _empty_figure("Argument Flow", "No data yet")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=labels,
            color=[BG_COLOR] * len(labels),
            line=dict(color=TEXT_COLOR, width=0.5),
        ),
        link=dict(source=source, target=target, value=values, color=colors),
    ))

    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        title=dict(text="Argument Flow", font=dict(color=TEXT_COLOR, size=12), x=0.5),
        font=dict(color=TEXT_COLOR),
    )
    return fig


# ── 8.4 Agent Influence Radar Chart ───────────────────────────────────

def build_agent_radar(
    evidence_log: list[dict[str, Any]],
    events: list[MonitorEvent],
) -> go.Figure:
    """
    Radar chart: 5 axes — Evidence Submitted, Avg EVID-Q, Rebuttal count,
    Rounds Active, Influence Score.
    """
    agents: dict[str, dict[str, float]] = {}
    for entry in evidence_log:
        name = entry.get("agent_name", "Unknown")
        if name not in agents:
            agents[name] = {
                "submitted": 0, "avg_evid_q": 0.0, "total_evid_q": 0.0,
                "rounds": set(), "rebuttals": 0,
            }
        agents[name]["submitted"] += 1
        agents[name]["total_evid_q"] += entry.get("evid_q", 0)
        agents[name]["rounds"].add(entry.get("round", 0))

    # Count rebuttals per agent
    for ev in events:
        if ev.event_type == MonitorEventType.REBUTTAL_FILED:
            name = ev.agent_name
            if name in agents:
                agents[name]["rebuttals"] += 1

    if not agents:
        return _empty_figure("Agent Influence", "No data yet")

    categories = [
        "Evidence Submitted", "Avg EVID-Q", "Rebuttal Count",
        "Rounds Active", "Influence Score",
    ]

    fig = go.Figure()
    colours = px.colors.qualitative.Set2
    for i, (name, data) in enumerate(agents.items()):
        avg_eq = data["total_evid_q"] / max(data["submitted"], 1)
        influence = avg_eq * data["submitted"] * 0.2
        vals = [
            data["submitted"],
            avg_eq,
            data["rebuttals"],
            len(data["rounds"]),
            influence,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories,
            fill="toself",
            name=name,
            line_color=colours[i % len(colours)],
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=BG_COLOR,
            radialaxis=dict(visible=True, color=TEXT_COLOR, gridcolor=GRID_COLOR),
            angularaxis=dict(color=TEXT_COLOR, gridcolor=GRID_COLOR),
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        title=dict(text="Agent Influence", font=dict(color=TEXT_COLOR, size=12), x=0.5),
        legend=dict(font=dict(color=TEXT_COLOR)),
    )
    return fig


# ── 8.5 Calibration Reliability Diagram ──────────────────────────────

def build_calibration_diagram(
    posteriors: list[float],
) -> go.Figure:
    """
    Calibration diagram: confidence bins vs observed accuracy.
    Shows a perfect-calibration diagonal and the actual calibration bars.
    """
    n_bins = 10
    bins = [i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    totals = [0] * n_bins

    for p in posteriors:
        idx = min(int(p * n_bins), n_bins - 1)
        totals[idx] += 1
        # For calibration: treat posterior as "predicted probability"
        # In a real system this would compare to ground truth
        counts[idx] += 1

    accuracy = [
        (counts[i] / totals[i]) if totals[i] > 0 else 0
        for i in range(n_bins)
    ]
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]

    fig = go.Figure()

    # perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#555555", dash="dash"),
        name="Perfect Calibration",
    ))

    # actual bars
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=[t for t in totals],
        width=0.08,
        marker_color="#9B89C4",
        name="Confidence Distribution",
        opacity=0.7,
    ))

    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(
            title="Confidence", range=[0, 1],
            color=TEXT_COLOR, gridcolor=GRID_COLOR,
        ),
        yaxis=dict(
            title="Count", color=TEXT_COLOR, gridcolor=GRID_COLOR,
        ),
        margin=dict(l=40, r=20, t=30, b=30),
        height=300,
        title=dict(
            text="Calibration Reliability",
            font=dict(color=TEXT_COLOR, size=12), x=0.5,
        ),
        legend=dict(font=dict(color=TEXT_COLOR)),
    )
    return fig


# ── 8.6 Debate Timeline Event Plot ───────────────────────────────────

def build_timeline(events: list[MonitorEvent]) -> go.Figure:
    """
    Horizontal Gantt-style timeline of debate events.
    ARISTOTLE interventions appear as white markers.
    """
    if not events:
        return _empty_figure("Debate Timeline", "No events yet")

    x: list[int] = []
    y: list[str] = []
    colors: list[str] = []
    hover_texts: list[str] = []

    _COLOR_MAP = {
        MonitorEventType.SESSION_START: "#00BFFF",
        MonitorEventType.SPECIALIST_INSTANTIATED: "#4682B4",
        MonitorEventType.SUPPORT_EVIDENCE: "#2ECC71",
        MonitorEventType.ATTACK_EVIDENCE: "#E74C3C",
        MonitorEventType.REBUTTAL_FILED: "#FFA500",
        MonitorEventType.BAYESIAN_UPDATE: "#9B89C4",
        MonitorEventType.ROUND_COMPLETE: "#888888",
        MonitorEventType.VERDICT_RENDERED: "#FFD700",
        MonitorEventType.DEBATE_COMPLETE: "#FFD700",
    }

    for i, ev in enumerate(events):
        x.append(i)
        y.append(ev.event_type.value.replace("_", " ").title())
        color = _COLOR_MAP.get(ev.event_type, "#FFFFFF")
        if "intervention" in ev.event_type.value:
            color = "#FFFFFF"
        colors.append(color)
        hover_texts.append(
            f"Event: {ev.event_type.value}\n"
            f"Round: {ev.round_num}\n"
            f"Agent: {ev.agent_name}\n"
            f"{ev.chat_message[:100]}"
        )

    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=10, color=colors, line=dict(width=1, color="#444")),
        hoverinfo="text",
        hovertext=hover_texts,
    ))

    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(title="Event Sequence", color=TEXT_COLOR, gridcolor=GRID_COLOR),
        yaxis=dict(color=TEXT_COLOR, gridcolor=GRID_COLOR),
        margin=dict(l=140, r=20, t=30, b=30),
        height=350,
        title=dict(text="Debate Timeline", font=dict(color=TEXT_COLOR, size=12), x=0.5),
    )
    return fig


# ── helper ────────────────────────────────────────────────────────────

def _empty_figure(title: str, message: str) -> go.Figure:
    """Return an empty dark-themed placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, text=message,
        showarrow=False,
        font=dict(color="#777777", size=14),
        xref="paper", yref="paper",
    )
    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=20),
        height=250,
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=12), x=0.5),
    )
    return fig
