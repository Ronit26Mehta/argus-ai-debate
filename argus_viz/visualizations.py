"""
Interactive Plotly Visualizations for Argus-Viz.

All chart functions return plotly.graph_objects.Figure objects
rendered by Streamlit via st.plotly_chart().

Charts:
    - Posterior evolution (line)
    - Evidence waterfall (bar)
    - CDAG network graph (scatter + edges)
    - Specialist radar (polar)
    - Confidence histogram
    - Debate timeline (Gantt-style)
    - Evidence polarity donut
    - Round heatmap
    - Debate flow DAG (full lifecycle graph)
    - Dashboard composer
"""

from __future__ import annotations

import math
from typing import Any, Optional

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0e1117",
    "paper": "#1a1f2e",
    "grid": "#2a3040",
    "text": "#e0e0e0",
    "accent_cyan": "#00d4ff",
    "accent_magenta": "#ff00d4",
    "accent_amber": "#ffbf00",
    "support_green": "#00ff88",
    "attack_red": "#ff4466",
    "rebuttal_orange": "#ff8800",
    "neutral": "#888888",
    "verdict_supported": "#00ff88",
    "verdict_rejected": "#ff4466",
    "verdict_undecided": "#ffbf00",
    "proposition": "#00d4ff",
    "bayesian_update": "#b388ff",
}

DARK_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["paper"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)


def _apply_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent dark styling to a figure."""
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, font=dict(size=18, color=COLORS["accent_cyan"])),
    )
    return fig


# ============================================================================
# 1. Posterior Evolution
# ============================================================================

def plot_posterior_evolution(rounds_data: list[dict]) -> go.Figure:
    """Animated line chart of posterior probability across debate rounds."""
    rounds = [0] + [r["round"] for r in rounds_data]
    posteriors = [rounds_data[0].get("posterior_before", 0.5)] + [
        r["posterior_after"] for r in rounds_data
    ]

    fig = go.Figure()

    # Confidence band
    upper = [min(1.0, p + 0.1) for p in posteriors]
    lower = [max(0.0, p - 0.1) for p in posteriors]

    fig.add_trace(go.Scatter(
        x=rounds + rounds[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.08)",
        line=dict(width=0),
        name="Confidence Band",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main posterior line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=posteriors,
        mode="lines+markers",
        name="Posterior",
        line=dict(color=COLORS["accent_cyan"], width=3, shape="spline"),
        marker=dict(size=10, color=COLORS["accent_cyan"],
                    line=dict(width=2, color=COLORS["bg"])),
        hovertemplate="Round %{x}<br>Posterior: %{y:.3f}<extra></extra>",
    ))

    # Decision threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text="Neutral (0.5)")

    # Support / Reject zones
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(0,255,136,0.05)",
                  line_width=0, annotation_text="Supported Zone")
    fig.add_hrect(y0=0.0, y1=0.3, fillcolor="rgba(255,68,102,0.05)",
                  line_width=0, annotation_text="Rejected Zone")

    _apply_dark_layout(fig, "📈 Posterior Probability Evolution")
    fig.update_yaxes(range=[0, 1], title="Posterior Probability")
    fig.update_xaxes(title="Debate Round", dtick=1)

    return fig


# ============================================================================
# 2. Evidence Waterfall
# ============================================================================

def plot_evidence_waterfall(rounds_data: list[dict]) -> go.Figure:
    """Waterfall chart showing cumulative evidence impact per round."""
    labels = []
    values = []
    colors = []

    for r in rounds_data:
        rnd = r["round"]
        support = r.get("support_count", 0)
        attack = r.get("attack_count", 0)

        if support > 0:
            labels.append(f"R{rnd} Support (+{support})")
            values.append(support)
            colors.append(COLORS["support_green"])
        if attack > 0:
            labels.append(f"R{rnd} Attack (-{attack})")
            values.append(-attack)
            colors.append(COLORS["attack_red"])

    if not labels:
        labels, values, colors = ["No Data"], [0], [COLORS["neutral"]]

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        connector=dict(line=dict(color=COLORS["grid"])),
        increasing=dict(marker=dict(color=COLORS["support_green"])),
        decreasing=dict(marker=dict(color=COLORS["attack_red"])),
        totals=dict(marker=dict(color=COLORS["accent_cyan"])),
        textposition="outside",
        text=[f"{v:+d}" for v in values],
        hovertemplate="%{x}<br>Impact: %{y:+d}<extra></extra>",
    ))

    _apply_dark_layout(fig, "📊 Evidence Impact Waterfall")
    fig.update_yaxes(title="Evidence Count")
    fig.update_xaxes(title="")

    return fig


# ============================================================================
# 3. CDAG Network Graph
# ============================================================================

def plot_cdag_network(graph_data: dict) -> go.Figure:
    """Interactive network graph of the CDAG."""
    import networkx as nx

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.add_annotation(text="No graph data available", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _apply_dark_layout(fig, "🕸️ C-DAG Network")
        return fig

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        if e["source"] in G.nodes and e["target"] in G.nodes:
            G.add_edge(e["source"], e["target"], **e)

    # Layout
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except Exception:
        pos = nx.shell_layout(G)

    # Edge traces
    edge_traces = []
    for e in edges:
        if e["source"] not in pos or e["target"] not in pos:
            continue
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]

        etype = e.get("edge_type", "supports")
        if "support" in etype.lower():
            color = COLORS["support_green"]
        elif "attack" in etype.lower():
            color = COLORS["attack_red"]
        elif "rebut" in etype.lower():
            color = COLORS["rebuttal_orange"]
        else:
            color = COLORS["neutral"]

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=1.5, color=color),
            hoverinfo="none",
            showlegend=False,
        ))

    # Node traces by type
    type_configs = {
        "Proposition": {"color": COLORS["proposition"], "size": 22, "symbol": "diamond"},
        "Evidence": {"color": COLORS["support_green"], "size": 14, "symbol": "circle"},
        "Rebuttal": {"color": COLORS["rebuttal_orange"], "size": 12, "symbol": "triangle-up"},
    }

    node_traces = []
    for node_type, cfg in type_configs.items():
        typed = [n for n in nodes if n["type"] == node_type and n["id"] in pos]
        if not typed:
            continue

        xs = [pos[n["id"]][0] for n in typed]
        ys = [pos[n["id"]][1] for n in typed]

        colors_list = []
        for n in typed:
            if node_type == "Evidence":
                colors_list.append(
                    COLORS["support_green"] if n.get("polarity", 0) > 0
                    else COLORS["attack_red"]
                )
            else:
                colors_list.append(cfg["color"])

        texts = [n["text"][:60] + "..." if len(n["text"]) > 60 else n["text"]
                 for n in typed]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=node_type,
            marker=dict(
                size=cfg["size"],
                color=colors_list,
                symbol=cfg["symbol"],
                line=dict(width=1.5, color=COLORS["bg"]),
            ),
            text=[n["type"][0] for n in typed],
            textfont=dict(size=8, color="white"),
            hovertext=texts,
            hoverinfo="text",
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    _apply_dark_layout(fig, "🕸️ C-DAG Network Graph")
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig


# ============================================================================
# 4. Specialist Radar
# ============================================================================

def plot_specialist_radar(rounds_data: list[dict]) -> go.Figure:
    """Radar chart comparing specialist contributions."""
    specialist_stats: dict[str, dict] = {}

    for r in rounds_data:
        for e in r.get("evidence", []):
            name = e.get("specialist", "Unknown")
            if name not in specialist_stats:
                specialist_stats[name] = {
                    "total": 0, "support": 0, "attack": 0,
                    "avg_confidence": 0.0, "conf_sum": 0.0,
                }
            specialist_stats[name]["total"] += 1
            specialist_stats[name]["conf_sum"] += e.get("confidence", 0.5)
            if e.get("polarity", 0) > 0:
                specialist_stats[name]["support"] += 1
            elif e.get("polarity", 0) < 0:
                specialist_stats[name]["attack"] += 1

    if not specialist_stats:
        fig = go.Figure()
        fig.add_annotation(text="No specialist data", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _apply_dark_layout(fig, "🎯 Specialist Contributions")
        return fig

    for s in specialist_stats.values():
        s["avg_confidence"] = s["conf_sum"] / max(s["total"], 1)

    categories = ["Total Evidence", "Supporting", "Attacking", "Avg Confidence"]
    fig = go.Figure()

    palette = [COLORS["accent_cyan"], COLORS["accent_magenta"],
               COLORS["accent_amber"], COLORS["support_green"],
               COLORS["rebuttal_orange"], "#7c4dff"]

    def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
        """Convert hex color to rgba() string for Plotly compatibility."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for i, (name, stats) in enumerate(specialist_stats.items()):
        color = palette[i % len(palette)]
        values = [
            stats["total"],
            stats["support"],
            stats["attack"],
            stats["avg_confidence"] * 10,  # Scale confidence for visibility
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.15),
            line=dict(color=color, width=2),
            name=name,
        ))

    _apply_dark_layout(fig, "🎯 Specialist Contributions Radar")
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["paper"],
            angularaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
            radialaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"],
                           visible=True),
        ),
    )

    return fig


# ============================================================================
# 5. Confidence Histogram
# ============================================================================

def plot_confidence_histogram(rounds_data: list[dict]) -> go.Figure:
    """Distribution of evidence confidence scores."""
    confidences: list[float] = []
    polarities: list[str] = []

    for r in rounds_data:
        for e in r.get("evidence", []):
            conf = e.get("confidence", 0.0)
            if conf > 0:
                confidences.append(conf)
                polarities.append(
                    "Supporting" if e.get("polarity", 0) > 0 else "Attacking"
                )

    if not confidences:
        fig = go.Figure()
        fig.add_annotation(text="No confidence data", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _apply_dark_layout(fig, "📊 Evidence Confidence Distribution")
        return fig

    support_conf = [c for c, p in zip(confidences, polarities) if p == "Supporting"]
    attack_conf = [c for c, p in zip(confidences, polarities) if p == "Attacking"]

    fig = go.Figure()

    if support_conf:
        fig.add_trace(go.Histogram(
            x=support_conf,
            name="Supporting",
            marker_color=COLORS["support_green"],
            opacity=0.7,
            nbinsx=15,
        ))

    if attack_conf:
        fig.add_trace(go.Histogram(
            x=attack_conf,
            name="Attacking",
            marker_color=COLORS["attack_red"],
            opacity=0.7,
            nbinsx=15,
        ))

    _apply_dark_layout(fig, "📊 Evidence Confidence Distribution")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Confidence Score", range=[0, 1])
    fig.update_yaxes(title="Count")

    return fig


# ============================================================================
# 6. Debate Timeline
# ============================================================================

def plot_debate_timeline(rounds_data: list[dict]) -> go.Figure:
    """Gantt-style horizontal timeline of agent actions per round."""
    tasks = []
    colors = {
        "Specialist": COLORS["accent_cyan"],
        "Refuter": COLORS["rebuttal_orange"],
        "Bayesian": COLORS["bayesian_update"],
    }

    for r in rounds_data:
        rnd = r["round"]
        base_x = (rnd - 1) * 3

        # Specialist phase
        specialists_active = list(r.get("specialist_breakdown", {}).keys())
        for j, sp_name in enumerate(specialists_active):
            tasks.append({
                "Agent": sp_name,
                "Start": base_x,
                "End": base_x + 1,
                "Round": rnd,
                "Type": "Specialist",
                "Detail": f"{r['specialist_breakdown'][sp_name]['total']} evidence",
            })

        # Refuter phase
        if r.get("total_rebuttals", 0) > 0:
            tasks.append({
                "Agent": "Refuter",
                "Start": base_x + 1,
                "End": base_x + 2,
                "Round": rnd,
                "Type": "Refuter",
                "Detail": f"{r['total_rebuttals']} rebuttals",
            })

        # Bayesian update
        tasks.append({
            "Agent": "Bayesian Update",
            "Start": base_x + 2,
            "End": base_x + 3,
            "Round": rnd,
            "Type": "Bayesian",
            "Detail": f"P: {r['posterior_after']:.3f}",
        })

    if not tasks:
        fig = go.Figure()
        fig.add_annotation(text="No timeline data", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _apply_dark_layout(fig, "⏱️ Debate Timeline")
        return fig

    fig = go.Figure()

    for task in tasks:
        color = colors.get(task["Type"], COLORS["neutral"])
        fig.add_trace(go.Bar(
            x=[task["End"] - task["Start"]],
            y=[task["Agent"]],
            base=[task["Start"]],
            orientation="h",
            marker=dict(color=color, opacity=0.8,
                       line=dict(width=1, color=COLORS["bg"])),
            name=f"R{task['Round']} {task['Agent']}",
            hovertext=f"Round {task['Round']}: {task['Detail']}",
            hoverinfo="text",
            showlegend=False,
        ))

    _apply_dark_layout(fig, "⏱️ Debate Timeline")
    fig.update_xaxes(title="Timeline Phase")
    fig.update_yaxes(title="")
    fig.update_layout(barmode="stack", height=max(300, len(tasks) * 30))

    return fig


# ============================================================================
# 7. Evidence Polarity Donut
# ============================================================================

def plot_evidence_polarity_donut(rounds_data: list[dict]) -> go.Figure:
    """Donut chart of supporting vs attacking evidence."""
    total_support = sum(r.get("support_count", 0) for r in rounds_data)
    total_attack = sum(r.get("attack_count", 0) for r in rounds_data)

    if total_support == 0 and total_attack == 0:
        total_support, total_attack = 1, 1  # Placeholder

    fig = go.Figure(go.Pie(
        labels=["Supporting", "Attacking"],
        values=[total_support, total_attack],
        hole=0.55,
        marker=dict(
            colors=[COLORS["support_green"], COLORS["attack_red"]],
            line=dict(color=COLORS["bg"], width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value} items (%{percent})<extra></extra>",
    ))

    total = total_support + total_attack
    fig.add_annotation(
        text=f"<b>{total}</b><br>Evidence",
        showarrow=False,
        font=dict(size=18, color=COLORS["text"]),
    )

    _apply_dark_layout(fig, "🎯 Evidence Polarity Balance")

    return fig


# ============================================================================
# 8. Round Heatmap
# ============================================================================

def plot_round_heatmap(rounds_data: list[dict]) -> go.Figure:
    """Heatmap of specialist × round evidence density."""
    all_specialists: set[str] = set()
    for r in rounds_data:
        for e in r.get("evidence", []):
            all_specialists.add(e.get("specialist", "Unknown"))

    if not all_specialists:
        fig = go.Figure()
        fig.add_annotation(text="No heatmap data", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _apply_dark_layout(fig, "🗺️ Evidence Heatmap")
        return fig

    spec_list = sorted(all_specialists)
    z_matrix: list[list[int]] = []

    for spec in spec_list:
        row = []
        for r in rounds_data:
            count = sum(
                1 for e in r.get("evidence", [])
                if e.get("specialist", "") == spec
            )
            row.append(count)
        z_matrix.append(row)

    round_labels = [f"Round {r['round']}" for r in rounds_data]

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=round_labels,
        y=spec_list,
        colorscale=[
            [0, COLORS["paper"]],
            [0.5, COLORS["accent_cyan"]],
            [1, COLORS["accent_magenta"]],
        ],
        hoverongaps=False,
        hovertemplate="Specialist: %{y}<br>%{x}<br>Evidence: %{z}<extra></extra>",
    ))

    _apply_dark_layout(fig, "🗺️ Specialist × Round Heatmap")

    return fig


# ============================================================================
# 9. Debate Flow Graph (Full Lifecycle DAG)
# ============================================================================

def plot_debate_flow_graph(result: dict) -> go.Figure:
    """
    Full debate lifecycle directed acyclic graph.

    Trace the entire debate journey from start to end:
    Proposition → Specialists → Evidence → Rebuttals → Bayesian Updates → Verdict

    Uses networkx for hierarchical layout and Plotly for rendering.
    """
    import networkx as nx

    G = nx.DiGraph()

    rounds_data = result.get("rounds", [])
    verdict = result.get("verdict", {})
    proposition = result.get("proposition", "Proposition")

    if not rounds_data:
        fig = go.Figure()
        fig.add_annotation(text="No debate data to visualize", showarrow=False,
                           font=dict(size=16, color=COLORS["text"]),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _apply_dark_layout(fig, "🔀 Debate Flow Graph")
        return fig

    # --- Build the DAG ---
    # Layer 0: Proposition
    prop_id = "PROP"
    G.add_node(prop_id, label=proposition[:50], layer=0,
               node_type="Proposition", size=28,
               color=COLORS["proposition"])

    node_meta: dict[str, dict] = {
        prop_id: {"type": "Proposition", "text": proposition,
                  "color": COLORS["proposition"], "size": 28, "symbol": "diamond"},
    }

    node_counter = 0

    for r in rounds_data:
        rnd = r["round"]

        # Layer 1: Specialist agent nodes (one per specialist per round)
        specialist_names_in_round: set[str] = set()
        for ev in r.get("evidence", []):
            specialist_names_in_round.add(ev.get("specialist", "Unknown"))

        specialist_node_map: dict[str, str] = {}
        for sp_name in specialist_names_in_round:
            node_counter += 1
            sp_node_id = f"SP_{rnd}_{sp_name.replace(' ','_')}"
            G.add_node(sp_node_id, label=f"{sp_name}\n(R{rnd})",
                       layer=1 + (rnd - 1) * 4,
                       node_type="Specialist", size=18)
            node_meta[sp_node_id] = {
                "type": "Specialist", "text": f"{sp_name} — Round {rnd}",
                "color": COLORS["accent_cyan"], "size": 18, "symbol": "square",
            }
            # Edge: Proposition → Specialist
            G.add_edge(prop_id, sp_node_id, etype="activates")
            specialist_node_map[sp_name] = sp_node_id

        # Layer 2: Evidence nodes
        for ev in r.get("evidence", []):
            node_counter += 1
            ev_id = ev.get("id", f"EV_{node_counter}")
            polarity = ev.get("polarity", 0)
            confidence = ev.get("confidence", 0.5)
            sp_name = ev.get("specialist", "Unknown")

            ev_color = COLORS["support_green"] if polarity > 0 else COLORS["attack_red"]
            ev_label = ev.get("text", "")[:40]

            G.add_node(ev_id, label=ev_label,
                       layer=2 + (rnd - 1) * 4,
                       node_type="Evidence", size=10 + confidence * 10)
            node_meta[ev_id] = {
                "type": "Evidence", "text": ev.get("text", ""),
                "color": ev_color,
                "size": 10 + confidence * 10,
                "symbol": "circle",
                "polarity": polarity,
                "confidence": confidence,
            }

            # Edge: Specialist → Evidence
            sp_node = specialist_node_map.get(sp_name)
            if sp_node:
                etype = "SUPPORTS" if polarity > 0 else "ATTACKS"
                G.add_edge(sp_node, ev_id, etype=etype, weight=confidence)

            # Edge: Evidence → Proposition (conceptual)
            G.add_edge(ev_id, prop_id, etype="SUPPORTS" if polarity > 0 else "ATTACKS",
                       weight=confidence)

        # Layer 3: Rebuttal nodes
        for rb in r.get("rebuttals", []):
            node_counter += 1
            rb_id = rb.get("id", f"RB_{node_counter}")
            rb_label = rb.get("text", "")[:40]
            strength = rb.get("strength", 0.5)

            G.add_node(rb_id, label=rb_label,
                       layer=3 + (rnd - 1) * 4,
                       node_type="Rebuttal", size=10 + strength * 8)
            node_meta[rb_id] = {
                "type": "Rebuttal", "text": rb.get("text", ""),
                "color": COLORS["rebuttal_orange"],
                "size": 10 + strength * 8,
                "symbol": "triangle-up",
                "strength": strength,
            }

            # Edge: Rebuttal → targeted evidence (if known)
            target = rb.get("target_id", "")
            if target and target in G.nodes:
                G.add_edge(rb_id, target, etype="REBUTS", weight=strength)

        # Layer 4: Bayesian update checkpoint
        bu_id = f"BU_{rnd}"
        posterior = r.get("posterior_after", 0.5)
        G.add_node(bu_id, label=f"P={posterior:.3f}\n(R{rnd})",
                   layer=4 + (rnd - 1) * 4,
                   node_type="BayesianUpdate", size=16)
        node_meta[bu_id] = {
            "type": "BayesianUpdate",
            "text": f"Bayesian Update — Round {rnd}\nPosterior: {posterior:.3f}",
            "color": COLORS["bayesian_update"],
            "size": 16, "symbol": "hexagon",
            "posterior": posterior,
        }

        # Edges from evidence to Bayesian update
        for ev in r.get("evidence", []):
            ev_id = ev.get("id", "")
            if ev_id in G.nodes:
                G.add_edge(ev_id, bu_id, etype="updates")

    # Final layer: Verdict
    v_label = verdict.get("label", "undecided")
    v_posterior = verdict.get("posterior", 0.5)
    v_id = "VERDICT"
    v_color = (COLORS["verdict_supported"] if v_label == "supported"
               else COLORS["verdict_rejected"] if v_label == "rejected"
               else COLORS["verdict_undecided"])

    G.add_node(v_id, label=f"VERDICT\n{v_label.upper()}\nP={v_posterior:.3f}",
               layer=max(nx.get_node_attributes(G, "layer").values()) + 1 if G.nodes else 1,
               node_type="Verdict", size=30)
    node_meta[v_id] = {
        "type": "Verdict",
        "text": f"Verdict: {v_label} (P={v_posterior:.3f})\n{verdict.get('reasoning', '')[:200]}",
        "color": v_color, "size": 30, "symbol": "star",
    }

    # Edge from last Bayesian update to verdict
    last_bu = f"BU_{len(rounds_data)}"
    if last_bu in G.nodes:
        G.add_edge(last_bu, v_id, etype="renders")

    # --- Compute layout ---
    try:
        pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
        # Rotate: swap x/y so flow goes top-to-bottom
        pos = {n: (p[0], -p[1]) for n, p in pos.items()}
    except Exception:
        pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # --- Render edges ---
    edge_traces = []
    edge_type_colors = {
        "SUPPORTS": COLORS["support_green"],
        "ATTACKS": COLORS["attack_red"],
        "REBUTS": COLORS["rebuttal_orange"],
        "activates": COLORS["accent_cyan"],
        "updates": COLORS["bayesian_update"],
        "renders": COLORS["accent_amber"],
    }

    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        etype = data.get("etype", "")
        ecolor = edge_type_colors.get(etype, COLORS["neutral"])
        weight = data.get("weight", 1.0)

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(0.8, weight * 2.5), color=ecolor),
            hoverinfo="none",
            showlegend=False,
            opacity=0.6,
        ))

    # --- Render nodes by type ---
    type_order = ["Proposition", "Specialist", "Evidence",
                  "Rebuttal", "BayesianUpdate", "Verdict"]
    node_traces = []

    for ntype in type_order:
        typed_nodes = [nid for nid in G.nodes if node_meta.get(nid, {}).get("type") == ntype]
        if not typed_nodes:
            continue

        xs = [pos[n][0] for n in typed_nodes if n in pos]
        ys = [pos[n][1] for n in typed_nodes if n in pos]
        valid_nodes = [n for n in typed_nodes if n in pos]

        colors_list = [node_meta[n]["color"] for n in valid_nodes]
        sizes = [node_meta[n]["size"] for n in valid_nodes]
        symbols = [node_meta[n].get("symbol", "circle") for n in valid_nodes]
        hover_texts = [node_meta[n]["text"][:120] for n in valid_nodes]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=ntype,
            marker=dict(
                size=sizes,
                color=colors_list,
                symbol=symbols[0] if len(set(symbols)) == 1 else symbols,
                line=dict(width=1.5, color=COLORS["bg"]),
            ),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    _apply_dark_layout(fig, "🔀 Debate Flow Graph — Full Lifecycle DAG")
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    # Add legend annotations
    legend_items = [
        ("◆ Proposition", COLORS["proposition"]),
        ("■ Specialist", COLORS["accent_cyan"]),
        ("● Support Evidence", COLORS["support_green"]),
        ("● Attack Evidence", COLORS["attack_red"]),
        ("▲ Rebuttal", COLORS["rebuttal_orange"]),
        ("⬡ Bayesian Update", COLORS["bayesian_update"]),
        ("★ Verdict", COLORS["accent_amber"]),
    ]

    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=1.02, y=1.0 - i * 0.07,
            xref="paper", yref="paper",
            text=f'<span style="color:{color}">{label}</span>',
            showarrow=False,
            font=dict(size=11),
            align="left",
        )

    fig.update_layout(
        height=max(600, len(G.nodes) * 15),
        showlegend=False,
    )

    return fig


# ============================================================================
# 10. Dashboard Composer
# ============================================================================

def create_dashboard(result: dict) -> dict[str, go.Figure]:
    """
    Create all charts from a debate result.

    Returns:
        Dict mapping chart name to Figure.
    """
    rounds_data = result.get("rounds", [])
    graph_data = result.get("graph_data", {"nodes": [], "edges": []})

    return {
        "posterior_evolution": plot_posterior_evolution(rounds_data),
        "evidence_waterfall": plot_evidence_waterfall(rounds_data),
        "cdag_network": plot_cdag_network(graph_data),
        "specialist_radar": plot_specialist_radar(rounds_data),
        "confidence_histogram": plot_confidence_histogram(rounds_data),
        "debate_timeline": plot_debate_timeline(rounds_data),
        "evidence_polarity": plot_evidence_polarity_donut(rounds_data),
        "round_heatmap": plot_round_heatmap(rounds_data),
        "debate_flow_graph": plot_debate_flow_graph(result),
    }
