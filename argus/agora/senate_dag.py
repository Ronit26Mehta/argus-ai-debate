"""
AGORA Senate Lifecycle DAG — Plotly-based deliberation visualisation.

Renders the complete epistemic structure of an AGORA session as a
dark-themed, layered directed acyclic graph:

    Tier 1: Proposition       (cyan diamond)
    Tier 2: Senators          (colour-coded squares by category)
    Tier 3: Evidence          (green/red/orange circles by polarity)
    Tier 4: Challenges & EA   (amber triangles + white rulings)
    Tier 5: Coalitions + Verdict (gold star, purple clusters)
    Tier 6: Minority Report   (crimson hexagon)

Constructed from an AgoraResult after the session completes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import plotly.graph_objects as go

from argus.agora.models import (
    AgoraResult, SenatorCategory, EvidencePolarity,
    ChallengeOutcome, RecordEntryType,
)

# ── colour palette ─────────────────────────────────────────────────────

BG_COLOR        = "#1A1A2E"
PROPOSITION_CLR = "#00BFFF"
VERDICT_CLR     = "#FFD700"
MINORITY_CLR    = "#FF4444"
EDGE_CLR        = "#555577"
GRID_CLR        = "#333355"

# Category → colour
CATEGORY_COLORS: dict[str, str] = {
    "domain_expert":           "#4682B4",
    "adversarial_challenger":  "#E74C3C",
    "synthesis_agent":         "#9B89C4",
    "normative_analyst":       "#2ECC71",
    "historical_contextualist":"#FFA500",
    "devils_advocate":         "#FF6B6B",
    "epistemic_auditor":       "#FFFFFF",
    "cross_domain_integrator": "#00BCD4",
}

POLARITY_COLORS = {
    "supports":  "#2ECC71",
    "attacks":   "#E74C3C",
    "qualifies": "#FFA500",
}

CHALLENGE_CLR    = "#FFA500"
EA_RULING_CLR    = "#FFFFFF"
COALITION_CLR    = "#9B89C4"

# ── tier Y-positions ──────────────────────────────────────────────────

TIER_Y = {
    1: 1.00,   # proposition
    2: 0.82,   # senators
    3: 0.58,   # evidence
    4: 0.38,   # challenges + EA rulings
    5: 0.15,   # coalitions + verdict
    6: 0.00,   # minority report
}

# ── symbols ───────────────────────────────────────────────────────────

SYM_PROPOSITION = "diamond"
SYM_SENATOR     = "square"
SYM_EVIDENCE    = "circle"
SYM_CHALLENGE   = "triangle-up"
SYM_EA_RULING   = "hexagon"
SYM_COALITION   = "pentagon"
SYM_VERDICT     = "star"
SYM_MINORITY    = "hexagram"

# ── internal structures ──────────────────────────────────────────────

@dataclass
class _Node:
    id: str
    label: str
    tier: int
    x: float = 0.0
    y: float = 0.0
    color: str = "#FFFFFF"
    symbol: str = "circle"
    size: int = 12
    hover: str = ""

@dataclass
class _Edge:
    source_id: str
    target_id: str
    color: str = EDGE_CLR
    dash: str = "solid"
    width: float = 1.5
    label: str = ""


def build_senate_dag(result: AgoraResult) -> go.Figure:
    """Build the complete Senate Lifecycle DAG from an AgoraResult.

    Returns a fully-configured Plotly ``go.Figure`` ready for
    ``st.plotly_chart()``.
    """
    nodes: dict[str, _Node] = {}
    edges: list[_Edge] = []

    # ── Tier 1: Proposition ───────────────────────────────────────────
    prop_id = "proposition"
    posterior = result.majority_opinion.posterior_probability
    nodes[prop_id] = _Node(
        id=prop_id,
        label=f"{posterior:.0%}",
        tier=1, x=0.5, y=TIER_Y[1],
        color=PROPOSITION_CLR,
        symbol=SYM_PROPOSITION,
        size=24,
        hover=f"Proposition: {result.proposition[:120]}\nPosterior: {posterior:.2%}",
    )

    # ── Tier 2: Senators ──────────────────────────────────────────────
    senator_x: dict[str, float] = {}
    for idx, sc in enumerate(result.scorecards):
        x = (idx + 1) / (len(result.scorecards) + 1)
        senator_x[sc.senator_id] = x
        cat_color = CATEGORY_COLORS.get(sc.category.value, "#4682B4")
        traj_str = ""
        if sc.position_trajectory:
            start = sc.position_trajectory[0]
            end = sc.position_trajectory[-1]
            traj_str = f"\nPosition: {start:.2f} → {end:.2f}"

        nodes[sc.senator_id] = _Node(
            id=sc.senator_id,
            label=sc.senator_name[:15],
            tier=2, x=x, y=TIER_Y[2],
            color=cat_color,
            symbol=SYM_SENATOR,
            size=14,
            hover=(
                f"Senator: {sc.senator_name}\n"
                f"Category: {sc.category.display_name}\n"
                f"Evidence: {sc.evidence_submitted} | "
                f"Challenges: {sc.challenges_issued}↗ {sc.challenges_received}↙\n"
                f"ECS: {sc.epistemic_contribution_score:.3f}"
                f"{traj_str}"
            ),
        )
        edges.append(_Edge(
            source_id=prop_id, target_id=sc.senator_id,
            color=EDGE_CLR, dash="dash", label="ASSIGNED",
        ))

    # ── Tier 3: Evidence ──────────────────────────────────────────────
    evidence_counters: dict[str, int] = {}
    for ev in result.docket_items:
        count = evidence_counters.get(ev.senator_id, 0)
        evidence_counters[ev.senator_id] = count + 1

        base_x = senator_x.get(ev.senator_id, 0.5)
        offset = (count % 5 - 2) * 0.03
        x = max(0.02, min(base_x + offset, 0.98))
        y = TIER_Y[3] - count * 0.015

        pol_color = POLARITY_COLORS.get(ev.polarity.value, "#FFA500")

        nodes[ev.id] = _Node(
            id=ev.id,
            label="",
            tier=3, x=x, y=y,
            color=pol_color,
            symbol=SYM_EVIDENCE,
            size=8,
            hover=(
                f"{ev.polarity.value.title()} Evidence [{ev.evidence_id_display}]\n"
                f"Senator: {ev.senator_name}\n"
                f"DEW: {ev.dew_score:.2f} | Confidence: {ev.confidence_score:.2f}\n"
                f"{ev.claim_text[:120]}"
            ),
        )
        if ev.senator_id in nodes:
            edges.append(_Edge(
                source_id=ev.senator_id, target_id=ev.id,
                color=pol_color, width=1.0,
            ))

    # ── Tier 4: Challenges + EA Rulings ───────────────────────────────
    challenge_entries = [
        e for e in result.senate_record_entries
        if e.entry_type in (
            RecordEntryType.CHALLENGE_ISSUED,
            RecordEntryType.EA_RULING,
        )
    ]
    for idx, entry in enumerate(challenge_entries):
        ch_id = f"challenge_{entry.id}"
        x = (idx + 1) / (len(challenge_entries) + 1) if challenge_entries else 0.5
        y_offset = (idx % 3) * 0.02

        if entry.entry_type == RecordEntryType.CHALLENGE_ISSUED:
            nodes[ch_id] = _Node(
                id=ch_id,
                label="",
                tier=4, x=x, y=TIER_Y[4] + y_offset,
                color=CHALLENGE_CLR,
                symbol=SYM_CHALLENGE,
                size=8,
                hover=f"Challenge by {entry.senator_name}\n{entry.content[:120]}",
            )
            # Edge from challenger senator
            if entry.senator_id in nodes:
                edges.append(_Edge(
                    source_id=entry.senator_id, target_id=ch_id,
                    color=CHALLENGE_CLR, dash="dot", width=1.0,
                ))
        else:
            # EA ruling
            nodes[ch_id] = _Node(
                id=ch_id,
                label="",
                tier=4, x=x, y=TIER_Y[4] - 0.03 + y_offset,
                color=EA_RULING_CLR,
                symbol=SYM_EA_RULING,
                size=7,
                hover=f"EA Ruling\n{entry.content[:120]}",
            )

    # ── Tier 5: Coalitions + Verdict ──────────────────────────────────
    for idx, coal in enumerate(result.coalitions):
        coal_id = f"coalition_{coal.id}"
        x = 0.15 + idx * 0.15
        indep = "⚠ LOW" if coal.is_low_independence else "✓ OK"
        nodes[coal_id] = _Node(
            id=coal_id,
            label=coal.name[:12],
            tier=5, x=min(x, 0.45), y=TIER_Y[5] + 0.02,
            color=COALITION_CLR,
            symbol=SYM_COALITION,
            size=12,
            hover=(
                f"Coalition: {coal.name}\n"
                f"Members: {', '.join(coal.member_names)}\n"
                f"EIS: {coal.epistemic_independence_score:.2f} ({indep})\n"
                f"Similarity: {coal.similarity_score:.2f}"
            ),
        )
        # Edges from coalition members
        for mid in coal.member_ids:
            if mid in nodes:
                edges.append(_Edge(
                    source_id=mid, target_id=coal_id,
                    color=COALITION_CLR, dash="dot", width=0.8,
                ))

    # Verdict node
    verdict_id = "verdict"
    verdict_label = result.majority_opinion.verdict_label.value
    nodes[verdict_id] = _Node(
        id=verdict_id,
        label=f"{verdict_label}\n{posterior:.0%}",
        tier=5, x=0.7, y=TIER_Y[5],
        color=VERDICT_CLR,
        symbol=SYM_VERDICT,
        size=28,
        hover=(
            f"VERDICT: {verdict_label}\n"
            f"Posterior: {posterior:.2%}\n"
            f"Senators: {result.num_senators} | "
            f"Rounds: {result.num_rounds} | "
            f"Evidence: {result.num_evidence}\n"
            f"Duration: {result.duration_seconds:.0f}s"
        ),
    )
    # Edge from proposition to verdict
    edges.append(_Edge(
        source_id=prop_id, target_id=verdict_id,
        color=VERDICT_CLR, width=3.0, label="CONCLUDES",
    ))

    # ── Tier 6: Minority Report ───────────────────────────────────────
    if result.minority_report.minority_senator_ids:
        min_id = "minority_report"
        nodes[min_id] = _Node(
            id=min_id,
            label="Minority",
            tier=6, x=0.5, y=TIER_Y[6],
            color=MINORITY_CLR,
            symbol=SYM_MINORITY,
            size=18,
            hover=(
                f"MINORITY REPORT\n"
                f"Claim: {result.minority_report.minority_claim[:120]}\n"
                f"Senators: {', '.join(result.minority_report.minority_senator_names)}\n"
                f"Evidence: {len(result.minority_report.supporting_evidence_ids)} items\n"
                f"Sustained challenges: {len(result.minority_report.sustained_challenges)}"
            ),
        )
        # Edges from minority senators
        for sid in result.minority_report.minority_senator_ids:
            if sid in nodes:
                edges.append(_Edge(
                    source_id=sid, target_id=min_id,
                    color=MINORITY_CLR, dash="dash", width=1.2,
                ))

    # ═══════════════════════════════════════════════════════════════════
    # Build Plotly figure
    # ═══════════════════════════════════════════════════════════════════

    return _render_figure(nodes, edges, title="🏛️ Senate Lifecycle DAG")


def build_live_senate_dag(
    engine: Any,
    proposition: str,
) -> go.Figure:
    """Build a partial DAG from the live SocraticEngine state mid-session.

    Shows Tiers 1-3 (Proposition → Senators → Evidence) incrementally
    as the session runs. Challenges (Tier 4) are included if present.

    Args:
        engine: The running SocraticEngine instance with exposed
            .senate, .live_senators, .docket, .record attributes.
        proposition: The proposition text.

    Returns:
        A Plotly ``go.Figure`` for the partial DAG.
    """
    nodes: dict[str, _Node] = {}
    edges: list[_Edge] = []

    # ── Tier 1: Proposition ───────────────────────────────────────────
    prop_id = "proposition"
    nodes[prop_id] = _Node(
        id=prop_id,
        label="⟳",
        tier=1, x=0.5, y=TIER_Y[1],
        color=PROPOSITION_CLR,
        symbol=SYM_PROPOSITION,
        size=24,
        hover=f"Proposition: {proposition[:120]}\n(session in progress…)",
    )

    # ── Tier 2: Senators ──────────────────────────────────────────────
    senator_x: dict[str, float] = {}
    live_senators = engine.live_senators or {}
    senator_list = list(live_senators.items())
    total = len(senator_list)

    for idx, (sid, ls) in enumerate(senator_list):
        x = (idx + 1) / (total + 1)
        senator_x[sid] = x
        cat_color = CATEGORY_COLORS.get(ls.spec.category.value, "#4682B4")
        pos = ls.current_position

        nodes[sid] = _Node(
            id=sid,
            label=ls.spec.name[:15],
            tier=2, x=x, y=TIER_Y[2],
            color=cat_color,
            symbol=SYM_SENATOR,
            size=14,
            hover=(
                f"Senator: {ls.spec.name}\n"
                f"Category: {ls.spec.category.display_name}\n"
                f"Current position: {pos:.2f}\n"
                f"Evidence submitted: {ls.evidence_submitted}"
            ),
        )
        edges.append(_Edge(
            source_id=prop_id, target_id=sid,
            color=EDGE_CLR, dash="dash",
        ))

    # ── Tier 3: Evidence (from live docket) ───────────────────────────
    docket = engine.docket
    if docket:
        evidence_counters: dict[str, int] = {}
        try:
            all_items = docket.all_items
        except Exception:
            all_items = []

        for ev in all_items:
            count = evidence_counters.get(ev.senator_id, 0)
            evidence_counters[ev.senator_id] = count + 1

            base_x = senator_x.get(ev.senator_id, 0.5)
            offset = (count % 5 - 2) * 0.03
            x = max(0.02, min(base_x + offset, 0.98))
            y = TIER_Y[3] - count * 0.015

            pol_color = POLARITY_COLORS.get(ev.polarity.value, "#FFA500")

            nodes[ev.id] = _Node(
                id=ev.id,
                label="",
                tier=3, x=x, y=y,
                color=pol_color,
                symbol=SYM_EVIDENCE,
                size=8,
                hover=(
                    f"{ev.polarity.value.title()} [{ev.evidence_id_display}]\n"
                    f"Senator: {ev.senator_name}\n"
                    f"DEW: {ev.dew_score:.2f}\n"
                    f"{ev.claim_text[:100]}"
                ),
            )
            if ev.senator_id in nodes:
                edges.append(_Edge(
                    source_id=ev.senator_id, target_id=ev.id,
                    color=pol_color, width=1.0,
                ))

    # ── Tier 4: Challenges (from live record) ─────────────────────────
    record = engine.record
    if record:
        try:
            entries = [
                e for e in record.entries
                if e.entry_type == RecordEntryType.CHALLENGE_ISSUED
            ]
        except Exception:
            entries = []

        for idx, entry in enumerate(entries):
            ch_id = f"challenge_{entry.id}"
            x = (idx + 1) / (len(entries) + 1)
            nodes[ch_id] = _Node(
                id=ch_id,
                label="",
                tier=4, x=x, y=TIER_Y[4],
                color=CHALLENGE_CLR,
                symbol=SYM_CHALLENGE,
                size=8,
                hover=f"Challenge by {entry.senator_name}\n{entry.content[:100]}",
            )
            if entry.senator_id in nodes:
                edges.append(_Edge(
                    source_id=entry.senator_id, target_id=ch_id,
                    color=CHALLENGE_CLR, dash="dot", width=1.0,
                ))

    return _render_figure(nodes, edges, title="🏛️ Senate DAG (Live)")


# ═══════════════════════════════════════════════════════════════════════
# Shared figure renderer
# ═══════════════════════════════════════════════════════════════════════

def _render_figure(
    nodes: dict[str, _Node],
    edges: list[_Edge],
    title: str = "🏛️ Senate Lifecycle DAG",
) -> go.Figure:
    """Render nodes and edges into a Plotly figure."""
    fig = go.Figure()

    # ── edges ──────────────────────────────────────────────────────────
    for edge in edges:
        src = nodes.get(edge.source_id)
        tgt = nodes.get(edge.target_id)
        if not src or not tgt:
            continue
        fig.add_trace(go.Scatter(
            x=[src.x, tgt.x],
            y=[src.y, tgt.y],
            mode="lines",
            line=dict(color=edge.color, width=edge.width, dash=edge.dash),
            hoverinfo="text",
            hovertext=edge.label,
            showlegend=False,
        ))

    # ── tier labels ───────────────────────────────────────────────────
    tier_labels = {
        1: "PROPOSITION",
        2: "SENATE",
        3: "EVIDENCE",
        4: "CHALLENGES",
        5: "VERDICT",
        6: "MINORITY",
    }
    for tier, label in tier_labels.items():
        # Only show tiers with nodes
        if not any(n.tier == tier for n in nodes.values()):
            continue
        fig.add_annotation(
            x=-0.08, y=TIER_Y[tier],
            text=label,
            showarrow=False,
            font=dict(color="#666688", size=8),
            textangle=-90 if tier not in (1, 6) else 0,
        )

    # ── nodes ─────────────────────────────────────────────────────────
    for node in nodes.values():
        fig.add_trace(go.Scatter(
            x=[node.x],
            y=[node.y],
            mode="markers+text",
            marker=dict(
                size=node.size,
                color=node.color,
                symbol=node.symbol,
                line=dict(width=1, color="#FFFFFF")
                if "open" not in node.symbol
                else dict(width=2, color=node.color),
            ),
            text=[node.label],
            textposition="bottom center",
            textfont=dict(color="#CCCCCC", size=8),
            hoverinfo="text",
            hovertext=node.hover,
            showlegend=False,
        ))

    # ── layout ────────────────────────────────────────────────────────
    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.15, 1.15],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.08, 1.12],
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=550,
        title=dict(
            text=title,
            font=dict(color="#CCCCCC", size=14),
            x=0.5,
        ),
    )

    return fig

