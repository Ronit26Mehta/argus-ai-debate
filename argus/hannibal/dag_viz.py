"""
HANNIBAL Battle DAG — Plotly-based campaign visualisation.

Renders the epistemic battle structure as a dark-themed, layered
directed acyclic graph that grows incrementally:

    Tier 1: Proposition       (cyan diamond)
    Tier 2: Forces            (coloured squares — PF green, OF red, FF blue)
    Tier 3: Agents            (small circles — force-coloured)
    Tier 4: Evidence          (green/red dots — support/counter)
    Tier 5: Skirmish Results  (amber triangles — winner indicated)
    Tier 6: Engagement/Theatre (steel-blue hexagons)
    Tier 7: Campaign Verdict  (gold star)

Grows node-by-node in real time via :meth:`process_event`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import plotly.graph_objects as go

from argus.hannibal.models import ForceType

# ── colour palette ─────────────────────────────────────────────────────

BG_COLOR = "#0B141A"
PROPOSITION_COLOR = "#00BFFF"
FORCE_COLORS: dict[str, str] = {
    "proposition": "#2ECC71",
    "opposition": "#E74C3C",
    "faction_1": "#3498DB",
    "faction_2": "#E67E22",
    "faction_3": "#9B59B6",
}
EVIDENCE_SUPPORT_COLOR = "#2ECC71"
EVIDENCE_COUNTER_COLOR = "#E74C3C"
SKIRMISH_COLOR = "#FFA500"
ENGAGEMENT_COLOR = "#4682B4"
VERDICT_COLOR = "#FFD700"
EDGE_DEFAULT_COLOR = "#555555"
EDGE_EVIDENCE_COLOR = "#888888"
EDGE_WINNER_COLOR = "#FFD700"

# ── shapes ─────────────────────────────────────────────────────────────

SHAPE_PROPOSITION = "diamond"
SHAPE_FORCE = "square"
SHAPE_AGENT = "circle"
SHAPE_EVIDENCE = "circle"
SHAPE_SKIRMISH = "triangle-up"
SHAPE_ENGAGEMENT = "hexagon"
SHAPE_VERDICT = "star"

# ── tier Y positions ──────────────────────────────────────────────────

TIER_Y = {
    1: 1.00,   # proposition
    2: 0.85,   # forces
    3: 0.70,   # agents
    4: 0.50,   # evidence
    5: 0.30,   # skirmish results
    6: 0.15,   # engagements / theatres
    7: 0.00,   # campaign verdict
}


@dataclass
class _Node:
    """Internal node for layout."""
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
    """Internal edge."""
    source_id: str
    target_id: str
    color: str = "#888888"
    dash: str = "solid"
    width: float = 1.5
    label: str = ""


@dataclass
class BattleDAGState:
    """Accumulated state of the Battle DAG."""
    nodes: dict[str, _Node] = field(default_factory=dict)
    edges: list[_Edge] = field(default_factory=list)
    proposition_id: str = "prop"
    force_x: dict[str, float] = field(default_factory=dict)
    agent_x: dict[str, float] = field(default_factory=dict)
    evidence_count: int = 0
    skirmish_count: int = 0
    engagement_count: int = 0


class BattleDAG:
    """Builds and incrementally updates the HANNIBAL Battle DAG.

    Usage (inside Streamlit)::

        dag = BattleDAG()
        # call dag.add_* methods as campaign progresses
        fig = dag.build_figure()
        st.plotly_chart(fig, width="stretch")
    """

    def __init__(self) -> None:
        self.state = BattleDAGState()

    # ── incremental adders ────────────────────────────────────────────

    def add_proposition(self, text: str) -> None:
        """Add the root proposition node (Tier 1)."""
        self.state.proposition_id = "prop"
        self.state.nodes["prop"] = _Node(
            id="prop",
            label="Proposition",
            tier=1, x=0.5, y=TIER_Y[1],
            color=PROPOSITION_COLOR,
            symbol=SHAPE_PROPOSITION,
            size=22,
            hover=f"Proposition:\n{text[:120]}",
        )

    def add_force(self, force_type: str, position: str = "",
                  agent_count: int = 0, force_name: str = "") -> None:
        """Add a force node (Tier 2)."""
        n = len(self.state.force_x)
        # Re-layout all forces evenly
        self.state.force_x[force_type] = 0.0  # placeholder
        total = len(self.state.force_x)
        for idx, ft in enumerate(self.state.force_x):
            self.state.force_x[ft] = (idx + 1) / (total + 1)

        x = self.state.force_x[force_type]
        color = FORCE_COLORS.get(force_type, "#888888")
        try:
            display = force_name or ForceType(force_type).display_name
            abbr = "".join([w[0].upper() for w in display.split(" ") if w])[:3] if force_name else ForceType(force_type).abbreviation
        except ValueError:
            display = force_name or force_type
            abbr = "".join([w[0].upper() for w in display.split(" ") if w])[:3]

        self.state.nodes[f"force_{force_type}"] = _Node(
            id=f"force_{force_type}",
            label=abbr,
            tier=2, x=x, y=TIER_Y[2],
            color=color,
            symbol=SHAPE_FORCE,
            size=18,
            hover=f"{display}\nAgents: {agent_count}\n{position[:80]}",
        )
        self.state.edges.append(_Edge(
            source_id="prop",
            target_id=f"force_{force_type}",
            color=EDGE_DEFAULT_COLOR,
            dash="dash",
        ))

    def add_agent(self, agent_id: str, agent_name: str,
                  force_type: str, role: str = "") -> None:
        """Add an agent node (Tier 3)."""
        force_x = self.state.force_x.get(force_type, 0.5)
        # Stack agents around force x
        n = sum(1 for k in self.state.agent_x if k.startswith(force_type))
        offset = (n - 1) * 0.04
        x = max(0.02, min(force_x + offset, 0.98))
        self.state.agent_x[f"{force_type}_{agent_id}"] = x

        color = FORCE_COLORS.get(force_type, "#888888")
        self.state.nodes[agent_id] = _Node(
            id=agent_id,
            label="",
            tier=3, x=x, y=TIER_Y[3],
            color=color,
            symbol=SHAPE_AGENT,
            size=8,
            hover=f"{agent_name}\nRole: {role}\nForce: {force_type}",
        )
        self.state.edges.append(_Edge(
            source_id=f"force_{force_type}",
            target_id=agent_id,
            color=color,
            width=1.0,
        ))

    def add_evidence(self, evidence_id: str, agent_id: str,
                     agent_name: str, force_type: str,
                     claim: str = "", evid_q: float = 0.5,
                     is_counter: bool = False) -> None:
        """Add an evidence node (Tier 4)."""
        self.state.evidence_count += 1
        agent_x = 0.5
        for k, v in self.state.agent_x.items():
            if agent_id in k:
                agent_x = v
                break

        # Spread evidence around agent
        n = self.state.evidence_count
        offset = ((n % 5) - 2) * 0.03
        x = max(0.02, min(agent_x + offset, 0.98))
        y = TIER_Y[4] + ((n % 3) - 1) * 0.03

        color = EVIDENCE_COUNTER_COLOR if is_counter else EVIDENCE_SUPPORT_COLOR
        self.state.nodes[evidence_id] = _Node(
            id=evidence_id,
            label="",
            tier=4, x=x, y=y,
            color=color,
            symbol=SHAPE_EVIDENCE,
            size=7,
            hover=(
                f"{'Counter' if is_counter else 'Support'} Evidence\n"
                f"Agent: {agent_name}\n"
                f"EVID-Q: {evid_q:.2f}\n"
                f"{claim[:100]}"
            ),
        )
        if agent_id in self.state.nodes:
            self.state.edges.append(_Edge(
                source_id=agent_id,
                target_id=evidence_id,
                color=color,
                width=1.0,
            ))

    def add_skirmish_result(self, skirmish_id: str, label: str,
                            winner: str, confidence: float,
                            force_a: str, force_b: str, winner_name_override: str = "") -> None:
        """Add a skirmish result node (Tier 5)."""
        self.state.skirmish_count += 1
        n = self.state.skirmish_count
        total_expected = max(n, 4)
        x = n / (total_expected + 1)

        winner_color = FORCE_COLORS.get(winner, SKIRMISH_COLOR)
        if winner_name_override:
            winner_abbr = "".join([w[0].upper() for w in winner_name_override.split(" ") if w])[:3]
        else:
            try:
                winner_abbr = ForceType(winner).abbreviation
            except ValueError:
                winner_abbr = winner[:3].upper()

        self.state.nodes[skirmish_id] = _Node(
            id=skirmish_id,
            label=winner_abbr,
            tier=5, x=x, y=TIER_Y[5],
            color=winner_color,
            symbol=SHAPE_SKIRMISH,
            size=12,
            hover=(
                f"{label}\n"
                f"Winner: {winner_abbr}\n"
                f"Confidence: {confidence:.2f}"
            ),
        )

        # Connect to forces involved
        for ft in [force_a, force_b]:
            fid = f"force_{ft}"
            if fid in self.state.nodes:
                self.state.edges.append(_Edge(
                    source_id=fid,
                    target_id=skirmish_id,
                    color=FORCE_COLORS.get(ft, EDGE_DEFAULT_COLOR),
                    width=1.0,
                    dash="dot",
                ))

    def add_engagement_result(self, engagement_id: str, label: str,
                              winner: str, margin: float,
                              child_skirmish_ids: list[str] | None = None) -> None:
        """Add an engagement result node (Tier 6)."""
        self.state.engagement_count += 1
        n = self.state.engagement_count
        total_expected = max(n, 2)
        x = n / (total_expected + 1)

        winner_color = FORCE_COLORS.get(winner, ENGAGEMENT_COLOR)

        self.state.nodes[engagement_id] = _Node(
            id=engagement_id,
            label=label[:12],
            tier=6, x=x, y=TIER_Y[6],
            color=winner_color,
            symbol=SHAPE_ENGAGEMENT,
            size=14,
            hover=f"{label}\nWinner: {winner}\nMargin: {margin:.2f}",
        )

        # Connect child skirmishes → engagement
        for sid in (child_skirmish_ids or []):
            if sid in self.state.nodes:
                self.state.edges.append(_Edge(
                    source_id=sid,
                    target_id=engagement_id,
                    color=EDGE_DEFAULT_COLOR,
                ))

    def add_verdict(self, verdict_label: str, winner: str,
                    strength: float, narrative: str = "") -> None:
        """Add the campaign verdict node (Tier 7)."""
        winner_color = FORCE_COLORS.get(winner, VERDICT_COLOR)
        try:
            winner_display = ForceType(winner).display_name
        except ValueError:
            winner_display = winner

        self.state.nodes["verdict"] = _Node(
            id="verdict",
            label=f"{strength:.0%}",
            tier=7, x=0.5, y=TIER_Y[7],
            color=VERDICT_COLOR,
            symbol=SHAPE_VERDICT,
            size=26,
            hover=(
                f"Campaign Verdict: {verdict_label}\n"
                f"Winner: {winner_display}\n"
                f"Strength: {strength:.0%}\n"
                f"{narrative[:120]}"
            ),
        )

        # Connect engagements/skirmishes → verdict
        for nid, node in self.state.nodes.items():
            if node.tier in (5, 6) and nid != "verdict":
                self.state.edges.append(_Edge(
                    source_id=nid,
                    target_id="verdict",
                    color=EDGE_WINNER_COLOR,
                    width=2.0,
                ))

    # ── figure builder ────────────────────────────────────────────────

    def build_figure(self) -> go.Figure:
        """Render the current DAG state as a Plotly figure."""
        fig = go.Figure()

        # ── edges ─────────────────────────────────────────────────────
        for edge in self.state.edges:
            src = self.state.nodes.get(edge.source_id)
            tgt = self.state.nodes.get(edge.target_id)
            if not src or not tgt:
                continue
            fig.add_trace(go.Scatter(
                x=[src.x, tgt.x],
                y=[src.y, tgt.y],
                mode="lines",
                line=dict(
                    color=edge.color,
                    width=edge.width,
                    dash=edge.dash,
                ),
                hoverinfo="none",
                showlegend=False,
            ))

        # ── nodes (grouped by tier for legend) ────────────────────────
        tier_names = {
            1: "Proposition", 2: "Forces", 3: "Agents",
            4: "Evidence", 5: "Skirmishes", 6: "Engagements",
            7: "Verdict",
        }
        tiers_rendered: set[int] = set()

        for node in self.state.nodes.values():
            show_legend = node.tier not in tiers_rendered
            tiers_rendered.add(node.tier)

            fig.add_trace(go.Scatter(
                x=[node.x],
                y=[node.y],
                mode="markers+text",
                marker=dict(
                    size=node.size,
                    color=node.color,
                    symbol=node.symbol,
                    line=dict(width=1, color="#FFFFFF"),
                ),
                text=[node.label],
                textposition="bottom center",
                textfont=dict(color="#CCCCCC", size=9),
                hoverinfo="text",
                hovertext=node.hover,
                showlegend=show_legend,
                name=tier_names.get(node.tier, ""),
            ))

        # ── layout ────────────────────────────────────────────────────
        fig.update_layout(
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-0.1, 1.1],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-0.08, 1.08],
            ),
            margin=dict(l=10, r=10, t=35, b=10),
            height=450,
            title=dict(
                text="HANNIBAL Battle DAG",
                font=dict(color="#CCCCCC", size=13),
                x=0.5,
            ),
            showlegend=True,
            legend=dict(font=dict(size=9, color="#AAAAAA")),
        )
        return fig
