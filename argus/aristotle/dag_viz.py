"""
Full Lifecycle DAG — Plotly-based debate visualisation.

Renders the complete epistemic structure of the debate as a dark-themed,
layered directed acyclic graph:

    Tier 1: Proposition (cyan diamond)
    Tier 2: Specialists (steel-blue squares)
    Tier 3: Evidence    (green/red circles)
    Tier 4: Rebuttals + Bayesian updates (amber triangles, lavender circles)
    Tier 5: Verdict     (gold star)

Grows node-by-node in real time as :class:`MonitorEvent` items arrive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import plotly.graph_objects as go

from argus.aristotle.models import MonitorEvent, MonitorEventType

# ── colour palette ─────────────────────────────────────────────────────

BG_COLOR = "#1A1A2E"
PROPOSITION_COLOR = "#00BFFF"
SPECIALIST_COLOR = "#4682B4"
SUPPORT_COLOR = "#2ECC71"
ATTACK_COLOR = "#E74C3C"
REBUTTAL_COLOR = "#FFA500"
BAYESIAN_COLOR = "#9B89C4"
VERDICT_COLOR = "#FFD700"
INTERVENTION_COLOR = "#FFFFFF"
EDGE_ASSIGNED_COLOR = "#888888"
EDGE_REFINES_COLOR = "#00BCD4"
EDGE_CONCLUDES_COLOR = "#FFD700"
EDGE_UPDATES_COLOR = "#9B89C4"
EDGE_INTERVENES_COLOR = "#FFFFFF"
GRID_LINE_COLOR = "#333355"

# ── node shape symbols (Plotly marker notation) ───────────────────────

SHAPE_PROPOSITION = "diamond"
SHAPE_SPECIALIST = "square"
SHAPE_EVIDENCE = "circle"
SHAPE_REBUTTAL = "triangle-up"
SHAPE_BAYESIAN = "circle-open"
SHAPE_VERDICT = "star"
SHAPE_INTERVENTION = "hexagon"

# ── tier Y-positions ──────────────────────────────────────────────────

TIER_Y = {
    1: 1.0,   # proposition
    2: 0.80,  # specialists
    3: 0.55,  # evidence
    4: 0.30,  # rebuttals + bayesian
    5: 0.05,  # verdict
}


@dataclass
class _Node:
    """Internal node record for layout computation."""
    id: str
    label: str
    tier: int
    x: float = 0.0
    y: float = 0.0
    color: str = "#FFFFFF"
    symbol: str = "circle"
    size: int = 12
    hover: str = ""
    round_num: int = 0


@dataclass
class _Edge:
    """Internal edge record."""
    source_id: str
    target_id: str
    color: str = "#888888"
    dash: str = "solid"
    width: float = 1.5
    label: str = ""


@dataclass
class DAGState:
    """Accumulated state of the Full Lifecycle DAG."""
    nodes: dict[str, _Node] = field(default_factory=dict)
    edges: list[_Edge] = field(default_factory=list)
    specialists_x: dict[str, float] = field(default_factory=dict)
    evidence_counters: dict[str, int] = field(default_factory=dict)
    bayesian_count: int = 0
    round_dividers: list[tuple[int, float]] = field(default_factory=list)
    current_round: int = 0
    proposition_id: str = ""


class FullLifecycleDAG:
    """
    Builds and incrementally updates the Plotly ``go.Figure`` for the
    Full Lifecycle DAG.

    Usage (inside the Streamlit interface)::

        dag = FullLifecycleDAG()
        for event in event_stream:
            dag.process_event(event)
        fig = dag.build_figure()
        st.plotly_chart(fig, width="stretch", key="dag")
    """

    def __init__(self) -> None:
        self.state = DAGState()

    # ── public API ────────────────────────────────────────────────────

    def process_event(self, event: MonitorEvent) -> None:
        """Ingest a single :class:`MonitorEvent` and update the graph state."""
        handler = _EVENT_HANDLERS.get(event.event_type)
        if handler:
            handler(self.state, event)

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
                hoverinfo="text",
                hovertext=edge.label,
                showlegend=False,
            ))

        # ── round dividers ────────────────────────────────────────────
        for rnd, posterior in self.state.round_dividers:
            y = _round_divider_y(rnd)
            fig.add_shape(
                type="line", x0=-0.1, x1=1.1, y0=y, y1=y,
                line=dict(color=GRID_LINE_COLOR, width=1, dash="dot"),
            )
            fig.add_annotation(
                x=1.05, y=y,
                text=f"R{rnd} — {posterior:.0%}",
                showarrow=False,
                font=dict(color="#777777", size=9),
            )

        # ── nodes ─────────────────────────────────────────────────────
        for node in self.state.nodes.values():
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
                textfont=dict(color="#CCCCCC", size=9),
                hoverinfo="text",
                hovertext=node.hover,
                showlegend=False,
            ))

        # ── layout ────────────────────────────────────────────────────
        fig.update_layout(
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-0.15, 1.25],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-0.05, 1.10],
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=520,
            title=dict(
                text="Full Lifecycle DAG",
                font=dict(color="#CCCCCC", size=14),
                x=0.5,
            ),
        )
        return fig


# ── event handlers ────────────────────────────────────────────────────

def _handle_session_start(state: DAGState, event: MonitorEvent) -> None:
    state.proposition_id = event.node_id
    state.nodes[event.node_id] = _Node(
        id=event.node_id,
        label=f"{event.posterior:.0%}",
        tier=1, x=0.5, y=TIER_Y[1],
        color=PROPOSITION_COLOR,
        symbol=SHAPE_PROPOSITION,
        size=22,
        hover=f"Proposition: {event.node_text[:80]}…\nPrior: {event.posterior:.2%}",
    )


def _handle_specialist(state: DAGState, event: MonitorEvent) -> None:
    n = len(state.specialists_x)
    total_expected = max(n + 1, 2)
    # recalculate all x positions for even spacing
    x = (n + 1) / (total_expected + 1)
    state.specialists_x[event.agent_id] = x
    # re-layout existing specialists
    for idx, aid in enumerate(state.specialists_x):
        state.specialists_x[aid] = (idx + 1) / (len(state.specialists_x) + 1)

    actual_x = state.specialists_x[event.agent_id]
    state.nodes[event.agent_id] = _Node(
        id=event.agent_id,
        label=event.agent_name,
        tier=2, x=actual_x, y=TIER_Y[2],
        color=SPECIALIST_COLOR,
        symbol=SHAPE_SPECIALIST,
        size=16,
        hover=f"Specialist: {event.agent_name}",
    )
    state.edges.append(_Edge(
        source_id=state.proposition_id,
        target_id=event.agent_id,
        color=EDGE_ASSIGNED_COLOR,
        dash="dash",
        label="ASSIGNED",
    ))


def _handle_evidence(state: DAGState, event: MonitorEvent, is_support: bool) -> None:
    agent_x = state.specialists_x.get(event.agent_id, 0.5)
    count = state.evidence_counters.get(event.agent_id, 0)
    state.evidence_counters[event.agent_id] = count + 1

    # Spread evidence around the agent's x
    offset = (count % 3 - 1) * 0.04
    x = max(0.02, min(agent_x + offset, 0.98))

    # Y within tier 3 — lower for later rounds
    base_y = TIER_Y[3]
    y = base_y - event.round_num * 0.03

    color = SUPPORT_COLOR if is_support else ATTACK_COLOR
    edge_color = color

    state.nodes[event.node_id] = _Node(
        id=event.node_id,
        label="",
        tier=3, x=x, y=y,
        color=color,
        symbol=SHAPE_EVIDENCE,
        size=10,
        hover=(
            f"{'Support' if is_support else 'Attack'} Evidence\n"
            f"Agent: {event.agent_name}\n"
            f"EVID-Q: {event.evid_q:.2f}\n"
            f"Round: {event.round_num}\n"
            f"{event.node_text[:100]}"
        ),
        round_num=event.round_num,
    )
    state.edges.append(_Edge(
        source_id=event.agent_id,
        target_id=event.node_id,
        color=edge_color,
        label=f"EVID-Q: {event.evid_q:.2f}",
    ))


def _handle_support(state: DAGState, event: MonitorEvent) -> None:
    _handle_evidence(state, event, is_support=True)


def _handle_attack(state: DAGState, event: MonitorEvent) -> None:
    _handle_evidence(state, event, is_support=False)


def _handle_rebuttal(state: DAGState, event: MonitorEvent) -> None:
    # Position rebuttal near the refuter agent if known, else centre
    agent_x = state.specialists_x.get(event.agent_id, 0.5)
    reb_x = max(0.05, min(agent_x + 0.06, 0.95))

    state.nodes[event.node_id] = _Node(
        id=event.node_id,
        label="",
        tier=4, x=reb_x, y=TIER_Y[4] + 0.05,
        color=REBUTTAL_COLOR,
        symbol=SHAPE_REBUTTAL,
        size=9,
        hover=f"Rebuttal by {event.agent_name}\n{event.node_text[:100]}",
        round_num=event.round_num,
    )

    # Edge: refuter agent → rebuttal node
    if event.agent_id and event.agent_id in state.nodes:
        state.edges.append(_Edge(
            source_id=event.agent_id,
            target_id=event.node_id,
            color=REBUTTAL_COLOR,
            dash="dash",
            label="REBUTS",
        ))

    # Edge: rebuttal → target evidence node (if provided)
    target_id = event.details.get("target_id", "") if event.details else ""
    if target_id and target_id in state.nodes:
        state.edges.append(_Edge(
            source_id=event.node_id,
            target_id=target_id,
            color=ATTACK_COLOR,
            dash="dot",
            width=1.0,
            label="CHALLENGES",
        ))


def _handle_bayesian(state: DAGState, event: MonitorEvent) -> None:
    state.bayesian_count += 1
    node_id = f"bayesian_{state.bayesian_count}"
    y = TIER_Y[4] - event.round_num * 0.03
    state.nodes[node_id] = _Node(
        id=node_id,
        label=f"{event.posterior:.0%}",
        tier=4, x=0.5, y=y,
        color=BAYESIAN_COLOR,
        symbol=SHAPE_BAYESIAN,
        size=14,
        hover=f"Bayesian Update — Round {event.round_num}\nPosterior: {event.posterior:.2%}",
        round_num=event.round_num,
    )
    # Link from previous bayesian update
    if state.bayesian_count > 1:
        prev_id = f"bayesian_{state.bayesian_count - 1}"
        state.edges.append(_Edge(
            source_id=prev_id,
            target_id=node_id,
            color=EDGE_UPDATES_COLOR,
            label="UPDATES",
        ))


def _handle_round_complete(state: DAGState, event: MonitorEvent) -> None:
    state.current_round = event.round_num
    state.round_dividers.append((event.round_num, event.posterior))


def _handle_verdict(state: DAGState, event: MonitorEvent) -> None:
    node_id = "verdict"
    state.nodes[node_id] = _Node(
        id=node_id,
        label=f"{event.posterior:.0%}",
        tier=5, x=0.5, y=TIER_Y[5],
        color=VERDICT_COLOR,
        symbol=SHAPE_VERDICT,
        size=26,
        hover=f"Verdict: {event.chat_message}\nConfidence: {event.posterior:.2%}",
    )
    # Link from last bayesian update
    if state.bayesian_count > 0:
        last_bayes = f"bayesian_{state.bayesian_count}"
        state.edges.append(_Edge(
            source_id=last_bayes,
            target_id=node_id,
            color=EDGE_CONCLUDES_COLOR,
            width=3.0,
            label="CONCLUDES",
        ))


def _handle_intervention(state: DAGState, event: MonitorEvent) -> None:
    node_id = f"intervention_{event.id}"
    state.nodes[node_id] = _Node(
        id=node_id,
        label=event.intervention_type.replace("_", " ").title(),
        tier=4, x=0.15, y=TIER_Y[4],
        color=INTERVENTION_COLOR,
        symbol=SHAPE_INTERVENTION,
        size=12,
        hover=(
            f"ARISTOTLE Intervention\n"
            f"Type: {event.intervention_type}\n"
            f"Reason: {event.intervention_reason}\n"
            f"Round: {event.round_num}"
        ),
        round_num=event.round_num,
    )


def _handle_agent_replaced(state: DAGState, event: MonitorEvent) -> None:
    _handle_intervention(state, event)
    # Also add the new specialist
    _handle_specialist(state, MonitorEvent(
        event_type=MonitorEventType.SPECIALIST_INSTANTIATED,
        agent_id=event.agent_id,
        agent_name=event.agent_name,
    ))


def _round_divider_y(round_num: int) -> float:
    """Compute y-position for round divider line."""
    return max(0.1, TIER_Y[3] - round_num * 0.05)


_EVENT_HANDLERS: dict[MonitorEventType, Any] = {
    MonitorEventType.SESSION_START: _handle_session_start,
    MonitorEventType.SPECIALIST_INSTANTIATED: _handle_specialist,
    MonitorEventType.SUPPORT_EVIDENCE: _handle_support,
    MonitorEventType.ATTACK_EVIDENCE: _handle_attack,
    MonitorEventType.REBUTTAL_FILED: _handle_rebuttal,
    MonitorEventType.BAYESIAN_UPDATE: _handle_bayesian,
    MonitorEventType.ROUND_COMPLETE: _handle_round_complete,
    MonitorEventType.VERDICT_RENDERED: _handle_verdict,
    MonitorEventType.INTERVENTION_CONVERGENCE: _handle_intervention,
    MonitorEventType.INTERVENTION_STAGNATION: _handle_intervention,
    MonitorEventType.INTERVENTION_DRIFT: _handle_intervention,
    MonitorEventType.INTERVENTION_CONFIDENCE_GUARD: _handle_intervention,
    MonitorEventType.INTERVENTION_EVIDENCE_DROUGHT: _handle_intervention,
    MonitorEventType.INTERVENTION_BUDGET_OVERFLOW: _handle_intervention,
    MonitorEventType.INTERVENTION_DEADLOCK: _handle_intervention,
    MonitorEventType.AGENT_REPLACED: _handle_agent_replaced,
}
