"""Lifecycle DAG tracking and visualization for ARGUS Sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StageState:
    stage: str
    status: str = "pending"  # pending | running | completed | failed
    started_at: str | None = None
    finished_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class LifecycleDAG:
    """Directed lifecycle graph across all blueprint modules."""

    ORDER = [
        "seed",
        "verichain_injection",
        "debate_flow",
        "chronos",
        "phalanx",
        "mneme",
        "fractal",
        "mirror",
        "pulse",
        "verichain_commit",
    ]

    EDGES = [
        ("seed", "verichain_injection"),
        ("verichain_injection", "debate_flow"),
        ("debate_flow", "chronos"),
        ("seed", "chronos"),
        ("seed", "phalanx"),
        ("chronos", "mneme"),
        ("phalanx", "mneme"),
        ("chronos", "fractal"),
        ("fractal", "mirror"),
        ("mirror", "pulse"),
        ("pulse", "verichain_commit"),
    ]

    def __init__(self) -> None:
        self.stages = {name: StageState(stage=name) for name in self.ORDER}

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start(self, stage: str, details: dict[str, Any] | None = None) -> None:
        node = self.stages[stage]
        node.status = "running"
        node.started_at = self._now_iso()
        if details:
            node.details.update(details)

    def complete(self, stage: str, details: dict[str, Any] | None = None) -> None:
        node = self.stages[stage]
        node.status = "completed"
        node.finished_at = self._now_iso()
        if details:
            node.details.update(details)

    def fail(self, stage: str, error: str) -> None:
        node = self.stages[stage]
        node.status = "failed"
        node.finished_at = self._now_iso()
        node.details["error"] = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {name: vars(state) for name, state in self.stages.items()},
            "edges": list(self.EDGES),
        }

    def to_plotly(self) -> Any:
        """Return a plotly Figure if plotly is available, else None."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        positions = {
            "seed": (0.0, 0.0),
            "verichain_injection": (1.0, 0.8),
            "debate_flow": (2.0, 0.8),
            "chronos": (3.0, 1.2),
            "phalanx": (3.0, -0.8),
            "mneme": (4.0, 0.2),
            "fractal": (5.0, 1.0),
            "mirror": (6.0, 0.8),
            "pulse": (7.0, 0.4),
            "verichain_commit": (8.0, 0.4),
        }

        color_map = {
            "pending": "#8B95A1",
            "running": "#00B4D8",
            "completed": "#2DC653",
            "failed": "#D90429",
        }

        fig = go.Figure()

        for source, target in self.EDGES:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color="#4F5D75", width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        x_nodes = []
        y_nodes = []
        labels = []
        colors = []
        hover_text = []

        for name in self.ORDER:
            stage = self.stages[name]
            x, y = positions[name]
            x_nodes.append(x)
            y_nodes.append(y)
            labels.append(name)
            colors.append(color_map.get(stage.status, "#8B95A1"))
            hover_text.append(
                f"{name}<br>status={stage.status}<br>"
                f"start={stage.started_at or '-'}<br>end={stage.finished_at or '-'}"
            )

        fig.add_trace(
            go.Scatter(
                x=x_nodes,
                y=y_nodes,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=20, color=colors, line=dict(color="#0B132B", width=1)),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
            )
        )

        fig.update_layout(
            title="Sandbox Lifecycle DAG",
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=360,
        )
        return fig
