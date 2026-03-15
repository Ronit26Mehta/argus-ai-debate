"""
PULSE Visualization — dual-theme operational dashboards.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

import numpy as np

if TYPE_CHECKING:
    from argus.pulse.dashboard import DashboardReport

DARK = {"bg": "#0e1117", "paper": "#1a1f2e", "text": "#e0e0e0", "title": "#00d4ff",
        "grid": "#2a3040", "bar": "#00d4ff", "bar2": "#ff00d4", "bar3": "#00ff88",
        "warning": "#ffbf00", "critical": "#ff4466", "good": "#00ff88"}

LIGHT = {"bg": "#ffffff", "paper": "#f8f9fa", "text": "#1a1a1a", "title": "#0066cc",
         "grid": "#e0e0e0", "bar": "#0066cc", "bar2": "#cc0066", "bar3": "#00aa44",
         "warning": "#cc8800", "critical": "#cc0033", "good": "#00aa44"}


def _t(theme: str = "dark") -> dict:
    return LIGHT if theme == "light" else DARK


def plot_latency_histogram(
    values: list[float],
    stage_name: str = "debate",
    theme: str = "dark",
    height: int = 400,
) -> Any:
    """Plot latency distribution with percentile markers."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    t = _t(theme)
    fig = go.Figure()

    if not values:
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=t["text"]))
        fig.update_layout(paper_bgcolor=t["bg"])
        return fig

    arr = np.array(values)
    fig.add_trace(go.Histogram(
        x=values, nbinsx=30,
        marker=dict(color=f"rgba({int(t['bar'][1:3],16)},{int(t['bar'][3:5],16)},{int(t['bar'][5:7],16)},0.6)",
                    line=dict(color=t["bar"], width=1)),
        name="Latency",
    ))

    for p, color, label in [(50, t["bar3"], "P50"), (95, t["warning"], "P95"), (99, t["critical"], "P99")]:
        val = float(np.percentile(arr, p))
        fig.add_vline(x=val, line_dash="dash", line_color=color,
                      annotation_text=f"{label}={val:.0f}ms",
                      annotation_font=dict(size=10, color=color))

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"]),
        title=dict(text=f"⏱️ {stage_name} Latency Distribution",
                   font=dict(size=16, color=t["title"])),
        xaxis=dict(title="Latency (ms)", gridcolor=t["grid"]),
        yaxis=dict(title="Count", gridcolor=t["grid"]),
        height=height, showlegend=False,
    )
    return fig


def plot_token_usage(
    input_tokens: list[int],
    output_tokens: list[int],
    theme: str = "dark",
    height: int = 400,
) -> Any:
    """Plot token usage over time."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    t = _t(theme)
    fig = go.Figure()

    x = list(range(len(input_tokens)))
    fig.add_trace(go.Bar(x=x, y=input_tokens, name="Input Tokens",
                         marker_color=t["bar"]))
    fig.add_trace(go.Bar(x=x, y=output_tokens, name="Output Tokens",
                         marker_color=t["bar2"]))

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"]),
        title=dict(text="🔤 Token Usage", font=dict(size=16, color=t["title"])),
        xaxis=dict(title="Request #", gridcolor=t["grid"]),
        yaxis=dict(title="Tokens", gridcolor=t["grid"]),
        barmode="group", height=height,
    )
    return fig


def plot_accuracy_trend(
    accuracies: list[float],
    theme: str = "dark",
    height: int = 400,
) -> Any:
    """Plot accuracy trend over time."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    t = _t(theme)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=accuracies, mode="lines+markers",
        line=dict(color=t["bar3"], width=2),
        marker=dict(size=6, color=t["bar3"]),
        name="Accuracy",
    ))
    fig.add_hline(y=0.7, line_dash="dot", line_color=t["warning"],
                  annotation_text="Target (0.7)")

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"]),
        title=dict(text="📈 Accuracy Trend", font=dict(size=16, color=t["title"])),
        xaxis=dict(title="Debate #", gridcolor=t["grid"]),
        yaxis=dict(title="Accuracy", range=[0, 1], gridcolor=t["grid"]),
        height=height, showlegend=False,
    )
    return fig


def plot_failure_taxonomy(
    taxonomy_data: dict[str, int],
    theme: str = "dark",
    height: int = 400,
) -> Any:
    """Plot failure type distribution."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    t = _t(theme)
    fig = go.Figure()

    categories = list(taxonomy_data.keys())
    counts = list(taxonomy_data.values())

    colors = [t["critical"], t["warning"], t["bar"], t["bar2"], t["bar3"], t["grid"]]

    fig.add_trace(go.Bar(
        x=categories, y=counts,
        marker_color=colors[:len(categories)],
    ))

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"]),
        title=dict(text="🔴 Failure Taxonomy", font=dict(size=16, color=t["title"])),
        xaxis=dict(title="Failure Category", gridcolor=t["grid"]),
        yaxis=dict(title="Count", gridcolor=t["grid"]),
        height=height, showlegend=False,
    )
    return fig


def generate_dashboard_html(
    report: "DashboardReport",
    theme: str = "dark",
) -> str:
    """Generate a standalone HTML dashboard."""
    t = _t(theme)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PULSE Operational Dashboard</title>
    <style>
        body {{ background: {t['bg']}; color: {t['text']}; font-family: 'Inter', sans-serif; padding: 20px; }}
        .card {{ background: {t['paper']}; border-radius: 12px; padding: 20px; margin: 15px 0; }}
        h1 {{ color: {t['title']}; }}
        h2 {{ color: {t['title']}; font-size: 1.2em; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric .value {{ font-size: 2em; font-weight: bold; color: {t['bar']}; }}
        .metric .label {{ font-size: 0.9em; opacity: 0.7; }}
        .anomaly {{ border-left: 4px solid {t['warning']}; padding-left: 12px; margin: 8px 0; }}
        .anomaly.critical {{ border-color: {t['critical']}; }}
        .rec {{ padding: 8px 0; }}
    </style>
</head>
<body>
    <h1>📡 PULSE Operational Intelligence Dashboard</h1>
    <p>Generated: {report.generated_at}</p>

    <div class="card">
        <h2>📊 Metrics Summary</h2>"""

    counters = report.metrics_snapshot.get("counters", {})
    for name, value in counters.items():
        html += f'<div class="metric"><div class="value">{value:.0f}</div><div class="label">{name}</div></div>'

    html += """</div><div class="card"><h2>⚠️ Anomalies</h2>"""
    if report.anomalies:
        for a in report.anomalies:
            cls = "anomaly critical" if a.severity == "critical" else "anomaly"
            html += f'<div class="{cls}"><strong>{a.metric_name}</strong>: {a.description}</div>'
    else:
        html += "<p>No anomalies detected.</p>"

    html += """</div><div class="card"><h2>💡 Recommendations</h2>"""
    for rec in report.recommendations:
        html += f'<div class="rec">{rec}</div>'

    html += """</div></body></html>"""
    return html
