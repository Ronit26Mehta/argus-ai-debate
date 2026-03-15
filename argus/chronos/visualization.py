"""
CHRONOS Visualizations — dual-theme (dark + light) temporal posterior plots.

Provides:
    - plot_temporal_posterior(): Timeline with credible bands
    - plot_drift_timeline(): Inflection points with causal annotations

All functions return plotly Figure objects and support both dark and
light themes for proper rendering in different contexts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

if TYPE_CHECKING:
    from argus.chronos.temporal_posterior import TemporalPosterior
    from argus.chronos.drift_detector import BeliefDriftReport

logger = logging.getLogger(__name__)

# ── Theme Constants ──────────────────────────────────────────────────────────

DARK_THEME = {
    "bg": "#0e1117",
    "paper": "#1a1f2e",
    "grid": "#2a3040",
    "text": "#e0e0e0",
    "title": "#00d4ff",
    "posterior_line": "#00d4ff",
    "band_95": "rgba(0, 212, 255, 0.12)",
    "band_50": "rgba(0, 212, 255, 0.25)",
    "inflection_up": "#00ff88",
    "inflection_down": "#ff4466",
    "prior_line": "#ffbf00",
    "evidence_marker": "#b388ff",
    "annotation_bg": "rgba(26, 31, 46, 0.9)",
}

LIGHT_THEME = {
    "bg": "#ffffff",
    "paper": "#f8f9fa",
    "grid": "#e0e0e0",
    "text": "#1a1a1a",
    "title": "#0066cc",
    "posterior_line": "#0066cc",
    "band_95": "rgba(0, 102, 204, 0.10)",
    "band_50": "rgba(0, 102, 204, 0.20)",
    "inflection_up": "#00aa44",
    "inflection_down": "#cc0033",
    "prior_line": "#cc8800",
    "evidence_marker": "#6633cc",
    "annotation_bg": "rgba(248, 249, 250, 0.9)",
}


def _get_theme(theme: str = "dark") -> dict[str, str]:
    """Get theme dictionary."""
    return LIGHT_THEME if theme == "light" else DARK_THEME


def _check_plotly() -> None:
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for CHRONOS visualizations. "
            "Install with: pip install plotly"
        )


def _apply_layout(fig: Any, title: str, theme: dict[str, str]) -> Any:
    """Apply theme layout to figure."""
    fig.update_layout(
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["paper"],
        font=dict(family="Inter, sans-serif", color=theme["text"], size=12),
        margin=dict(l=60, r=30, t=60, b=50),
        title=dict(
            text=title,
            font=dict(size=18, color=theme["title"]),
        ),
        xaxis=dict(
            gridcolor=theme["grid"],
            zerolinecolor=theme["grid"],
        ),
        yaxis=dict(
            gridcolor=theme["grid"],
            zerolinecolor=theme["grid"],
        ),
    )
    return fig


def plot_temporal_posterior(
    temporal_posterior: "TemporalPosterior",
    theme: str = "dark",
    show_bands: bool = True,
    show_prior: bool = True,
    height: int = 500,
) -> Any:
    """
    Plot temporal posterior with credible bands.

    Creates a timeline showing the posterior probability evolution with
    95% and 50% credible interval bands.

    Args:
        temporal_posterior: TemporalPosterior object
        theme: 'dark' or 'light'
        show_bands: Show credible interval bands
        show_prior: Show prior probability reference line
        height: Figure height in pixels

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    t = _get_theme(theme)
    fig = go.Figure()

    if not temporal_posterior.snapshots:
        fig.add_annotation(
            text="No temporal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=t["text"]),
        )
        return _apply_layout(fig, "⏳ Temporal Posterior", t)

    times = temporal_posterior.times
    values = temporal_posterior.values

    # 95% credible band
    if show_bands:
        band_95 = temporal_posterior.get_credible_band(0.95)
        fig.add_trace(go.Scatter(
            x=band_95.times + band_95.times[::-1],
            y=band_95.upper + band_95.lower[::-1],
            fill="toself",
            fillcolor=t["band_95"],
            line=dict(width=0),
            name="95% CI",
            showlegend=True,
            hoverinfo="skip",
        ))

        # 50% credible band
        band_50 = temporal_posterior.get_credible_band(0.50)
        fig.add_trace(go.Scatter(
            x=band_50.times + band_50.times[::-1],
            y=band_50.upper + band_50.lower[::-1],
            fill="toself",
            fillcolor=t["band_50"],
            line=dict(width=0),
            name="50% CI",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Posterior line
    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode="lines+markers",
        name="Posterior P(θ|E,t)",
        line=dict(color=t["posterior_line"], width=2.5),
        marker=dict(size=4, color=t["posterior_line"]),
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "Posterior: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Prior reference
    if show_prior:
        fig.add_hline(
            y=temporal_posterior.prior,
            line_dash="dash",
            line_color=t["prior_line"],
            annotation_text=f"Prior = {temporal_posterior.prior:.2f}",
            annotation_font=dict(color=t["prior_line"], size=11),
            annotation_position="bottom right",
        )

    # Decision thresholds
    fig.add_hline(y=0.5, line_dash="dot", line_color=t["grid"], opacity=0.5)

    _apply_layout(fig, "⏳ Temporal Posterior Evolution", t)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Posterior P(θ|E,t)",
        yaxis=dict(range=[0, 1]),
        height=height,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            bgcolor=t["annotation_bg"],
        ),
    )

    return fig


def plot_drift_timeline(
    drift_report: "BeliefDriftReport",
    temporal_posterior: Optional["TemporalPosterior"] = None,
    theme: str = "dark",
    height: int = 550,
) -> Any:
    """
    Plot belief drift timeline with inflection point annotations.

    Shows the posterior curve with highlighted inflection points and
    causal attribution annotations.

    Args:
        drift_report: BeliefDriftReport from BeliefDriftDetector
        temporal_posterior: Optional temporal posterior for curve overlay
        theme: 'dark' or 'light'
        height: Figure height in pixels

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    t = _get_theme(theme)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Posterior & Inflection Points", "Drift Magnitude"],
        vertical_spacing=0.12,
    )

    if not drift_report.inflections:
        fig.add_annotation(
            text="No inflection points detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=t["text"]),
        )
        return _apply_layout(fig, "📊 Belief Drift Analysis", t)

    # Plot posterior curve if available
    if temporal_posterior and temporal_posterior.snapshots:
        times = temporal_posterior.times
        values = temporal_posterior.values

        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name="Posterior",
            line=dict(color=t["posterior_line"], width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P=%{y:.3f}<extra></extra>",
        ), row=1, col=1)

    # Inflection points
    for ip in drift_report.inflections:
        color = t["inflection_up"] if ip.direction == "up" else t["inflection_down"]
        symbol = "triangle-up" if ip.direction == "up" else "triangle-down"

        fig.add_trace(go.Scatter(
            x=[ip.time],
            y=[ip.posterior_after],
            mode="markers",
            marker=dict(
                size=14,
                color=color,
                symbol=symbol,
                line=dict(width=2, color=t["bg"]),
            ),
            name=f"Inflection ({ip.direction})",
            showlegend=True,
            hovertemplate=(
                f"<b>Inflection ({ip.direction})</b><br>"
                f"Before: {ip.posterior_before:.3f}<br>"
                f"After: {ip.posterior_after:.3f}<br>"
                f"Magnitude: {ip.magnitude:.3f}<br>"
                "<extra></extra>"
            ),
        ), row=1, col=1)

        # Annotation with top causal evidence
        if ip.causal_attributions:
            top_cause = ip.causal_attributions[0]
            anno_text = (
                f"Δ={ip.magnitude:.2f} {ip.direction}<br>"
                f"Cause: {top_cause.evidence_text[:30]}..."
            )
            fig.add_annotation(
                x=ip.time, y=ip.posterior_after,
                text=anno_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                font=dict(size=9, color=t["text"]),
                bgcolor=t["annotation_bg"],
                bordercolor=color,
                borderwidth=1,
                row=1, col=1,
            )

    # Drift magnitude subplot
    for ip in drift_report.inflections:
        color = t["inflection_up"] if ip.direction == "up" else t["inflection_down"]
        fig.add_trace(go.Bar(
            x=[ip.time],
            y=[ip.magnitude],
            marker=dict(color=color),
            name=f"Δ={ip.magnitude:.3f}",
            showlegend=False,
            hovertemplate=(
                f"<b>{ip.time.strftime('%Y-%m-%d')}</b><br>"
                f"Magnitude: {ip.magnitude:.4f}<br>"
                "<extra></extra>"
            ),
        ), row=2, col=1)

    _apply_layout(fig, "📊 Belief Drift Analysis", t)
    fig.update_layout(
        height=height,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            bgcolor=t["annotation_bg"],
        ),
    )
    fig.update_yaxes(title_text="Posterior", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    return fig
