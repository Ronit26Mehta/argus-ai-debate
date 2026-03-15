"""
PHALANX Visualizations — dual-theme population posterior and bias heatmaps.
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
    from argus.phalanx.posterior import PopulationPosterior
    from argus.phalanx.consensus import PolarisationIndex

DARK_THEME = {
    "bg": "#0e1117", "paper": "#1a1f2e", "grid": "#2a3040",
    "text": "#e0e0e0", "title": "#00d4ff",
    "hist_color": "rgba(0, 212, 255, 0.6)", "hist_line": "#00d4ff",
    "mean_line": "#ff00d4", "median_line": "#ffbf00",
    "q_band": "rgba(0, 255, 136, 0.15)",
    "heatmap_low": "#1a1f2e", "heatmap_high": "#ff00d4",
    "anno_bg": "rgba(26, 31, 46, 0.9)",
}

LIGHT_THEME = {
    "bg": "#ffffff", "paper": "#f8f9fa", "grid": "#e0e0e0",
    "text": "#1a1a1a", "title": "#0066cc",
    "hist_color": "rgba(0, 102, 204, 0.5)", "hist_line": "#0066cc",
    "mean_line": "#cc0066", "median_line": "#cc8800",
    "q_band": "rgba(0, 170, 68, 0.12)",
    "heatmap_low": "#f0f0ff", "heatmap_high": "#cc0066",
    "anno_bg": "rgba(248, 249, 250, 0.9)",
}


def _get_theme(theme: str = "dark") -> dict:
    return LIGHT_THEME if theme == "light" else DARK_THEME


def _check_plotly():
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required: pip install plotly")


def plot_population_posterior(
    population_posterior: "PopulationPosterior",
    polarisation_index: Optional["PolarisationIndex"] = None,
    theme: str = "dark",
    bins: int = 30,
    height: int = 500,
) -> Any:
    """
    Plot population belief distribution.

    Shows histogram of beliefs with mean/median lines and
    quartile bands. Annotates polarisation index if available.

    Args:
        population_posterior: PopulationPosterior object
        polarisation_index: Optional PolarisationIndex for annotation
        theme: 'dark' or 'light'
        bins: Number of histogram bins
        height: Figure height in pixels

    Returns:
        Plotly Figure
    """
    _check_plotly()
    t = _get_theme(theme)
    fig = go.Figure()

    beliefs = population_posterior.beliefs
    if not beliefs:
        fig.add_annotation(
            text="No population data", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=t["text"]),
        )
        fig.update_layout(paper_bgcolor=t["bg"], plot_bgcolor=t["paper"])
        return fig

    # Histogram
    fig.add_trace(go.Histogram(
        x=beliefs, nbinsx=bins,
        marker=dict(color=t["hist_color"], line=dict(color=t["hist_line"], width=1)),
        name="Belief Distribution",
        hovertemplate="Belief: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    dist = population_posterior.distribution

    # Mean line
    fig.add_vline(x=dist.mean, line_dash="dash", line_color=t["mean_line"],
                  annotation_text=f"Mean = {dist.mean:.3f}",
                  annotation_font=dict(color=t["mean_line"], size=11))

    # Median line
    fig.add_vline(x=dist.median, line_dash="dot", line_color=t["median_line"],
                  annotation_text=f"Median = {dist.median:.3f}",
                  annotation_font=dict(color=t["median_line"], size=11))

    # IQR band
    fig.add_vrect(x0=dist.q25, x1=dist.q75, fillcolor=t["q_band"],
                  line_width=0, annotation_text="IQR")

    # Decision threshold
    fig.add_vline(x=0.5, line_dash="dot", line_color=t["grid"], opacity=0.5)

    # PI annotation
    if polarisation_index:
        pi = polarisation_index
        fig.add_annotation(
            text=(f"Polarisation Index: {pi.value:.3f}<br>"
                  f"({pi.interpretation.replace('_', ' ').title()})"),
            xref="paper", yref="paper", x=0.98, y=0.95,
            showarrow=False, font=dict(size=12, color=t["title"]),
            bgcolor=t["anno_bg"], bordercolor=t["title"], borderwidth=1,
        )

    title = f"👥 Population Posterior (N={population_posterior.size})"
    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"], size=12),
        title=dict(text=title, font=dict(size=18, color=t["title"])),
        xaxis=dict(title="Posterior Belief P(θ|E)", range=[0, 1],
                   gridcolor=t["grid"]),
        yaxis=dict(title="Count", gridcolor=t["grid"]),
        height=height, showlegend=False,
        margin=dict(l=60, r=30, t=60, b=50),
    )

    return fig


def plot_bias_heatmap(
    persona_data: list[dict[str, Any]],
    theme: str = "dark",
    height: int = 500,
) -> Any:
    """
    Plot persona × bias strength heatmap.

    Args:
        persona_data: List of dicts with 'name' and 'bias_strengths'
        theme: 'dark' or 'light'
        height: Figure height

    Returns:
        Plotly Figure
    """
    _check_plotly()
    t = _get_theme(theme)
    fig = go.Figure()

    if not persona_data:
        fig.add_annotation(
            text="No persona data", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=t["text"]),
        )
        fig.update_layout(paper_bgcolor=t["bg"], plot_bgcolor=t["paper"])
        return fig

    # Collect all bias names
    all_biases = set()
    for pd in persona_data:
        all_biases.update(pd.get("bias_strengths", {}).keys())
    bias_names = sorted(all_biases)

    if not bias_names:
        return fig

    # Build matrix (limit to 50 personas for readability)
    display_data = persona_data[:50]
    persona_names = [d.get("name", f"P-{i}") for i, d in enumerate(display_data)]
    z = []
    for pd in display_data:
        row = [pd.get("bias_strengths", {}).get(b, 0.0) for b in bias_names]
        z.append(row)

    fig.add_trace(go.Heatmap(
        z=z, x=bias_names, y=persona_names,
        colorscale=[[0, t["heatmap_low"]], [1, t["heatmap_high"]]],
        colorbar=dict(title="Bias Strength"),
        hovertemplate="Persona: %{y}<br>Bias: %{x}<br>Strength: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["paper"],
        font=dict(family="Inter, sans-serif", color=t["text"], size=12),
        title=dict(text="🧠 Cognitive Bias Heatmap", font=dict(size=18, color=t["title"])),
        xaxis=dict(title="Cognitive Bias"),
        yaxis=dict(title="Persona", autorange="reversed"),
        height=height, margin=dict(l=100, r=30, t=60, b=50),
    )

    return fig
