"""
Reusable Streamlit UI Components for Argus-Viz.

Provides styled cards, badges, and custom CSS for a premium dark-theme look.
"""

from __future__ import annotations

import streamlit as st
from typing import Any


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def inject_custom_css():
    """Inject premium dark-themed CSS into Streamlit."""
    st.markdown("""
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #131829 100%);
        border-right: 1px solid #1e2640;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff;
    }

    /* Card styling */
    .verdict-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .verdict-card.supported {
        border-color: #00ff88;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.15);
    }

    .verdict-card.rejected {
        border-color: #ff4466;
        box-shadow: 0 4px 20px rgba(255, 68, 102, 0.15);
    }

    .verdict-card.undecided {
        border-color: #ffbf00;
        box-shadow: 0 4px 20px rgba(255, 191, 0, 0.15);
    }

    .verdict-label {
        font-size: 36px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 8px;
    }

    .verdict-posterior {
        font-size: 48px;
        font-weight: 300;
        font-family: 'JetBrains Mono', monospace;
    }

    .verdict-reasoning {
        font-size: 14px;
        color: #a0a0a0;
        margin-top: 16px;
        line-height: 1.6;
    }

    /* Evidence card */
    .evidence-card {
        background: #151a28;
        border: 1px solid #252a3a;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #888;
    }

    .evidence-card.support {
        border-left-color: #00ff88;
    }

    .evidence-card.attack {
        border-left-color: #ff4466;
    }

    .evidence-text {
        font-size: 14px;
        line-height: 1.5;
        color: #e0e0e0;
    }

    .evidence-meta {
        font-size: 12px;
        color: #808080;
        margin-top: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-support { background: rgba(0,255,136,0.15); color: #00ff88; }
    .badge-attack { background: rgba(255,68,102,0.15); color: #ff4466; }
    .badge-rebuttal { background: rgba(255,136,0,0.15); color: #ff8800; }

    /* Round summary */
    .round-header {
        background: linear-gradient(90deg, #1a1f2e 0%, transparent 100%);
        border-left: 3px solid #00d4ff;
        padding: 8px 16px;
        font-size: 15px;
        font-weight: 600;
        color: #00d4ff;
        margin: 12px 0 4px 0;
    }

    /* Agent status */
    .agent-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
    }

    .agent-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    .agent-dot.active { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
    .agent-dot.idle { background: #555; }
    .agent-dot.error { background: #ff4466; }

    /* Metrics row */
    .metrics-row {
        display: flex;
        gap: 16px;
        margin: 12px 0;
    }

    .metric-box {
        background: #151a28;
        border: 1px solid #252a3a;
        border-radius: 8px;
        padding: 12px 20px;
        flex: 1;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        color: #00d4ff;
    }

    .metric-label {
        font-size: 11px;
        color: #808080;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2a3050; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Verdict Card
# ---------------------------------------------------------------------------

def render_verdict_card(verdict: dict):
    """Render a large styled verdict metric card."""
    label = verdict.get("label", "undecided")
    posterior = verdict.get("posterior", 0.5)
    reasoning = verdict.get("reasoning", "")
    confidence = verdict.get("confidence", 0.0)

    color_map = {
        "supported": "#00ff88",
        "rejected": "#ff4466",
        "undecided": "#ffbf00",
    }
    color = color_map.get(label, "#ffbf00")

    st.markdown(f"""
    <div class="verdict-card {label}">
        <div class="verdict-label" style="color: {color};">{label}</div>
        <div class="verdict-posterior" style="color: {color};">{posterior:.3f}</div>
        <div style="color: #808080; font-size: 12px; margin-top: 4px;">
            Confidence: {confidence:.2f}
        </div>
        <div class="verdict-reasoning">{reasoning[:500]}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Evidence Card
# ---------------------------------------------------------------------------

def render_evidence_card(evidence: dict):
    """Render a single evidence item card."""
    polarity = evidence.get("polarity", 0)
    css_class = "support" if polarity > 0 else "attack"
    badge_class = "badge-support" if polarity > 0 else "badge-attack"
    badge_text = "SUPPORT" if polarity > 0 else "ATTACK"

    text = evidence.get("text", "")
    specialist = evidence.get("specialist", "Unknown")
    confidence = evidence.get("confidence", 0.0)
    explanation = evidence.get("explanation", "")

    st.markdown(f"""
    <div class="evidence-card {css_class}">
        <span class="badge {badge_class}">{badge_text}</span>
        <span class="badge" style="background: rgba(0,212,255,0.15); color: #00d4ff;">
            {specialist}
        </span>
        <div class="evidence-text" style="margin-top: 8px;">{text}</div>
        {"<div class='evidence-text' style='color: #999; font-style: italic; margin-top: 4px;'>" + explanation + "</div>" if explanation else ""}
        <div class="evidence-meta">confidence: {confidence:.2f}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Round Summary
# ---------------------------------------------------------------------------

def render_round_summary(round_data: dict):
    """Render a collapsible round summary."""
    rnd = round_data.get("round", 0)
    posterior = round_data.get("posterior_after", 0.5)
    total_ev = round_data.get("total_evidence", 0)
    support = round_data.get("support_count", 0)
    attack = round_data.get("attack_count", 0)
    rebuttals = round_data.get("total_rebuttals", 0)

    with st.expander(f"📌 Round {rnd} — Posterior: {posterior:.3f} | Evidence: {total_ev} | Rebuttals: {rebuttals}", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Posterior", f"{posterior:.3f}")
        cols[1].metric("Support", f"+{support}", delta=f"+{support}", delta_color="normal")
        cols[2].metric("Attack", f"-{attack}", delta=f"-{attack}", delta_color="inverse")
        cols[3].metric("Rebuttals", f"{rebuttals}")

        for e in round_data.get("evidence", []):
            render_evidence_card(e)


# ---------------------------------------------------------------------------
# Agent Status
# ---------------------------------------------------------------------------

def render_agent_status(agent_name: str, status: str = "idle"):
    """Render an agent activity indicator."""
    dot_class = status  # active, idle, error
    st.markdown(f"""
    <div class="agent-status">
        <span class="agent-dot {dot_class}"></span>
        <span style="color: #e0e0e0; font-size: 13px;">{agent_name}</span>
        <span style="color: #606060; font-size: 11px; margin-left: auto;">{status.upper()}</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Config Summary
# ---------------------------------------------------------------------------

def render_debate_config_summary(config: dict):
    """Render a summary of the current debate configuration."""
    st.markdown("""
    <div style="background: #151a28; border: 1px solid #252a3a; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
        <div style="font-size: 13px; color: #00d4ff; font-weight: 600; margin-bottom: 8px;">⚙️ DEBATE CONFIGURATION</div>
    """, unsafe_allow_html=True)

    details = [
        f"**Provider:** {config.get('provider', 'N/A')}",
        f"**Model:** {config.get('model', 'N/A')}",
        f"**Rounds:** {config.get('max_rounds', 3)}",
        f"**Specialists:** {config.get('num_specialists', 1)}",
        f"**Refuter:** {'✅' if config.get('refuter_enabled', True) else '❌'}",
        f"**Prior:** {config.get('prior', 0.5):.2f}",
    ]
    st.markdown(" | ".join(details))
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Metrics Row
# ---------------------------------------------------------------------------

def render_metrics_row(metrics: list[tuple[str, str, str]]):
    """
    Render a row of styled metric boxes.

    Args:
        metrics: List of (value, label, color) tuples.
    """
    cols = st.columns(len(metrics))
    for col, (value, label, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div style="background:#151a28;border:1px solid #252a3a;border-radius:8px;'
                f'padding:16px 20px;text-align:center;">'
                f'<div style="font-size:28px;font-weight:600;'
                f'font-family:JetBrains Mono,monospace;color:{color};">{value}</div>'
                f'<div style="font-size:11px;color:#808080;text-transform:uppercase;'
                f'letter-spacing:1px;margin-top:4px;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
