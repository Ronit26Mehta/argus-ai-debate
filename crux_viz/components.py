"""
Reusable Streamlit UI Components for CRUX-Viz.

Provides styled cards, badges, and custom CSS for a premium dark-theme look
with CRUX-specific magenta accent color and protocol-specific card types.
"""

from __future__ import annotations

import streamlit as st
from typing import Any


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def inject_custom_css():
    """Inject premium dark-themed CSS with CRUX magenta accents into Streamlit."""
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
        border-right: 1px solid #2a1840;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ff00d4;
    }

    /* Verdict card */
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

    .evidence-card.support { border-left-color: #00ff88; }
    .evidence-card.attack  { border-left-color: #ff4466; }

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

    /* Claim Bundle card */
    .cb-card {
        background: linear-gradient(135deg, #150d28 0%, #0e1117 100%);
        border: 1px solid #3a1a5a;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 5px solid #ff00d4;
        box-shadow: 0 2px 12px rgba(255, 0, 212, 0.10);
    }

    .cb-card.supports { border-left-color: #00ff88; }
    .cb-card.attacks  { border-left-color: #ff4466; }
    .cb-card.neutral  { border-left-color: #888888; }

    .cb-id {
        font-size: 10px;
        color: #606060;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 6px;
    }

    .cb-claim-text {
        font-size: 14px;
        color: #e0e0e0;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .cb-meta {
        font-size: 11px;
        color: #909090;
        font-family: 'JetBrains Mono', monospace;
    }

    /* BRP card */
    .brp-card {
        background: linear-gradient(135deg, #1a1230 0%, #0e1117 100%);
        border: 1px solid #4a2a7a;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 5px solid #b388ff;
        box-shadow: 0 2px 12px rgba(179, 136, 255, 0.10);
    }

    /* Auction card */
    .auction-card {
        background: linear-gradient(135deg, #1a1500 0%, #0e1117 100%);
        border: 1px solid #4a3a00;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 5px solid #ff8800;
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

    .badge-support   { background: rgba(0,255,136,0.15); color: #00ff88; }
    .badge-attack    { background: rgba(255,68,102,0.15); color: #ff4466; }
    .badge-neutral   { background: rgba(136,136,136,0.15); color: #888888; }
    .badge-rebuttal  { background: rgba(255,136,0,0.15); color: #ff8800; }
    .badge-crux      { background: rgba(255,0,212,0.15); color: #ff00d4; }
    .badge-brp       { background: rgba(179,136,255,0.15); color: #b388ff; }
    .badge-auction   { background: rgba(255,136,0,0.15); color: #ff8800; }
    .badge-edr       { background: rgba(153,102,255,0.15); color: #9966ff; }

    /* Round summary */
    .round-header {
        background: linear-gradient(90deg, #1a1f2e 0%, transparent 100%);
        border-left: 3px solid #ff00d4;
        padding: 8px 16px;
        font-size: 15px;
        font-weight: 600;
        color: #ff00d4;
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

    .agent-dot.active { background: #ff00d4; box-shadow: 0 0 6px #ff00d4; }
    .agent-dot.idle   { background: #555; }
    .agent-dot.error  { background: #ff4466; }

    /* Metrics row */
    .metrics-row { display: flex; gap: 16px; margin: 12px 0; }

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
        color: #ff00d4;
    }

    .metric-label {
        font-size: 11px;
        color: #808080;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* CRUX Session stats box */
    .crux-stats-box {
        background: linear-gradient(135deg, #150d28 0%, #1a1f2e 100%);
        border: 1px solid #3a1a5a;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #3a1a5a; border-radius: 3px; }
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
        <span class="badge" style="background: rgba(255,0,212,0.15); color: #ff00d4;">
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
    cb_count = round_data.get("claim_bundles_issued", 0)

    with st.expander(
        f"📌 Round {rnd} — Posterior: {posterior:.3f} | Evidence: {total_ev} | "
        f"CBs: {cb_count} | Rebuttals: {rebuttals}",
        expanded=False,
    ):
        cols = st.columns(5)
        cols[0].metric("Posterior", f"{posterior:.3f}")
        cols[1].metric("Support", f"+{support}", delta=f"+{support}", delta_color="normal")
        cols[2].metric("Attack", f"-{attack}", delta=f"-{attack}", delta_color="inverse")
        cols[3].metric("Rebuttals", f"{rebuttals}")
        cols[4].metric("Claim Bundles", f"{cb_count}")

        for e in round_data.get("evidence", []):
            render_evidence_card(e)


# ---------------------------------------------------------------------------
# Claim Bundle Card
# ---------------------------------------------------------------------------

def render_claim_bundle_card(cb: dict):
    """Render a CRUX Claim Bundle card."""
    polarity_raw = cb.get("polarity", 0)
    if isinstance(polarity_raw, str):
        polarity_val = 1 if polarity_raw.lower() in ("supports", "support") else (
            -1 if polarity_raw.lower() in ("attacks", "attack") else 0
        )
    else:
        polarity_val = int(polarity_raw)

    css_class = "supports" if polarity_val > 0 else ("attacks" if polarity_val < 0 else "neutral")
    badge_class = "badge-support" if polarity_val > 0 else ("badge-attack" if polarity_val < 0 else "badge-neutral")
    polarity_label = "SUPPORTS" if polarity_val > 0 else ("ATTACKS" if polarity_val < 0 else "NEUTRAL")

    cb_id = cb.get("cb_id", "")[:16] + "..."
    claim = cb.get("claim_text", "")[:200]
    posterior = cb.get("posterior", 0.5)
    credibility = cb.get("issuer_credibility", 0.0)
    agent = cb.get("issuer_agent", "Unknown")
    challenged = not cb.get("challenge_open", True)

    challenge_badge = (
        '<span class="badge" style="background:rgba(255,68,102,0.12);color:#ff4466;">CHALLENGED</span>'
        if challenged else
        '<span class="badge" style="background:rgba(255,191,0,0.12);color:#ffbf00;">OPEN</span>'
    )

    st.markdown(f"""
    <div class="cb-card {css_class}">
        <div class="cb-id">CB: {cb_id}</div>
        <span class="badge {badge_class}">{polarity_label}</span>
        <span class="badge badge-crux">{agent.split("-")[-1] if "-" in agent else agent}</span>
        {challenge_badge}
        <div class="cb-claim-text" style="margin-top: 8px;">{claim}</div>
        <div class="cb-meta">posterior: {posterior:.3f} | credibility: {credibility:.2f}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CRUX Session Stats Card
# ---------------------------------------------------------------------------

def render_crux_session_stats(stats: dict):
    """Render a CRUX session statistics overview box."""
    st.markdown("""
    <div class="crux-stats-box">
        <div style="font-size: 13px; color: #ff00d4; font-weight: 600; margin-bottom: 10px;">
            ⚡ CRUX SESSION STATISTICS
        </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].metric("Claim Bundles", stats.get("num_claim_bundles", 0))
    cols[1].metric("Challenges", stats.get("num_challenges", 0))
    cols[2].metric("BRP Sessions", stats.get("num_brp_sessions", 0))
    cols[3].metric("Auctions", stats.get("num_auctions", 0))

    col2 = st.columns(3)
    col2[0].metric("Unchallenged", stats.get("num_unchallenged", 0))
    col2[1].metric("Avg DFS", f"{stats.get('avg_dfs', 0.0):.3f}")
    col2[2].metric("Credibility Updates", stats.get("total_credibility_updates", 0))

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# BRP Session Card
# ---------------------------------------------------------------------------

def render_brp_session_card(brp: dict):
    """Render a Belief Reconciliation Protocol session card."""
    resolution = brp.get("resolution") or {}
    reconciled_posterior = resolution.get("reconciled_posterior", 0.5)
    contributors = resolution.get("contributor_agents", [])
    strategy = resolution.get("strategy", "weighted_average")
    original_ids = resolution.get("original_cb_ids", [])

    st.markdown(f"""
    <div class="brp-card">
        <span class="badge badge-brp">BRP MERGE</span>
        <div style="margin-top: 8px; color: #e0e0e0; font-size: 14px;">
            Reconciled <b>{len(original_ids)}</b> Claim Bundles →
            Posterior: <b style="color: #b388ff;">{reconciled_posterior:.3f}</b>
        </div>
        <div class="evidence-meta" style="margin-top: 6px;">
            Contributors: {", ".join(contributors[:4]) or "N/A"} | Strategy: {strategy}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Auction Card
# ---------------------------------------------------------------------------

def render_auction_card(auction: dict):
    """Render a Challenger Auction result card."""
    unchallenged = auction.get("unchallenged", True)
    winner = auction.get("winner") or "None"
    num_bids = auction.get("num_bids", 0)
    cb_id = (auction.get("cb_id") or "")[:16]

    status_badge = (
        '<span class="badge badge-neutral">UNCHALLENGED</span>'
        if unchallenged else
        '<span class="badge badge-auction">CHALLENGED</span>'
    )

    st.markdown(f"""
    <div class="auction-card">
        <span class="badge badge-auction">AUCTION</span>
        {status_badge}
        <div style="margin-top: 8px; color: #e0e0e0; font-size: 14px;">
            CB: <span style="font-family: 'JetBrains Mono', monospace; color: #ff8800;">{cb_id}...</span>
        </div>
        <div class="evidence-meta" style="margin-top: 6px;">
            Bids: {num_bids} | Winner: {winner.split("-")[-1] if winner and "-" in winner else winner}
        </div>
    </div>
    """, unsafe_allow_html=True)


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

def render_crux_config_summary(config: dict):
    """Render a summary of the current CRUX debate configuration."""
    st.markdown("""
    <div style="background: #150d28; border: 1px solid #3a1a5a; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
        <div style="font-size: 13px; color: #ff00d4; font-weight: 600; margin-bottom: 8px;">⚡ CRUX CONFIGURATION</div>
    """, unsafe_allow_html=True)

    details = [
        f"**Provider:** {config.get('provider', 'N/A')}",
        f"**Model:** {config.get('model', 'N/A')}",
        f"**Rounds:** {config.get('max_rounds', 3)}",
        f"**Specialists:** {config.get('num_specialists', 1)}",
        f"**Refuter:** {'✅' if config.get('refuter_enabled', True) else '❌'}",
        f"**Prior:** {config.get('prior', 0.5):.2f}",
        f"**Contradiction θ:** {config.get('contradiction_threshold', 0.20):.2f}",
        f"**EDR:** {'✅' if config.get('enable_edr', True) else '❌'}",
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
