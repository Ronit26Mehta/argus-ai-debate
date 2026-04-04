"""
HANNIBAL War Room — Three-Panel Streamlit Interface.

Layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Battle Status Bar (full width)                                  │
    ├────────────────┬───────────────────────┬─────────────────────────┤
    │  LEFT  (30%)   │  CENTER (40%)         │  RIGHT (30%)           │
    │  Command Post  │  Battle Chamber       │  Analytics Suite       │
    │  - Input       │  - Live combat feed   │  - Force Posteriors    │
    │  - Battle Map  │  - WhatsApp bubbles   │  - Tournament Tree     │
    │  - Controls    │  - Phase indicator    │  - Evidence Heatmap    │
    │  - Verdict     │                       │  - Scorecards          │
    └────────────────┴───────────────────────┴─────────────────────────┘

Incremental live updates via st.empty() placeholders.

Launch:
    $ streamlit run argus/hannibal/hannibal_app.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from html import escape as _html_escape
from typing import Any

os.environ.setdefault("ARGUS_DEFAULT_PROVIDER", "gemini")
os.environ.setdefault("ARGUS_DEFAULT_MODEL", "gemini-2.0-flash")

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="ARGUS × HANNIBAL",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from argus.hannibal.models import (
    CampaignPhase,
    ForceType,
    HannibalResult,
    HannibalSessionConfig,
)
from argus.hannibal import HANNIBAL

logger = logging.getLogger(__name__)

# ── War Room Dark-theme CSS ───────────────────────────────────────────

_CSS = """
<style>
    .stApp { background-color: #0B141A; color: #E9EDEF; }
    div[data-testid="stSidebar"] { background: #111B21; }

    /* Panel boundaries */
    div[data-testid="column"] {
        background-color: #111B21;
        border: 1px solid #2A3942;
        border-radius: 8px;
        padding: 10px 16px 16px 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .battle-status-bar {
        background: linear-gradient(135deg, #1a0a0a, #2a1515);
        border: 1px solid #5C2020;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 14px;
        color: #E9EDEF;
        margin-bottom: 12px;
    }
    .battle-status-bar .dot-green  { color: #2ECC71; }
    .battle-status-bar .dot-amber  { color: #FFA500; }
    .battle-status-bar .dot-red    { color: #E74C3C; }
    .battle-status-bar .dot-gold   { color: #FFD700; }

    .battle-map-card {
        background: linear-gradient(135deg, #1a1a1a, #2d1a1a);
        border: 1px solid #5C2020;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 12.5px;
        font-family: monospace;
        white-space: pre-wrap;
        color: #E9EDEF;
    }

    .wa-msg-row { display: flex; margin: 3px 8px; align-items: flex-end; }
    .wa-msg-row.wa-left  { justify-content: flex-start; }
    .wa-msg-row.wa-right { justify-content: flex-end; }
    .wa-avatar {
        width: 28px; height: 28px; border-radius: 50%;
        background: #2A3942; display: flex; align-items: center;
        justify-content: center; font-size: 16px;
        margin-right: 6px; flex-shrink: 0;
    }
    .wa-bubble {
        max-width: 88%; padding: 6px 10px 4px 10px; border-radius: 8px;
        position: relative; word-wrap: break-word;
        box-shadow: 0 1px 1px rgba(0,0,0,0.25);
    }
    .wa-bubble-left { background: #1F2C34; border-top-left-radius: 0; color: #E9EDEF; }
    .wa-bubble-right { background: #005C4B; border-top-right-radius: 0; color: #E9EDEF; }
    .wa-name { font-size: 12px; font-weight: 600; color: #2ECC71; margin-bottom: 2px; }
    .wa-text { font-size: 13.5px; line-height: 1.45; word-break: break-word; }
    .wa-time { font-size: 10px; color: #667781; text-align: right; margin-top: 2px; }

    .force-badge {
        display: inline-block; font-size: 10px; font-weight: 700;
        padding: 1px 8px; border-radius: 10px; margin-right: 5px;
    }
    .force-pf { background: #2ECC71; color: #0B141A; }
    .force-of { background: #E74C3C; color: white; }
    .force-ff { background: #3498DB; color: white; }

    .result-card {
        background: linear-gradient(135deg, #1a0a0a, #2a1a15);
        border: 1px solid #FFD700;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    }
    .result-card h3 { color: #FFD700; margin: 0 0 8px 0; }

    .posterior-gauge {
        background: #1F2C34; border-radius: 4px;
        padding: 6px 10px; margin: 4px 0; font-size: 12px;
    }
    .posterior-bar {
        height: 8px; border-radius: 4px;
        transition: width 0.3s ease;
    }

    .live-metric {
        background: #1F2C34; border: 1px solid #2A3942;
        border-radius: 6px; padding: 8px 12px; margin: 4px 0;
        font-size: 12px; color: #E9EDEF;
    }
    .live-metric .label { color: #667781; font-size: 11px; }
    .live-metric .value { font-size: 18px; font-weight: 700; }
</style>
"""

# ═══════════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════════

_DEFAULTS: dict[str, Any] = {
    "messages": [],
    "result": None,
    "mode": "input",
    "running": False,
    # Live tracking (accumulated during campaign)
    "_live_posteriors": {},
    "_live_skirmish_count": 0,
    "_live_total_skirmishes": 0,
    "_live_skirmish_log": [],
    "_live_tree_state": None,
    # LLM config
    "provider": os.environ.get("ARGUS_DEFAULT_PROVIDER", "gemini"),
    "model": os.environ.get("ARGUS_DEFAULT_MODEL", "gemini-2.0-flash"),
    "model_source": "env",
    "local_server_url": "http://localhost:8080",
    "local_model_name": "local-model",
    "local_server_status": "",
    "max_skirmish_rounds": 3,
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════════
# Sidebar — LLM Configuration
# ═══════════════════════════════════════════════════════════════════════

def _test_local_connection(url: str) -> str:
    import urllib.request
    for endpoint in ["/api/tags", "/v1/models"]:
        test_url = url.rstrip("/") + endpoint
        try:
            req = urllib.request.Request(test_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("models", data.get("data", []))
                names = [m.get("name", m.get("id", "?")) for m in models]
                if names:
                    return f"✅ Connected — models: {', '.join(names[:5])}"
                return "✅ Connected (no models listed)"
        except Exception:
            continue
    return "❌ Connection failed. Check server URL."


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("⚔️ HANNIBAL Configuration")

        model_source = st.radio(
            "Model Source",
            options=["env", "local"],
            format_func=lambda x: "🌐 Cloud / Registry" if x == "env" else "🖥️ Local LLM",
            index=0 if st.session_state.get("model_source", "env") == "env" else 1,
            key="sb_model_source",
        )
        st.session_state["model_source"] = model_source
        st.divider()

        if model_source == "local":
            st.subheader("🖥️ Local Server")
            local_url = st.text_input(
                "Server URL", value=st.session_state["local_server_url"],
                key="sb_local_url",
            )
            st.session_state["local_server_url"] = local_url
            local_model = st.text_input(
                "Model Name", value=st.session_state["local_model_name"],
                key="sb_local_model",
            )
            st.session_state["local_model_name"] = local_model
            if st.button("🔌 Test Connection", width='stretch'):
                with st.spinner("Testing…"):
                    status = _test_local_connection(local_url)
                st.session_state["local_server_status"] = status
            status_msg = st.session_state.get("local_server_status", "")
            if status_msg:
                (st.success if status_msg.startswith("✅") else st.error)(status_msg)
            st.caption("Supports Ollama, llama-server, vLLM, LM Studio.")
        else:
            st.subheader("🌐 Provider & Model")
            try:
                from argus.core.llm import list_providers
                available = list_providers()
            except Exception:
                available = ["openai", "anthropic", "gemini", "ollama",
                             "groq", "mistral", "deepseek"]
            provider = st.selectbox("Provider", available, key="sb_provider",
                index=available.index(st.session_state["provider"])
                if st.session_state["provider"] in available else 0,
            )
            model = st.text_input("Model", value=st.session_state["model"],
                                   key="sb_model")
            st.session_state["provider"] = provider
            st.session_state["model"] = model

        st.divider()
        st.header("⚙️ Battle Settings")
        max_rounds = st.slider(
            "Max Skirmish Rounds", min_value=2, max_value=3,
            value=st.session_state.get("max_skirmish_rounds", 3),
            help="2 = faster, 3 = deeper battles",
            key="sb_max_rounds",
        )
        st.session_state["max_skirmish_rounds"] = max_rounds
        st.divider()
        st.caption("ARGUS × HANNIBAL v1.0")


# ═══════════════════════════════════════════════════════════════════════
# LLM Factory
# ═══════════════════════════════════════════════════════════════════════

def _get_llm():
    if st.session_state.get("model_source") == "local":
        from argus.core.llm.openai import OpenAILLM
        base_url = st.session_state.get("local_server_url", "http://localhost:8080").rstrip("/")
        model_name = st.session_state.get("local_model_name", "local-model")
        return OpenAILLM(
            model=model_name,
            base_url=f"{base_url}/v1",
            api_key="not-needed",
            max_tokens=2048, timeout=300.0,
        )
    from argus.core.llm import get_llm
    return get_llm(
        provider=st.session_state["provider"],
        model=st.session_state["model"],
    )


# ═══════════════════════════════════════════════════════════════════════
# Status Bar
# ═══════════════════════════════════════════════════════════════════════

def _render_status_bar() -> None:
    mode = st.session_state["mode"]
    if mode == "input":
        html = '<span class="dot-green">●</span> War Room Ready — enter a proposition to begin'
    elif mode == "running":
        sc = st.session_state.get("_live_skirmish_count", 0)
        total = st.session_state.get("_live_total_skirmishes", 0)
        html = (f'<span class="dot-amber">●</span> Campaign in progress — '
                f'skirmish {sc}/{total}')
    elif mode == "complete":
        result: HannibalResult | None = st.session_state.get("result")
        if result:
            v = result.verdict
            html = (
                f'<span class="dot-gold">★</span> Campaign Complete — '
                f'{v.verdict_label.value} ({v.winning_force.display_name}, '
                f'{v.campaign_strength_label.value})')
        else:
            html = '<span class="dot-gold">★</span> Campaign Complete'
    else:
        html = '<span class="dot-green">●</span> HANNIBAL Ready'
    st.markdown(f'<div class="battle-status-bar">{html}</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Chat Helpers
# ═══════════════════════════════════════════════════════════════════════

def _msg_html(msg: dict) -> str:
    role = msg["role"]
    content = _html_escape(msg.get("content", "")).replace("\n", "<br>")
    ts = msg.get("timestamp", "")
    label = msg.get("agent_label", "HANNIBAL" if role == "assistant" else "You")

    if role == "user":
        return (
            f'<div class="wa-msg-row wa-right">'
            f'<div class="wa-bubble wa-bubble-right">'
            f'<div class="wa-text">{content}</div>'
            f'<div class="wa-time">{ts}</div>'
            f'</div></div>')
    return (
        f'<div class="wa-msg-row wa-left">'
        f'<div class="wa-avatar">⚔️</div>'
        f'<div class="wa-bubble wa-bubble-left">'
        f'<div class="wa-name">{_html_escape(label)}</div>'
        f'<div class="wa-text">{content}</div>'
        f'<div class="wa-time">{ts}</div>'
        f'</div></div>')


def _add_msg(role: str, content: str, agent_label: str = "HANNIBAL") -> None:
    now = datetime.now(timezone.utc)
    st.session_state["messages"].append({
        "role": role,
        "content": content,
        "timestamp": now.strftime("%I:%M %p"),
        "agent_label": agent_label if role == "assistant" else "You",
    })


def _render_all_chat_html() -> str:
    """Build one big HTML block for all messages — fast re-render."""
    parts = [_msg_html(m) for m in st.session_state["messages"]]
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Live Plotly Builders (lightweight, inline)
# ═══════════════════════════════════════════════════════════════════════

_FORCE_COLORS = {
    "proposition": "#2ECC71",
    "opposition": "#E74C3C",
    "faction_1": "#3498DB",
    "faction_2": "#E67E22",
    "faction_3": "#9B59B6",
}

def _build_live_posterior_fig(posteriors: dict[str, list[float]]):
    """Build a compact Force Posterior timeline from live data."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure()
    for force_val, history in posteriors.items():
        if not history:
            continue
        try:
            ft = ForceType(force_val)
            name = ft.display_name
            color = _FORCE_COLORS.get(force_val, "#888888")
        except ValueError:
            name = force_val
            color = "#888888"
        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=history,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
        ))
    fig.update_layout(
        template="plotly_dark",
        title="Force Posterior Timeline",
        height=260,
        xaxis_title="Skirmish",
        yaxis_title="Posterior",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=10, t=35, b=30),
        showlegend=True,
        legend=dict(font=dict(size=9)),
        plot_bgcolor="#0B141A",
        paper_bgcolor="#0B141A",
    )
    return fig


def _build_live_skirmish_bar(skirmish_log: list[dict]):
    """Build a bar chart showing skirmish winners and confidence."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not skirmish_log:
        return None

    labels = [s.get("label", f"S{i+1}")[:12] for i, s in enumerate(skirmish_log)]
    confs = [s.get("confidence", 0) for s in skirmish_log]
    colors = [_FORCE_COLORS.get(s.get("winner", ""), "#888") for s in skirmish_log]

    fig = go.Figure(go.Bar(
        x=labels, y=confs, marker_color=colors,
        text=[f"{c:.2f}" for c in confs],
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Skirmish Results (Confidence)",
        height=220,
        xaxis_title="", yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=10, t=35, b=30),
        plot_bgcolor="#0B141A",
        paper_bgcolor="#0B141A",
    )
    return fig


def _build_live_tree_fig(tree_state: dict):
    """Build the Tournament Tree bracket from live tree state data."""
    if not tree_state:
        return None
    try:
        from argus.hannibal.war_room import TournamentTreeViz
        viz = TournamentTreeViz(tree_state)
        return viz.build_figure()
    except Exception:
        return None

def _build_live_dag_fig(dag):
    """Build the Battle DAG figure."""
    if dag is None:
        return None
    try:
        return dag.build_figure()
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════
# Campaign Execution — with live placeholder updates
# ═══════════════════════════════════════════════════════════════════════

def _run_campaign(
    proposition: str,
    chat_placeholder,
    status_placeholder,
    posterior_placeholder,
    skirmish_placeholder,
    tree_placeholder,
    dag_placeholder,
    progress_placeholder,
    transcript_placeholder,
) -> None:
    """Run the full campaign with incremental UI updates."""
    llm = _get_llm()
    config = HannibalSessionConfig(
        max_skirmish_rounds=st.session_state.get("max_skirmish_rounds", 3),
    )

    # Reset live state
    st.session_state["_live_posteriors"] = {}
    st.session_state["_live_skirmish_count"] = 0
    st.session_state["_live_total_skirmishes"] = 0
    st.session_state["_live_skirmish_log"] = []
    st.session_state["_live_transcripts"] = []
    st.session_state["_live_force_names"] = {}
    st.session_state["_live_tree_state"] = None
    
    from argus.hannibal.dag_viz import BattleDAG
    dag = BattleDAG()
    dag.add_proposition(proposition)
    st.session_state["_live_dag"] = dag
    
    st.session_state["mode"] = "running"
    st.session_state["running"] = True

    _add_msg("assistant",
             "Initiating HANNIBAL campaign — analysing proposition depth…",
             "HANNIBAL · PDA")
    chat_placeholder.markdown(_render_all_chat_html(), unsafe_allow_html=True)

    campaign = HANNIBAL(llm=llm, config=config)

    def _update_chat():
        chat_placeholder.markdown(_render_all_chat_html(), unsafe_allow_html=True)

    def _update_posteriors():
        posteriors = st.session_state["_live_posteriors"]
        if not posteriors:
            return
        fig = _build_live_posterior_fig(posteriors)
        if fig:
            posterior_placeholder.plotly_chart(fig, width='stretch',
                                               key=f"lp_{time.time()}")

    def _update_skirmish_bar():
        log = st.session_state["_live_skirmish_log"]
        if not log:
            return
        fig = _build_live_skirmish_bar(log)
        if fig:
            skirmish_placeholder.plotly_chart(fig, width='stretch',
                                              key=f"sb_{time.time()}")

    def _update_progress():
        sc = st.session_state["_live_skirmish_count"]
        total = st.session_state["_live_total_skirmishes"]
        if total > 0:
            pct = sc / total
            progress_placeholder.progress(pct, text=f"⚔️ Skirmish {sc}/{total}")
        else:
            progress_placeholder.empty()

    def _update_tree():
        tree_state = st.session_state.get("_live_tree_state")
        if not tree_state:
            return
        fig = _build_live_tree_fig(tree_state)
        if fig:
            tree_placeholder.plotly_chart(fig, width='stretch',
                                          key=f"lt_{time.time()}")

    def _update_dag():
        dag_inst = st.session_state.get("_live_dag")
        if not dag_inst:
            return
        fig = _build_live_dag_fig(dag_inst)
        if fig:
            dag_placeholder.plotly_chart(fig, width='stretch',
                                         key=f"dag_{time.time()}")

    def _update_transcripts():
        t_list = st.session_state.get("_live_transcripts", [])
        if not t_list:
            return
        
        # Build markdown transcript
        out = ["#### 📜 Agent Transcripts & Evidence"]
        for t in reversed(t_list):
            out.append(f"**{t['label']}** (Winner: {t['winner']})")
            out.append(f"> *{t['adjudication']}*")
            out.append("")
            # Evidence A
            for e in t.get("evidence_a", []):
                out.append(f"- **[Support] {e.get('agent_name')}** ({e.get('force_type')}): {e.get('claim_text')}")
            # Evidence B
            for e in t.get("evidence_b", []):
                out.append(f"- **[Counter] {e.get('agent_name')}** ({e.get('force_type')}): {e.get('claim_text')}")
            out.append("---")
            
        transcript_placeholder.markdown("\n".join(out))

    def _phase_callback(phase, label, details):
        """Live update all panels on each campaign event."""

        if phase == CampaignPhase.ANALYSIS and details.get("status") == "complete":
            bm = details.get("battle_map", {})
            polarity = bm.get("polarity", "bipolar").upper()
            num_forces = bm.get("num_forces", 2)
            depth = bm.get("depth_score", {}).get("aggregate", 0)
            
            # Format detailed breakdown
            msg = [f"**Battle Map ready:** {polarity} ({num_forces} forces), depth={depth:.2f}\n"]
            msg.append("🛡️ **Identified Factions:**")
            stances = bm.get("faction_positions", {})
            names = bm.get("faction_names", {})
            for lbl, stance in stances.items():
                dyn_name = names.get(lbl, lbl.upper())
                msg.append(f"  - **{dyn_name}**: *{stance}*")
                
            _add_msg("assistant", "\n".join(msg), "HANNIBAL · PDA")
            _update_chat()

        elif phase == CampaignPhase.DEPLOYMENT and details.get("status") == "complete":
            n_forces = details.get("num_forces", 0)
            n_agents = details.get("total_agents", 0)
            
            msg = [f"**Forces finally deployed:** {n_forces} forces commanding {n_agents} agents total.\n"]
            for f_data in details.get("forces_data", []):
                fname = f_data.get("force_name", f_data.get("force_type", "").upper())
                msg.append(f"⚔️ **{fname}** ({len(f_data.get('agents', []))} units)")
                for ag in f_data.get("agents", []):
                    msg.append(f"  - `{ag.get('role_abbr', '?')}` {ag.get('name', 'Agent')} ({ag.get('domain_expertise', 'General')})")
                    
            _add_msg("assistant", "\n".join(msg), "HANNIBAL · Force Deployment")
            _update_chat()
            
            # DAG: forces and agents
            dag_inst = st.session_state.get("_live_dag")
            if dag_inst:
                for f_data in details.get("forces_data", []):
                    ft_val = f_data.get("force_type", "")
                    f_name = f_data.get("force_name", "")
                    st.session_state["_live_force_names"][ft_val] = f_name
                    dag_inst.add_force(ft_val, position=f_data.get("position_description", ""), agent_count=f_data.get("force_size", 0), force_name=f_name)
                    for ag in f_data.get("agents", []):
                        dag_inst.add_agent(ag.get("id", ""), ag.get("name", ""), ft_val, role=ag.get("role", ""))
                _update_dag()

        elif phase == CampaignPhase.BATTLE:
            if details.get("status") == "constructed":
                total = details.get("total_skirmishes", 0)
                st.session_state["_live_total_skirmishes"] = total
                _add_msg("assistant",
                         f"Tournament Tree built: {total} skirmishes scheduled.",
                         "HANNIBAL · Tournament")
                _update_chat()
                _update_progress()
                return

            if details.get("type") == "skirmish":
                winner = details.get("winner", "?")
                conf = details.get("confidence", 0)
                progress = details.get("progress", 0)

                # 1) Track posteriors FIRST, then render
                posteriors = details.get("force_posteriors", {})
                for fv, val in posteriors.items():
                    st.session_state["_live_posteriors"].setdefault(fv, [])
                    st.session_state["_live_posteriors"][fv].append(val)

                # 2) Track skirmish results
                sc = st.session_state["_live_skirmish_count"] + 1
                st.session_state["_live_skirmish_count"] = sc
                st.session_state["_live_skirmish_log"].append({
                    "label": label,
                    "winner": winner,
                    "confidence": conf,
                })

                # 3) Chat message
                try:
                    winner_name = ForceType(winner).abbreviation
                except ValueError:
                    winner_name = winner[:3].upper()
                    
                dyn_winner_name = st.session_state["_live_force_names"].get(winner, "")
                if dyn_winner_name:
                    winner_name = dyn_winner_name
                    
                _add_msg("assistant",
                         f"{label}: Winner={winner_name} (conf={conf:.2f}) "
                         f"[{progress:.0%}]",
                         "HANNIBAL · Battle")

                # Transcripts
                ev_a = details.get("evidence_a", [])
                ev_b = details.get("evidence_b", [])
                adj = details.get("adjudication_summary", "")
                st.session_state["_live_transcripts"].append({
                    "label": label,
                    "winner": winner_name,
                    "evidence_a": ev_a,
                    "evidence_b": ev_b,
                    "adjudication": adj
                })

                # DAG update
                dag_inst = st.session_state.get("_live_dag")
                if dag_inst:
                    for ev in ev_a + ev_b:
                        dag_inst.add_evidence(
                            evidence_id=ev.get("evidence_id", f"ev_{time.time()}_{ev.get('agent_name')}"),
                            agent_id=ev.get("agent_id", ""),
                            agent_name=ev.get("agent_name", ""),
                            force_type=ev.get("force_type", ""),
                            claim=ev.get("claim_text", ""),
                            evid_q=ev.get("effective_weight", 0.5),
                            is_counter=ev.get("is_counter_evidence", False)
                        )

                    from argus.hannibal.models import _uid
                    sid = _uid("skr")
                    
                    # Try to capture the two competing forces for edge linking
                    fa_val = ev_a[0].get("force_type") if ev_a else "proposition"
                    fb_val = ev_b[0].get("force_type") if ev_b else "opposition"
                    
                    dag_inst.add_skirmish_result(sid, label, winner, conf, fa_val, fb_val, winner_name_override=dyn_winner_name)
                    _update_dag()

                # 4) Update ALL live panels
                _update_chat()
                _update_posteriors()
                _update_skirmish_bar()
                _update_progress()
                _update_transcripts()

                # 5) Refresh tree (grab live state from campaign)
                try:
                    tree_state = campaign._last_tree_state
                    if tree_state:
                        st.session_state["_live_tree_state"] = tree_state
                        _update_tree()
                except AttributeError:
                    pass

        elif phase == CampaignPhase.RESOLUTION:
            if details.get("type") in ("engagement", "theatre"):
                winner = details.get("winner", "")
                try:
                    winner_name = ForceType(winner).abbreviation
                except ValueError:
                    winner_name = winner[:3].upper()
                _add_msg("assistant",
                         f"{label} resolved: {winner_name} prevails.",
                         "HANNIBAL · Resolution")
                _update_chat()

        elif phase == CampaignPhase.COMPLETE:
            _add_msg("assistant",
                     "Campaign complete — assembling results…",
                     "HANNIBAL · Field Marshal")
            _update_chat()
            progress_placeholder.progress(1.0, text="✅ Campaign Complete")
            
            dag_inst = st.session_state.get("_live_dag")
            if dag_inst:
                verdict_data = details.get("verdict", {})
                winner_val = verdict_data.get("winning_force") or "proposition"
                strength = verdict_data.get("campaign_strength_score", 0.9)
                dag_inst.add_verdict(verdict_data.get("verdict_label", "Supported"), winner_val, strength)
                _update_dag()

    # ── Execute ───────────────────────────────────────────────────
    result = campaign.run(proposition, phase_callback=_phase_callback)

    # Ensure live posteriors are stored on result
    if st.session_state["_live_posteriors"]:
        result.force_posterior_history = dict(st.session_state["_live_posteriors"])

    st.session_state["result"] = result
    st.session_state["mode"] = "complete"
    st.session_state["running"] = False

    _add_msg("assistant", result.chat_card(), "HANNIBAL · Campaign Verdict")
    _update_chat()


# ═══════════════════════════════════════════════════════════════════════
# Main Layout
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_sidebar()
    _render_status_bar()

    # ── Three-panel layout ────────────────────────────────────────
    left_col, center_col, right_col = st.columns([0.28, 0.42, 0.30])

    # ── CENTER: Battle Chamber ────────────────────────────────────
    with center_col:
        st.subheader("⚔️ Battle Chamber")
        chat_placeholder = st.empty()
        st.divider()
        transcript_placeholder = st.empty()
        
        # Render existing messages
        if st.session_state["messages"]:
            chat_placeholder.markdown(
                _render_all_chat_html(), unsafe_allow_html=True)
                
        # Final transcripts
        t_list = st.session_state.get("_live_transcripts", [])
        if t_list:
            out = ["#### 📜 Agent Transcripts & Evidence"]
            for t in reversed(t_list):
                out.append(f"**{t['label']}** (Winner: {t['winner']})")
                out.append(f"> *{t['adjudication']}*")
                out.append("")
                for e in t.get("evidence_a", []):
                    out.append(f"- **[Support] {e.get('agent_name')}** ({e.get('force_type')}): {e.get('claim_text')}")
                for e in t.get("evidence_b", []):
                    out.append(f"- **[Counter] {e.get('agent_name')}** ({e.get('force_type')}): {e.get('claim_text')}")
                out.append("---")
            transcript_placeholder.markdown("\n".join(out))

    # ── RIGHT: Analytics (placeholders for live updates) ──────────
    with right_col:
        st.subheader("📊 Analytics Suite")
        progress_placeholder = st.empty()
        posterior_placeholder = st.empty()
        skirmish_placeholder = st.empty()
        tree_placeholder = st.empty()
        dag_placeholder = st.empty()

        # If we're in complete state, show final analytics
        result = st.session_state.get("result")
        if result:
            progress_placeholder.progress(1.0, text="✅ Campaign Complete")

            # Final posterior timeline
            if result.force_posterior_history:
                fig = _build_live_posterior_fig(result.force_posterior_history)
                if fig:
                    posterior_placeholder.plotly_chart(
                        fig, width='stretch', key="final_fp")

            # Final skirmish bar
            slog = st.session_state.get("_live_skirmish_log", [])
            if slog:
                fig = _build_live_skirmish_bar(slog)
                if fig:
                    skirmish_placeholder.plotly_chart(
                        fig, width='stretch', key="final_sb")

            # Final Tournament Tree
            tree_state = result.battle_map_summary.get("tree_state")
            if tree_state:
                fig = _build_live_tree_fig(tree_state)
                if fig:
                    tree_placeholder.plotly_chart(
                        fig, width='stretch', key="final_tree")

            # Final Battle DAG
            dag_inst = st.session_state.get("_live_dag")
            if dag_inst:
                fig = _build_live_dag_fig(dag_inst)
                if fig:
                    dag_placeholder.plotly_chart(
                        fig, width='stretch', key="final_dag")

            # Evidence EVID-Q
            if result.decisive_evidence and result.decisive_evidence.items:
                with st.expander("⚖️ Evidence EVID-Q Scores", expanded=False):
                    try:
                        from argus.hannibal.war_room import build_evidence_heatmap
                        ev_data = [e.to_dict() for e in result.decisive_evidence.items]
                        fig = build_evidence_heatmap(ev_data)
                        st.plotly_chart(fig, width='stretch',
                                        key="evid_q_chart")
                    except Exception:
                        st.caption("Evidence chart unavailable.")

            # Force Scorecards
            if result.scorecards:
                with st.expander("🎯 Force Scorecards", expanded=False):
                    for sc in result.scorecards:
                        w, l, d = sc.skirmishes_won, sc.skirmishes_lost, sc.skirmishes_drawn
                        st.markdown(
                            f"**{sc.force_type.display_name}** — "
                            f"W:{w} L:{l} D:{d} | "
                            f"Evidence: {sc.evidence_submitted} | "
                            f"Avg EVID-Q: `{sc.avg_evid_q:.3f}` | "
                            f"BES: `{sc.battle_efficiency_score:.4f}`")

            # CANNAE Matrix
            if result.encirclement_report:
                with st.expander("🔄 CANNAE Dominance Matrix", expanded=False):
                    try:
                        from argus.hannibal.war_room import build_cannae_matrix
                        matrix = result.encirclement_report.get("dominance_matrix", {})
                        labels = result.encirclement_report.get("force_labels", [])
                        if matrix and labels:
                            fig = build_cannae_matrix(matrix, labels)
                            st.plotly_chart(fig, width='stretch',
                                            key="cannae_matrix")
                    except Exception:
                        pass

            # What Would Change
            if result.what_would_change:
                with st.expander("🔮 What Would Change This Verdict", expanded=False):
                    for i, item in enumerate(result.what_would_change, 1):
                        st.write(f"{i}. {item}")

            # Campaign Log
            if result.campaign_log:
                with st.expander("📖 Campaign Log (Field Manual)", expanded=False):
                    for entry in result.campaign_log[-25:]:
                        st.text(entry.to_field_manual_line())
                    st.caption(f"Log seal: {result.log_seal_hash[:16]}…")

        elif st.session_state["mode"] == "input":
            st.caption("📊 Enter a proposition to begin the campaign.\n\n"
                       "Live charts will appear here during battle.")

    # ── LEFT: Command Post ────────────────────────────────────────
    with left_col:
        st.subheader("🏰 Command Post")
        mode = st.session_state["mode"]

        if mode == "input":
            proposition = st.text_area(
                "Proposition",
                placeholder=(
                    "Enter a proposition for battle…\n"
                    "e.g., 'Nuclear fusion will be commercially viable by 2040'"),
                height=100,
                key="proposition_input",
            )
            st.info("⚔️ **Sequential mode** — optimised for your hardware.")

            if st.button("🚀 Launch Campaign", width='stretch',
                          type="primary"):
                if proposition and proposition.strip():
                    _add_msg("user", proposition.strip())
                    _run_campaign(
                        proposition.strip(),
                        chat_placeholder,
                        st.empty(),  # status (unused in this branch)
                        posterior_placeholder,
                        skirmish_placeholder,
                        tree_placeholder,
                        dag_placeholder,
                        progress_placeholder,
                        transcript_placeholder,
                    )
                    st.rerun()
                else:
                    st.warning("Please enter a proposition.")

        elif mode == "complete":
            result = st.session_state.get("result")
            if result:
                v = result.verdict
                st.markdown(
                    f'<div class="result-card">'
                    f'<h3>{v.verdict_label.value.upper()}</h3>'
                    f'<p>Winner: {v.winning_force.display_name}</p>'
                    f'<p>Strength: {v.campaign_strength_label.value} '
                    f'({v.campaign_strength_score:.0%})</p>'
                    f'<p>Skirmishes: {result.num_skirmishes} | '
                    f'Evidence: {result.total_evidence}</p>'
                    f'<p>Duration: {result.duration_seconds:.0f}s</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Force Posterior Gauges
                if result.force_posterior_history:
                    st.markdown("**Force Posteriors:**")
                    for fv, history in result.force_posterior_history.items():
                        try:
                            ft = ForceType(fv)
                            name = ft.display_name
                            color = ft.color_hex
                        except ValueError:
                            name = fv
                            color = "#888"
                        final = history[-1] if history else 0.5
                        pct = int(final * 100)
                        st.markdown(
                            f'<div class="posterior-gauge">'
                            f'{name}: {final:.2f}'
                            f'<div class="posterior-bar" style="'
                            f'width: {pct}%; background: {color};"></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Verdict narrative
                if result.verdict.narrative:
                    with st.expander("📋 Verdict Narrative", expanded=True):
                        st.write(result.verdict.narrative)

                if result.minority_record and result.minority_record.narrative:
                    with st.expander("📋 Minority Record"):
                        st.write(result.minority_record.narrative)
                        if result.minority_record.conditions_to_prevail:
                            st.write("**Conditions to Prevail:**")
                            for c in result.minority_record.conditions_to_prevail:
                                st.write(f"- {c}")

                if result.armistice_fired:
                    st.warning(
                        f"⚠️ Armistice Protocol: "
                        f"{result.armistice_option.value if result.armistice_option else 'N/A'}\n\n"
                        f"{result.armistice_details}")

                if st.button("🔄 New Campaign", width='stretch'):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()


main()
