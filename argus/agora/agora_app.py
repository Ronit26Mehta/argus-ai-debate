"""
AGORA Sandbox — Three-panel Streamlit interface.

Layout (from the AGORA Protocol specification):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Status Bar (full width)                                        │
    ├────────────────┬───────────────────────┬─────────────────────────┤
    │  LEFT  (30%)   │  CENTER (40%)         │  RIGHT (30%)           │
    │  Senate Floor  │  Senate Chamber       │  Analytics Pane        │
    │  - Input       │  - Evidence stream    │  - Position trajectory │
    │  - Controls    │  - Coalitions         │  - DEW distribution    │
    │  - Quorum      │  - Phase indicator    │  - Senate Record       │
    │  - Results     │  - Senate chart       │  - Scorecards          │
    └────────────────┴───────────────────────┴─────────────────────────┘

Supports:
    - Cloud providers (OpenAI, Gemini, Anthropic, etc.)
    - Local LLMs (Ollama, llama-server, vLLM, LM Studio)
    - Unbounded time sessions (agents complete naturally)
    - Time-bounded sessions with configurable limits

Launch:
    $ streamlit run argus/agora/agora_app.py
    $ agora-chat   (via console script)
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
    page_title="ARGUS × AGORA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from argus.agora.models import (
    AgoraResult,
    AgoraSessionConfig,
    SenateSpec,
    SessionPhase,
    StoppingTrigger,
)
from argus.agora.senate_gen import SenateGenerator
from argus.agora.socratic import SocraticEngine

logger = logging.getLogger(__name__)

# ── Dark-theme CSS (Senate-style) ─────────────────────────────────────

_CSS = """
<style>
    /* ── whole page ── */
    .stApp { background-color: #0B141A; color: #E9EDEF; }
    div[data-testid="stSidebar"] { background: #111B21; }

    /* ── status bar ── */
    .agora-status-bar {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2A3942;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 14px;
        color: #8696A0;
        margin-bottom: 12px;
    }
    .agora-status-bar .dot-green  { color: #00A884; }
    .agora-status-bar .dot-amber  { color: #FFA500; }
    .agora-status-bar .dot-blue   { color: #53BDEB; }
    .agora-status-bar .dot-gold   { color: #FFD700; }

    /* ── senate card ── */
    .senate-card {
        background: #1a1a2e;
        border: 1px solid #2A3942;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 13px;
    }
    .senate-card h4 { color: #53BDEB; margin: 0 0 6px 0; font-size: 14px; }

    /* ── evidence stream ── */
    .evidence-item {
        background: #1F2C34;
        border-left: 3px solid #00A884;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 12.5px;
    }
    .evidence-item.attack { border-left-color: #FF6B6B; }
    .evidence-item.qualifies { border-left-color: #FFA500; }

    /* ── WhatsApp-style bubbles ── */
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
    .wa-name { font-size: 12px; font-weight: 600; color: #00A884; margin-bottom: 2px; }
    .wa-text { font-size: 13.5px; line-height: 1.45; word-break: break-word; }
    .wa-time { font-size: 10px; color: #667781; text-align: right; margin-top: 2px; }

    /* ── result card ── */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 1px solid #FFD700;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    }
    .result-card h3 { color: #FFD700; margin: 0 0 8px 0; }
</style>
"""

# ═══════════════════════════════════════════════════════════════════════
# Session State Initialisation
# ═══════════════════════════════════════════════════════════════════════

_DEFAULTS: dict[str, Any] = {
    "messages": [],
    "senate": None,
    "result": None,
    "mode": "input",         # input | preview | running | complete
    "running": False,
    "max_rounds": 5,
    "time_limit": 0,         # 0 = unbounded
    "session_mode": "unbounded",  # unbounded | time_bounded
    "stopping_triggers": ["unbounded"],
    # LLM config
    "provider": os.environ.get("ARGUS_DEFAULT_PROVIDER", "gemini"),
    "model": os.environ.get("ARGUS_DEFAULT_MODEL", "gemini-2.0-flash"),
    "model_source": "env",
    "local_server_url": "http://localhost:8080",
    "local_model_name": "local-model",
    "local_server_status": "",
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════════
# Sidebar — LLM Configuration
# ═══════════════════════════════════════════════════════════════════════

def _test_local_connection(url: str) -> str:
    """Ping a local OpenAI-compatible / Ollama server and return status."""
    import urllib.request
    import urllib.error

    # Try Ollama endpoint first
    for endpoint in ["/api/tags", "/v1/models"]:
        test_url = url.rstrip("/") + endpoint
        try:
            req = urllib.request.Request(test_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                if "models" in data:
                    models = [m.get("name", m.get("id", "?")) for m in data["models"]]
                elif "data" in data:
                    models = [m.get("id", "?") for m in data["data"]]
                else:
                    models = []
                if models:
                    return f"✅ Connected — models: {', '.join(models[:5])}"
                return "✅ Connected (no models listed)"
        except (urllib.error.URLError, Exception):
            continue
    return "❌ Connection failed. Check server URL."


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ AGORA Configuration")

        # ── Model source toggle ───────────────────────────────────
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
                help="Base URL of your OpenAI-compatible server (e.g. http://localhost:8080)",
            )
            st.session_state["local_server_url"] = local_url

            local_model = st.text_input(
                "Model Name", value=st.session_state["local_model_name"],
                key="sb_local_model",
            )
            st.session_state["local_model_name"] = local_model

            if st.button("🔌 Test Connection", use_container_width=True):
                with st.spinner("Testing…"):
                    status = _test_local_connection(local_url)
                st.session_state["local_server_status"] = status
            status_msg = st.session_state.get("local_server_status", "")
            if status_msg:
                if status_msg.startswith("✅"):
                    st.success(status_msg)
                else:
                    st.error(status_msg)
            st.caption("Supports Ollama, llama-server, vLLM, LM Studio.")
        else:
            st.subheader("🌐 Provider & Model")
            try:
                from argus.core.llm import list_providers
                available = list_providers()
            except Exception:
                available = ["openai", "anthropic", "gemini", "ollama", "groq", "mistral", "deepseek"]

            provider = st.selectbox("Provider", available, key="sb_provider",
                index=available.index(st.session_state["provider"]) if st.session_state["provider"] in available else 0,
            )
            model = st.text_input("Model", value=st.session_state["model"], key="sb_model")
            st.session_state["provider"] = provider
            st.session_state["model"] = model

        st.divider()

        # ── Session Settings ──────────────────────────────────────
        st.header("🏛️ Session Settings")

        session_mode = st.radio(
            "Session Mode",
            options=["unbounded", "time_bounded"],
            format_func=lambda x: "♾️ Unbounded (complete naturally)" if x == "unbounded" else "⏱️ Time-Bounded",
            key="sb_session_mode",
        )
        st.session_state["session_mode"] = session_mode

        if session_mode == "time_bounded":
            time_limit = st.slider(
                "Time Limit (minutes)", min_value=1, max_value=120, value=15,
                key="sb_time_limit",
            )
            st.session_state["time_limit"] = time_limit * 60
        else:
            st.session_state["time_limit"] = 0

        max_rounds = st.slider(
            "Max Rounds per Phase", min_value=1, max_value=15,
            value=st.session_state.get("max_rounds", 5),
            help="Rounds within evidence and cross-examination phases",
            key="sb_max_rounds",
        )
        st.session_state["max_rounds"] = max_rounds

        st.divider()
        st.caption("ARGUS × AGORA v1.0")


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
        html = '<span class="dot-green">●</span> Ready — enter a proposition to begin'
    elif mode == "preview":
        html = '<span class="dot-blue">●</span> Senate generated — review and launch session'
    elif mode == "running":
        html = '<span class="dot-amber">●</span> Session in progress — agents are deliberating'
        session_mode = st.session_state.get("session_mode", "unbounded")
        if session_mode == "unbounded":
            html += ' (♾️ unbounded)'
    elif mode == "complete":
        result: AgoraResult | None = st.session_state.get("result")
        if result:
            verdict = result.majority_opinion.verdict_label.value
            post = f"{result.majority_opinion.posterior_probability:.0%}"
            html = f'<span class="dot-gold">★</span> Session Complete — {verdict} ({post})'
        else:
            html = '<span class="dot-gold">★</span> Session Complete'
    else:
        html = '<span class="dot-green">●</span> AGORA Ready'

    st.markdown(f'<div class="agora-status-bar">{html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Chat Helpers
# ═══════════════════════════════════════════════════════════════════════

def _render_single_message(container: Any, msg: dict[str, str]) -> None:
    role = msg["role"]
    content = _html_escape(msg.get("content", ""))
    content = content.replace("\n", "<br>")
    ts = msg.get("timestamp", "")
    label = msg.get("agent_label", "AGORA" if role == "assistant" else "You")

    if role == "user":
        html = (
            f'<div class="wa-msg-row wa-right">'
            f'<div class="wa-bubble wa-bubble-right">'
            f'<div class="wa-text">{content}</div>'
            f'<div class="wa-time">{ts}</div>'
            f'</div></div>'
        )
    else:
        html = (
            f'<div class="wa-msg-row wa-left">'
            f'<div class="wa-avatar">🏛️</div>'
            f'<div class="wa-bubble wa-bubble-left">'
            f'<div class="wa-name">{_html_escape(label)}</div>'
            f'<div class="wa-text">{content}</div>'
            f'<div class="wa-time">{ts}</div>'
            f'</div></div>'
        )
    container.markdown(html, unsafe_allow_html=True)


def _add_message(role: str, content: str, agent_label: str = "AGORA", container: Any = None) -> None:
    now = datetime.now(timezone.utc)
    msg = {
        "role": role,
        "content": content,
        "timestamp": now.strftime("%I:%M %p"),
        "agent_label": agent_label if role == "assistant" else "You",
    }
    st.session_state["messages"].append(msg)
    if container:
        _render_single_message(container, msg)


def _render_chat(container: Any) -> None:
    with container:
        for msg in st.session_state["messages"]:
            _render_single_message(st, msg)


# ═══════════════════════════════════════════════════════════════════════
# Session Execution
# ═══════════════════════════════════════════════════════════════════════

def _build_config() -> AgoraSessionConfig:
    """Build session config from sidebar state."""
    session_mode = st.session_state.get("session_mode", "unbounded")
    time_limit = st.session_state.get("time_limit", 0)
    max_rounds = st.session_state.get("max_rounds", 5)

    if session_mode == "unbounded":
        triggers = [StoppingTrigger.UNBOUNDED]
        time_limit_seconds = None
    else:
        triggers = [StoppingTrigger.TIME_BOUNDARY, StoppingTrigger.CONVERGENCE]
        time_limit_seconds = float(time_limit) if time_limit > 0 else None

    return AgoraSessionConfig(
        active_triggers=triggers,
        max_rounds=max_rounds,
        time_limit_seconds=time_limit_seconds,
    )


def _run_session(proposition: str, chat_container: Any, analytics_placeholder: Any) -> None:
    """Execute a full AGORA session."""
    llm = _get_llm()
    config = _build_config()

    st.session_state["mode"] = "running"
    st.session_state["running"] = True

    # Generate Senate
    _add_message("assistant", "Generating Senate composition…", "AGORA", chat_container)
    gen = SenateGenerator(llm=llm)
    senate = gen.generate(proposition, config)
    st.session_state["senate"] = senate
    _add_message("assistant", senate.preview_card_text(), "AGORA · Senate", chat_container)

    # Run full session
    session_mode = st.session_state.get("session_mode", "unbounded")
    mode_label = "UNBOUNDED" if session_mode == "unbounded" else "TIME-BOUNDED"
    _add_message(
        "assistant",
        f"Session started ({mode_label}). {senate.senate_size} senators deliberating…",
        "AGORA · Socratic Engine",
        chat_container
    )

    engine = SocraticEngine(llm=llm, config=config)

    def _on_round(phase, round_num, entry):
        if entry:
            # We use an inline import to avoid circular dependencies if any
            from argus.agora.models import RecordEntryType
            if entry.entry_type == RecordEntryType.SENATOR_STATEMENT:
                _add_message("assistant", f"{entry.content}", f"AGORA · {entry.senator_name}", chat_container)
            elif entry.entry_type == RecordEntryType.EVIDENCE_SUBMISSION:
                _add_message("assistant", f"Submitted evidence:\n{entry.content}", f"AGORA · {entry.senator_name}", chat_container)
            elif entry.entry_type == RecordEntryType.CHALLENGE_ISSUED:
                _add_message("assistant", f"Challenge issued:\n{entry.content}", "AGORA · Challenge", chat_container)
            elif entry.entry_type == RecordEntryType.CHALLENGE_RULING:
                _add_message("assistant", f"Ruling:\n{entry.content}", "AGORA · Epistemic Auditor", chat_container)
        elif phase and round_num > 0:
            _add_message(
                "assistant",
                f"Phase {phase.phase_number} ({phase.value}) — Round {round_num} complete.",
                f"AGORA · {phase.value}",
                chat_container
            )
            
            # Live Update Analytics
            if engine.position_trajectories and engine.live_senators:
                with analytics_placeholder.container():
                    st.subheader("📊 Live Trajectories & DAG")
                    try:
                        import plotly.graph_objects as go
                        from argus.agora.senate_dag import build_live_senate_dag

                        # Render the live DAG
                        dag_fig = build_live_senate_dag(engine, proposition)
                        st.plotly_chart(dag_fig, width="stretch", key=f"live_dag_{phase.name}_{round_num}")

                        # Render trajectories
                        fig = go.Figure()
                        for sid, traj in engine.position_trajectories.items():
                            name = engine.live_senators[sid].spec.name if sid in engine.live_senators else sid
                            fig.add_trace(go.Scatter(
                                x=list(range(len(traj))),
                                y=traj,
                                mode="lines+markers",
                                name=name[:20],
                                line=dict(width=1.5),
                                marker=dict(size=4),
                            ))
                        fig.update_layout(
                            height=250,
                            template="plotly_dark",
                            margin=dict(l=40, r=10, t=30, b=30),
                            xaxis_title="Round",
                            yaxis_title="Position",
                            yaxis=dict(range=[0, 1]),
                            showlegend=True,
                            legend=dict(font=dict(size=9)),
                        )
                        st.plotly_chart(fig, width="stretch", key=f"live_pos_{phase.name}_{round_num}")
                    except ImportError:
                        st.warning("Install plotly for live charts.")
                    except Exception as e:
                        st.warning(f"Live chart error: {e}")

    result = engine.run_session(
        senate=senate,
        proposition=proposition,
        config=config,
        round_callback=_on_round,
    )

    st.session_state["result"] = result
    st.session_state["mode"] = "complete"
    st.session_state["running"] = False

    _add_message("assistant", result.chat_card(), "AGORA · Verdict", chat_container)


# ═══════════════════════════════════════════════════════════════════════
# Main Layout
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_sidebar()
    _render_status_bar()

    # ── Three-panel layout ────────────────────────────────────────
    left_col, center_col, right_col = st.columns([0.30, 0.40, 0.30])
    
    # Initialize analytics placeholder in the right column early
    analytics_placeholder = right_col.empty()

    # ── CENTER: Senate Chamber ────────────────────────────────────
    with center_col:
        st.subheader("📜 Senate Chamber")
        chat_container = st.container(height=600)
        _render_chat(chat_container)

    # ── LEFT: Senate Floor ────────────────────────────────────────
    with left_col:
        st.subheader("🏛️ Senate Floor")

        mode = st.session_state["mode"]

        if mode in ("input", "preview"):
            proposition = st.text_area(
                "Proposition",
                placeholder="Enter a proposition to deliberate…\ne.g., 'Is nuclear fusion commercially viable by 2040?'",
                height=100,
                key="proposition_input",
            )

            session_mode = st.session_state.get("session_mode", "unbounded")
            if session_mode == "unbounded":
                st.info("♾️ **Unbounded mode** — agents will complete all 5 phases naturally.")
            else:
                tl = st.session_state.get("time_limit", 0)
                st.info(f"⏱️ **Time-bounded** — session ends after {tl // 60} minutes.")

            if st.button("🚀 Launch AGORA Session", use_container_width=True, type="primary"):
                if proposition and proposition.strip():
                    _add_message("user", proposition.strip(), container=chat_container)
                    _run_session(proposition.strip(), chat_container, analytics_placeholder)
                    st.rerun()
                else:
                    st.warning("Please enter a proposition.")

        elif mode == "running":
            st.info("🔄 Session in progress…")
            if st.button("🔨 Gavel — Stop Now", use_container_width=True, type="secondary"):
                st.session_state["mode"] = "complete"
                st.session_state["running"] = False
                st.rerun()

        elif mode == "complete":
            result: AgoraResult | None = st.session_state.get("result")
            if result:
                st.markdown(
                    f'<div class="result-card">'
                    f'<h3>{result.majority_opinion.verdict_label.value.upper()}</h3>'
                    f'<p>Posterior: {result.majority_opinion.posterior_probability:.0%}</p>'
                    f'<p>Senators: {result.num_senators} | Rounds: {result.num_rounds}</p>'
                    f'<p>Duration: {result.duration_seconds:.0f}s | Evidence: {result.num_evidence}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if result.minority_report.narrative:
                    with st.expander("📋 Minority Report"):
                        st.write(result.minority_report.narrative)
                        if result.minority_report.what_would_change:
                            st.write("**What Would Change This:**")
                            for item in result.minority_report.what_would_change:
                                st.write(f"- {item}")

                if result.quorum_details:
                    st.caption(f"Quorum: {'✅' if result.quorum_met else '❌'} {result.quorum_details}")

                if st.button("🔄 New Session", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

    # ── RIGHT: Analytics Pane ─────────────────────────────────────
    with right_col:
        result = st.session_state.get("result")
        senate = st.session_state.get("senate")

        if result:
            with analytics_placeholder.container():
                st.subheader("📊 Final Analytics")

            # Senate Lifecycle DAG
            with st.expander("🏛️ Senate Lifecycle DAG", expanded=True):
                try:
                    from argus.agora.senate_dag import build_senate_dag
                    dag_fig = build_senate_dag(result)
                    st.plotly_chart(dag_fig, width="stretch", key="senate_dag")
                except ImportError:
                    st.info("Install plotly for DAG visualisation.")
                except Exception as exc:
                    st.warning(f"DAG render error: {exc}")

            # Position Trajectory
            with st.expander("📈 Position Trajectories", expanded=True):
                if result.position_trajectories:
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        for sid, traj in result.position_trajectories.items():
                            name = sid
                            for sc in result.scorecards:
                                if sc.senator_id == sid:
                                    name = sc.senator_name
                                    break
                            fig.add_trace(go.Scatter(
                                x=list(range(len(traj))),
                                y=traj,
                                mode="lines+markers",
                                name=name[:20],
                                line=dict(width=1.5),
                                marker=dict(size=4),
                            ))
                        fig.update_layout(
                            height=250,
                            template="plotly_dark",
                            margin=dict(l=40, r=10, t=30, b=30),
                            xaxis_title="Round",
                            yaxis_title="Position",
                            yaxis=dict(range=[0, 1]),
                            showlegend=True,
                            legend=dict(font=dict(size=9)),
                        )
                        st.plotly_chart(fig, width="stretch", key="pos_traj")
                    except ImportError:
                        st.info("Install plotly for position trajectory charts.")

            # Evidence DEW Distribution
            with st.expander("⚖️ Evidence DEW Scores"):
                if result.docket_items:
                    try:
                        import plotly.graph_objects as go
                        dew_scores = [i.dew_score for i in result.docket_items]
                        polarities = [i.polarity.value for i in result.docket_items]
                        colors = [
                            "#00A884" if p == "supports"
                            else "#FF6B6B" if p == "attacks"
                            else "#FFA500"
                            for p in polarities
                        ]
                        fig = go.Figure(go.Bar(
                            x=list(range(len(dew_scores))),
                            y=dew_scores,
                            marker_color=colors,
                        ))
                        fig.update_layout(
                            height=200,
                            template="plotly_dark",
                            margin=dict(l=40, r=10, t=10, b=30),
                            xaxis_title="Evidence #",
                            yaxis_title="DEW",
                        )
                        st.plotly_chart(fig, width="stretch", key="dew_dist")
                    except ImportError:
                        st.info("Install plotly for DEW charts.")

            # Senator Scorecards
            with st.expander("🎯 Senator Scorecards"):
                if result.scorecards:
                    sorted_cards = sorted(
                        result.scorecards,
                        key=lambda s: s.epistemic_contribution_score,
                        reverse=True,
                    )
                    for sc in sorted_cards:
                        st.markdown(
                            f"**{sc.senator_name}** ({sc.category.abbreviation}) — "
                            f"ECS: `{sc.epistemic_contribution_score:.3f}` | "
                            f"Evidence: {sc.evidence_submitted} | "
                            f"Challenges: {sc.challenges_issued}↗ {sc.challenges_received}↙"
                        )

            # Coalitions
            with st.expander("🤝 Coalitions"):
                if result.coalitions:
                    for c in result.coalitions:
                        emoji = "⚠️" if c.is_low_independence else "✅"
                        st.markdown(
                            f"{emoji} **{c.name}** ({c.size} members)\n\n"
                            f"EIS: {c.epistemic_independence_score:.2f} | "
                            f"Similarity: {c.similarity_score:.2f}\n\n"
                            f"Members: {', '.join(c.member_names)}"
                        )
                else:
                    st.caption("No coalitions detected.")

            # Senate Record
            with st.expander("📖 Senate Record (Summary)"):
                if result.senate_record_entries:
                    from argus.agora.models import SenateRecordEntry
                    for entry in result.senate_record_entries[-20:]:
                        st.text(entry.to_hansard_line())

        elif senate:
            st.caption("Senate generated. Launch session to see analytics.")
            with st.expander("🏛️ Senate Preview"):
                st.text(senate.preview_card_text())
        else:
            st.caption("Enter a proposition to begin.")


main()
