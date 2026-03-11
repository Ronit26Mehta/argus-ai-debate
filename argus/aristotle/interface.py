"""
Layer 4 — Single-Pane Split Streamlit Interface.

Layout (§6 of the ARISTOTLE protocol):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Status Bar (full width)                                         │
    ├────────────────────┬─────────────────────────────────────────────┤
    │  LEFT  (38%)       │  RIGHT (62%)                                │
    │  ARISTOTLE chat    │  Zone A: Full Lifecycle DAG (55%)           │
    │                    │  Zone B: Belief + Heatmap (35%)             │
    │  [input box]       │  Zone C: Expander (more panels)            │
    └────────────────────┴─────────────────────────────────────────────┘

Launch:
    $ streamlit run argus/aristotle/interface.py
    or
    $ argus aristotle run   (via CLI entry-point)

Session-state event queue drives 800ms refresh cycle.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape as _html_escape
from pathlib import Path
from typing import Any

os.environ.setdefault("ARGUS_DEFAULT_PROVIDER", "gemini")
os.environ.setdefault("ARGUS_DEFAULT_MODEL", "gemini-2.0-flash")
os.environ.setdefault("GOOGLE_API_KEY", "AIzaSyBJF6kR2xOG2WuVGRfV3DVMz241PVn3wRI")

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="ARGUS × ARISTOTLE",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from argus.aristotle.models import (
    AgentSpec,
    DebateFrame,
    MonitorEvent,
    MonitorEventType,
    RefuterIntensity,
    SynthesisResult,
    TopologySpec,
)
from argus.aristotle.framing import FramingEngine
from argus.aristotle.topology import TopologyBuilder
from argus.aristotle.monitor import ExecutionMonitor
from argus.aristotle.synthesis import SynthesisEngine
from argus.aristotle.dag_viz import FullLifecycleDAG
from argus.aristotle import panels

logger = logging.getLogger(__name__)

# Module-level containers for live rendering during debate execution.
# Populated each run by main(); the round_callback reads them.
_LIVE_CONTAINERS: dict[str, Any] = {}

# ── Dark-theme CSS (WhatsApp-style) ───────────────────────────────────

_CSS = """
<style>
    /* ── whole page ── */
    .stApp { background-color: #0B141A; color: #E9EDEF; }
    div[data-testid="stSidebar"] { background: #111B21; }

    /* ── status bar ── */
    .status-bar {
        background: #1F2C34;
        border: 1px solid #2A3942;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 14px;
        color: #8696A0;
        margin-bottom: 12px;
    }
    .status-bar .dot-green  { color: #00A884; }
    .status-bar .dot-amber  { color: #FFA500; }
    .status-bar .dot-blue   { color: #53BDEB; }
    .status-bar .dot-gold   { color: #FFD700; }

    /* __ hide Streamlit defaults __ */
    .stChatMessage { background: transparent !important; border: none !important; }

    /* ── WhatsApp-style message rows ── */
    .wa-msg-row {
        display: flex;
        margin: 3px 8px;
        align-items: flex-end;
    }
    .wa-msg-row.wa-left  { justify-content: flex-start; }
    .wa-msg-row.wa-right { justify-content: flex-end; }

    .wa-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #2A3942;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        margin-right: 6px;
        flex-shrink: 0;
    }

    /* ── Bubbles ── */
    .wa-bubble {
        max-width: 88%;
        padding: 6px 10px 4px 10px;
        border-radius: 8px;
        position: relative;
        word-wrap: break-word;
        box-shadow: 0 1px 1px rgba(0,0,0,0.25);
    }
    .wa-bubble-left {
        background: #1F2C34;
        border-top-left-radius: 0;
        color: #E9EDEF;
    }
    .wa-bubble-right {
        background: #005C4B;
        border-top-right-radius: 0;
        color: #E9EDEF;
    }

    /* tail triangles */
    .wa-bubble-left::before {
        content: '';
        position: absolute;
        top: 0; left: -8px;
        border-width: 0 8px 8px 0;
        border-style: solid;
        border-color: transparent #1F2C34 transparent transparent;
    }
    .wa-bubble-right::after {
        content: '';
        position: absolute;
        top: 0; right: -8px;
        border-width: 0 0 8px 8px;
        border-style: solid;
        border-color: transparent transparent transparent #005C4B;
    }

    .wa-name {
        font-size: 12px;
        font-weight: 600;
        color: #00A884;
        margin-bottom: 2px;
    }
    .wa-text {
        font-size: 13.5px;
        line-height: 1.45;
        word-break: break-word;
    }
    .wa-text strong { color: #ffffff; }
    .wa-text em     { color: #adbac7; }

    .wa-time {
        font-size: 10px;
        color: #667781;
        text-align: right;
        margin-top: 2px;
        margin-bottom: -2px;
    }

    /* round badge inside bubbles */
    .wa-round-badge {
        display: inline-block;
        background: #00A884;
        color: #0B141A;
        font-size: 10px;
        font-weight: 700;
        padding: 1px 8px;
        border-radius: 10px;
        margin-bottom: 4px;
    }

    /* system pill (date headers, system msgs) */
    .wa-system-pill {
        text-align: center;
        margin: 12px 0 8px 0;
    }
    .wa-system-pill span {
        background: #182229;
        color: #8696A0;
        font-size: 11.5px;
        padding: 5px 14px;
        border-radius: 8px;
        box-shadow: 0 1px 1px rgba(0,0,0,0.15);
    }
</style>
"""

# ═══════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ═══════════════════════════════════════════════════════════════════════

_DEFAULTS: dict[str, Any] = {
    "messages": [],         # list[dict] — {role, content}
    "event_queue": [],      # list[MonitorEvent]
    "frame": None,          # DebateFrame | None
    "topology": None,       # TopologySpec | None
    "synthesis": None,      # SynthesisResult | None
    "posteriors": [],       # list[float]
    "evidence_log": [],     # list[dict]
    "debate_result": None,  # dict | None
    "dag": None,            # FullLifecycleDAG | None
    "round_snapshots": [],  # list[dict] — per-round data for inline plots
    "mode": "query",        # query | refinement | monitoring | synthesis
    "running": False,
    "max_rounds": 3,        # user-configurable max rounds
    "provider": os.environ.get("ARGUS_DEFAULT_PROVIDER", os.environ.get("LLM_PROVIDER", "gemini")),
    "model": os.environ.get("ARGUS_DEFAULT_MODEL", os.environ.get("LLM_MODEL", "gemini-2.0-flash")),
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Always sync provider/model with env defaults (overrides stale session state)
st.session_state["provider"] = _DEFAULTS["provider"]
st.session_state["model"] = _DEFAULTS["model"]


# ═══════════════════════════════════════════════════════════════════════
# Sidebar — LLM provider selection
# ═══════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.header("LLM Configuration")
        try:
            from argus.core.llm import list_providers
            available = list_providers()
        except Exception:
            available = [
                "openai", "anthropic", "gemini", "ollama", "cohere",
                "mistral", "groq", "deepseek", "together", "openrouter",
                "fireworks", "anyscale", "perplexity", "replicate",
                "huggingface", "vllm", "llama_cpp", "cerebras",
                "ai21", "aleph_alpha", "bedrock", "azure", "vertex",
                "palm", "inflection", "writer", "sambanova", "nebius",
            ]

        # Ensure providers are in the list
        for p in ("mistral", "openrouter", "gemini"):
            if p not in available:
                available.append(p)

        # Force provider/model into widget state before rendering
        target_provider = _DEFAULTS["provider"]
        target_model = _DEFAULTS["model"]
        if "sb_provider" not in st.session_state:
            st.session_state["sb_provider"] = target_provider
        if "sb_model" not in st.session_state:
            st.session_state["sb_model"] = target_model

        provider_idx = available.index(st.session_state["sb_provider"]) if st.session_state["sb_provider"] in available else 0

        provider = st.selectbox(
            "Provider",
            available,
            index=provider_idx,
            key="sb_provider",
        )
        model = st.text_input("Model", value=st.session_state["sb_model"], key="sb_model")

        st.session_state["provider"] = provider
        st.session_state["model"] = model

        st.divider()

        # Debate rounds configuration
        st.header("Debate Settings")
        max_rounds = st.slider(
            "Max Rounds",
            min_value=1,
            max_value=8,
            value=st.session_state.get("max_rounds", 3),
            help="Maximum number of debate rounds (1 = quick, 8 = thorough)",
            key="sb_max_rounds",
        )
        st.session_state["max_rounds"] = max_rounds

        st.divider()
        st.caption("ARGUS × ARISTOTLE v4.0")
        st.caption(
            "All LLM providers supported by argus-debate-ai's registry "
            "can be selected above, or set via environment variables "
            "ARGUS_DEFAULT_PROVIDER / ARGUS_DEFAULT_MODEL."
        )


# ═══════════════════════════════════════════════════════════════════════
# LLM factory
# ═══════════════════════════════════════════════════════════════════════

def _get_llm():
    from argus.core.llm import get_llm
    return get_llm(
        provider=st.session_state["provider"],
        model=st.session_state["model"],
    )


# ═══════════════════════════════════════════════════════════════════════
# Status bar
# ═══════════════════════════════════════════════════════════════════════

def _render_status_bar() -> None:
    mode = st.session_state["mode"]
    posteriors = st.session_state["posteriors"]
    events = st.session_state["event_queue"]

    if mode == "query":
        html = '<span class="dot-green">●</span> Ready — ask a question to begin'
    elif mode == "refinement":
        html = '<span class="dot-blue">●</span> Topology ready — refine or launch debate'
    elif mode == "monitoring":
        rounds = max(ev.round_num for ev in events) if events else 0
        topo = st.session_state.get("topology")
        est = topo.estimated_rounds if topo else "?"
        post = f"{posteriors[-1]:.0%}" if posteriors else "—"
        n_ev = len(st.session_state["evidence_log"])
        html = (
            f'<span class="dot-green">●</span> '
            f'Round {rounds} of {est} │ Posterior: {post} │ '
            f'Status: DEBATING │ Evidence nodes: {n_ev}'
        )
    elif mode == "synthesis":
        syn = st.session_state.get("synthesis")
        post = f"{syn.jury_confidence:.0%}" if syn else "—"
        html = (
            f'<span class="dot-gold">★</span> '
            f'Debate Complete │ Final Posterior: {post}'
        )
    else:
        html = '<span class="dot-green">●</span> Ready'

    st.markdown(f'<div class="status-bar">{html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Chat helpers
# ═══════════════════════════════════════════════════════════════════════


def _md_to_html(text: str) -> str:
    """Minimal Markdown-to-HTML for WhatsApp-style chat bubbles."""
    text = _html_escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = text.replace('\n', '<br>')
    return text


def _add_message(
    role: str,
    content: str,
    *,
    msg_type: str = "text",
    round_data: dict | None = None,
    agent_label: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    st.session_state["messages"].append({
        "role": role,
        "content": content,
        "timestamp": now.strftime("%I:%M %p"),
        "msg_type": msg_type,
        "round_data": round_data,
        "agent_label": agent_label or ("ARISTOTLE" if role == "assistant" else "You"),
    })


def _render_chat(container) -> None:
    with container:
        for idx, msg in enumerate(st.session_state["messages"]):
            role = msg["role"]
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            msg_type = msg.get("msg_type", "text")
            agent_label = msg.get("agent_label", "ARISTOTLE" if role == "assistant" else "You")

            content_html = _md_to_html(content)

            if role == "user":
                # ── user bubble (right-aligned, green) ──
                html = (
                    f'<div class="wa-msg-row wa-right">'
                    f'<div class="wa-bubble wa-bubble-right">'
                    f'<div class="wa-text">{content_html}</div>'
                    f'<div class="wa-time">{timestamp}</div>'
                    f'</div></div>'
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                # ── assistant bubble (left-aligned, dark) ──
                avatar = "📊" if msg_type == "round_summary" else "🏛️"
                badge = ""
                if msg_type == "round_summary":
                    rd = msg.get("round_data") or {}
                    rn = rd.get("round_num", "")
                    badge = f'<span class="wa-round-badge">Round {rn}</span> '

                html = (
                    f'<div class="wa-msg-row wa-left">'
                    f'<div class="wa-avatar">{avatar}</div>'
                    f'<div class="wa-bubble wa-bubble-left">'
                    f'<div class="wa-name">{badge}{_html_escape(agent_label)}</div>'
                    f'<div class="wa-text">{content_html}</div>'
                    f'<div class="wa-time">{timestamp}</div>'
                    f'</div></div>'
                )
                st.markdown(html, unsafe_allow_html=True)

                # ── inline per-round plots ──
                if msg_type == "round_summary" and msg.get("round_data"):
                    rd = msg["round_data"]
                    posteriors = rd.get("posteriors", [])
                    evidence_log = rd.get("evidence_log", [])
                    round_num = rd.get("round_num", 0)

                    plot_cols = st.columns(2)
                    with plot_cols[0]:
                        if len(posteriors) > 1:
                            fig_belief = panels.build_belief_trajectory(posteriors)
                            fig_belief.update_layout(height=200, margin=dict(l=30, r=10, t=25, b=20))
                            st.plotly_chart(
                                fig_belief, width="stretch",
                                key=f"chat_belief_r{round_num}_{idx}",
                            )
                    with plot_cols[1]:
                        if evidence_log:
                            fig_heat = panels.build_evidence_heatmap(evidence_log)
                            fig_heat.update_layout(height=200, margin=dict(l=60, r=10, t=25, b=20))
                            st.plotly_chart(
                                fig_heat, width="stretch",
                                key=f"chat_heat_r{round_num}_{idx}",
                            )


# ═══════════════════════════════════════════════════════════════════════
# Debate execution with live per-round rendering
# ═══════════════════════════════════════════════════════════════════════

def _run_debate_blocking() -> None:
    """Execute L3 + L5 and populate session state.

    Uses a ``round_callback`` so that each completed round immediately
    renders a WhatsApp-style chat bubble with inline plots, giving the
    user a semi-live view of the debate while it is still in progress.
    """
    frame: DebateFrame = st.session_state["frame"]
    topology: TopologySpec = st.session_state["topology"]
    llm = _get_llm()

    # Incremental DAG built by the callback
    live_dag = FullLifecycleDAG()

    # Reset per-debate accumulators
    st.session_state["round_snapshots"] = []
    st.session_state["event_queue"] = []

    # ── round callback: fires after every round ───────────────────
    def _on_round_complete(
        round_num: int,
        posteriors: list[float],
        evidence_log: list[dict],
        round_events: list[MonitorEvent],
    ) -> None:
        """Render one round's data live in the chat + right pane."""
        try:
            # Feed round events into incremental DAG
            for ev in round_events:
                live_dag.process_event(ev)

            # Build snapshot for this round
            snapshot = {
                "round_num": round_num,
                "posteriors": list(posteriors),
                "evidence_log": list(evidence_log),
                "round_events_text": [
                    ev.chat_message for ev in round_events if ev.chat_message
                ],
            }
            st.session_state["round_snapshots"].append(snapshot)
            st.session_state["posteriors"] = list(posteriors)
            st.session_state["evidence_log"] = list(evidence_log)
            st.session_state["event_queue"].extend(round_events)
            st.session_state["dag"] = live_dag

            # ── Build message text ────────────────────────────────
            post_str = f"{posteriors[-1]:.2%}" if posteriors else "N/A"
            round_items = [
                e for e in evidence_log if e.get("round") == round_num
            ]
            new_ev = len(round_items)

            # Per-agent breakdown
            agent_map: dict[str, list[dict]] = {}
            for e in round_items:
                agent_map.setdefault(
                    e.get("agent_name", "Unknown"), [],
                ).append(e)

            lines = [
                f"**Round {round_num} Complete**",
                f"Posterior: {post_str} | New evidence: {new_ev}",
            ]
            for ag_name, items in agent_map.items():
                avg_q = sum(i.get("evid_q", 0) for i in items) / max(len(items), 1)
                sups = sum(1 for i in items if i.get("polarity", 0) >= 0)
                atks = len(items) - sups
                lines.append(
                    f"  📄 {ag_name}: {len(items)} items "
                    f"(↑{sups} ↓{atks}, avg EVID-Q: {avg_q:.2f})"
                )

            for evt_text in snapshot["round_events_text"]:
                lines.append(f"• {evt_text}")

            content = "\n".join(lines)

            # Store message persistently (will survive rerun)
            _add_message(
                "assistant",
                content,
                msg_type="round_summary",
                round_data=snapshot,
                agent_label=f"ARISTOTLE · Round {round_num}",
            )

            # ── LIVE render to chat container ─────────────────────
            chat = _LIVE_CONTAINERS.get("chat")
            if chat:
                with chat:
                    timestamp = datetime.now(timezone.utc).strftime("%I:%M %p")
                    content_html = _md_to_html(content)
                    badge = f'<span class="wa-round-badge">Round {round_num}</span> '
                    html = (
                        f'<div class="wa-msg-row wa-left">'
                        f'<div class="wa-avatar">📊</div>'
                        f'<div class="wa-bubble wa-bubble-left">'
                        f'<div class="wa-name">{badge}'
                        f'{_html_escape(f"ARISTOTLE · Round {round_num}")}</div>'
                        f'<div class="wa-text">{content_html}</div>'
                        f'<div class="wa-time">{timestamp}</div>'
                        f'</div></div>'
                    )
                    st.markdown(html, unsafe_allow_html=True)

                    # Inline plots
                    pcols = st.columns(2)
                    with pcols[0]:
                        if len(posteriors) > 1:
                            fig = panels.build_belief_trajectory(posteriors)
                            fig.update_layout(
                                height=200,
                                margin=dict(l=30, r=10, t=25, b=20),
                            )
                            st.plotly_chart(
                                fig,
                                width="stretch",
                                key=f"live_bel_{round_num}",
                            )
                    with pcols[1]:
                        round_ev = [
                            e for e in evidence_log
                            if e.get("round", 0) <= round_num
                        ]
                        if round_ev:
                            fig = panels.build_evidence_heatmap(round_ev)
                            fig.update_layout(
                                height=200,
                                margin=dict(l=60, r=10, t=25, b=20),
                            )
                            st.plotly_chart(
                                fig,
                                width="stretch",
                                key=f"live_heat_{round_num}",
                            )

            # ── LIVE update right-pane slots ──────────────────────
            dag_slot = _LIVE_CONTAINERS.get("dag_slot")
            if dag_slot:
                fig = live_dag.build_figure()
                dag_slot.plotly_chart(fig, width="stretch")

            belief_slot = _LIVE_CONTAINERS.get("belief_slot")
            if belief_slot and len(posteriors) > 1:
                fig = panels.build_belief_trajectory(posteriors)
                belief_slot.plotly_chart(fig, width="stretch")

            heat_slot = _LIVE_CONTAINERS.get("heat_slot")
            if heat_slot and evidence_log:
                fig = panels.build_evidence_heatmap(evidence_log)
                heat_slot.plotly_chart(fig, width="stretch")

        except Exception:
            logger.debug("Round callback render failed", exc_info=True)

    # ── Execute debate with callback ──────────────────────────────
    monitor = ExecutionMonitor(llm=llm)
    debate_result, events = monitor.execute(
        topology=topology,
        frame=frame,
        llm=llm,
        round_callback=_on_round_complete,
    )

    st.session_state["debate_result"] = debate_result
    st.session_state["posteriors"] = debate_result.get("posteriors", [])
    st.session_state["evidence_log"] = debate_result.get("evidence_log", [])

    # Rebuild full DAG from all events (includes jury/verdict nodes
    # that round callbacks did not see)
    dag = FullLifecycleDAG()
    for ev in events:
        dag.process_event(ev)
    st.session_state["dag"] = dag
    st.session_state["event_queue"] = events

    # L5 — synthesis
    synth_engine = SynthesisEngine(llm=llm)
    synthesis = synth_engine.synthesise(
        frame=frame,
        topology=topology,
        debate_result=debate_result,
        events=events,
    )
    st.session_state["synthesis"] = synthesis
    st.session_state["mode"] = "synthesis"
    st.session_state["running"] = False


# ═══════════════════════════════════════════════════════════════════════
# User input handler
# ═══════════════════════════════════════════════════════════════════════

def _handle_input(user_text: str) -> None:
    _add_message("user", user_text)
    mode = st.session_state["mode"]

    if mode == "query":
        _handle_query(user_text)
    elif mode == "refinement":
        _handle_refinement(user_text)
    elif mode == "monitoring":
        _add_message(
            "assistant",
            "The debate is in progress. I'll let you know when there are updates.",
        )
    elif mode == "synthesis":
        _handle_followup(user_text)


def _handle_query(query: str) -> None:
    """Layer 1 → Layer 2: frame the query and build topology."""
    llm = _get_llm()

    # L1
    framing = FramingEngine(llm=llm)
    with st.spinner("ARISTOTLE is analysing your question…"):
        frame = framing.frame(query)
    st.session_state["frame"] = frame

    # Framing narrative
    narrative = (
        f"I have read your question. Here is how I understand it:\n\n"
        f"**Domain:** {frame.primary_domain}\n"
        f"**Debate type:** {frame.debate_type.value.replace('_', ' ').title()}\n"
        f"**Controversy level:** {frame.controversy_score:.2f}\n"
        f"**Evidence landscape:** {frame.evidence_density.upper()}\n"
    )
    if frame.literature_probe:
        lp = frame.literature_probe
        narrative += (
            f"\nI scanned {lp.total_scanned} relevant studies: "
            f"{lp.num_support} support, {lp.num_against} challenge, "
            f"{lp.num_partial} partial.\n"
        )
    narrative += f"\n**Starting prior:** {frame.prior_probability:.2f}\n\nNow I am building your debate team."
    _add_message("assistant", narrative)

    # L2
    builder = TopologyBuilder(llm=llm)
    with st.spinner("Building debate topology…"):
        topology = builder.build(frame)
    st.session_state["topology"] = topology

    _add_message("assistant", topology.preview_card_text())
    st.session_state["mode"] = "refinement"


def _handle_refinement(instruction: str) -> None:
    """User adjusts topology; or clicks launch."""
    low = instruction.strip().lower()
    if low in ("launch", "launch debate", "start", "run", "go"):
        _launch_debate()
        return

    # Interpret adjustment via LLM and apply actual topology modifications
    topo: TopologySpec = st.session_state["topology"]
    frame: DebateFrame = st.session_state["frame"]
    llm = _get_llm()

    _REFINEMENT_SYSTEM = (
        "You are ARISTOTLE's Topology Refiner. The user wants to adjust the "
        "debate topology. Analyse their instruction and respond with a JSON "
        "object describing EXACTLY which modifications to apply. Supported "
        "modification keys (all optional):\n"
        "  add_specialist: {name, domain_mandate, epistemic_prior, persona_description}\n"
        "  remove_specialist_index: int (1-based index)\n"
        "  add_refuter: {name, domain_mandate, epistemic_prior, persona_description}\n"
        "  remove_refuter_index: int (1-based)\n"
        "  set_rounds: int\n"
        "  set_jury_size: int\n"
        "  set_refuter_intensity: 'low' | 'medium' | 'high'\n"
        "  acknowledgement: str (brief description of what changed)\n\n"
        "Output ONLY valid JSON, no extra text."
    )

    prompt = (
        f"Current topology:\n{topo.preview_card_text()}\n\n"
        f"User instruction: {instruction}\n"
    )
    try:
        response = llm.generate(
            prompt=prompt,
            system_prompt=_REFINEMENT_SYSTEM,
            temperature=0.2,
            max_tokens=400,
        )
        import json as _json
        import re as _re
        raw = response.strip()
        if raw.startswith("```"):
            raw = _re.sub(r"^```(?:json)?\s*", "", raw)
            raw = _re.sub(r"\s*```$", "", raw)
        mods = _json.loads(raw)
    except Exception:
        _add_message(
            "assistant",
            "I have noted your adjustment. Type **launch** to start the debate.",
        )
        return

    # Apply modifications to the live TopologySpec
    changed: list[str] = []

    if "add_specialist" in mods:
        spec_data = mods["add_specialist"]
        new_agent = AgentSpec(
            name=spec_data.get("name", f"Specialist-{len(topo.specialists)+1}"),
            role="specialist",
            domain_mandate=spec_data.get("domain_mandate", frame.primary_domain),
            epistemic_prior=float(spec_data.get("epistemic_prior", 0.5)),
            persona_description=spec_data.get("persona_description", ""),
        )
        topo.specialists.append(new_agent)
        changed.append(f"Added specialist: {new_agent.name}")

    if "remove_specialist_index" in mods:
        idx = int(mods["remove_specialist_index"]) - 1
        if 0 <= idx < len(topo.specialists):
            removed = topo.specialists.pop(idx)
            changed.append(f"Removed specialist: {removed.name}")

    if "add_refuter" in mods:
        spec_data = mods["add_refuter"]
        new_agent = AgentSpec(
            name=spec_data.get("name", f"Refuter-{len(topo.refuters)+1}"),
            role="refuter",
            domain_mandate=spec_data.get("domain_mandate", "challenge methodology"),
            epistemic_prior=float(spec_data.get("epistemic_prior", 0.5)),
            persona_description=spec_data.get("persona_description", ""),
        )
        topo.refuters.append(new_agent)
        changed.append(f"Added refuter: {new_agent.name}")

    if "remove_refuter_index" in mods:
        idx = int(mods["remove_refuter_index"]) - 1
        if 0 <= idx < len(topo.refuters):
            removed = topo.refuters.pop(idx)
            changed.append(f"Removed refuter: {removed.name}")

    if "set_rounds" in mods:
        new_rounds = max(1, min(int(mods["set_rounds"]), topo.max_rounds))
        topo.estimated_rounds = new_rounds
        changed.append(f"Set rounds to {new_rounds}")

    if "set_jury_size" in mods:
        topo.jury.size = max(1, min(int(mods["set_jury_size"]), 9))
        changed.append(f"Set jury size to {topo.jury.size}")

    if "set_refuter_intensity" in mods:
        try:
            topo.refuter_intensity = RefuterIntensity(mods["set_refuter_intensity"])
            changed.append(f"Set refuter intensity to {topo.refuter_intensity.value}")
        except ValueError:
            pass

    st.session_state["topology"] = topo

    ack = mods.get("acknowledgement", "")
    if changed:
        summary = "**Topology updated:**\n" + "\n".join(f"- {c}" for c in changed)
        if ack:
            summary += f"\n\n{ack}"
        summary += f"\n\nType **launch** when ready."
        _add_message("assistant", summary)
    elif ack:
        _add_message("assistant", ack + "\n\nType **launch** when ready.")
    else:
        _add_message(
            "assistant",
            "I have noted your adjustment. Type **launch** to start the debate.",
        )


def _launch_debate() -> None:
    """Kick off L3 execution with live per-round rendering."""
    # Apply sidebar max_rounds to topology before launch
    topo: TopologySpec = st.session_state["topology"]
    user_max = st.session_state.get("max_rounds", 3)
    topo.max_rounds = user_max
    topo.estimated_rounds = min(topo.estimated_rounds, user_max)

    _add_message(
        "assistant",
        f"Launching the debate now (up to {user_max} rounds). "
        "Round results will appear live below…",
    )
    st.session_state["mode"] = "monitoring"
    st.session_state["running"] = True

    # Create right-pane live-update placeholders
    right_col = _LIVE_CONTAINERS.get("right_col")
    if right_col:
        with right_col:
            _LIVE_CONTAINERS["dag_slot"] = st.empty()
            _bl, _br = st.columns(2)
            with _bl:
                _LIVE_CONTAINERS["belief_slot"] = st.empty()
            with _br:
                _LIVE_CONTAINERS["heat_slot"] = st.empty()

    with st.spinner("ARISTOTLE debate in progress…"):
        _run_debate_blocking()

    # Post-debate: synthesis message (round bubbles already injected live)
    syn = st.session_state.get("synthesis")
    if syn:
        _add_message("assistant", syn.chat_card(), agent_label="ARISTOTLE · Verdict")

    # Save all results to timestamped folder
    _save_debate_results()

    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# Save debate results to timestamped folder
# ═══════════════════════════════════════════════════════════════════════

def _save_debate_results() -> None:
    """Persist plots, JSON data and transcript to argus/outputs/<ts>_<topic>/."""
    frame: DebateFrame | None = st.session_state.get("frame")
    synthesis: SynthesisResult | None = st.session_state.get("synthesis")
    debate_result = st.session_state.get("debate_result")

    if not frame or not debate_result:
        return

    # ── folder name ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_raw = (frame.primary_proposition or frame.raw_query or "debate")[:40]
    topic_slug = re.sub(r'[^\w\s-]', '', topic_raw).strip().replace(' ', '_')
    folder_name = f"{ts}_{topic_slug}"

    output_dir = Path("argus") / "outputs" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── JSON exports ──────────────────────────────────────────────
    try:
        safe_result = {
            k: v for k, v in debate_result.items()
            if k not in ("graph_obj", "verdict_obj", "ledger")
        }
        (output_dir / "debate_result.json").write_text(
            json.dumps(safe_result, indent=2, default=str), encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save debate_result.json: %s", exc)

    if synthesis:
        try:
            (output_dir / "synthesis.json").write_text(
                json.dumps(synthesis.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save synthesis.json: %s", exc)

    try:
        (output_dir / "frame.json").write_text(
            json.dumps(frame.to_dict(), indent=2, default=str), encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save frame.json: %s", exc)

    # ── topology ──────────────────────────────────────────────────
    topo = st.session_state.get("topology")
    if topo:
        try:
            (output_dir / "topology.json").write_text(
                json.dumps(topo.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save topology.json: %s", exc)

    # ── final aggregate plots (HTML) ──────────────────────────────
    posteriors = st.session_state.get("posteriors", [])
    evidence_log = st.session_state.get("evidence_log", [])
    events = st.session_state.get("event_queue", [])

    _safe_write_plot(
        lambda: panels.build_belief_trajectory(posteriors) if posteriors else None,
        plots_dir / "belief_trajectory.html",
    )
    _safe_write_plot(
        lambda: panels.build_evidence_heatmap(evidence_log) if evidence_log else None,
        plots_dir / "evidence_heatmap.html",
    )
    _safe_write_plot(
        lambda: panels.build_sankey(events, evidence_log) if (events or evidence_log) else None,
        plots_dir / "argument_flow_sankey.html",
    )
    _safe_write_plot(
        lambda: panels.build_agent_radar(evidence_log, events) if (evidence_log or events) else None,
        plots_dir / "agent_radar.html",
    )
    _safe_write_plot(
        lambda: panels.build_calibration_diagram(posteriors) if posteriors else None,
        plots_dir / "calibration.html",
    )
    _safe_write_plot(
        lambda: panels.build_timeline(events) if events else None,
        plots_dir / "timeline.html",
    )

    dag = st.session_state.get("dag")
    _safe_write_plot(
        lambda: dag.build_figure() if dag else None,
        plots_dir / "lifecycle_dag.html",
    )

    # ── per-round plots ───────────────────────────────────────────
    for snap in st.session_state.get("round_snapshots", []):
        r = snap["round_num"]
        _safe_write_plot(
            lambda s=snap: panels.build_belief_trajectory(s["posteriors"]) if len(s["posteriors"]) > 1 else None,
            plots_dir / f"round_{r}_belief.html",
        )
        _safe_write_plot(
            lambda s=snap: panels.build_evidence_heatmap(s["evidence_log"]) if s["evidence_log"] else None,
            plots_dir / f"round_{r}_heatmap.html",
        )

    # ── chat transcript ───────────────────────────────────────────
    try:
        lines: list[str] = []
        for msg in st.session_state.get("messages", []):
            role = msg.get("role", "unknown").upper()
            ts_str = msg.get("timestamp", "")
            content = msg.get("content", "")
            lines.append(f"[{ts_str}] {role}:\n{content}\n")
        (output_dir / "chat_transcript.txt").write_text(
            "\n".join(lines), encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save chat transcript: %s", exc)

    # ── optional PNG export (needs kaleido) ───────────────────────
    try:
        import kaleido  # noqa: F401
        for name, fig_fn in [
            ("belief_trajectory", lambda: panels.build_belief_trajectory(posteriors) if posteriors else None),
            ("evidence_heatmap", lambda: panels.build_evidence_heatmap(evidence_log) if evidence_log else None),
            ("timeline", lambda: panels.build_timeline(events) if events else None),
            ("lifecycle_dag", lambda: dag.build_figure() if dag else None),
        ]:
            fig = fig_fn()
            if fig:
                fig.write_image(str(plots_dir / f"{name}.png"))
    except ImportError:
        pass  # kaleido not installed; HTML exports are sufficient
    except Exception as exc:
        logger.warning("PNG export failed: %s", exc)

    logger.info("Debate results saved to %s", output_dir)


def _safe_write_plot(fig_fn, path: Path) -> None:
    """Call *fig_fn*; if it returns a figure, write it as HTML."""
    try:
        fig = fig_fn()
        if fig is not None:
            fig.write_html(str(path))
    except Exception as exc:
        logger.warning("Failed to save %s: %s", path.name, exc)


def _handle_followup(question: str) -> None:
    """Post-debate follow-up question handling."""
    llm = _get_llm()
    syn: SynthesisResult = st.session_state["synthesis"]
    frame: DebateFrame = st.session_state["frame"]

    context = (
        f"Proposition: {frame.primary_proposition or frame.raw_query}\n"
        f"Verdict: {syn.verdict_label.value} ({syn.jury_confidence:.1%})\n"
        f"Narrative: {syn.reasoning_narrative[:500]}\n"
    )
    prompt = (
        f"The user has a follow-up question about a completed debate.\n\n"
        f"DEBATE CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"Answer concisely, referencing debate evidence where possible."
    )
    try:
        response = llm.generate(prompt=prompt, temperature=0.4, max_tokens=500)
        _add_message("assistant", response.strip())
    except Exception:
        _add_message(
            "assistant",
            "I was unable to process your follow-up. Please try rephrasing your question.",
        )


# ═══════════════════════════════════════════════════════════════════════
# Right-pane visualisation
# ═══════════════════════════════════════════════════════════════════════

def _render_right_pane(container) -> None:
    with container:
        # Zone A — Full Lifecycle DAG (55%)
        dag = st.session_state.get("dag")
        if dag is not None:
            fig = dag.build_figure()
            st.plotly_chart(fig, width="stretch", key="dag_chart")
        else:
            st.markdown(
                '<div style="text-align:center; color:#555; padding:60px 0;">'
                "Your debate visualization will appear here once you ask a question."
                "</div>",
                unsafe_allow_html=True,
            )

        # Zone B — Belief Trajectory + Evidence Heatmap
        posteriors = st.session_state.get("posteriors", [])
        evidence_log = st.session_state.get("evidence_log", [])

        b_left, b_right = st.columns(2)
        with b_left:
            if posteriors:
                fig_belief = panels.build_belief_trajectory(posteriors)
                st.plotly_chart(fig_belief, width="stretch", key="belief_chart")
        with b_right:
            if evidence_log:
                fig_heat = panels.build_evidence_heatmap(evidence_log)
                st.plotly_chart(fig_heat, width="stretch", key="heatmap_chart")

        # Zone C — More panels (expander)
        events = st.session_state.get("event_queue", [])
        if events or evidence_log:
            with st.expander("More Panels", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    fig_sankey = panels.build_sankey(events, evidence_log)
                    st.plotly_chart(fig_sankey, width="stretch", key="sankey_chart")
                with c2:
                    fig_radar = panels.build_agent_radar(evidence_log, events)
                    st.plotly_chart(fig_radar, width="stretch", key="radar_chart")

                c3, c4 = st.columns(2)
                with c3:
                    if posteriors:
                        fig_calib = panels.build_calibration_diagram(posteriors)
                        st.plotly_chart(fig_calib, width="stretch", key="calib_chart")
                with c4:
                    fig_timeline = panels.build_timeline(events)
                    st.plotly_chart(fig_timeline, width="stretch", key="timeline_chart")

        # Zone D — Final Synthesis Verdict
        synthesis = st.session_state.get("synthesis")
        if synthesis is not None:
            st.divider()
            st.subheader("Final Verdict")
            verdict_color = {
                "SUPPORTED": "#2ecc71", "PARTIALLY_SUPPORTED": "#f39c12",
                "NOT_SUPPORTED": "#e74c3c", "INCONCLUSIVE": "#95a5a6",
            }
            label = getattr(synthesis.verdict_label, "value", str(synthesis.verdict_label))
            color = verdict_color.get(label.upper(), "#3498db")
            st.markdown(
                f'<div style="background:{color}22; border-left:4px solid {color}; '
                f'padding:12px 16px; border-radius:6px; margin-bottom:12px;">'
                f'<span style="font-size:1.3em; font-weight:bold; color:{color};">'
                f'{label.replace("_", " ").title()}</span>'
                f'<span style="margin-left:16px; color:#ccc;">'
                f'Confidence: {synthesis.jury_confidence:.1%}</span></div>',
                unsafe_allow_html=True,
            )
            if synthesis.reasoning_narrative:
                with st.expander("Reasoning Narrative", expanded=True):
                    st.markdown(synthesis.reasoning_narrative)
            if hasattr(synthesis, "recommendation") and synthesis.recommendation:
                st.info(f"**Recommendation:** {synthesis.recommendation}")


# ═══════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_sidebar()
    _render_status_bar()

    left, right = st.columns([0.38, 0.62], gap="medium")

    # Store column references for live rendering during debate
    _LIVE_CONTAINERS["right_col"] = right

    # Left pane — chat
    chat_container = left.container(height=700)
    _LIVE_CONTAINERS["chat"] = chat_container
    _render_chat(chat_container)

    user_input = left.chat_input(
        "Type your question here…",
        disabled=st.session_state.get("running", False),
    )
    if user_input:
        _handle_input(user_input)
        st.rerun()

    # Right pane — visualisation
    _render_right_pane(right)

    # Welcome message
    if not st.session_state["messages"]:
        _add_message(
            "assistant",
            "Hello. I am **ARISTOTLE**. I am here to help you find the truth "
            "about complex and contested claims.\n\nWhat would you like to know?",
        )
        st.rerun()


if __name__ == "__main__":
    main()
