"""
CRUX-Viz: Interactive CRUX Protocol Visualization Sandbox.

Main Streamlit application. Launch with:
    crux-viz
    python -m crux_viz
    streamlit run crux_viz/app.py
"""

from __future__ import annotations

import os
import json
import streamlit as st

# Load .env so GROQ_API_KEY and other keys are available
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

# Must be first Streamlit call
st.set_page_config(
    page_title="CRUX-Viz | Protocol Sandbox",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from crux_viz.components import (
    inject_custom_css,
    render_verdict_card,
    render_evidence_card,
    render_round_summary,
    render_agent_status,
    render_crux_config_summary,
    render_metrics_row,
    render_claim_bundle_card,
    render_crux_session_stats,
    render_brp_session_card,
    render_auction_card,
)
from crux_viz.visualizations import (
    plot_posterior_evolution,
    plot_evidence_polarity_donut,
    plot_claim_bundle_timeline,
    plot_session_stats_radar,
    plot_edr_checkpoints,
    plot_brp_summary,
    plot_credibility_snapshot,
    plot_auction_summary,
    create_crux_full_dashboard,
    _synthetic_crux_flow,
)
from crux_viz.protocol_explainer import (
    render_crux_pipeline_diagram,
    render_all_primitive_explanations,
    render_crux_algorithm_explanation,
    render_crux_flow_with_data,
)
from crux_viz.crux_engine import StreamingCRUXEngine, SpecialistDef


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize session state with defaults."""
    defaults = {
        "crux_result": None,
        "debate_running": False,
        "live_rounds": [],
        "live_claim_bundles": [],
        "live_auctions": [],
        "live_brp_sessions": [],
        "current_phase": "idle",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render the sidebar configuration panel. Returns config dict."""
    with st.sidebar:
        st.markdown("# ⚡ CRUX-Viz")
        st.markdown("*Claim-Routed Uncertainty eXchange*")
        st.markdown("---")

        # --- LLM Configuration ---
        st.markdown("### 🤖 LLM Configuration")

        try:
            from argus.core.llm import list_providers
            providers = list_providers()
        except Exception:
            providers = [
                "openai", "anthropic", "gemini", "ollama", "groq",
                "mistral", "cohere", "deepseek", "together",
            ]

        provider = st.selectbox(
            "Provider",
            options=providers,
            index=providers.index("groq") if "groq" in providers else 0,
            help="LLM provider for the CRUX debate.",
        )

        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.0-flash",
            "ollama": "llama3.1",
            "groq": "llama-3.3-70b-versatile",
            "mistral": "mistral-large-latest",
            "cohere": "command-r-plus",
            "deepseek": "deepseek-chat",
        }
        model = st.text_input(
            "Model Name",
            value=default_models.get(provider, ""),
            help="Model identifier. Leave empty for provider default.",
        )

        # Auto-load API key from environment for the selected provider
        _env_key_map = {
            "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY", "cohere": "COHERE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY", "together": "TOGETHER_API_KEY",
        }
        _env_val = os.environ.get(_env_key_map.get(provider, ""), "")
        api_key = st.text_input(
            "API Key",
            value=_env_val,
            type="password",
            help="API key for the provider (auto-loaded from .env / environment).",
        )

        st.markdown("---")

        # --- Debate Parameters ---
        st.markdown("### ⚙️ Debate Parameters")

        max_rounds = st.slider(
            "Max Rounds", min_value=1, max_value=10, value=3,
            help="Number of debate rounds.",
        )

        prior = st.slider(
            "Prior Probability", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Starting belief probability.",
        )

        jury_threshold = st.slider(
            "Jury Decision Threshold", min_value=0.5, max_value=0.95, value=0.7, step=0.05,
            help="Posterior threshold for 'supported' verdict.",
        )

        refuter_enabled = st.toggle(
            "Enable Refuter", value=True,
            help="Enable the Refuter agent to generate rebuttals.",
        )

        st.markdown("---")

        # --- CRUX-Specific Parameters ---
        st.markdown("### ⚡ CRUX Protocol Settings")

        contradiction_threshold = st.slider(
            "Contradiction Threshold (θ)", min_value=0.05, max_value=0.50, value=0.20, step=0.05,
            help="Minimum posterior difference to trigger Belief Reconciliation Protocol (BRP).",
        )

        enable_edr = st.toggle(
            "Enable EDR (Epistemic Dead Reckoning)", value=True,
            help="Create EDR checkpoints for reconnection sync.",
        )

        auction_timeout = st.slider(
            "Auction Timeout (s)", min_value=5, max_value=60, value=30, step=5,
            help="Challenger Auction timeout seconds.",
        )

        st.markdown("---")

        # --- Specialist Configuration ---
        st.markdown("### 🔬 Specialists")

        num_specialists = st.slider(
            "Number of Specialists", min_value=1, max_value=6, value=3,
            help="How many specialist agents participate.",
        )

        default_specialists = [
            ("Bull Analyst", "optimistic", "Find strong supporting evidence and positive indicators for the proposition."),
            ("Bear Analyst", "critical", "Find counter-evidence, risks, and reasons the proposition may be false."),
            ("Technical Analyst", "data-driven", "Provide quantitative data, statistics, and empirical evidence."),
            ("Domain Expert", "expert", "Leverage deep domain knowledge to assess the proposition."),
            ("Devil's Advocate", "contrarian", "Challenge assumptions and explore alternative explanations."),
            ("Synthesis Analyst", "balanced", "Weigh both sides and identify the strongest arguments."),
        ]

        specialists: list[dict] = []
        for i in range(num_specialists):
            default = default_specialists[i] if i < len(default_specialists) else (
                f"Specialist {i+1}", "general", "Analyze the proposition from your perspective."
            )

            with st.expander(f"Specialist {i+1}: {default[0]}", expanded=(i < 2)):
                name = st.text_input(f"Name##sp{i}", value=default[0], key=f"sp_name_{i}")
                persona = st.text_input(f"Persona##sp{i}", value=default[1], key=f"sp_persona_{i}")
                instruction = st.text_area(
                    f"Instruction##sp{i}", value=default[2],
                    height=80, key=f"sp_instr_{i}")
                specialists.append({
                    "name": name, "persona": persona, "instruction": instruction,
                })

        st.markdown("---")

        # --- Proposition ---
        st.markdown("### 📝 Proposition")
        proposition = st.text_area(
            "Enter the claim to debate",
            value="Machine learning models can achieve human-level performance on complex reasoning tasks.",
            height=100,
            help="The proposition that CRUX agents will debate.",
        )

        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "max_rounds": max_rounds,
            "prior": prior,
            "jury_threshold": jury_threshold,
            "refuter_enabled": refuter_enabled,
            "num_specialists": num_specialists,
            "specialists": specialists,
            "proposition": proposition,
            # CRUX-specific
            "contradiction_threshold": contradiction_threshold,
            "enable_edr": enable_edr,
            "auction_timeout": auction_timeout,
        }


# ---------------------------------------------------------------------------
# Debate Runner
# ---------------------------------------------------------------------------

def run_debate(config: dict):
    """Execute the CRUX debate and update session state with results."""
    if config["api_key"]:
        key_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "together": "TOGETHER_API_KEY",
        }
        env_var = key_env_map.get(config["provider"], f"{config['provider'].upper()}_API_KEY")
        os.environ[env_var] = config["api_key"]

    # Initialize LLM
    try:
        from argus.core.llm import get_llm
        llm = get_llm(
            provider=config["provider"],
            model=config["model"] or None,
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {e}")
        st.info("Make sure you have the correct API key set and the provider is installed.")
        return

    # Build specialist definitions
    specialist_defs = [
        SpecialistDef(
            name=s["name"],
            persona=s["persona"],
            instruction=s["instruction"],
        )
        for s in config["specialists"]
    ]

    # Create engine
    engine = StreamingCRUXEngine(
        llm=llm,
        specialists=specialist_defs,
        max_rounds=config["max_rounds"],
        refuter_enabled=config["refuter_enabled"],
        jury_threshold=config["jury_threshold"],
        prior=config["prior"],
        contradiction_threshold=config["contradiction_threshold"],
        enable_edr=config["enable_edr"],
        auction_timeout_seconds=config["auction_timeout"],
    )

    # --- Live UI containers ---
    phase_status   = st.empty()
    progress_bar   = st.progress(0)

    live_col1, live_col2 = st.columns(2)
    live_posterior_container = live_col1.empty()
    live_flow_container      = live_col2.empty()

    round_log_container = st.container()

    class StreamlitCRUXCallback:
        def __init__(self):
            self.rounds_so_far: list[dict] = []
            self.claim_bundles: list[dict] = []
            self.auctions: list[dict] = []
            self.brp_sessions: list[dict] = []
            self.proposition = config["proposition"]

        def on_phase_change(self, phase: str):
            phase_labels = {
                "initializing": "🔧 Initializing CRUX session...",
                "debate": "⚡ Debate in progress...",
                "reconciliation": "🔀 Running Belief Reconciliation Protocol...",
                "verdict": "⚖️ Jury rendering final verdict...",
            }
            phase_status.info(phase_labels.get(phase, f"📌 Phase: {phase}"))
            st.session_state["current_phase"] = phase

        def on_round_start(self, round_num: int, total_rounds: int):
            progress = round_num / total_rounds
            progress_bar.progress(progress, text=f"⚡ Round {round_num}/{total_rounds}")
            phase_status.info(f"🔄 **Round {round_num}** — Specialists gathering evidence...")

        def on_specialist_evidence(self, specialist: str, evidence: list[dict]):
            phase_status.info(f"🔬 **{specialist}** submitted {len(evidence)} evidence items")

        def on_rebuttal(self, rebuttals: list[dict]):
            if rebuttals:
                phase_status.info(f"⚔️ **Refuter** generated {len(rebuttals)} rebuttals")

        def on_claim_bundle(self, cb: dict):
            self.claim_bundles.append(cb)
            st.session_state["live_claim_bundles"] = list(self.claim_bundles)

        def on_auction(self, auction: dict):
            self.auctions.append(auction)
            st.session_state["live_auctions"] = list(self.auctions)

        def on_brp(self, brp: dict):
            self.brp_sessions.append(brp)
            st.session_state["live_brp_sessions"] = list(self.brp_sessions)
            phase_status.warning(
                f"🔀 **BRP Triggered** — Contradiction Δ={brp.get('contradiction_delta', 0):.3f}"
            )

        def on_round_complete(self, round_data: dict):
            self.rounds_so_far.append(round_data)
            rnd = round_data["round"]

            # Update live posterior chart
            with live_posterior_container.container():
                if self.rounds_so_far:
                    fig = plot_posterior_evolution(self.rounds_so_far)
                    st.plotly_chart(fig, use_container_width=True, key=f"live_posterior_{rnd}")

            # Update live CRUX flow
            with live_flow_container.container():
                flow_fig = _synthetic_crux_flow(
                    self.claim_bundles, self.brp_sessions, self.auctions
                )
                if flow_fig:
                    st.plotly_chart(flow_fig, use_container_width=True, key=f"live_flow_{rnd}")

            with round_log_container:
                render_round_summary(round_data)

        def on_verdict(self, verdict: dict):
            progress_bar.progress(1.0, text="✅ CRUX Debate Complete!")
            phase_status.success(
                f"**Verdict: {verdict['label'].upper()}** | "
                f"Posterior: {verdict['posterior']:.3f}"
            )

    callback = StreamlitCRUXCallback()

    st.session_state["debate_running"] = True
    try:
        result = engine.run_debate(config["proposition"], callback=callback)
        st.session_state["crux_result"] = result
        st.session_state["live_rounds"] = callback.rounds_so_far
    except Exception as e:
        st.error(f"❌ CRUX Debate failed: {e}")
        import traceback
        st.code(traceback.format_exc(), language="text")
    finally:
        st.session_state["debate_running"] = False


# ---------------------------------------------------------------------------
# Tab 1: CRUX Arena
# ---------------------------------------------------------------------------

def render_crux_arena(config: dict):
    """Render the CRUX Arena tab."""
    st.markdown("## ⚡ CRUX Arena")

    render_crux_config_summary(config)

    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button(
            "🚀 **Run CRUX Debate**",
            disabled=st.session_state.get("debate_running", False),
            use_container_width=True,
            type="primary",
        )

    with col2:
        if not config["api_key"] and not any(
            os.environ.get(k) for k in [
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                "GROQ_API_KEY", "MISTRAL_API_KEY",
            ]
        ):
            st.warning("⚠️ No API key set. Enter one in the sidebar or set it as an environment variable.")

    if run_clicked:
        run_debate(config)

    result = st.session_state.get("crux_result")
    if result:
        st.markdown("---")
        st.markdown("## 📋 Final Result")
        render_verdict_card(result["verdict"])

        rounds = result.get("rounds", [])
        total_ev = sum(r.get("total_evidence", 0) for r in rounds)
        total_rb = sum(r.get("total_rebuttals", 0) for r in rounds)
        total_cbs = result.get("crux_session_stats", {}).get("num_claim_bundles", 0)
        duration = result.get("duration_seconds", 0)

        render_metrics_row([
            (str(len(rounds)), "Rounds", "#ff00d4"),
            (str(total_ev), "Evidence Items", "#00ff88"),
            (str(total_cbs), "Claim Bundles", "#00d4ff"),
            (str(total_rb), "Rebuttals", "#ff8800"),
            (f"{duration:.1f}s", "Duration", "#b388ff"),
        ])

        # CRUX session stats
        st.markdown("---")
        st.markdown("### ⚡ CRUX Session Statistics")
        render_crux_session_stats(result.get("crux_session_stats", {}))

        # Evidence cards
        st.markdown("### 📑 All Evidence")
        for ev in result.get("all_evidence", []):
            render_evidence_card(ev)

        # BRP sessions
        if result.get("brp_sessions"):
            st.markdown("---")
            st.markdown("### 🔀 BRP Reconciliation Sessions")
            for brp in result["brp_sessions"]:
                render_brp_session_card(brp)

        # Auctions
        if result.get("auctions"):
            st.markdown("---")
            st.markdown("### 🏆 Challenger Auctions")
            for auction in result["auctions"]:
                render_auction_card(auction)


# ---------------------------------------------------------------------------
# Tab 2: Analysis Dashboard
# ---------------------------------------------------------------------------

def render_analysis_dashboard():
    """Render the full Analysis Dashboard tab."""
    st.markdown("## 📊 CRUX Analysis Dashboard")

    result = st.session_state.get("crux_result")
    if not result:
        st.info("🔬 Run a CRUX debate first to see the analysis dashboard.")
        return

    charts = create_crux_full_dashboard(result)

    # Row 1: Posterior + Polarity
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(charts["posterior_evolution"], use_container_width=True, key="dash_posterior")
    with col2:
        st.plotly_chart(charts["evidence_polarity"], use_container_width=True, key="dash_polarity")

    # Row 2: CB Timeline + Session Radar
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts["cb_timeline"], use_container_width=True, key="dash_cb_timeline")
    with col2:
        st.plotly_chart(charts["session_radar"], use_container_width=True, key="dash_session_radar")

    # Row 3: BRP Summary + Credibility Snapshot
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts["brp_summary"], use_container_width=True, key="dash_brp")
    with col2:
        st.plotly_chart(charts["credibility_snapshot"], use_container_width=True, key="dash_credibility")

    # Row 4: Auction Summary + EDR Checkpoints
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts["auction_summary"], use_container_width=True, key="dash_auction")
    with col2:
        st.plotly_chart(charts["edr_checkpoints"], use_container_width=True, key="dash_edr")

    # Row 5: Full CRUX Debate Flow (full width)
    st.markdown("---")
    st.plotly_chart(charts["crux_debate_flow"], use_container_width=True, key="dash_crux_flow")

    # Claim Bundles detail
    st.markdown("---")
    st.markdown("### 📦 All Claim Bundles")
    for cb in result.get("claim_bundles", []):
        render_claim_bundle_card(cb)


# ---------------------------------------------------------------------------
# Tab 3: CRUX Flow
# ---------------------------------------------------------------------------

def render_crux_flow_tab():
    """Render the live CRUX Flow visualization tab."""
    st.markdown("## 🔄 CRUX Debate Flow")

    result = st.session_state.get("crux_result")
    if result:
        claim_bundles = result.get("claim_bundles", [])
        brp_sessions  = result.get("brp_sessions", [])
        auctions      = result.get("auctions", [])

        flow_fig = _synthetic_crux_flow(claim_bundles, brp_sessions, auctions)
        if flow_fig:
            st.plotly_chart(flow_fig, use_container_width=True, key="flow_tab_main")

        st.markdown("---")
        render_crux_flow_with_data(result)
    else:
        # Show empty flow placeholder
        flow_fig = _synthetic_crux_flow([], [], [])
        if flow_fig:
            st.plotly_chart(flow_fig, use_container_width=True, key="flow_tab_empty")
        st.info("🔬 Run a CRUX debate to see the live debate flow visualization.")


# ---------------------------------------------------------------------------
# Tab 4: Protocol Explainer
# ---------------------------------------------------------------------------

def render_protocol_tab():
    """Render the Protocol explanation tab."""
    st.markdown("## 📖 CRUX Protocol — 7 Primitives")
    st.markdown(
        "*Claim-Routed Uncertainty eXchange (CRUX) v1.0 — "
        "A novel inter-agent communication protocol for Bayesian multi-agent reasoning.*"
    )

    # Sankey pipeline
    pipeline_fig = render_crux_pipeline_diagram()
    st.plotly_chart(pipeline_fig, use_container_width=True, key="protocol_sankey")

    st.markdown("---")

    # Step explanations
    st.markdown("## 🔬 Step-by-Step Primitive Explanations")
    render_all_primitive_explanations()

    # Algorithm
    st.markdown("---")
    render_crux_algorithm_explanation()


# ---------------------------------------------------------------------------
# Tab 5: Raw Data
# ---------------------------------------------------------------------------

def render_raw_data_tab():
    """Render the Raw Data export tab."""
    st.markdown("## 📋 Raw Data Export")

    result = st.session_state.get("crux_result")
    if not result:
        st.info("🔬 Run a CRUX debate first to see the raw data.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Debate Result")
        export = {
            k: v for k, v in result.items()
            if k not in ("claim_bundles", "brp_sessions", "auctions")
        }
        st.json(export)

    with col2:
        st.markdown("### CRUX Session Stats")
        st.json(result.get("crux_session_stats", {}))

        st.markdown("### Credibility Ledger")
        st.json(result.get("credibility_ledger", []))

        st.markdown("### EDR Checkpoints")
        st.json(result.get("edr_checkpoints", []))

    st.markdown("---")
    st.markdown("### Claim Bundles")
    for cb in result.get("claim_bundles", []):
        st.json(cb)

    st.markdown("---")
    download_data = json.dumps(result, indent=2, default=str)
    st.download_button(
        "📥 Download Full CRUX Result (JSON)",
        data=download_data,
        file_name="crux_debate_result.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_app():
    """Main CRUX-Viz application entry point."""
    inject_custom_css()
    init_session_state()

    # Sidebar
    config = render_sidebar()

    # Main area tabs
    tab_arena, tab_dashboard, tab_flow, tab_protocol, tab_data = st.tabs([
        "⚡ CRUX Arena",
        "📊 Analysis Dashboard",
        "🔄 CRUX Flow",
        "📖 Protocol",
        "📋 Raw Data",
    ])

    with tab_arena:
        render_crux_arena(config)

    with tab_dashboard:
        render_analysis_dashboard()

    with tab_flow:
        render_crux_flow_tab()

    with tab_protocol:
        render_protocol_tab()

    with tab_data:
        render_raw_data_tab()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main_app()
else:
    # When loaded by Streamlit
    main_app()
