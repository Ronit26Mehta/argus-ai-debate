"""
Argus-Viz: Interactive Debate Visualization Sandbox.

Main Streamlit application. Launch with:
    argus-viz
    python -m argus_viz
    streamlit run argus_viz/app.py
"""

from __future__ import annotations

import os
import json
import streamlit as st

# Must be first Streamlit call
st.set_page_config(
    page_title="Argus-Viz | Debate Sandbox",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from argus_viz.components import (
    inject_custom_css,
    render_verdict_card,
    render_evidence_card,
    render_round_summary,
    render_agent_status,
    render_debate_config_summary,
    render_metrics_row,
)
from argus_viz.visualizations import (
    plot_posterior_evolution,
    plot_evidence_waterfall,
    plot_cdag_network,
    plot_specialist_radar,
    plot_confidence_histogram,
    plot_debate_timeline,
    plot_evidence_polarity_donut,
    plot_round_heatmap,
    plot_debate_flow_graph,
    create_dashboard,
)
from argus_viz.flow_explainer import (
    render_debate_flow_diagram,
    render_all_step_explanations,
    render_algorithm_explanation,
    render_flow_with_data,
)
from argus_viz.debate_engine import StreamingDebateEngine, SpecialistDef


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize session state with defaults."""
    defaults = {
        "debate_result": None,
        "debate_running": False,
        "live_rounds": [],
        "live_evidence": [],
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
        st.markdown("# ⚔️ Argus-Viz")
        st.markdown("*Interactive Debate Sandbox*")
        st.markdown("---")

        # --- LLM Configuration ---
        st.markdown("### 🤖 LLM Configuration")

        # Provider selection
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
            index=providers.index("gemini") if "gemini" in providers else 0,
            help="Select the LLM provider to use for the debate.",
        )

        # Model name
        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.0-flash",
            "ollama": "llama3.1",
            "groq": "llama-3.1-70b-versatile",
            "mistral": "mistral-large-latest",
            "cohere": "command-r-plus",
            "deepseek": "deepseek-chat",
        }
        model = st.text_input(
            "Model Name",
            value=default_models.get(provider, ""),
            help="Model identifier. Leave empty for provider default.",
        )

        # API Key
        api_key = st.text_input(
            "API Key",
            type="password",
            help="API key for the selected provider. Set here or via environment variable.",
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
            help="The proposition that specialists will debate.",
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
        }


# ---------------------------------------------------------------------------
# Debate Runner
# ---------------------------------------------------------------------------

def run_debate(config: dict):
    """Execute the debate and update session state with results."""
    # Set API key in environment
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
    engine = StreamingDebateEngine(
        llm=llm,
        specialists=specialist_defs,
        max_rounds=config["max_rounds"],
        refuter_enabled=config["refuter_enabled"],
        jury_threshold=config["jury_threshold"],
        prior=config["prior"],
    )

    # --- Live UI containers ---
    status_container = st.empty()
    progress_bar = st.progress(0)

    # Two side-by-side live charts: posterior + flow graph
    live_col1, live_col2 = st.columns(2)
    live_chart_container = live_col1.empty()
    live_flow_container = live_col2.empty()

    round_log_container = st.container()

    # Streaming callback that updates UI
    class StreamlitCallback:
        def __init__(self):
            self.rounds_so_far: list[dict] = []
            self.proposition = config["proposition"]

        def on_round_start(self, round_num: int, total_rounds: int):
            progress = round_num / total_rounds
            progress_bar.progress(progress, text=f"⚔️ Round {round_num}/{total_rounds}")
            status_container.info(f"🔄 **Round {round_num}** — Specialists gathering evidence...")

        def on_specialist_evidence(self, specialist: str, evidence: list[dict]):
            status_container.info(f"🔬 **{specialist}** submitted {len(evidence)} evidence items")

        def on_rebuttal(self, rebuttals: list[dict]):
            if rebuttals:
                status_container.info(f"⚔️ **Refuter** generated {len(rebuttals)} rebuttals")

        def on_round_complete(self, round_data: dict):
            self.rounds_so_far.append(round_data)
            rnd = round_data["round"]

            # Update live posterior chart
            with live_chart_container.container():
                if self.rounds_so_far:
                    fig = plot_posterior_evolution(self.rounds_so_far)
                    st.plotly_chart(fig, use_container_width=True, key=f"live_posterior_{rnd}")

            # Update live debate flow graph incrementally
            with live_flow_container.container():
                partial_result = {
                    "proposition": self.proposition,
                    "rounds": list(self.rounds_so_far),
                    "verdict": {
                        "label": "in progress",
                        "posterior": round_data.get("posterior_after", 0.5),
                        "reasoning": f"Debate in progress — {rnd} round(s) completed",
                    },
                    "graph_data": {"nodes": [], "edges": []},
                }
                flow_fig = plot_debate_flow_graph(partial_result)
                st.plotly_chart(flow_fig, use_container_width=True, key=f"live_flow_{rnd}")

            # Log round details
            with round_log_container:
                render_round_summary(round_data)

        def on_verdict(self, verdict: dict):
            progress_bar.progress(1.0, text="✅ Debate Complete!")
            status_container.success(
                f"**Verdict: {verdict['label'].upper()}** | "
                f"Posterior: {verdict['posterior']:.3f}"
            )

    callback = StreamlitCallback()

    # Run debate
    st.session_state["debate_running"] = True
    try:
        result = engine.run_debate(config["proposition"], callback=callback)
        st.session_state["debate_result"] = result
        st.session_state["live_rounds"] = callback.rounds_so_far
    except Exception as e:
        st.error(f"❌ Debate failed: {e}")
        import traceback
        st.code(traceback.format_exc(), language="text")
    finally:
        st.session_state["debate_running"] = False


# ---------------------------------------------------------------------------
# Tab: Debate Arena
# ---------------------------------------------------------------------------

def render_debate_arena(config: dict):
    """Render the Debate Arena tab."""
    st.markdown("## ⚔️ Debate Arena")

    # Config summary
    render_debate_config_summary(config)

    # Run button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button(
            "🚀 **Run Debate**",
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

    # Show verdict if available
    result = st.session_state.get("debate_result")
    if result:
        st.markdown("---")
        st.markdown("## 📋 Final Result")
        render_verdict_card(result["verdict"])

        # Quick metrics
        rounds = result.get("rounds", [])
        total_ev = sum(r.get("total_evidence", 0) for r in rounds)
        total_rb = sum(r.get("total_rebuttals", 0) for r in rounds)
        duration = result.get("duration_seconds", 0)

        render_metrics_row([
            (str(len(rounds)), "Rounds", "#00d4ff"),
            (str(total_ev), "Evidence Items", "#00ff88"),
            (str(total_rb), "Rebuttals", "#ff8800"),
            (f"{duration:.1f}s", "Duration", "#b388ff"),
        ])

        # All evidence cards
        st.markdown("### 📑 All Evidence")
        for ev in result.get("all_evidence", []):
            render_evidence_card(ev)

        # Debate Flow Graph — full lifecycle visualization
        st.markdown("---")
        st.markdown("### 🔀 Debate Flow Graph")
        flow_fig = plot_debate_flow_graph(result)
        st.plotly_chart(flow_fig, use_container_width=True, key="arena_final_flow")


# ---------------------------------------------------------------------------
# Tab: Analysis Dashboard
# ---------------------------------------------------------------------------

def render_analysis_dashboard():
    """Render the Analysis Dashboard tab."""
    st.markdown("## 📊 Analysis Dashboard")

    result = st.session_state.get("debate_result")
    if not result:
        st.info("🔬 Run a debate first to see the analysis dashboard.")
        return

    charts = create_dashboard(result)

    # Row 1: Posterior + Polarity
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(charts["posterior_evolution"], use_container_width=True, key="dash_posterior")
    with col2:
        st.plotly_chart(charts["evidence_polarity"], use_container_width=True, key="dash_polarity")

    # Row 2: Waterfall + Radar
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts["evidence_waterfall"], use_container_width=True, key="dash_waterfall")
    with col2:
        st.plotly_chart(charts["specialist_radar"], use_container_width=True, key="dash_radar")

    # Row 3: Timeline + Heatmap
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts["debate_timeline"], use_container_width=True, key="dash_timeline")
    with col2:
        st.plotly_chart(charts["round_heatmap"], use_container_width=True, key="dash_heatmap")

    # Row 4: Confidence + CDAG
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts["confidence_histogram"], use_container_width=True, key="dash_confidence")
    with col2:
        st.plotly_chart(charts["cdag_network"], use_container_width=True, key="dash_cdag")

    # Row 5: Full debate flow graph (full width)
    st.markdown("---")
    st.plotly_chart(charts["debate_flow_graph"], use_container_width=True, key="dash_flow_graph")


# ---------------------------------------------------------------------------
# Tab: Debate Flow
# ---------------------------------------------------------------------------

def render_debate_flow_tab():
    """Render the Debate Flow explanation tab."""
    st.markdown("## 🗺️ Debate Flow — How ARGUS Works")

    # Pipeline diagram
    fig = render_debate_flow_diagram()
    st.plotly_chart(fig, use_container_width=True, key="flow_tab_sankey")

    st.markdown("---")

    # Step explanations
    st.markdown("## 📖 Step-by-Step Explanation")
    render_all_step_explanations()

    # Algorithm
    st.markdown("---")
    render_algorithm_explanation()

    # Data overlay if debate ran
    result = st.session_state.get("debate_result")
    if result:
        st.markdown("---")
        render_flow_with_data(result)


# ---------------------------------------------------------------------------
# Tab: Raw Data
# ---------------------------------------------------------------------------

def render_raw_data_tab():
    """Render the Raw Data tab."""
    st.markdown("## 📋 Raw Data Export")

    result = st.session_state.get("debate_result")
    if not result:
        st.info("🔬 Run a debate first to see the raw data.")
        return

    # JSON export
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Debate Result")
        # Make a serializable copy (remove non-serializable items)
        export = {k: v for k, v in result.items() if k != "graph_data"}
        st.json(export)

    with col2:
        st.markdown("### Graph Summary")
        st.json(result.get("graph_summary", {}))

        st.markdown("### Graph Data")
        graph_data = result.get("graph_data", {})
        st.metric("Nodes", len(graph_data.get("nodes", [])))
        st.metric("Edges", len(graph_data.get("edges", [])))

        if st.button("Show Full Graph Data"):
            st.json(graph_data)

    # Download button
    st.markdown("---")
    download_data = json.dumps(result, indent=2, default=str)
    st.download_button(
        "📥 Download Full Result (JSON)",
        data=download_data,
        file_name="argus_debate_result.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_app():
    """Main application entry point."""
    inject_custom_css()
    init_session_state()

    # Sidebar
    config = render_sidebar()

    # Main area tabs
    tab_arena, tab_dashboard, tab_flow, tab_data = st.tabs([
        "⚔️ Debate Arena",
        "📊 Analysis Dashboard",
        "🗺️ Debate Flow",
        "📋 Raw Data",
    ])

    with tab_arena:
        render_debate_arena(config)

    with tab_dashboard:
        render_analysis_dashboard()

    with tab_flow:
        render_debate_flow_tab()

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
