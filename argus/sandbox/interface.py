"""ARGUS Sandbox Streamlit interface with credential-gated launch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from argus_viz.visualizations import create_dashboard

st.set_page_config(
    page_title="ARGUS Sandbox",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

from argus.sandbox.lifecycle import LifecycleDAG
from argus.sandbox.orchestrator import ArgusSandboxRunner, SandboxConfig


def _defaults() -> None:
    if "events" not in st.session_state:
        st.session_state["events"] = []
    if "summary" not in st.session_state:
        st.session_state["summary"] = None
    if "stage_payloads" not in st.session_state:
        st.session_state["stage_payloads"] = {}
    if "runner" not in st.session_state:
        st.session_state["runner"] = None
    if "lifecycle" not in st.session_state:
        st.session_state["lifecycle"] = LifecycleDAG()
    if "provider" not in st.session_state:
        st.session_state["provider"] = os.environ.get("ARGUS_DEFAULT_PROVIDER", "gemini")
    if "model" not in st.session_state:
        st.session_state["model"] = os.environ.get("ARGUS_DEFAULT_MODEL", "gemini-2.0-flash")
    if "model_source" not in st.session_state:
        st.session_state["model_source"] = os.environ.get("ARGUS_SANDBOX_MODEL_SOURCE", "env")
    if "local_server_url" not in st.session_state:
        st.session_state["local_server_url"] = os.environ.get("ARGUS_SANDBOX_LOCAL_URL", "http://localhost:8080")
    if "local_model_name" not in st.session_state:
        st.session_state["local_model_name"] = os.environ.get("ARGUS_SANDBOX_LOCAL_MODEL", "local-model")
    if "local_server_status" not in st.session_state:
        st.session_state["local_server_status"] = ""


def _test_local_connection(url: str) -> str:
    endpoint = url.rstrip("/") + "/v1/models"
    try:
        req = urllib.request.Request(endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        models = [m.get("id", "?") for m in data.get("data", [])]
        if models:
            return f"Connected: {', '.join(models)}"
        return "Connected: no models listed"
    except urllib.error.URLError as exc:
        return f"Connection failed: {exc.reason}"
    except Exception as exc:
        return f"Connection failed: {exc}"


def _get_llm() -> Any:
    if st.session_state.get("model_source") == "local":
        from argus.core.llm.openai import OpenAILLM

        base_url = st.session_state.get("local_server_url", "http://localhost:8080").rstrip("/")
        model_name = st.session_state.get("local_model_name", "local-model")
        return OpenAILLM(
            model=model_name,
            base_url=f"{base_url}/v1",
            api_key="not-needed",
            max_tokens=2048,
            timeout=120.0,
        )

    from argus.core.llm import get_llm

    return get_llm(
        provider=st.session_state["provider"],
        model=st.session_state["model"],
    )


def _apply_event(event: dict[str, Any]) -> None:
    stage = event.get("stage", "system")
    message = event.get("message", "")
    payload = event.get("payload", {})
    st.session_state["events"].append(event)
    st.session_state["events"] = st.session_state["events"][-300:]

    lifecycle: LifecycleDAG = st.session_state["lifecycle"]
    if stage in lifecycle.stages:
        if message == "completed":
            lifecycle.complete(stage, payload)
        elif message == "run_failed":
            lifecycle.fail(stage, payload.get("error", "unknown"))
        else:
            lifecycle.start(stage, payload)

    if stage not in ("system",):
        st.session_state["stage_payloads"][stage] = payload


def _render_status() -> None:
    lifecycle: LifecycleDAG = st.session_state["lifecycle"]
    dag_fig = lifecycle.to_plotly()
    if dag_fig is not None:
        st.plotly_chart(dag_fig, use_container_width=True)
    else:
        st.info("Install plotly to render lifecycle DAG.")


def _render_events() -> None:
    st.subheader("Incremental Output")
    for event in reversed(st.session_state["events"][-40:]):
        st.write(f"[{event.get('ts', '')}] {event.get('stage', 'system')} :: {event.get('message', '')}")


def _render_summary() -> None:
    summary = st.session_state.get("summary")
    if not summary:
        return
    st.subheader("Result")
    outputs = summary.get("outputs", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", outputs.get("verdict", "-"))
    c2.metric("Posterior", f"{outputs.get('posterior', 0.0):.3f}")
    c3.metric("Population Mean", f"{outputs.get('population_mean', 0.0):.3f}")
    c4.metric("Consequences", outputs.get("mirror_consequences", 0))

    with st.expander("Full Summary JSON", expanded=False):
        st.json(summary)


def _load_stage_artifact(summary: dict[str, Any], stage_name: str) -> dict[str, Any] | None:
    try:
        root = Path(summary["paths"]["root"]) / "stages"
        stage_file = root / f"{stage_name}.json"
        if stage_file.exists():
            return json.loads(stage_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _plot_chronos_from_raw(chronos_data: dict[str, Any]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.12)
    raw = chronos_data.get("result", {}).get("temporal_posterior", {})
    snaps = raw.get("snapshots", [])

    if not snaps:
        fig.add_annotation(
            text="No temporal posterior data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#E2E8F0"),
        )
    else:
        x = [s.get("time", "") for s in snaps]
        y = [float(s.get("posterior", 0.5)) for s in snaps]

        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#38BDF8", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=[int(s.get("num_evidence", 0)) for s in snaps],
                marker=dict(color="#2DD4BF"),
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=0.5, line_color="#64748B", line_dash="dot", row=1, col=1)

        for inf in chronos_data.get("result", {}).get("drift_report", {}).get("inflections", []):
            fig.add_vline(x=inf.get("time", ""), line_color="#F59E0B", line_dash="dash", row=1, col=1)

    fig.update_layout(
        title="CHRONOS Temporal Posterior",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        height=440,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _plot_phalanx_from_raw(phalanx_data: dict[str, Any]) -> go.Figure:
    beliefs = [float(v) for v in phalanx_data.get("beliefs", [])]
    fig = go.Figure()

    if not beliefs:
        fig.add_annotation(
            text="No PHALANX beliefs found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#E2E8F0"),
        )
    else:
        fig.add_trace(
            go.Histogram(
                x=beliefs,
                nbinsx=25,
                marker=dict(color="rgba(56,189,248,0.65)", line=dict(color="#38BDF8", width=1)),
                name="Belief Distribution",
            )
        )
        fig.add_vline(
            x=float(phalanx_data.get("population_mean", 0.5)),
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text="mean",
        )

    fig.update_layout(
        title=f"PHALANX Population Posterior ({phalanx_data.get('consensus_type', 'UNKNOWN')})",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(range=[0, 1], title="Belief"),
        yaxis=dict(title="Count"),
        height=440,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _plot_fractal_from_raw(fractal_data: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    leaves = fractal_data.get("num_leaves", 0)
    max_depth = fractal_data.get("max_depth", 0)
    root = float(fractal_data.get("root_posterior", 0.5))
    fig.add_trace(
        go.Bar(
            x=["Root Posterior", "Leaf Count", "Max Depth"],
            y=[root, leaves, max_depth],
            marker=dict(color=["#38BDF8", "#2DD4BF", "#A78BFA"]),
            text=[f"{root:.3f}", str(leaves), str(max_depth)],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="FRACTAL Decomposition Summary",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        height=440,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _plot_mirror_from_raw(mirror_data: dict[str, Any]) -> go.Figure:
    graph = mirror_data.get("graph", {})
    nodes = graph.get("nodes", [])
    fig = go.Figure()

    if not nodes:
        fig.add_annotation(
            text="No MIRROR consequence nodes",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#E2E8F0"),
        )
    else:
        labels = [n.get("text", "consequence")[:40] for n in nodes[:12]]
        marginals = [float(n.get("marginal_probability", 0.0)) for n in nodes[:12]]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=marginals,
                marker=dict(color="#FB7185"),
            )
        )

    fig.update_layout(
        title="MIRROR Consequence Probabilities",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(title="Consequence", tickangle=-30),
        yaxis=dict(title="Marginal Probability", range=[0, 1]),
        height=440,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _render_module_visuals() -> None:
    summary = st.session_state.get("summary")
    if not summary:
        return

    st.subheader("Module Visualizations")
    chronos_data = _load_stage_artifact(summary, "chronos")
    phalanx_data = _load_stage_artifact(summary, "phalanx")
    fractal_data = _load_stage_artifact(summary, "fractal")
    mirror_data = _load_stage_artifact(summary, "mirror")

    c1, c2 = st.columns(2)
    with c1:
        if chronos_data:
            st.plotly_chart(_plot_chronos_from_raw(chronos_data), use_container_width=True)
        else:
            st.info("Chronos data not available yet.")

    with c2:
        if phalanx_data:
            st.plotly_chart(_plot_phalanx_from_raw(phalanx_data), use_container_width=True)
        else:
            st.info("Phalanx data not available yet.")

    c3, c4 = st.columns(2)
    with c3:
        if fractal_data:
            st.plotly_chart(_plot_fractal_from_raw(fractal_data), use_container_width=True)
        else:
            st.info("Fractal tree not available yet.")

    with c4:
        if mirror_data:
            st.plotly_chart(_plot_mirror_from_raw(mirror_data), use_container_width=True)
        else:
            st.info("Mirror graph not available yet.")


def _render_debate_flow() -> None:
    summary = st.session_state.get("summary")
    if not summary:
        return

    debate_data = _load_stage_artifact(summary, "debate-flow")
    if not debate_data:
        debate_data = _load_stage_artifact(summary, "debate_flow")
    if not debate_data:
        st.warning("Debate flow data not available yet.")
        return

    st.subheader("Full Debate Flow")

    verdict = debate_data.get("verdict", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Label", verdict.get("label", "undecided"))
    c2.metric("Posterior", f"{float(verdict.get('posterior', 0.5)):.3f}")
    c3.metric("Confidence", f"{float(verdict.get('confidence', 0.5)):.3f}")
    c4.metric("Rounds", len(debate_data.get("rounds", [])))

    rounds = debate_data.get("rounds", [])
    with st.expander("Round-by-Round Details", expanded=True):
        if not rounds:
            st.info("No round snapshots available.")
        for round_data in rounds:
            round_num = round_data.get("round", "?")
            st.markdown(f"### Round {round_num}")
            st.write(
                f"Posterior {float(round_data.get('posterior_before', 0.5)):.3f} -> "
                f"{float(round_data.get('posterior_after', 0.5)):.3f} | "
                f"Support {round_data.get('support_count', 0)} | "
                f"Attack {round_data.get('attack_count', 0)} | "
                f"Rebuttals {len(round_data.get('rebuttals', []))}"
            )

            ev = round_data.get("evidence", [])
            if ev:
                st.caption("Evidence")
                for item in ev[:8]:
                    pol = "SUPPORT" if item.get("polarity", 0) > 0 else "ATTACK"
                    st.write(
                        f"- [{item.get('specialist', 'Unknown')}] {pol} | "
                        f"conf={float(item.get('confidence', 0.5)):.2f} | "
                        f"{item.get('claim', '')[:180]}"
                    )

            rb = round_data.get("rebuttals", [])
            if rb:
                st.caption("Rebuttals")
                for item in rb[:4]:
                    st.write(f"- {item.get('text', '')[:200]}")

            st.divider()

    charts = create_dashboard(debate_data)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(charts["posterior_evolution"], use_container_width=True)
    with r1c2:
        st.plotly_chart(charts["evidence_waterfall"], use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(charts["cdag_network"], use_container_width=True)
    with r2c2:
        st.plotly_chart(charts["specialist_radar"], use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(charts["debate_timeline"], use_container_width=True)
    with r3c2:
        st.plotly_chart(charts["round_heatmap"], use_container_width=True)

    st.plotly_chart(charts["debate_flow_graph"], use_container_width=True)


def _run_pipeline(
    proposition: str,
    prior: float,
    source_text: str,
    domain: str,
    storage_dir: str,
    max_rounds: int,
    population_size: int,
) -> None:
    llm = _get_llm()
    runner = ArgusSandboxRunner(
        SandboxConfig(
            storage_dir=storage_dir,
            max_rounds=max_rounds,
            population_size=population_size,
        ),
        llm=llm,
    )
    st.session_state["runner"] = runner
    st.session_state["events"] = []
    st.session_state["summary"] = None
    st.session_state["stage_payloads"] = {}
    st.session_state["lifecycle"] = LifecycleDAG()

    stream = runner.run_iter(
        proposition=proposition,
        prior=prior,
        source_text=source_text.strip() or None,
        domain=domain,
    )

    try:
        while True:
            event = next(stream)
            _apply_event(event)
    except StopIteration as stop:
        st.session_state["summary"] = stop.value


def main() -> None:
    _defaults()

    st.title("ARGUS Sandbox")
    st.caption("Realistic, local-storage, all-modules evolution pipeline")

    with st.sidebar:
        st.subheader("LLM Configuration")

        model_source = st.radio(
            "Model Source",
            options=["env", "local"],
            format_func=lambda x: "Environment / Registry" if x == "env" else "Local Server (OpenAI-compatible)",
            index=0 if st.session_state.get("model_source", "env") == "env" else 1,
        )
        st.session_state["model_source"] = model_source

        if model_source == "local":
            local_url = st.text_input("Server URL", value=st.session_state.get("local_server_url", "http://localhost:8080"))
            local_model = st.text_input("Model Name", value=st.session_state.get("local_model_name", "local-model"))
            st.session_state["local_server_url"] = local_url
            st.session_state["local_model_name"] = local_model

            if st.button("Test Local Connection", use_container_width=True):
                st.session_state["local_server_status"] = _test_local_connection(local_url)

            status = st.session_state.get("local_server_status", "")
            if status:
                if status.startswith("Connected"):
                    st.success(status)
                else:
                    st.error(status)
        else:
            try:
                from argus.core.llm import list_providers

                providers = list_providers()
            except Exception:
                providers = ["openai", "anthropic", "gemini", "ollama", "groq", "mistral"]

            default_provider = st.session_state.get("provider", "gemini")
            provider_index = providers.index(default_provider) if default_provider in providers else 0
            selected_provider = st.selectbox("Provider", providers, index=provider_index)
            selected_model = st.text_input("Model", value=st.session_state.get("model", "gemini-2.0-flash"))
            st.session_state["provider"] = selected_provider
            st.session_state["model"] = selected_model

        st.divider()
        st.subheader("Run Configuration")
        proposition = st.text_area(
            "Proposition",
            value="Urban congestion pricing reduces net air pollution burden in metro regions.",
            height=100,
        )
        source_text = st.text_area(
            "Source Text (optional, SEED input)",
            value="",
            height=130,
        )
        domain = st.text_input("Domain", value="policy")
        prior = st.slider("Prior", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
        max_rounds = st.slider("Max Rounds", min_value=1, max_value=8, value=4)
        population_size = st.slider("Population Size", min_value=20, max_value=300, value=120, step=10)
        default_storage = os.environ.get("ARGUS_SANDBOX_STORAGE", "./argus_sandbox_runs")
        storage_dir = st.text_input("Storage Directory", value=default_storage)

        run_clicked = st.button("Start Sandbox Run", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running end-to-end sandbox pipeline..."):
            _run_pipeline(
                proposition=proposition,
                prior=prior,
                source_text=source_text,
                domain=domain,
                storage_dir=storage_dir,
                max_rounds=max_rounds,
                population_size=population_size,
            )

    left, right = st.columns([0.45, 0.55])
    with left:
        _render_events()
    with right:
        _render_status()

    _render_debate_flow()
    _render_summary()
    _render_module_visuals()


if __name__ == "__main__":
    main()
