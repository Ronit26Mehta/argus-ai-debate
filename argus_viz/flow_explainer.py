"""
Debate Flow Explainer for Argus-Viz.

Visualizes the entire debate pipeline with interactive diagrams
and step-by-step explanations.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Flow diagram (static pipeline overview)
# ---------------------------------------------------------------------------

def render_debate_flow_diagram():
    """
    Render an interactive Plotly Sankey diagram showing the complete
    ARGUS debate pipeline.
    """
    labels = [
        "Proposition",       # 0
        "Moderator",         # 1
        "Specialist 1",      # 2
        "Specialist 2",      # 3
        "Specialist N",      # 4
        "Evidence Pool",     # 5
        "Refuter",           # 6
        "Rebuttals",         # 7
        "CDAG Graph",        # 8
        "Bayesian Updater",  # 9
        "Stopping Check",    # 10
        "Jury",              # 11
        "Verdict",           # 12
    ]

    # source → target flows
    sources = [0, 1, 1, 1, 2, 3, 4, 5, 6, 5, 7, 8, 9, 10, 10, 11]
    targets = [1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 9, 10, 1,  11, 12]
    values  = [10,8, 8, 8, 6, 6, 6, 8, 5, 10,5, 10,10,4,  10, 10]
    link_labels = [
        "Submit", "Assign", "Assign", "Assign",
        "Evidence", "Evidence", "Evidence",
        "Challenge", "Counter",
        "Aggregate", "Aggregate",
        "Update", "Evaluate",
        "Continue", "Final",
        "Render",
    ]

    node_colors = [
        "#00d4ff",   # Proposition
        "#b388ff",   # Moderator
        "#00ff88",   # Specialist 1
        "#00ff88",   # Specialist 2
        "#00ff88",   # Specialist N
        "#00d4ff",   # Evidence Pool
        "#ff8800",   # Refuter
        "#ff4466",   # Rebuttals
        "#7c4dff",   # CDAG Graph
        "#b388ff",   # Bayesian Updater
        "#ffbf00",   # Stopping Check
        "#00d4ff",   # Jury
        "#00ff88",   # Verdict
    ]

    link_colors = [
        "rgba(0,212,255,0.25)",   # Proposition → Moderator
        "rgba(0,255,136,0.20)",   # Moderator → Specialists
        "rgba(0,255,136,0.20)",
        "rgba(0,255,136,0.20)",
        "rgba(0,255,136,0.20)",   # Specialists → Evidence
        "rgba(0,255,136,0.20)",
        "rgba(0,255,136,0.20)",
        "rgba(255,136,0,0.25)",   # Evidence → Refuter
        "rgba(255,68,102,0.25)",  # Refuter → Rebuttals
        "rgba(0,212,255,0.20)",   # Evidence → CDAG
        "rgba(255,68,102,0.20)",  # Rebuttals → CDAG
        "rgba(179,136,255,0.25)", # CDAG → Bayesian
        "rgba(255,191,0,0.25)",   # Bayesian → Stopping
        "rgba(179,136,255,0.15)", # Loop back
        "rgba(0,212,255,0.25)",   # Stopping → Jury
        "rgba(0,255,136,0.30)",   # Jury → Verdict
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="#0e1117", width=2),
            label=labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
            hovertemplate="%{source.label} → %{target.label}<br>%{label}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(text="ARGUS Debate Pipeline", font=dict(size=18, color="#00d4ff")),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(family="Inter, sans-serif", color="#e0e0e0", size=12),
        height=500,
    )

    return fig


# ---------------------------------------------------------------------------
# Step explanations
# ---------------------------------------------------------------------------

STEP_EXPLANATIONS = {
    "proposition": {
        "title": "📝 1. Proposition",
        "icon": "📝",
        "description": """
The debate begins with a **Proposition** — the central claim to evaluate.
It includes:
- **Text**: The statement being debated (e.g., *"Drug X reduces symptoms by >20%"*)
- **Prior probability**: Initial belief (0.0 = certainly false, 1.0 = certainly true)
- **Domain**: Field of expertise (e.g., clinical, financial, general)
""",
    },
    "specialists": {
        "title": "🔬 2. Specialist Agents",
        "icon": "🔬",
        "description": """
**Specialist agents** are domain experts who gather evidence.
Each specialist has a unique persona and instruction that shapes their analysis.

For example:
- **Bull Analyst**: Finds supporting evidence
- **Bear Analyst**: Finds counter-evidence
- **Technical Analyst**: Provides data-driven analysis

Each specialist generates 2 evidence items per round as structured JSON.
""",
    },
    "evidence": {
        "title": "📊 3. Evidence Collection",
        "icon": "📊",
        "description": """
Evidence items are added to the **C-DAG (Conceptual Debate Graph)** with:
- **Polarity**: +1 (supports) or -1 (attacks) the proposition
- **Confidence**: 0.0–1.0, how sure the specialist is
- **Relevance**: How relevant to the proposition
- **Quality**: Evidence quality assessment

Evidence items are linked to the proposition via **SUPPORTS** or **ATTACKS** edges.
""",
    },
    "refuter": {
        "title": "⚔️ 4. Refuter Agent",
        "icon": "⚔️",
        "description": """
The **Refuter** challenges existing evidence by:
1. Analyzing evidence for methodological weaknesses
2. Finding contradicting facts
3. Generating **Rebuttals** — counter-arguments that target specific evidence

Rebuttals are linked to evidence via **REBUTS** edges and reduce the effective weight
of the targeted evidence.
""",
    },
    "bayesian_update": {
        "title": "📐 5. Bayesian Belief Update",
        "icon": "📐",
        "description": """
After each round, the system computes a new **posterior probability** using
Bayesian belief propagation in log-odds space:

```
posterior = σ(log-odds(prior) + Σᵢ wᵢ · log(LRᵢ))
```

Where:
- **σ** is the logistic (sigmoid) function
- **LRᵢ** is the likelihood ratio for evidence item i
- **wᵢ** = polarity × confidence × relevance × quality

This moves the posterior toward support (>0.5) or rejection (<0.5).
""",
    },
    "jury": {
        "title": "⚖️ 6. Jury Verdict",
        "icon": "⚖️",
        "description": """
After all rounds are complete, the **Jury agent** renders the final verdict:

- **Supported** (posterior ≥ threshold): Evidence strongly favors the proposition
- **Rejected** (posterior ≤ 1-threshold): Evidence strongly opposes the proposition
- **Undecided**: Not enough evidence for a confident decision

The Jury also provides:
- **Reasoning**: Natural language explanation of the verdict
- **Top support/attack**: Most impactful evidence items
- **Confidence**: How certain the jury is of its verdict
""",
    },
}


def explain_step(step_name: str):
    """Render rich markdown explanation of a debate stage."""
    step = STEP_EXPLANATIONS.get(step_name, {})
    if not step:
        st.warning(f"Unknown step: {step_name}")
        return

    st.markdown(f"### {step['title']}")
    st.markdown(step["description"])


def render_all_step_explanations():
    """Render all step explanations in order."""
    for key in ["proposition", "specialists", "evidence",
                "refuter", "bayesian_update", "jury"]:
        explain_step(key)
        st.markdown("---")


# ---------------------------------------------------------------------------
# Algorithm explanation
# ---------------------------------------------------------------------------

def render_algorithm_explanation():
    """Show the Bayesian belief propagation formula with explanation."""
    st.markdown("### 📐 Bayesian Belief Propagation Algorithm")

    st.markdown("""
The core algorithm converts prior beliefs + evidence into a posterior probability.

**In log-odds space:**
""")

    st.latex(r"\text{posterior} = \sigma\left(\log\text{-odds}(\text{prior}) + \sum_{i} w_i \cdot \log(LR_i)\right)")

    st.markdown("""
**Where:**

| Symbol | Meaning |
|--------|---------|
| σ | Logistic sigmoid function: σ(x) = 1/(1+e⁻ˣ) |
| LRᵢ | Likelihood ratio for evidence i |
| wᵢ | Weight = polarity × confidence × relevance × quality |
| prior | Initial belief probability (0–1) |

**Evidence weighting:**
""")

    st.latex(r"w_i = \text{polarity}_i \times \text{confidence}_i \times \text{relevance}_i \times \text{quality}_i")

    st.markdown("""
- **polarity = +1**: evidence supports the proposition
- **polarity = -1**: evidence attacks the proposition
- Rebuttals reduce the effective weight of the targeted evidence
""")


# ---------------------------------------------------------------------------
# Flow with actual debate data
# ---------------------------------------------------------------------------

def render_flow_with_data(result: dict):
    """Overlay actual debate data onto the flow diagram via metrics."""
    if not result:
        st.info("Run a debate to see data overlaid on the flow diagram.")
        return

    rounds = result.get("rounds", [])
    verdict = result.get("verdict", {})
    config = result.get("config", {})

    st.markdown("### 📊 Debate Data Summary")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    total_ev = sum(r.get("total_evidence", 0) for r in rounds)
    total_rb = sum(r.get("total_rebuttals", 0) for r in rounds)
    total_support = sum(r.get("support_count", 0) for r in rounds)
    total_attack = sum(r.get("attack_count", 0) for r in rounds)

    col1.metric("Total Evidence", total_ev)
    col2.metric("Total Rebuttals", total_rb)
    col3.metric("Support / Attack", f"{total_support} / {total_attack}")
    col4.metric("Final Posterior", f"{verdict.get('posterior', 0.5):.3f}")

    # Per-round breakdown
    st.markdown("#### Round-by-Round Posterior Trajectory")
    for r in rounds:
        rnd = r["round"]
        before = r.get("posterior_before", 0.5)
        after = r.get("posterior_after", 0.5)
        delta = after - before
        direction = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"

        st.markdown(
            f"**Round {rnd}**: {before:.3f} → {after:.3f} "
            f"({delta:+.3f}) {direction} | "
            f"+{r.get('support_count', 0)} support, "
            f"-{r.get('attack_count', 0)} attack, "
            f"{r.get('total_rebuttals', 0)} rebuttals"
        )

    # Verdict
    st.markdown("---")
    label = verdict.get("label", "undecided")
    emoji = "✅" if label == "supported" else "❌" if label == "rejected" else "🤔"
    st.markdown(f"### {emoji} Final Verdict: **{label.upper()}** (P={verdict.get('posterior', 0.5):.3f})")
    if verdict.get("reasoning"):
        st.markdown(f"*{verdict['reasoning'][:400]}*")
