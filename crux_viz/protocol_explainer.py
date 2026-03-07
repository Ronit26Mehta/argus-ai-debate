"""
CRUX Protocol Explainer for CRUX-Viz.

Visualizes the full CRUX 7-primitive pipeline with interactive Sankey diagrams
and detailed step-by-step explanations of each protocol primitive.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from typing import Any, Optional


# ---------------------------------------------------------------------------
# CRUX pipeline Sankey diagram
# ---------------------------------------------------------------------------

def render_crux_pipeline_diagram() -> Any:
    """
    Render an interactive Plotly Sankey diagram showing the complete
    CRUX protocol pipeline with all 7 primitives.
    """
    labels = [
        "Proposition",          # 0
        "EAC Registry",         # 1 — Epistemic Agent Cards
        "Agent Pool",           # 2
        "Claim Bundles",        # 3 — CB
        "Dialectical Router",   # 4 — DR
        "Challenger Auction",   # 5 — CA
        "Evidence + Rebuttals", # 6
        "CRUX BRP",             # 7 — BRP
        "Credibility Ledger",   # 8 — CL
        "EDR Checkpoint",       # 9 — EDR
        "Bayesian Updater",     # 10
        "Jury",                 # 11
        "Verdict",              # 12
    ]

    sources = [ 0,  1,  2,  3,  4,  5,  3,  6,  7,  7,  8,  9, 10, 10, 11]
    targets = [ 1,  2,  3,  4,  5,  6,  7,  7,  9, 10,  9, 10, 11, 10, 12]
    values  = [10,  8,  9,  8,  6,  6,  5,  8,  4,  8,  3,  8,  8,  3, 10]

    link_labels = [
        "Submit", "Register EAC", "Emit CBs",
        "Route to Challenger", "Award Contract", "Generate Evidence",
        "Detect Contradiction", "Reconcile Beliefs",
        "Log State", "Update Posterior",
        "Sync Delta", "Evaluate",
        "Render Verdict", "Continue Loop", "Final",
    ]

    node_colors = [
        "#00d4ff",   # Proposition
        "#ff00d4",   # EAC Registry ← CRUX magenta
        "#ff00d4",   # Agent Pool
        "#00d4ff",   # Claim Bundles
        "#ff8800",   # Dialectical Router
        "#ff8800",   # Challenger Auction
        "#00ff88",   # Evidence + Rebuttals
        "#b388ff",   # CRUX BRP
        "#ffbf00",   # Credibility Ledger
        "#9966ff",   # EDR Checkpoint
        "#b388ff",   # Bayesian Updater
        "#00d4ff",   # Jury
        "#00ff88",   # Verdict
    ]

    link_colors = [
        "rgba(0,212,255,0.25)",    # Proposition → EAC
        "rgba(255,0,212,0.20)",    # EAC → Agents
        "rgba(0,212,255,0.20)",    # Agents → CBs
        "rgba(255,136,0,0.25)",    # CBs → DR
        "rgba(255,136,0,0.20)",    # DR → CA
        "rgba(0,255,136,0.20)",    # CA → Evidence
        "rgba(179,136,255,0.25)",  # CBs → BRP
        "rgba(179,136,255,0.25)",  # Evidence → BRP
        "rgba(255,191,0,0.20)",    # BRP → CL
        "rgba(179,136,255,0.20)",  # BRP → Bayesian
        "rgba(153,102,255,0.20)",  # CL → EDR
        "rgba(179,136,255,0.20)",  # EDR → Bayesian
        "rgba(0,212,255,0.25)",    # Bayesian → Jury
        "rgba(179,136,255,0.15)",  # Loop back
        "rgba(0,255,136,0.30)",    # Jury → Verdict
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
        title=dict(
            text="⚡ CRUX Protocol Pipeline — 7 Primitives",
            font=dict(size=18, color="#ff00d4"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(family="Inter, sans-serif", color="#e0e0e0", size=12),
        height=560,
    )

    return fig


# ---------------------------------------------------------------------------
# Step explanations — all 7 CRUX primitives
# ---------------------------------------------------------------------------

CRUX_PRIMITIVE_EXPLANATIONS: dict[str, dict] = {
    "eac": {
        "title": "🪪 Primitive 1: Epistemic Agent Card (EAC)",
        "badge": "EAC",
        "color": "#ff00d4",
        "description": """
An **Epistemic Agent Card (EAC)** is the identity document for every agent in CRUX.
It replaces simple agent IDs with rich epistemic metadata:

| Field | Description |
|-------|-------------|
| `agent_id` | Unique agent identifier |
| `belief_domains` | Expertise domains (e.g., `["clinical", "statistics"]`) |
| `calibration.brier_score` | Historical prediction accuracy (lower = better) |
| `calibration.credibility_rating` | Current trust score (0–1) |
| `capabilities` | What the agent can do: emit claims, challenge, render verdicts |

**Why it matters:** Traditional systems treat all agents equally.
CRUX weights contributions by _calibrated credibility_, so a historically
more accurate agent has proportionally higher influence on the final belief state.
""",
    },
    "cb": {
        "title": "📦 Primitive 2: Claim Bundle (CB)",
        "badge": "CB",
        "color": "#00d4ff",
        "description": """
A **Claim Bundle (CB)** is the atomic epistemic unit in CRUX —
the currency of inter-agent communication.

Unlike simple text evidence, a CB packages:

```python
ClaimBundle(
    claim_text="Drug X reduces symptoms by >20%",
    polarity=Polarity.SUPPORTS,        # +1 / -1 / 0
    prior=0.50,                        # Belief before this evidence
    posterior=0.72,                    # Updated belief after
    confidence_distribution=BetaDist( # Uncertainty over confidence
        alpha=8.5, beta=3.2
    ),
    issuer_agent="specialist-bull",
    issuer_credibility=0.85,
    challenge_open=True,               # Accepts challengers?
)
```

**Key insight:** The `confidence_distribution` encodes *epistemic uncertainty* —
not just a point estimate, but how confident we are in the confidence itself.
""",
    },
    "dr": {
        "title": "🧭 Primitive 3: Dialectical Routing (DR)",
        "badge": "DR",
        "color": "#ff8800",
        "description": """
**Dialectical Routing (DR)** is CRUX's adversarial-aware agent selection mechanism.

When a Claim Bundle arrives, DR computes a **Dialectical Fitness Score (DFS)** for
each registered challenger:

```
DFS(agent, claim) = w₁ · domain_match
                  + w₂ · adversarial_potential
                  + w₃ · credibility_factor
                  + w₄ · recency_factor
```

| Component | Meaning |
|-----------|---------|
| `domain_match` | How relevant is the agent's expertise? |
| `adversarial_potential` | How likely to produce a strong counter-claim? |
| `credibility_factor` | Historical accuracy weight |
| `recency_factor` | Penalizes agents that haven't updated recently |

**Result:** CRUX routes each claim to the _most capable adversarial agent_,
not a random or round-robin selection.
""",
    },
    "brp": {
        "title": "🔀 Primitive 4: Belief Reconciliation Protocol (BRP)",
        "badge": "BRP",
        "color": "#b388ff",
        "description": """
The **Belief Reconciliation Protocol (BRP)** resolves contradictory claims.

**Trigger:** When the absolute difference in posteriors between two opposing
Claim Bundles exceeds the contradiction threshold θ:

```
|posterior_A - posterior_B| > θ
```

**Resolution:** BRP uses a credibility-weighted merge:

```
reconciled_posterior = Σᵢ (credibility_i × posterior_i) / Σᵢ credibility_i
```

The reconciled posterior is wrapped in a new **Reconciled Claim Bundle** with
a merged confidence distribution (convolution of the Beta distributions).

**Why it matters:** Without BRP, contradicting agents would produce unstable
oscillating posteriors. BRP produces a principled, stable merged belief.
""",
    },
    "cl": {
        "title": "📊 Primitive 5: Credibility Ledger (CL)",
        "badge": "CL",
        "color": "#ffbf00",
        "description": """
The **Credibility Ledger (CL)** is a persistent statistical trust layer
across debate sessions.

After each debate, CL updates each agent's track record:

| Update Rule | When Applied |
|-------------|-------------|
| Credibility ↑ | Agent's claim aligned with final verdict |
| Credibility ↓ | Agent's claim contradicted by final verdict |
| Suspension | Credibility drops below the floor (0.30) |

**Brier Score** is also tracked:
```
Brier = (predicted_posterior - actual_outcome)²
```

The CL persists across sessions, so agents that are consistently accurate
gain compounding influence in future debates.
""",
    },
    "edr": {
        "title": "📍 Primitive 6: Epistemic Dead Reckoning (EDR)",
        "badge": "EDR",
        "color": "#9966ff",
        "description": """
**Epistemic Dead Reckoning (EDR)** handles reconnection and state synchronization
when an agent disconnects mid-debate.

When an agent reconnects, EDR computes a **delta** from the last known checkpoint:

```python
EDRDelta(
    checkpoint_id="cp-round-2",
    agent_id="specialist-bull",
    missed_claim_bundles=[cb3, cb4, cb5],
    posterior_drift=0.12,        # How much the belief moved
    credibility_drift=-0.02,     # Change in credibility
    sync_strategy="replay",      # Or "fast_forward"
)
```

**Sync strategies:**
- **Replay**: Agent re-processes all missed CBs in order
- **Fast-forward**: Agent receives a summary delta and jumps to current state

**Why it matters:** In distributed or asynchronous systems, agents may go
offline. EDR ensures no agent is permanently left behind.
""",
    },
    "ca": {
        "title": "🏆 Primitive 7: Challenger Auction (CA)",
        "badge": "CA",
        "color": "#ff8800",
        "description": """
The **Challenger Auction (CA)** is the mechanism that selects the best
adversarial challenger for each open Claim Bundle.

**Process:**
1. A Claim Bundle is broadcast as _open to challenge_
2. Interested agents submit **Challenger Bids**:
   ```python
   ChallengerBid(
       bidder_agent="specialist-bear",
       estimated_dfs=0.82,
       domain_confidence=0.75,
       adversarial_strategy="counter_evidence",
   )
   ```
3. The auctioneer selects the bid with the highest estimated DFS
4. The winner is assigned to challenge the claim

**Unchallenged claims:** If no agents bid (or all below DFS threshold),
the claim is marked _unchallenged_ — it passes without rebuttal but
is flagged for later credibility review.
""",
    },
}


def render_all_primitive_explanations():
    """Render all 7 CRUX primitive explanations in order."""
    for key in ["eac", "cb", "dr", "brp", "cl", "edr", "ca"]:
        prim = CRUX_PRIMITIVE_EXPLANATIONS[key]
        badge_color = prim["color"]
        st.markdown(
            f'<span style="background:rgba({_hex_to_rgb(badge_color)},0.15);'
            f'color:{badge_color};padding:3px 10px;border-radius:12px;'
            f'font-size:11px;font-weight:700;letter-spacing:1px;">'
            f'PRIMITIVE {list(CRUX_PRIMITIVE_EXPLANATIONS.keys()).index(key) + 1}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {prim['title']}")
        st.markdown(prim["description"])
        st.markdown("---")


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' string."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "128,128,128"


# ---------------------------------------------------------------------------
# CRUX algorithm explanation
# ---------------------------------------------------------------------------

def render_crux_algorithm_explanation():
    """Render the CRUX Bayesian + BRP + DFS formulae."""
    st.markdown("### 📐 CRUX Mathematical Foundation")

    st.markdown("**1. Bayesian Belief Propagation (same as ARGUS):**")
    st.latex(
        r"\text{posterior} = \sigma\!\left(\log\text{-odds}(\text{prior})"
        r" + \sum_{i} w_i \cdot \log(LR_i)\right)"
    )

    st.markdown("""
| Symbol | Meaning |
|--------|---------|
| σ | Logistic sigmoid |
| LRᵢ | Likelihood ratio of evidence i |
| wᵢ | polarity × confidence × relevance × quality |
""")

    st.markdown("**2. CRUX Credibility-Weighted BRP Merge:**")
    st.latex(
        r"P_{\text{reconciled}} = "
        r"\frac{\sum_i \text{cred}_i \cdot P_i}{\sum_i \text{cred}_i}"
    )

    st.markdown("**3. Dialectical Fitness Score (DFS):**")
    st.latex(
        r"\text{DFS}(a, cb) = w_1 \cdot D_{\text{match}}"
        r" + w_2 \cdot A_{\text{potential}}"
        r" + w_3 \cdot C_{\text{factor}}"
        r" + w_4 \cdot R_{\text{recency}}"
    )

    st.markdown("""
**4. Brier Score Update (Credibility Ledger):**
""")
    st.latex(r"\text{Brier} = (P_{\text{predicted}} - O_{\text{actual}})^2")

    st.markdown("""
- **O = 1** if verdict = supported, **O = 0** if rejected
- Lower Brier score → higher accumulated credibility
""")


# ---------------------------------------------------------------------------
# Flow with actual data
# ---------------------------------------------------------------------------

def render_crux_flow_with_data(result: dict):
    """Overlay actual CRUX debate data onto the flow explanation."""
    if not result:
        st.info("Run a CRUX debate to see protocol data overlaid on the flow.")
        return

    rounds = result.get("rounds", [])
    verdict = result.get("verdict", {})
    crux_stats = result.get("crux_session_stats", {})
    claim_bundles = result.get("claim_bundles", [])
    brp_sessions = result.get("brp_sessions", [])
    auctions = result.get("auctions", [])

    st.markdown("### 📊 CRUX Debate Data Overlay")

    col1, col2, col3, col4 = st.columns(4)
    total_ev = sum(r.get("total_evidence", 0) for r in rounds)
    total_rb = sum(r.get("total_rebuttals", 0) for r in rounds)

    col1.metric("Total Evidence", total_ev)
    col2.metric("Claim Bundles", crux_stats.get("num_claim_bundles", 0))
    col3.metric("BRP Sessions", crux_stats.get("num_brp_sessions", 0))
    col4.metric("Auctions", crux_stats.get("num_auctions", 0))

    st.markdown("#### Round-by-Round CRUX Trajectory")
    for r in rounds:
        rnd = r["round"]
        before = r.get("posterior_before", 0.5)
        after = r.get("posterior_after", 0.5)
        delta = after - before
        cbs = r.get("claim_bundles_issued", 0)
        direction = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"

        st.markdown(
            f"**Round {rnd}**: {before:.3f} → {after:.3f} "
            f"({delta:+.3f}) {direction} | "
            f"+{r.get('support_count', 0)} support, "
            f"-{r.get('attack_count', 0)} attack, "
            f"{r.get('total_rebuttals', 0)} rebuttals, "
            f"**{cbs} CBs**"
        )

    st.markdown("---")
    label = verdict.get("label", "undecided")
    emoji = "✅" if label == "supported" else "❌" if label == "rejected" else "🤔"
    st.markdown(
        f"### {emoji} Final Verdict: **{label.upper()}** "
        f"(P={verdict.get('posterior', 0.5):.3f})"
    )
    if verdict.get("reasoning"):
        st.markdown(f"*{verdict['reasoning'][:400]}*")

    # CRUX-specific extras
    st.markdown("---")
    if brp_sessions:
        st.markdown(f"#### 🔀 BRP: {len(brp_sessions)} reconciliation(s) triggered")
        for j, brp in enumerate(brp_sessions):
            res = brp.get("resolution", {})
            st.markdown(
                f"- BRP-{j+1}: Δ={brp.get('contradiction_delta', 0):.3f} → "
                f"Reconciled P={res.get('reconciled_posterior', 0.5):.3f}"
            )
    else:
        st.markdown("#### 🔀 BRP: No reconciliations triggered (no contradictions above θ)")

    if auctions:
        challenged = sum(1 for a in auctions if not a.get("unchallenged"))
        st.markdown(
            f"#### 🏆 Auctions: {len(auctions)} conducted, "
            f"{challenged} with challengers, "
            f"{len(auctions) - challenged} unchallenged"
        )
