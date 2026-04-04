"""
fsociety visualizations — generates plots from debate results.

Produces:
  - Severity distribution bar chart
  - Agent contribution pie chart
  - Findings heatmap by agent × severity
  - Posterior convergence line chart
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_all_plots(
    debate_result: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Generate all visualizations from debate results."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not available — skipping visualizations")
        return []

    graphs_dir = output_dir / "graphs"
    heatmaps_dir = output_dir / "heatmaps"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    findings = debate_result.get("findings", [])
    posteriors = debate_result.get("posteriors", [])

    # ── Global style ──────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#111111",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#00ff41",
        "text.color": "#00ff41",
        "xtick.color": "#00cc33",
        "ytick.color": "#00cc33",
        "grid.color": "#1a1a1a",
        "font.family": "monospace",
    })

    # ── 1. Severity Distribution ──────────────────────────────────
    try:
        sev_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "INFO": 0}
        for f in findings:
            sev = f.get("severity", "P2").upper()
            if sev in sev_counts:
                sev_counts[sev] += 1
            elif sev == "CRITICAL":
                sev_counts["P0"] += 1
            elif sev == "HIGH":
                sev_counts["P1"] += 1
            elif sev == "MEDIUM":
                sev_counts["P2"] += 1
            elif sev == "LOW":
                sev_counts["P3"] += 1

        colors = ["#ff0000", "#ff6600", "#ffcc00", "#00cc33", "#0088ff"]
        labels = list(sev_counts.keys())
        values = list(sev_counts.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=14, fontweight="bold",
                    color="#00ff41",
                )

        ax.set_title("SEVERITY DISTRIBUTION", fontsize=16, fontweight="bold", pad=15)
        ax.set_ylabel("Finding Count", fontsize=12)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)

        path = graphs_dir / "severity_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)
        logger.info(f"Generated: {path}")
    except Exception as e:
        logger.warning(f"Severity chart failed: {e}")

    # ── 2. Agent Contribution ─────────────────────────────────────
    try:
        agent_counts: dict[str, int] = {}
        for f in findings:
            agent = f.get("agent", "unknown").upper()
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        if agent_counts:
            agent_colors = [
                "#00ff41", "#00cc33", "#009922", "#006611",
                "#ff6600", "#ffcc00", "#ff0000", "#0088ff",
                "#ff00ff", "#00ffff", "#ffff00", "#ff8800", "#8800ff",
            ]

            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(
                agent_counts.values(),
                labels=agent_counts.keys(),
                colors=agent_colors[:len(agent_counts)],
                autopct="%1.0f%%",
                pctdistance=0.8,
                startangle=90,
                textprops={"color": "#00ff41", "fontsize": 10},
            )
            for at in autotexts:
                at.set_color("#0a0a0a")
                at.set_fontweight("bold")

            ax.set_title("AGENT CONTRIBUTIONS", fontsize=16, fontweight="bold", pad=15)

            path = graphs_dir / "agent_contributions.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated.append(path)
            logger.info(f"Generated: {path}")
    except Exception as e:
        logger.warning(f"Agent contribution chart failed: {e}")

    # ── 3. Findings Heatmap (Agent × Severity) ───────────────────
    try:
        agents = sorted(set(f.get("agent", "unknown") for f in findings))
        sevs = ["P0", "P1", "P2", "P3"]
        matrix = []
        for agent in agents:
            row = []
            for sev in sevs:
                count = sum(
                    1 for f in findings
                    if f.get("agent", "unknown") == agent
                    and f.get("severity", "P2").upper() in (sev, {"P0": "CRITICAL", "P1": "HIGH", "P2": "MEDIUM", "P3": "LOW"}.get(sev, ""))
                )
                row.append(count)
            matrix.append(row)

        if matrix and agents:
            fig, ax = plt.subplots(figsize=(8, max(4, len(agents) * 0.8 + 2)))

            im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0)

            ax.set_xticks(range(len(sevs)))
            ax.set_xticklabels(sevs, fontsize=12)
            ax.set_yticks(range(len(agents)))
            ax.set_yticklabels([a.upper() for a in agents], fontsize=10)

            # Annotate cells
            for i in range(len(agents)):
                for j in range(len(sevs)):
                    val = matrix[i][j]
                    if val > 0:
                        ax.text(j, i, str(val), ha="center", va="center",
                                fontsize=14, fontweight="bold", color="#ffffff")

            ax.set_title("FINDINGS HEATMAP (Agent × Severity)", fontsize=14, fontweight="bold", pad=15)
            fig.colorbar(im, ax=ax, shrink=0.6, label="Count")

            path = heatmaps_dir / "agent_severity_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated.append(path)
            logger.info(f"Generated: {path}")
    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")

    # ── 4. Posterior Convergence ───────────────────────────────────
    try:
        if posteriors and len(posteriors) > 1:
            fig, ax = plt.subplots(figsize=(8, 4))

            rounds = list(range(len(posteriors)))
            ax.plot(rounds, posteriors, color="#00ff41", linewidth=2.5,
                    marker="o", markersize=8, markerfacecolor="#00cc33",
                    markeredgecolor="#ffffff", markeredgewidth=1.5)
            ax.fill_between(rounds, posteriors, alpha=0.15, color="#00ff41")

            # Threshold line
            ax.axhline(y=0.85, color="#ff6600", linestyle="--", linewidth=1, alpha=0.7, label="P0 Threshold (0.85)")

            ax.set_title("POSTERIOR CONVERGENCE", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Posterior Probability", fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(axis="both", alpha=0.2)

            path = graphs_dir / "posterior_convergence.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated.append(path)
            logger.info(f"Generated: {path}")
    except Exception as e:
        logger.warning(f"Posterior chart failed: {e}")

    logger.info(f"Generated {len(generated)} visualizations")
    return generated
