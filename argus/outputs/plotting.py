"""
Advanced Plotting Module for ARGUS SEC Debate Visualization.

Creates publication-quality, research-grade visualizations for debate results
including posterior evolution, evidence analysis, CDAG networks, and comparative
multi-stock dashboards.

Features:
- Publication-quality static plots (Matplotlib/Seaborn)
- Interactive visualizations (Plotly)
- Dark mode and light mode themes
- Colorblind-friendly palettes
- High-resolution export (300 DPI)

Example:
    from argus.outputs.plotting import DebatePlotter, PlotConfig
    
    plotter = DebatePlotter(PlotConfig(output_dir="./plots"))
    plotter.generate_all_plots(debate_result)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

# Optional imports for advanced features
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Color Palettes & Themes
# =============================================================================

class PlotTheme(str, Enum):
    """Available plot themes."""
    PUBLICATION = "publication"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"


# Professional color palette - colorblind friendly
COLORS = {
    # Core semantic colors
    "support": "#2ECC71",      # Emerald green - supporting evidence
    "attack": "#E74C3C",       # Alizarin red - attacking evidence
    "neutral": "#95A5A6",      # Concrete gray - neutral
    "rebuttal": "#F39C12",     # Orange - rebuttals
    
    # Primary palette
    "primary": "#3498DB",      # Peter River blue
    "secondary": "#9B59B6",    # Amethyst purple
    "tertiary": "#1ABC9C",     # Turquoise
    "accent": "#E74C3C",       # Alizarin
    
    # Specialist colors
    "bull": "#27AE60",         # Nephritis green
    "bear": "#C0392B",         # Pomegranate red
    "technical": "#2980B9",    # Belize blue
    "sec": "#8E44AD",          # Wisteria purple
    
    # Background colors
    "bg_dark": "#1A1A2E",      # Dark navy
    "bg_light": "#FAFBFC",     # Off white
    "grid_dark": "#2D2D44",    # Dark grid
    "grid_light": "#E5E5E5",   # Light grid
    
    # Text colors
    "text_dark": "#2C3E50",    # Midnight blue
    "text_light": "#ECF0F1",   # Clouds white
}

# Gradient colormaps
SUPPORT_ATTACK_CMAP = LinearSegmentedColormap.from_list(
    "support_attack", [COLORS["attack"], COLORS["neutral"], COLORS["support"]]
)

CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "confidence", ["#FEE5D9", "#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"]
)

# Specialist color mapping
SPECIALIST_COLORS = {
    "Bull Analyst": COLORS["bull"],
    "Bear Analyst": COLORS["bear"],
    "Technical Analyst": COLORS["technical"],
    "SEC Filing Analyst": COLORS["sec"],
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    output_dir: Path = field(default_factory=lambda: Path("./plots"))
    dpi: int = 300
    format: str = "png"  # png, pdf, svg
    theme: PlotTheme = PlotTheme.PUBLICATION
    interactive: bool = True
    figsize: tuple[float, float] = (12, 8)
    title_fontsize: int = 16
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def setup_plot_style(theme: PlotTheme = PlotTheme.PUBLICATION):
    """Apply publication-quality matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Base style parameters
    style_params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'medium',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    }
    
    if theme == PlotTheme.DARK:
        style_params.update({
            'axes.facecolor': COLORS["bg_dark"],
            'figure.facecolor': COLORS["bg_dark"],
            'axes.edgecolor': COLORS["grid_dark"],
            'axes.labelcolor': COLORS["text_light"],
            'text.color': COLORS["text_light"],
            'xtick.color': COLORS["text_light"],
            'ytick.color': COLORS["text_light"],
            'grid.color': COLORS["grid_dark"],
        })
    elif theme == PlotTheme.LIGHT:
        style_params.update({
            'axes.facecolor': COLORS["bg_light"],
            'figure.facecolor': '#FFFFFF',
            'axes.edgecolor': COLORS["grid_light"],
            'axes.labelcolor': COLORS["text_dark"],
            'text.color': COLORS["text_dark"],
            'grid.color': COLORS["grid_light"],
        })
    
    plt.rcParams.update(style_params)


# =============================================================================
# Data Extraction Utilities
# =============================================================================

def extract_posterior_timeline(result: dict) -> tuple[list[int], list[float], list[float]]:
    """Extract posterior values across rounds."""
    rounds = []
    before_values = []
    after_values = []
    
    for round_data in result.get("rounds", []):
        rounds.append(round_data.get("round_number", 0))
        before_values.append(round_data.get("posterior_before", 0.5))
        after_values.append(round_data.get("posterior_after", 0.5))
    
    return rounds, before_values, after_values


def extract_evidence_by_polarity(result: dict) -> tuple[int, int]:
    """Count supporting vs attacking evidence."""
    support_count = 0
    attack_count = 0
    
    for round_data in result.get("rounds", []):
        for evidence in round_data.get("evidence_added", []):
            if evidence.get("polarity", 0) > 0:
                support_count += 1
            else:
                attack_count += 1
    
    return support_count, attack_count


def extract_specialist_contributions(result: dict) -> dict[str, dict[int, int]]:
    """Extract evidence count per specialist per round."""
    contributions = {}
    
    for round_data in result.get("rounds", []):
        round_num = round_data.get("round_number", 0)
        for evidence in round_data.get("evidence_added", []):
            specialist = evidence.get("specialist", "Unknown")
            if specialist not in contributions:
                contributions[specialist] = {}
            if round_num not in contributions[specialist]:
                contributions[specialist][round_num] = 0
            contributions[specialist][round_num] += 1
    
    return contributions


def extract_evidence_confidence(result: dict) -> list[float]:
    """Extract confidence scores from graph nodes."""
    confidences = []
    graph = result.get("graph", {})
    
    for node in graph.get("nodes", []):
        if node.get("type") == "Evidence":
            confidences.append(node.get("confidence", 0.5))
    
    return confidences


def extract_graph_structure(result: dict) -> tuple[list[dict], list[dict]]:
    """Extract nodes and edges from graph data."""
    graph = result.get("graph", {})
    return graph.get("nodes", []), graph.get("edges", [])


# =============================================================================
# Static Plots (Matplotlib/Seaborn)
# =============================================================================

class DebatePlotter:
    """
    Main plotting interface for ARGUS debate results.
    
    Generates publication-quality visualizations for SEC debate analysis
    including posterior evolution, evidence distribution, specialist
    contributions, and CDAG network graphs.
    
    Example:
        >>> config = PlotConfig(output_dir="./plots", theme=PlotTheme.PUBLICATION)
        >>> plotter = DebatePlotter(config)
        >>> paths = plotter.generate_all_plots(debate_result)
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize the plotter with configuration."""
        self.config = config or PlotConfig()
        setup_plot_style(self.config.theme)
        logger.info(f"DebatePlotter initialized with output_dir: {self.config.output_dir}")
    
    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save figure with configured settings."""
        path = self.config.output_dir / f"{name}.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        logger.info(f"Saved plot: {path}")
        return path
    
    # -------------------------------------------------------------------------
    # Posterior Evolution Plot
    # -------------------------------------------------------------------------
    
    def plot_posterior_evolution(self, result: dict, title: Optional[str] = None) -> Path:
        """
        Create posterior probability evolution plot across debate rounds.
        
        Shows how the posterior probability changes round-by-round with
        gradient fill indicating confidence direction.
        
        Args:
            result: Debate result dictionary
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        symbol = result.get("symbol", "Unknown")
        rounds, before, after = extract_posterior_timeline(result)
        
        if not rounds:
            logger.warning("No rounds data available for posterior evolution plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create x-coordinates
        x = np.array(rounds)
        y_before = np.array(before)
        y_after = np.array(after)
        
        # Plot with gradient fill
        ax.fill_between(x, 0.5, y_after, where=y_after >= 0.5, 
                        alpha=0.3, color=COLORS["support"], interpolate=True)
        ax.fill_between(x, 0.5, y_after, where=y_after < 0.5,
                        alpha=0.3, color=COLORS["attack"], interpolate=True)
        
        # Main lines
        ax.plot(x, y_before, 'o--', color=COLORS["neutral"], linewidth=2,
                markersize=8, label="Before Round", alpha=0.7)
        ax.plot(x, y_after, 'o-', color=COLORS["primary"], linewidth=3,
                markersize=10, label="After Round", zorder=5)
        
        # Reference line at 0.5
        ax.axhline(y=0.5, color=COLORS["neutral"], linestyle=':', 
                   linewidth=1.5, alpha=0.7, label="Prior (0.5)")
        
        # Annotations for final posterior
        final_posterior = y_after[-1] if len(y_after) > 0 else 0.5
        verdict = result.get("verdict", {}).get("label", "N/A")
        ax.annotate(f"Final: {final_posterior:.4f}\n{verdict.upper()}", 
                    xy=(x[-1], final_posterior), xytext=(x[-1] + 0.3, final_posterior),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor=COLORS["primary"], alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color=COLORS["primary"]))
        
        # Styling
        ax.set_xlabel("Debate Round", fontsize=12, fontweight='medium')
        ax.set_ylabel("Posterior Probability", fontsize=12, fontweight='medium')
        ax.set_title(title or f"Posterior Evolution: {symbol}", fontsize=14, fontweight='bold')
        ax.set_xlim(0.8, max(x) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add subtle background regions
        ax.axhspan(0.5, 1.0, alpha=0.05, color=COLORS["support"])
        ax.axhspan(0, 0.5, alpha=0.05, color=COLORS["attack"])
        
        return self._save_figure(fig, f"{symbol.lower()}_posterior_evolution")
    
    # -------------------------------------------------------------------------
    # Evidence Polarity Distribution
    # -------------------------------------------------------------------------
    
    def plot_evidence_distribution(self, result: dict, title: Optional[str] = None) -> Path:
        """
        Create evidence polarity distribution donut chart.
        
        Shows the balance between supporting and attacking evidence
        with a clean donut chart design.
        
        Args:
            result: Debate result dictionary
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        symbol = result.get("symbol", "Unknown")
        support_count, attack_count = extract_evidence_by_polarity(result)
        total = support_count + attack_count
        
        if total == 0:
            logger.warning("No evidence data for distribution plot")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Donut chart
        sizes = [support_count, attack_count]
        colors = [COLORS["support"], COLORS["attack"]]
        labels = [f"Supporting\n({support_count})", f"Attacking\n({attack_count})"]
        explode = (0.02, 0.02)
        
        wedges, texts, autotexts = ax1.pie(
            sizes, colors=colors, labels=labels, autopct='%1.1f%%',
            startangle=90, explode=explode, pctdistance=0.75,
            textprops={'fontsize': 11, 'fontweight': 'medium'}
        )
        
        # Create donut effect
        centre_circle = plt.Circle((0, 0), 0.5, fc='white')
        ax1.add_patch(centre_circle)
        
        # Center text
        ax1.text(0, 0, f"{total}\nTotal", ha='center', va='center',
                fontsize=16, fontweight='bold', color=COLORS["text_dark"])
        
        ax1.set_title("Evidence Polarity Distribution", fontsize=13, fontweight='bold')
        
        # Bar chart comparison
        bar_x = ['Supporting', 'Attacking']
        bar_heights = [support_count, attack_count]
        bars = ax2.bar(bar_x, bar_heights, color=colors, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, height in zip(bars, bar_heights):
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    str(height), ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        # Calculate ratio
        ratio = support_count / attack_count if attack_count > 0 else float('inf')
        ratio_text = f"Support/Attack Ratio: {ratio:.2f}" if ratio != float('inf') else "Support/Attack: ∞"
        ax2.text(0.5, 0.95, ratio_text, transform=ax2.transAxes,
                ha='center', fontsize=11, fontweight='medium',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_ylabel("Evidence Count", fontsize=12, fontweight='medium')
        ax2.set_title("Evidence Count Comparison", fontsize=13, fontweight='bold')
        ax2.set_ylim(0, max(bar_heights) * 1.2)
        
        fig.suptitle(title or f"Evidence Analysis: {symbol}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, f"{symbol.lower()}_evidence_distribution")
    
    # -------------------------------------------------------------------------
    # Specialist Contributions
    # -------------------------------------------------------------------------
    
    def plot_specialist_contributions(self, result: dict, title: Optional[str] = None) -> Path:
        """
        Create stacked bar chart of specialist contributions per round.
        
        Shows how each specialist (Bull, Bear, Technical, SEC) contributes
        evidence across debate rounds.
        
        Args:
            result: Debate result dictionary
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        symbol = result.get("symbol", "Unknown")
        contributions = extract_specialist_contributions(result)
        
        if not contributions:
            logger.warning("No specialist data for contributions plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get all rounds and specialists
        all_rounds = sorted(set(
            r for spec_data in contributions.values() 
            for r in spec_data.keys()
        ))
        specialists = list(contributions.keys())
        
        # Build data matrix
        data = np.zeros((len(specialists), len(all_rounds)))
        for i, specialist in enumerate(specialists):
            for j, round_num in enumerate(all_rounds):
                data[i, j] = contributions[specialist].get(round_num, 0)
        
        # Create stacked bar
        x = np.arange(len(all_rounds))
        bottom = np.zeros(len(all_rounds))
        
        for i, specialist in enumerate(specialists):
            color = SPECIALIST_COLORS.get(specialist, COLORS["neutral"])
            bars = ax.bar(x, data[i], bottom=bottom, label=specialist,
                         color=color, edgecolor='white', linewidth=1)
            bottom += data[i]
        
        # Add total labels on top
        for j, total in enumerate(bottom):
            ax.text(j, total + 0.1, str(int(total)), ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel("Round", fontsize=12, fontweight='medium')
        ax.set_ylabel("Evidence Count", fontsize=12, fontweight='medium')
        ax.set_title(title or f"Specialist Contributions: {symbol}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Round {r}" for r in all_rounds])
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim(0, max(bottom) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        return self._save_figure(fig, f"{symbol.lower()}_specialist_contributions")
    
    # -------------------------------------------------------------------------
    # Confidence Distribution
    # -------------------------------------------------------------------------
    
    def plot_confidence_distribution(self, result: dict, title: Optional[str] = None) -> Path:
        """
        Create histogram and KDE plot of evidence confidence scores.
        
        Args:
            result: Debate result dictionary
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        symbol = result.get("symbol", "Unknown")
        confidences = extract_evidence_confidence(result)
        
        if not confidences:
            logger.warning("No confidence data for distribution plot")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram with KDE
        sns.histplot(confidences, bins=15, kde=True, ax=ax1,
                    color=COLORS["primary"], edgecolor='white', linewidth=1.2,
                    alpha=0.7)
        
        ax1.axvline(np.mean(confidences), color=COLORS["accent"], linestyle='--',
                   linewidth=2, label=f"Mean: {np.mean(confidences):.3f}")
        ax1.axvline(np.median(confidences), color=COLORS["secondary"], linestyle=':',
                   linewidth=2, label=f"Median: {np.median(confidences):.3f}")
        
        ax1.set_xlabel("Confidence Score", fontsize=12, fontweight='medium')
        ax1.set_ylabel("Count", fontsize=12, fontweight='medium')
        ax1.set_title("Confidence Distribution", fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.set_xlim(0, 1)
        
        # Box plot
        bp = ax2.boxplot([confidences], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COLORS["primary"])
        bp['boxes'][0].set_alpha(0.7)
        bp['medians'][0].set_color(COLORS["accent"])
        bp['medians'][0].set_linewidth(2)
        
        # Swarm plot overlay
        jitter = np.random.normal(0, 0.04, len(confidences))
        ax2.scatter(1 + jitter, confidences, alpha=0.6, color=COLORS["secondary"],
                   s=30, zorder=5)
        
        # Statistics annotation
        stats_text = f"n={len(confidences)}\nMin: {min(confidences):.3f}\nMax: {max(confidences):.3f}\nStd: {np.std(confidences):.3f}"
        ax2.text(1.3, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_ylabel("Confidence Score", fontsize=12, fontweight='medium')
        ax2.set_title("Confidence Box Plot", fontsize=13, fontweight='bold')
        ax2.set_xticklabels(['Evidence'])
        ax2.set_ylim(0, 1.05)
        
        fig.suptitle(title or f"Evidence Confidence Analysis: {symbol}", 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, f"{symbol.lower()}_confidence_distribution")
    
    # -------------------------------------------------------------------------
    # Round Heatmap
    # -------------------------------------------------------------------------
    
    def plot_round_heatmap(self, result: dict, title: Optional[str] = None) -> Path:
        """
        Create heatmap of evidence by specialist and round.
        
        Args:
            result: Debate result dictionary
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        symbol = result.get("symbol", "Unknown")
        contributions = extract_specialist_contributions(result)
        
        if not contributions:
            return None
        
        # Build matrix
        specialists = list(contributions.keys())
        all_rounds = sorted(set(
            r for spec_data in contributions.values() 
            for r in spec_data.keys()
        ))
        
        data = np.zeros((len(specialists), len(all_rounds)))
        for i, specialist in enumerate(specialists):
            for j, round_num in enumerate(all_rounds):
                data[i, j] = contributions[specialist].get(round_num, 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Labels
        ax.set_xticks(np.arange(len(all_rounds)))
        ax.set_yticks(np.arange(len(specialists)))
        ax.set_xticklabels([f"Round {r}" for r in all_rounds])
        ax.set_yticklabels(specialists)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add value annotations
        for i in range(len(specialists)):
            for j in range(len(all_rounds)):
                text = ax.text(j, i, int(data[i, j]), ha="center", va="center",
                              color="white" if data[i, j] > data.max()/2 else "black",
                              fontweight='bold')
        
        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel("Evidence Count", rotation=-90, va="bottom", fontsize=11)
        
        ax.set_title(title or f"Evidence Heatmap: {symbol}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, f"{symbol.lower()}_round_heatmap")
    
    # -------------------------------------------------------------------------
    # CDAG Network Graph
    # -------------------------------------------------------------------------
    
    def plot_cdag_network(self, result: dict, title: Optional[str] = None) -> Path:
        """
        Create CDAG network visualization using NetworkX.
        
        Shows the argument graph structure with propositions, evidence,
        and rebuttals colored by type.
        
        Args:
            result: Debate result dictionary
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available for network plot")
            return None
        
        symbol = result.get("symbol", "Unknown")
        nodes, edges = extract_graph_structure(result)
        
        if not nodes:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create graph
        G = nx.DiGraph()
        
        # Node color mapping
        node_colors = {
            "Proposition": COLORS["primary"],
            "Evidence": COLORS["support"],  # Will be overridden by polarity
            "Rebuttal": COLORS["rebuttal"],
        }
        
        # Add nodes
        color_list = []
        size_list = []
        labels = {}
        
        for node in nodes:
            node_id = node["id"]
            node_type = node.get("type", "Unknown")
            G.add_node(node_id)
            
            # Determine color based on type and polarity
            if node_type == "Evidence":
                polarity = node.get("polarity", 0)
                color = COLORS["support"] if polarity > 0 else COLORS["attack"]
            else:
                color = node_colors.get(node_type, COLORS["neutral"])
            
            color_list.append(color)
            
            # Size based on type
            if node_type == "Proposition":
                size_list.append(3000)
            else:
                size_list.append(800)
            
            # Label
            text = node.get("text", "")[:30] + "..." if len(node.get("text", "")) > 30 else node.get("text", "")
            labels[node_id] = text[:20]
        
        # Add edges
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        # Layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_list, 
                               node_size=size_list, alpha=0.9, edgecolors='white', 
                               linewidths=2)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=COLORS["neutral"],
                               alpha=0.5, arrows=True, arrowsize=20,
                               connectionstyle="arc3,rad=0.1")
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=COLORS["primary"], label='Proposition'),
            mpatches.Patch(color=COLORS["support"], label='Supporting Evidence'),
            mpatches.Patch(color=COLORS["attack"], label='Attacking Evidence'),
            mpatches.Patch(color=COLORS["rebuttal"], label='Rebuttal'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Stats annotation
        graph_summary = result.get("graph", {}).get("summary", {})
        stats_text = (
            f"Nodes: {len(nodes)}\n"
            f"Edges: {len(edges)}\n"
            f"Evidence: {graph_summary.get('num_evidence', 'N/A')}\n"
            f"Rebuttals: {graph_summary.get('num_rebuttals', 'N/A')}"
        )
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_title(title or f"CDAG Network: {symbol}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        return self._save_figure(fig, f"{symbol.lower()}_cdag_network")
    
    # -------------------------------------------------------------------------
    # Multi-Stock Comparison
    # -------------------------------------------------------------------------
    
    def plot_multi_stock_comparison(self, results: list[dict], title: Optional[str] = None) -> Path:
        """
        Create multi-stock comparison dashboard.
        
        Compares final posteriors, evidence counts, and verdicts across
        multiple debate results.
        
        Args:
            results: List of debate result dictionaries
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        if not results:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        symbols = [r.get("symbol", "?") for r in results]
        posteriors = [r.get("verdict", {}).get("posterior", 0.5) for r in results]
        verdicts = [r.get("verdict", {}).get("label", "N/A") for r in results]
        
        # 1. Final Posteriors Bar Chart
        ax1 = axes[0, 0]
        colors_bar = [COLORS["support"] if p > 0.5 else COLORS["attack"] for p in posteriors]
        bars = ax1.bar(symbols, posteriors, color=colors_bar, edgecolor='white', linewidth=2)
        ax1.axhline(y=0.5, color=COLORS["neutral"], linestyle='--', linewidth=2, label="Prior")
        for bar, post, verd in zip(bars, posteriors, verdicts):
            ax1.text(bar.get_x() + bar.get_width()/2, post + 0.02,
                    f"{post:.3f}\n({verd})", ha='center', fontsize=9, fontweight='bold')
        ax1.set_ylabel("Final Posterior", fontsize=12)
        ax1.set_title("Final Posteriors by Stock", fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 1.15)
        ax1.legend()
        
        # 2. Evidence Count Comparison
        ax2 = axes[0, 1]
        support_counts = []
        attack_counts = []
        for r in results:
            s, a = extract_evidence_by_polarity(r)
            support_counts.append(s)
            attack_counts.append(a)
        
        x = np.arange(len(symbols))
        width = 0.35
        ax2.bar(x - width/2, support_counts, width, label='Supporting', color=COLORS["support"])
        ax2.bar(x + width/2, attack_counts, width, label='Attacking', color=COLORS["attack"])
        ax2.set_xticks(x)
        ax2.set_xticklabels(symbols)
        ax2.set_ylabel("Evidence Count", fontsize=12)
        ax2.set_title("Evidence Distribution by Stock", fontsize=13, fontweight='bold')
        ax2.legend()
        
        # 3. Verdict Distribution Pie
        ax3 = axes[1, 0]
        verdict_counts = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        labels_pie = list(verdict_counts.keys())
        sizes_pie = list(verdict_counts.values())
        colors_pie = [COLORS["support"] if 'support' in l.lower() else 
                     COLORS["attack"] if 'reject' in l.lower() else 
                     COLORS["neutral"] for l in labels_pie]
        
        ax3.pie(sizes_pie, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11})
        ax3.set_title("Verdict Distribution", fontsize=13, fontweight='bold')
        
        # 4. Duration Comparison
        ax4 = axes[1, 1]
        durations = [r.get("duration_seconds", 0) / 60 for r in results]  # Convert to minutes
        ax4.barh(symbols, durations, color=COLORS["secondary"], edgecolor='white', linewidth=2)
        for i, d in enumerate(durations):
            ax4.text(d + 0.1, i, f"{d:.1f}m", va='center', fontsize=10)
        ax4.set_xlabel("Duration (minutes)", fontsize=12)
        ax4.set_title("Debate Duration by Stock", fontsize=13, fontweight='bold')
        
        fig.suptitle(title or "Multi-Stock SEC Debate Comparison", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, "multi_stock_comparison")
    
    # -------------------------------------------------------------------------
    # Radar Chart
    # -------------------------------------------------------------------------
    
    def plot_summary_radar(self, results: list[dict], title: Optional[str] = None) -> Path:
        """
        Create radar chart comparing multiple stocks across metrics.
        
        Args:
            results: List of debate result dictionaries
            title: Optional custom title
            
        Returns:
            Path to saved plot
        """
        if not results or len(results) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Metrics
        categories = ['Posterior', 'Confidence', 'Evidence\nCount', 'Support\nRatio', 'Duration\n(norm)']
        N = len(categories)
        
        # Compute values for each stock
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        max_duration = max(r.get("duration_seconds", 1) for r in results)
        
        for i, result in enumerate(results):
            symbol = result.get("symbol", "?")
            posterior = result.get("verdict", {}).get("posterior", 0.5)
            confidence = result.get("verdict", {}).get("confidence", 0.5)
            support, attack = extract_evidence_by_polarity(result)
            total_evidence = support + attack
            support_ratio = support / total_evidence if total_evidence > 0 else 0.5
            duration_norm = result.get("duration_seconds", 0) / max_duration
            
            values = [posterior, confidence, min(total_evidence / 50, 1), support_ratio, 1 - duration_norm]
            values += values[:1]
            
            color = plt.cm.Set2(i / len(results))
            ax.plot(angles, values, 'o-', linewidth=2, label=symbol, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.set_title(title or "Multi-Stock Metrics Comparison", fontsize=14, fontweight='bold', pad=20)
        
        return self._save_figure(fig, "summary_radar")
    
    # -------------------------------------------------------------------------
    # Generate All Plots
    # -------------------------------------------------------------------------
    
    def generate_all_plots(self, result: dict) -> list[Path]:
        """
        Generate all available plots for a single debate result.
        
        Args:
            result: Debate result dictionary
            
        Returns:
            List of paths to generated plots
        """
        paths = []
        
        # Individual plots
        plot_methods = [
            self.plot_posterior_evolution,
            self.plot_evidence_distribution,
            self.plot_specialist_contributions,
            self.plot_confidence_distribution,
            self.plot_round_heatmap,
            self.plot_cdag_network,
        ]
        
        for method in plot_methods:
            try:
                path = method(result)
                if path:
                    paths.append(path)
            except Exception as e:
                logger.error(f"Error in {method.__name__}: {e}")
        
        return paths
    
    def generate_all_comparison_plots(self, results: list[dict]) -> list[Path]:
        """
        Generate all comparison plots for multiple debate results.
        
        Args:
            results: List of debate result dictionaries
            
        Returns:
            List of paths to generated plots
        """
        paths = []
        
        # Multi-stock plots
        try:
            path = self.plot_multi_stock_comparison(results)
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Error in multi_stock_comparison: {e}")
        
        try:
            path = self.plot_summary_radar(results)
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Error in summary_radar: {e}")
        
        return paths


# =============================================================================
# Interactive Plots (Plotly)
# =============================================================================

class InteractivePlotter:
    """
    Interactive plotting using Plotly for web-based visualizations.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")
        self.config = config or PlotConfig()
        logger.info(f"InteractivePlotter initialized")
    
    def _save_html(self, fig: go.Figure, name: str) -> Path:
        """Save figure as HTML."""
        path = self.config.output_dir / f"{name}.html"
        fig.write_html(str(path))
        logger.info(f"Saved interactive plot: {path}")
        return path
    
    def plot_interactive_posterior(self, result: dict) -> Path:
        """Create interactive posterior evolution plot."""
        symbol = result.get("symbol", "Unknown")
        rounds, before, after = extract_posterior_timeline(result)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rounds, y=before,
            mode='lines+markers',
            name='Before Round',
            line=dict(color=COLORS["neutral"], dash='dash'),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds, y=after,
            mode='lines+markers',
            name='After Round',
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=12),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        
        fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["neutral"],
                     annotation_text="Prior (0.5)")
        
        fig.update_layout(
            title=f"Interactive Posterior Evolution: {symbol}",
            xaxis_title="Debate Round",
            yaxis_title="Posterior Probability",
            yaxis_range=[0, 1.05],
            template="plotly_white",
            hovermode='x unified',
            font=dict(family="Arial", size=12)
        )
        
        return self._save_html(fig, f"{symbol.lower()}_interactive_posterior")
    
    def plot_interactive_network(self, result: dict) -> Path:
        """Create interactive network graph."""
        symbol = result.get("symbol", "Unknown")
        nodes, edges = extract_graph_structure(result)
        
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX required for network layout")
            return None
        
        # Create NetworkX graph for layout
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        pos = nx.spring_layout(G, k=2, seed=42)
        
        # Edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = pos.get(edge["source"], (0, 0))
            x1, y1 = pos.get(edge["target"], (0, 0))
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=COLORS["neutral"]),
            hoverinfo='none',
            mode='lines'
        )
        
        # Node traces
        node_x = []
        node_y = []
        node_colors = []
        node_texts = []
        node_sizes = []
        
        for node in nodes:
            x, y = pos.get(node["id"], (0, 0))
            node_x.append(x)
            node_y.append(y)
            
            node_type = node.get("type", "Unknown")
            if node_type == "Proposition":
                node_colors.append(COLORS["primary"])
                node_sizes.append(30)
            elif node_type == "Evidence":
                polarity = node.get("polarity", 0)
                node_colors.append(COLORS["support"] if polarity > 0 else COLORS["attack"])
                node_sizes.append(15)
            else:
                node_colors.append(COLORS["rebuttal"])
                node_sizes.append(12)
            
            text = f"{node_type}<br>{node.get('text', '')[:50]}..."
            node_texts.append(text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_texts,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Interactive CDAG Network: {symbol}",
            showlegend=False,
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest'
        )
        
        return self._save_html(fig, f"{symbol.lower()}_interactive_network")
    
    def plot_dashboard(self, results: list[dict]) -> Path:
        """Create comprehensive interactive dashboard."""
        if not results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Final Posteriors", "Evidence Distribution",
                "Posterior Evolution (First Stock)", "Verdict Summary"
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        symbols = [r.get("symbol", "?") for r in results]
        posteriors = [r.get("verdict", {}).get("posterior", 0.5) for r in results]
        verdicts = [r.get("verdict", {}).get("label", "N/A") for r in results]
        
        # 1. Posteriors
        colors_bar = [COLORS["support"] if p > 0.5 else COLORS["attack"] for p in posteriors]
        fig.add_trace(
            go.Bar(x=symbols, y=posteriors, marker_color=colors_bar, name="Posterior"),
            row=1, col=1
        )
        
        # 2. Evidence counts
        support_counts = []
        attack_counts = []
        for r in results:
            s, a = extract_evidence_by_polarity(r)
            support_counts.append(s)
            attack_counts.append(a)
        
        fig.add_trace(
            go.Bar(x=symbols, y=support_counts, marker_color=COLORS["support"], name="Supporting"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=symbols, y=attack_counts, marker_color=COLORS["attack"], name="Attacking"),
            row=1, col=2
        )
        
        # 3. Posterior evolution for first stock
        if results:
            rounds, _, after = extract_posterior_timeline(results[0])
            fig.add_trace(
                go.Scatter(x=rounds, y=after, mode='lines+markers', 
                          line=dict(color=COLORS["primary"]), name=symbols[0]),
                row=2, col=1
            )
        
        # 4. Verdict pie
        verdict_counts = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(verdict_counts.keys()), values=list(verdict_counts.values()),
                   marker_colors=[COLORS["support"], COLORS["attack"], COLORS["neutral"]][:len(verdict_counts)]),
            row=2, col=2
        )
        
        fig.update_layout(
            title="SEC Debate Analysis Dashboard",
            template="plotly_white",
            height=800,
            showlegend=True
        )
        
        return self._save_html(fig, "debate_dashboard")


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_debate_plots(
    result: dict,
    output_dir: Optional[Path] = None,
    interactive: bool = True,
) -> list[Path]:
    """
    Generate all plots for a single debate result.
    
    Args:
        result: Debate result dictionary
        output_dir: Output directory for plots
        interactive: Whether to generate interactive plots
        
    Returns:
        List of paths to generated plots
    """
    config = PlotConfig(output_dir=output_dir or Path("./plots"))
    plotter = DebatePlotter(config)
    
    paths = plotter.generate_all_plots(result)
    
    if interactive and PLOTLY_AVAILABLE:
        interactive_plotter = InteractivePlotter(config)
        paths.append(interactive_plotter.plot_interactive_posterior(result))
        paths.append(interactive_plotter.plot_interactive_network(result))
    
    return [p for p in paths if p is not None]


def generate_comparison_plots(
    results: list[dict],
    output_dir: Optional[Path] = None,
    interactive: bool = True,
) -> list[Path]:
    """
    Generate comparison plots for multiple debate results.
    
    Args:
        results: List of debate result dictionaries
        output_dir: Output directory for plots
        interactive: Whether to generate interactive plots
        
    Returns:
        List of paths to generated plots
    """
    config = PlotConfig(output_dir=output_dir or Path("./plots"))
    plotter = DebatePlotter(config)
    
    paths = plotter.generate_all_comparison_plots(results)
    
    if interactive and PLOTLY_AVAILABLE:
        interactive_plotter = InteractivePlotter(config)
        paths.append(interactive_plotter.plot_dashboard(results))
    
    return [p for p in paths if p is not None]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PlotTheme",
    "PlotConfig",
    "DebatePlotter",
    "InteractivePlotter",
    "generate_debate_plots",
    "generate_comparison_plots",
    "COLORS",
    "SPECIALIST_COLORS",
]
