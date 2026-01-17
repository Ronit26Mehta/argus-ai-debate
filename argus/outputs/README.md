# ARGUS Outputs Module

## Overview

The `outputs/` module provides visualization and reporting capabilities for debate results, including publication-quality plots and structured reports.

## Components

| File | Description |
|------|-------------|
| `plotting.py` | Matplotlib/Plotly visualization (48KB) |
| `reports.py` | Report generation (Markdown, JSON, HTML) |

## Quick Start

```python
from argus.outputs import DebatePlotter, ReportGenerator, PlotConfig

# Generate plots
config = PlotConfig(output_dir="./plots", dpi=300, format="png")
plotter = DebatePlotter(config)

paths = plotter.generate_all_plots(debate_result)
print(f"Generated {len(paths)} plots")

# Generate report
generator = ReportGenerator()
report = generator.generate(debate_result)
report.save("debate_report.md")
```

## Plotting

```python
from argus.outputs import DebatePlotter, PlotConfig, PlotTheme

config = PlotConfig(
    output_dir="./plots",
    dpi=300,                      # Publication quality
    format="png",                 # png, pdf, svg
    theme=PlotTheme.PUBLICATION,  # publication, dark, light
    figsize=(12, 8),
)

plotter = DebatePlotter(config)

# Individual plot types
plotter.plot_posterior_evolution(result)     # Probability over rounds
plotter.plot_evidence_distribution(result)   # Support vs Attack
plotter.plot_specialist_contributions(result) # By specialist
plotter.plot_confidence_distribution(result)  # Histogram + KDE
plotter.plot_round_heatmap(result)           # Evidence matrix
plotter.plot_cdag_network(result)            # Network graph
plotter.plot_summary_radar(result)           # Multi-metric radar

# Multi-debate comparison
plotter.plot_multi_stock_comparison([result1, result2, result3])

# Interactive (Plotly)
plotter.plot_interactive_posterior(result)
plotter.plot_interactive_network(result)
plotter.plot_dashboard([result1, result2])  # HTML dashboard
```

## Report Generation

```python
from argus.outputs import ReportGenerator, ReportConfig

config = ReportConfig(
    include_evidence=True,
    include_rebuttals=True,
    include_graphs=True,
    max_evidence_per_section=10,
)

generator = ReportGenerator(config)
report = generator.generate(debate_result)

# Export formats
report.to_markdown("report.md")
report.to_json("report.json")
report.to_html("report.html")

# Get summary dict
summary = report.to_summary()
print(f"Verdict: {summary['verdict']}")
print(f"Confidence: {summary['confidence']}")
```

## Color Palettes

```python
from argus.outputs import COLORS, SPECIALIST_COLORS

# Main palette (colorblind-friendly)
COLORS = {
    "primary": "#2E86AB",    # Blue
    "secondary": "#A23B72",  # Magenta
    "success": "#F18F01",    # Orange
    "danger": "#C73E1D",     # Red
    "support": "#2E8B57",    # Green
    "attack": "#DC143C",     # Crimson
}

# Specialist-specific colors
SPECIALIST_COLORS = {
    "Bull Analyst": "#2E8B57",
    "Bear Analyst": "#DC143C",
    "Risk Analyst": "#FF8C00",
    "Technical Analyst": "#4169E1",
}
```

## Plot Themes

| Theme | Description |
|-------|-------------|
| `PUBLICATION` | Professional academic style |
| `DARK` | Dark background |
| `LIGHT` | Clean light theme |
| `MINIMAL` | Reduced chrome |

```python
from argus.outputs import PlotTheme

config = PlotConfig(theme=PlotTheme.DARK)
```
