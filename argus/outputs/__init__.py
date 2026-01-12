"""
ARGUS Outputs Module.

Provides report generation, output formatting, and visualization for debate results.
Supports JSON, Markdown, structured report formats, and publication-quality plots.

Example:
    >>> from argus.outputs import ReportGenerator, ReportConfig
    >>> 
    >>> generator = ReportGenerator()
    >>> report = generator.generate(debate_result)
    >>> 
    >>> # Export as JSON
    >>> json_output = report.to_json()
    >>> 
    >>> # Export as Markdown
    >>> md_output = report.to_markdown()
    
    >>> # Generate plots
    >>> from argus.outputs import DebatePlotter, PlotConfig
    >>> plotter = DebatePlotter(PlotConfig(output_dir="./plots"))
    >>> plotter.generate_all_plots(debate_result)
"""

from argus.outputs.reports import (
    ReportGenerator,
    ReportConfig,
    DebateReport,
    ReportFormat,
    generate_report,
    export_json,
    export_markdown,
)

from argus.outputs.plotting import (
    PlotTheme,
    PlotConfig,
    DebatePlotter,
    InteractivePlotter,
    generate_debate_plots,
    generate_comparison_plots,
    COLORS,
    SPECIALIST_COLORS,
)

__all__ = [
    # Reports
    "ReportGenerator",
    "ReportConfig",
    "DebateReport",
    "ReportFormat",
    "generate_report",
    "export_json",
    "export_markdown",
    # Plotting
    "PlotTheme",
    "PlotConfig",
    "DebatePlotter",
    "InteractivePlotter",
    "generate_debate_plots",
    "generate_comparison_plots",
    "COLORS",
    "SPECIALIST_COLORS",
]
