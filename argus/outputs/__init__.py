"""
ARGUS Outputs Module.

Provides report generation and output formatting for debate results.
Supports JSON, Markdown, and structured report formats.

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

__all__ = [
    "ReportGenerator",
    "ReportConfig",
    "DebateReport",
    "ReportFormat",
    "generate_report",
    "export_json",
    "export_markdown",
]
