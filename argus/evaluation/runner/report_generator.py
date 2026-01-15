"""
Report Generator for ARGUS Evaluation.

Generates formatted reports from benchmark results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from argus.evaluation.benchmarks.base import BenchmarkResult
from argus.evaluation.runner.results_aggregator import aggregate_results

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates evaluation reports in various formats.
    
    Supports markdown, JSON, and HTML output formats.
    
    Example:
        >>> generator = ReportGenerator(results)
        >>> generator.generate_markdown(\"report.md\")
        >>> generator.generate_html(\"report.html\")
    """
    
    def __init__(self, results: list[BenchmarkResult]):
        """Initialize with results.
        
        Args:
            results: List of BenchmarkResult objects
        """
        self.results = results
        self.summary = aggregate_results(results)
    
    def generate_markdown(self, output_path: Optional[Path] = None) -> str:
        """Generate markdown report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Markdown report string
        """
        lines = [
            "# ARGUS Evaluation Report",
            "",
            f"**Generated**: {datetime.utcnow().isoformat()}Z",
            "",
            f"**Total Runs**: {self.summary.get('total_runs', 0)}",
            f"**Overall Accuracy**: {self.summary.get('overall_accuracy', 0):.3f}",
            "",
            "---",
            "",
            "## Benchmark Results",
            "",
        ]
        
        # Benchmark table
        lines.append("| Benchmark | Runs | Mean Accuracy | Std Dev |")
        lines.append("|-----------|------|---------------|---------|")
        
        for name, data in self.summary.get("benchmarks", {}).items():
            lines.append(
                f"| {name} | {data['runs']} | "
                f"{data['mean_accuracy']:.3f} | {data['std_accuracy']:.3f} |"
            )
        
        lines.extend(["", "## ARGUS Metrics", ""])
        
        # Metrics table
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        
        for name, data in self.summary.get("metrics", {}).items():
            lines.append(
                f"| {name} | {data['mean']:.3f} | {data['std']:.3f} | "
                f"{data['min']:.3f} | {data['max']:.3f} |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            "## Metric Descriptions",
            "",
            "| Metric | Full Name | Description |",
            "|--------|-----------|-------------|",
            "| ARCIS | Argus Reasoning Coherence Index Score | Logical consistency across rounds |",
            "| EVID-Q | Evidence Quality Quotient | Evidence relevance × confidence × source quality |",
            "| DIALEC | Dialectical Depth Evaluation Coefficient | Attack/defense cycle sophistication |",
            "| REBUT-F | Rebuttal Effectiveness Factor | Rebuttal impact on opposing evidence |",
            "| CONV-S | Convergence Stability Score | Posterior convergence rate and stability |",
            "| PROV-I | Provenance Integrity Index | Citation chain completeness |",
            "| CALIB-M | Calibration Matrix Score | Confidence vs accuracy alignment |",
            "| EIG-U | Expected Information Gain Utilization | Uncertainty reduction efficiency |",
        ])
        
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def generate_json(self, output_path: Optional[Path] = None) -> str:
        """Generate JSON report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            JSON report string
        """
        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }
        
        report = json.dumps(report_data, indent=2, default=str)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def generate_html(self, output_path: Optional[Path] = None) -> str:
        """Generate HTML report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            HTML report string
        """
        html = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"UTF-8\">",
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
            "  <title>ARGUS Evaluation Report</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }",
            "    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "    h1 { color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }",
            "    h2 { color: #333; margin-top: 30px; }",
            "    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }",
            "    .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }",
            "    .stat-card .value { font-size: 2em; font-weight: bold; }",
            "    .stat-card .label { opacity: 0.9; margin-top: 5px; }",
            "    table { width: 100%; border-collapse: collapse; margin: 20px 0; }",
            "    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
            "    th { background: #f8f9fa; font-weight: 600; color: #333; }",
            "    tr:hover { background: #f5f5f5; }",
            "    .metric-high { color: #28a745; font-weight: 600; }",
            "    .metric-low { color: #dc3545; }",
            "    .metric-med { color: #ffc107; }",
            "    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <div class=\"container\">",
            f"    <h1>ARGUS Evaluation Report</h1>",
            f"    <p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>",
            "",
            "    <div class=\"stat-grid\">",
            f"      <div class=\"stat-card\"><div class=\"value\">{self.summary.get('total_runs', 0)}</div><div class=\"label\">Total Runs</div></div>",
            f"      <div class=\"stat-card\"><div class=\"value\">{self.summary.get('overall_accuracy', 0):.1%}</div><div class=\"label\">Overall Accuracy</div></div>",
            f"      <div class=\"stat-card\"><div class=\"value\">{len(self.summary.get('benchmarks', {}))}</div><div class=\"label\">Benchmarks</div></div>",
            f"      <div class=\"stat-card\"><div class=\"value\">{len(self.summary.get('metrics', {}))}</div><div class=\"label\">Metrics</div></div>",
            "    </div>",
            "",
            "    <h2>Benchmark Results</h2>",
            "    <table>",
            "      <thead><tr><th>Benchmark</th><th>Runs</th><th>Mean Accuracy</th><th>Std Dev</th></tr></thead>",
            "      <tbody>",
        ]
        
        for name, data in self.summary.get("benchmarks", {}).items():
            acc_class = "metric-high" if data['mean_accuracy'] > 0.7 else "metric-low" if data['mean_accuracy'] < 0.5 else "metric-med"
            html.append(
                f"        <tr><td>{name}</td><td>{data['runs']}</td>"
                f"<td class=\"{acc_class}\">{data['mean_accuracy']:.3f}</td>"
                f"<td>{data['std_accuracy']:.3f}</td></tr>"
            )
        
        html.extend([
            "      </tbody>",
            "    </table>",
            "",
            "    <h2>ARGUS Novel Metrics</h2>",
            "    <table>",
            "      <thead><tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>",
            "      <tbody>",
        ])
        
        for name, data in self.summary.get("metrics", {}).items():
            score_class = "metric-high" if data['mean'] > 0.7 else "metric-low" if data['mean'] < 0.4 else "metric-med"
            html.append(
                f"        <tr><td><strong>{name}</strong></td>"
                f"<td class=\"{score_class}\">{data['mean']:.3f}</td>"
                f"<td>{data['std']:.3f}</td>"
                f"<td>{data['min']:.3f}</td>"
                f"<td>{data['max']:.3f}</td></tr>"
            )
        
        html.extend([
            "      </tbody>",
            "    </table>",
            "",
            "    <div class=\"footer\">",
            "      <p>Generated by ARGUS Evaluation Framework</p>",
            "    </div>",
            "  </div>",
            "</body>",
            "</html>",
        ])
        
        report = "\n".join(html)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def generate_report(
    results: list[BenchmarkResult],
    output_path: Path,
    format: str = "markdown",
) -> str:
    """Convenience function to generate report.
    
    Args:
        results: List of BenchmarkResult objects
        output_path: Path to save report
        format: Report format (markdown, json, html)
        
    Returns:
        Report string
    """
    generator = ReportGenerator(results)
    
    if format == "json":
        return generator.generate_json(output_path)
    elif format == "html":
        return generator.generate_html(output_path)
    else:
        return generator.generate_markdown(output_path)
