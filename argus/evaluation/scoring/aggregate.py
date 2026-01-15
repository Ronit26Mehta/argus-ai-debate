"""
Score Aggregation and Comparison Utilities.

Provides tools for combining scores, comparing runs, and generating reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from argus.evaluation.scoring.metrics import ScoreCard, METRIC_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class CompositeScore:
    """Weighted composite of multiple metric scores.
    
    Allows custom weighting schemes for different evaluation contexts.
    
    Attributes:
        scores: Individual metric scores
        weights: Custom weights (defaults to METRIC_REGISTRY weights)
        composite: Final weighted score
    """
    scores: dict[str, float]
    weights: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    
    def __post_init__(self):
        if not self.weights:
            self.weights = {
                name: METRIC_REGISTRY[name].weight
                for name in self.scores
                if name in METRIC_REGISTRY
            }
        self.composite = self._compute()
    
    def _compute(self) -> float:
        """Compute weighted composite."""
        if not self.scores:
            return 0.0
        
        total = 0.0
        weight_sum = 0.0
        
        for name, score in self.scores.items():
            weight = self.weights.get(name, 1.0)
            total += score * weight
            weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 0.0
    
    def with_weights(self, weights: dict[str, float]) -> "CompositeScore":
        """Create new composite with different weights."""
        return CompositeScore(scores=self.scores.copy(), weights=weights)


@dataclass
class ScoreComparison:
    """Comparison between two score cards.
    
    Attributes:
        baseline: Baseline scores
        comparison: Comparison scores
        differences: Score differences (comparison - baseline)
        improvements: Metrics that improved
        regressions: Metrics that regressed
    """
    baseline: ScoreCard
    comparison: ScoreCard
    differences: dict[str, float] = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._compute_differences()
    
    def _compute_differences(self):
        """Compute score differences."""
        all_metrics = set(self.baseline.scores.keys()) | set(self.comparison.scores.keys())
        
        for metric in all_metrics:
            baseline_val = self.baseline.scores.get(metric, 0.0)
            comparison_val = self.comparison.scores.get(metric, 0.0)
            diff = comparison_val - baseline_val
            self.differences[metric] = diff
            
            # Determine if improvement or regression
            metric_def = METRIC_REGISTRY.get(metric)
            if metric_def:
                if metric_def.higher_is_better:
                    if diff > 0.01:
                        self.improvements.append(metric)
                    elif diff < -0.01:
                        self.regressions.append(metric)
                else:
                    if diff < -0.01:
                        self.improvements.append(metric)
                    elif diff > 0.01:
                        self.regressions.append(metric)
    
    @property
    def composite_change(self) -> float:
        """Change in composite score."""
        return self.comparison.composite_score - self.baseline.composite_score
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_composite": self.baseline.composite_score,
            "comparison_composite": self.comparison.composite_score,
            "composite_change": self.composite_change,
            "differences": self.differences,
            "improvements": self.improvements,
            "regressions": self.regressions,
        }
    
    def __str__(self) -> str:
        """Format as readable string."""
        lines = ["Score Comparison", "=" * 50]
        lines.append(f"Composite Change: {self.composite_change:+.3f}")
        lines.append("")
        
        if self.improvements:
            lines.append("Improvements:")
            for m in self.improvements:
                lines.append(f"  ↑ {m}: {self.differences[m]:+.3f}")
        
        if self.regressions:
            lines.append("Regressions:")
            for m in self.regressions:
                lines.append(f"  ↓ {m}: {self.differences[m]:+.3f}")
        
        return "\n".join(lines)


def generate_score_report(
    score_cards: list[ScoreCard],
    output_path: Optional[Path] = None,
    format: str = "markdown",
) -> str:
    """
    Generate a formatted report from multiple score cards.
    
    Args:
        score_cards: List of ScoreCard objects
        output_path: Optional path to save report
        format: Report format ('markdown', 'json', 'html')
        
    Returns:
        Formatted report string
    """
    if not score_cards:
        return "No scores to report."
    
    if format == "json":
        report = json.dumps([sc.to_dict() for sc in score_cards], indent=2)
    elif format == "html":
        report = _generate_html_report(score_cards)
    else:
        report = _generate_markdown_report(score_cards)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


def _generate_markdown_report(score_cards: list[ScoreCard]) -> str:
    """Generate markdown format report."""
    lines = [
        "# ARGUS Evaluation Report",
        "",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        f"Total Evaluations: {len(score_cards)}",
        "",
        "## Summary Statistics",
        "",
    ]
    
    # Compute statistics per metric
    metric_stats: dict[str, list[float]] = {}
    for sc in score_cards:
        for metric, score in sc.scores.items():
            if metric not in metric_stats:
                metric_stats[metric] = []
            metric_stats[metric].append(score)
    
    lines.append("| Metric | Mean | Min | Max | Std |")
    lines.append("|--------|------|-----|-----|-----|")
    
    import statistics
    for metric in sorted(metric_stats.keys()):
        values = metric_stats[metric]
        mean = statistics.mean(values)
        min_val = min(values)
        max_val = max(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {metric} | {mean:.3f} | {min_val:.3f} | {max_val:.3f} | {std:.3f} |")
    
    # Composite scores
    composites = [sc.composite_score for sc in score_cards]
    lines.append("")
    lines.append(f"**Overall Composite**: {statistics.mean(composites):.3f} ± {statistics.stdev(composites) if len(composites) > 1 else 0:.3f}")
    
    return "\n".join(lines)


def _generate_html_report(score_cards: list[ScoreCard]) -> str:
    """Generate HTML format report."""
    html = [
        "<!DOCTYPE html>",
        "<html><head><title>ARGUS Evaluation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        ".metric-high { color: green; font-weight: bold; }",
        ".metric-low { color: red; }",
        "</style></head><body>",
        "<h1>ARGUS Evaluation Report</h1>",
        f"<p>Generated: {datetime.utcnow().isoformat()}</p>",
        "<table>",
        "<tr><th>Metric</th><th>Score</th></tr>",
    ]
    
    if score_cards:
        avg_scores = {}
        for sc in score_cards:
            for m, s in sc.scores.items():
                avg_scores[m] = avg_scores.get(m, []) + [s]
        
        for metric in sorted(avg_scores.keys()):
            import statistics
            mean = statistics.mean(avg_scores[metric])
            css_class = "metric-high" if mean > 0.7 else "metric-low" if mean < 0.4 else ""
            html.append(f'<tr><td>{metric}</td><td class="{css_class}">{mean:.3f}</td></tr>')
    
    html.extend(["</table>", "</body></html>"])
    return "\n".join(html)
