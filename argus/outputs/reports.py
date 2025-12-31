"""
Report Generator for ARGUS.

Generates structured reports from debate results in various formats:
    - JSON (primary format for programmatic access)
    - Markdown (human-readable documentation)
    - Summary dictionaries (for dashboards)

Example:
    >>> from argus.outputs import ReportGenerator, ReportFormat
    >>> 
    >>> generator = ReportGenerator()
    >>> report = generator.generate(debate_result)
    >>> 
    >>> # Get JSON output
    >>> json_data = report.to_json()
    >>> 
    >>> # Get Markdown
    >>> markdown = report.to_markdown()
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from argus.orchestrator import DebateResult
    from argus.cdag.graph import CDAG
    from argus.cdag.nodes import Proposition, Evidence

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Available report formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    SUMMARY = "summary"


class ReportConfig(BaseModel):
    """Configuration for report generation.
    
    Attributes:
        include_evidence: Include full evidence details
        include_graph: Include graph structure
        include_metadata: Include execution metadata
        max_evidence_items: Maximum evidence items to include
    """
    include_evidence: bool = Field(
        default=True,
        description="Include evidence details in report",
    )
    include_graph: bool = Field(
        default=True,
        description="Include graph structure",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include execution metadata",
    )
    include_provenance: bool = Field(
        default=False,
        description="Include provenance trail",
    )
    max_evidence_items: int = Field(
        default=50,
        ge=1,
        description="Maximum evidence items to include",
    )
    pretty_print: bool = Field(
        default=True,
        description="Pretty print JSON output",
    )


@dataclass
class EvidenceSummary:
    """Summary of a piece of evidence."""
    id: str
    text: str
    type: str
    polarity: int
    confidence: float
    source: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerdictSummary:
    """Summary of a debate verdict."""
    verdict: str
    prior: float
    posterior: float
    confidence: float
    reasoning: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebateReport:
    """Complete report of a debate session.
    
    Contains all information about a completed debate including
    the proposition, verdict, evidence, and statistics.
    """
    # Core information
    proposition_id: str
    proposition_text: str
    verdict: VerdictSummary
    
    # Evidence
    supporting_evidence: list[EvidenceSummary] = field(default_factory=list)
    attacking_evidence: list[EvidenceSummary] = field(default_factory=list)
    rebuttals: list[dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    num_rounds: int = 0
    total_evidence: int = 0
    total_rebuttals: int = 0
    duration_seconds: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: dict[str, Any] = field(default_factory=dict)
    
    # Graph data (optional)
    graph_data: Optional[dict[str, Any]] = None
    
    # Provenance (optional)
    provenance_trail: Optional[list[dict[str, Any]]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.
        
        Returns:
            Dict representation suitable for JSON serialization
        """
        return {
            "proposition": {
                "id": self.proposition_id,
                "text": self.proposition_text,
            },
            "verdict": self.verdict.to_dict(),
            "evidence": {
                "supporting": [e.to_dict() for e in self.supporting_evidence],
                "attacking": [e.to_dict() for e in self.attacking_evidence],
                "rebuttals": self.rebuttals,
            },
            "statistics": {
                "num_rounds": self.num_rounds,
                "total_evidence": self.total_evidence,
                "total_rebuttals": self.total_rebuttals,
                "duration_seconds": self.duration_seconds,
            },
            "metadata": {
                "created_at": self.created_at.isoformat(),
                "config": self.config,
            },
            "graph": self.graph_data,
            "provenance": self.provenance_trail,
        }
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert report to JSON string.
        
        Args:
            indent: JSON indentation (None for compact)
            
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_markdown(self) -> str:
        """Convert report to Markdown format.
        
        Returns:
            Markdown string
        """
        lines = []
        
        # Title
        lines.append(f"# Debate Report: {self.proposition_text[:50]}...")
        lines.append("")
        lines.append(f"*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        # Verdict Section
        lines.append("## Verdict")
        lines.append("")
        lines.append(f"**{self.verdict.verdict.upper()}**")
        lines.append("")
        lines.append(f"- Prior Probability: {self.verdict.prior:.2%}")
        lines.append(f"- Posterior Probability: {self.verdict.posterior:.2%}")
        lines.append(f"- Confidence: {self.verdict.confidence:.2%}")
        lines.append("")
        
        if self.verdict.reasoning:
            lines.append("### Reasoning")
            lines.append(self.verdict.reasoning)
            lines.append("")
        
        # Proposition
        lines.append("## Proposition")
        lines.append("")
        lines.append(f"> {self.proposition_text}")
        lines.append("")
        
        # Supporting Evidence
        if self.supporting_evidence:
            lines.append("## Supporting Evidence")
            lines.append("")
            for i, evidence in enumerate(self.supporting_evidence, 1):
                lines.append(f"### {i}. {evidence.type.title()} Evidence")
                lines.append(f"> {evidence.text}")
                lines.append(f"- Confidence: {evidence.confidence:.2%}")
                if evidence.source:
                    lines.append(f"- Source: {evidence.source}")
                lines.append("")
        
        # Attacking Evidence
        if self.attacking_evidence:
            lines.append("## Attacking Evidence")
            lines.append("")
            for i, evidence in enumerate(self.attacking_evidence, 1):
                lines.append(f"### {i}. {evidence.type.title()} Evidence")
                lines.append(f"> {evidence.text}")
                lines.append(f"- Confidence: {evidence.confidence:.2%}")
                if evidence.source:
                    lines.append(f"- Source: {evidence.source}")
                lines.append("")
        
        # Statistics
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"- Debate Rounds: {self.num_rounds}")
        lines.append(f"- Total Evidence: {self.total_evidence}")
        lines.append(f"- Total Rebuttals: {self.total_rebuttals}")
        lines.append(f"- Duration: {self.duration_seconds:.2f}s")
        lines.append("")
        
        return "\n".join(lines)
    
    def to_summary(self) -> dict[str, Any]:
        """Get a compact summary of the report.
        
        Returns:
            Compact summary dict for dashboards
        """
        return {
            "proposition": self.proposition_text[:100],
            "verdict": self.verdict.verdict,
            "posterior": self.verdict.posterior,
            "confidence": self.verdict.confidence,
            "supporting_count": len(self.supporting_evidence),
            "attacking_count": len(self.attacking_evidence),
            "duration": self.duration_seconds,
        }


class ReportGenerator:
    """Generates reports from debate results.
    
    Transforms DebateResult objects into structured reports
    that can be exported in various formats.
    
    Example:
        >>> generator = ReportGenerator(config=ReportConfig(
        ...     include_evidence=True,
        ...     max_evidence_items=20,
        ... ))
        >>> 
        >>> report = generator.generate(debate_result)
        >>> print(report.to_json())
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the generator.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        logger.debug("Initialized ReportGenerator")
    
    def generate(
        self,
        result: "DebateResult",
        graph: Optional["CDAG"] = None,
    ) -> DebateReport:
        """Generate a report from a debate result.
        
        Args:
            result: DebateResult from orchestrator
            graph: Optional C-DAG for detailed info
            
        Returns:
            DebateReport object
        """
        # Use graph from result if not provided
        if graph is None:
            graph = getattr(result, "graph", None)
        
        # Extract verdict summary
        verdict = VerdictSummary(
            verdict=result.verdict.verdict if hasattr(result.verdict, "verdict") else str(result.verdict),
            prior=getattr(result.verdict, "prior", 0.5),
            posterior=getattr(result.verdict, "posterior", 0.5),
            confidence=getattr(result.verdict, "confidence", 0.0),
            reasoning=getattr(result.verdict, "reasoning", None),
        )
        
        # Extract evidence if graph available
        supporting = []
        attacking = []
        rebuttals = []
        
        if graph is not None and self.config.include_evidence:
            # Get proposition
            prop = graph.get_proposition(result.proposition_id)
            if prop:
                # Get supporting evidence
                support_nodes = graph.get_supporting_evidence(prop.id)
                for i, node in enumerate(support_nodes[:self.config.max_evidence_items]):
                    supporting.append(EvidenceSummary(
                        id=node.id,
                        text=node.text,
                        type=node.evidence_type.value if hasattr(node, "evidence_type") else "unknown",
                        polarity=1,
                        confidence=node.confidence,
                        source=getattr(node, "source_url", None),
                    ))
                
                # Get attacking evidence
                attack_nodes = graph.get_attacking_evidence(prop.id)
                for node in attack_nodes[:self.config.max_evidence_items]:
                    attacking.append(EvidenceSummary(
                        id=node.id,
                        text=node.text,
                        type=node.evidence_type.value if hasattr(node, "evidence_type") else "unknown",
                        polarity=-1,
                        confidence=node.confidence,
                        source=getattr(node, "source_url", None),
                    ))
        
        # Extract graph data if requested
        graph_data = None
        if graph is not None and self.config.include_graph:
            graph_data = self._extract_graph_data(graph)
        
        # Get proposition text
        prop_text = ""
        if graph is not None:
            prop = graph.get_proposition(result.proposition_id)
            if prop:
                prop_text = prop.text
        
        return DebateReport(
            proposition_id=result.proposition_id,
            proposition_text=prop_text,
            verdict=verdict,
            supporting_evidence=supporting,
            attacking_evidence=attacking,
            rebuttals=rebuttals,
            num_rounds=result.num_rounds,
            total_evidence=result.num_evidence,
            total_rebuttals=result.num_rebuttals,
            duration_seconds=result.duration_seconds,
            graph_data=graph_data,
            config=self.config.model_dump() if self.config.include_metadata else {},
        )
    
    def _extract_graph_data(self, graph: "CDAG") -> dict[str, Any]:
        """Extract graph structure for report.
        
        Args:
            graph: C-DAG graph
            
        Returns:
            Graph data dict
        """
        nodes = []
        edges = []
        
        # Extract nodes
        for node in graph.get_all_propositions():
            nodes.append({
                "id": node.id,
                "type": "proposition",
                "text": node.text[:100],
                "posterior": node.posterior,
            })
        
        for node in graph.get_all_evidence():
            nodes.append({
                "id": node.id,
                "type": "evidence",
                "text": node.text[:100],
                "confidence": node.confidence,
            })
        
        # Extract edges
        for edge in graph.get_all_edges():
            edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type.value if hasattr(edge, "edge_type") else "unknown",
                "weight": edge.weight,
            })
        
        return {
            "name": graph.name,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }
    
    def generate_batch(
        self,
        results: list["DebateResult"],
    ) -> list[DebateReport]:
        """Generate reports for multiple debate results.
        
        Args:
            results: List of DebateResults
            
        Returns:
            List of DebateReports
        """
        return [self.generate(r) for r in results]


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_report(
    result: "DebateResult",
    config: Optional[ReportConfig] = None,
) -> DebateReport:
    """Generate a report from a debate result.
    
    Args:
        result: DebateResult
        config: Optional report configuration
        
    Returns:
        DebateReport
    """
    generator = ReportGenerator(config)
    return generator.generate(result)


def export_json(
    result: "DebateResult",
    path: Optional[str] = None,
    config: Optional[ReportConfig] = None,
) -> str:
    """Export debate result as JSON.
    
    Args:
        result: DebateResult
        path: Optional file path to write
        config: Report configuration
        
    Returns:
        JSON string
    """
    report = generate_report(result, config)
    json_str = report.to_json()
    
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info(f"Exported JSON report to {path}")
    
    return json_str


def export_markdown(
    result: "DebateResult",
    path: Optional[str] = None,
    config: Optional[ReportConfig] = None,
) -> str:
    """Export debate result as Markdown.
    
    Args:
        result: DebateResult
        path: Optional file path to write
        config: Report configuration
        
    Returns:
        Markdown string
    """
    report = generate_report(result, config)
    md_str = report.to_markdown()
    
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_str)
        logger.info(f"Exported Markdown report to {path}")
    
    return md_str
