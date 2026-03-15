"""
ConsequenceInferenceAgent — generates consequence nodes from verdicts.
"""

from __future__ import annotations

import uuid
import logging
import random
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConsequenceNode:
    """
    A downstream consequence inferred from a debate verdict.

    Attributes:
        node_id: Unique consequence ID
        text: Consequence description
        parent_ids: IDs of parent consequences (or root verdict)
        conditional_probability: P(consequence | parent=True)
        inverse_probability: P(consequence | parent=False)
        category: Consequence category (economic, social, environmental, etc.)
        timeframe: When the consequence might manifest
        severity: Impact severity (0-1)
        confidence: Inference confidence
    """
    node_id: str = field(default_factory=lambda: f"cnode_{uuid.uuid4().hex[:8]}")
    text: str = ""
    parent_ids: list[str] = field(default_factory=list)
    conditional_probability: float = 0.5
    inverse_probability: float = 0.1
    category: str = "general"
    timeframe: str = "medium_term"  # immediate, short_term, medium_term, long_term
    severity: float = 0.5
    confidence: float = 0.5

    @property
    def impact_score(self) -> float:
        """Impact = P(consequence) × severity."""
        return self.conditional_probability * self.severity

    @property
    def sensitivity(self) -> float:
        """How sensitive is this consequence to the root verdict."""
        return abs(self.conditional_probability - self.inverse_probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text[:150],
            "conditional_p": round(self.conditional_probability, 4),
            "inverse_p": round(self.inverse_probability, 4),
            "category": self.category,
            "timeframe": self.timeframe,
            "severity": round(self.severity, 4),
            "impact_score": round(self.impact_score, 4),
            "sensitivity": round(self.sensitivity, 4),
        }


class ConsequenceInferenceAgent:
    """
    Generates consequence nodes from debate verdicts using
    forward inference and policy analysis patterns.

    Each agent acts as either an OPPORTUNITY agent (positive consequences)
    or a RISK agent (negative consequences).

    Example:
        >>> agent = ConsequenceInferenceAgent(role="risk")
        >>> consequences = agent.infer(
        ...     verdict="Ban single-use plastics → SUPPORTED (0.73)",
        ...     proposition="Ban single-use plastics",
        ... )
    """

    # Domain consequence templates
    CONSEQUENCE_TEMPLATES: dict[str, list[dict[str, Any]]] = {
        "economic": [
            {"pattern": "cost", "text": "Economic costs increase for affected industries",
             "cp": 0.7, "severity": 0.6},
            {"pattern": "job", "text": "Job displacement in affected sectors",
             "cp": 0.5, "severity": 0.7},
            {"pattern": "innovation", "text": "Innovation in alternative solutions accelerates",
             "cp": 0.6, "severity": 0.3},
            {"pattern": "market", "text": "Market restructuring creates new opportunities",
             "cp": 0.55, "severity": 0.4},
        ],
        "social": [
            {"pattern": "public", "text": "Public behavior patterns shift significantly",
             "cp": 0.65, "severity": 0.5},
            {"pattern": "equity", "text": "Equity implications for vulnerable populations",
             "cp": 0.4, "severity": 0.8},
            {"pattern": "awareness", "text": "Public awareness and education increase",
             "cp": 0.7, "severity": 0.2},
        ],
        "environmental": [
            {"pattern": "emission", "text": "Environmental impact reduction measurable within 5 years",
             "cp": 0.6, "severity": 0.5},
            {"pattern": "ecosystem", "text": "Ecosystem recovery processes begin",
             "cp": 0.45, "severity": 0.6},
        ],
        "policy": [
            {"pattern": "regulation", "text": "Regulatory frameworks require updates",
             "cp": 0.75, "severity": 0.4},
            {"pattern": "compliance", "text": "Compliance costs for implementation",
             "cp": 0.65, "severity": 0.5},
            {"pattern": "precedent", "text": "Sets precedent for similar future policies",
             "cp": 0.7, "severity": 0.3},
        ],
    }

    def __init__(
        self,
        role: str = "opportunity",  # opportunity or risk
        max_consequences: int = 8,
    ):
        self.role = role
        self.max_consequences = max_consequences

    def infer(
        self,
        verdict: str,
        proposition: str = "",
        verdict_posterior: float = 0.5,
    ) -> list[ConsequenceNode]:
        """
        Infer consequences from a debate verdict.

        Args:
            verdict: Verdict string
            proposition: Original proposition
            verdict_posterior: Posterior probability

        Returns:
            List of ConsequenceNode objects
        """
        consequences: list[ConsequenceNode] = []
        prop_lower = proposition.lower()

        for category, templates in self.CONSEQUENCE_TEMPLATES.items():
            for template in templates:
                # Score relevance
                relevance = 0.2

                # Keyword matching
                if template["pattern"] in prop_lower:
                    relevance += 0.4
                if any(w in prop_lower for w in category.split()):
                    relevance += 0.2

                # Random noise for diversity
                relevance += random.uniform(0, 0.2)

                if relevance < 0.3:
                    continue

                # Adjust for role
                cp = template["cp"]
                severity = template["severity"]

                if self.role == "risk":
                    severity = min(1.0, severity * 1.3)
                    cp = min(0.95, cp * 1.1)
                else:
                    severity = severity * 0.8

                # Adjust for verdict posterior  
                cp = cp * (0.5 + 0.5 * verdict_posterior)
                inverse_p = max(0.05, cp * 0.2)

                timeframes = ["immediate", "short_term", "medium_term", "long_term"]

                node = ConsequenceNode(
                    text=template["text"],
                    parent_ids=[],  # Will be linked to root
                    conditional_probability=cp,
                    inverse_probability=inverse_p,
                    category=category,
                    timeframe=random.choice(timeframes),
                    severity=severity,
                    confidence=relevance,
                )
                consequences.append(node)

        # Sort by impact and limit
        consequences.sort(key=lambda c: c.impact_score, reverse=True)
        consequences = consequences[:self.max_consequences]

        logger.info(
            f"MIRROR {self.role} agent inferred {len(consequences)} consequences"
        )
        return consequences
