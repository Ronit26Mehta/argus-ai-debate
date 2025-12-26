"""
Cite & Critique Prompting for ARGUS.

Implements structured prompting for citation extraction
and evidence evaluation with criticism.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class CiteResult:
    """
    Result from cite & critique prompting.
    
    Attributes:
        claim: Extracted claim text
        quote: Direct quote from source
        confidence: Confidence in the claim
        relevance: Relevance to query
        critique: Critique/limitations
        source_id: Source chunk ID
        polarity: Support (+1), attack (-1), neutral (0)
    """
    claim: str
    quote: str = ""
    confidence: float = 0.5
    relevance: float = 0.5
    critique: str = ""
    source_id: str = ""
    polarity: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim": self.claim,
            "quote": self.quote,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "critique": self.critique,
            "source_id": self.source_id,
            "polarity": self.polarity,
        }


def cite_and_critique(
    llm: "BaseLLM",
    query: str,
    source_text: str,
    source_id: str = "",
) -> CiteResult:
    """
    Extract claims with citations and critique.
    
    Uses structured prompting to:
    1. Find relevant information
    2. Extract specific claims with quotes
    3. Assess confidence and relevance
    4. Critique limitations
    
    Args:
        llm: LLM instance
        query: The query/proposition to evaluate against
        source_text: Source text to extract from
        source_id: Source identifier for provenance
        
    Returns:
        CiteResult with extracted claim and metadata
    """
    prompt = f"""Analyze this source text in relation to the query.

Query: {query}

Source Text:
\"\"\"
{source_text[:2000]}
\"\"\"

Instructions:
1. CITE: Find the most relevant passage and quote it directly
2. CLAIM: Extract a specific claim that addresses the query
3. CRITIQUE: Identify limitations, biases, or caveats
4. ASSESS: Rate confidence (0-1) and relevance (0-1)
5. POLARITY: Does it support (+1), attack (-1), or neutral (0)?

Return JSON:
{{
    "quote": "Direct quote from source (50-150 chars)",
    "claim": "The specific claim extracted",
    "critique": "Limitations or concerns",
    "confidence": 0.75,
    "relevance": 0.85,
    "polarity": "support" or "attack" or "neutral"
}}"""
    
    response = llm.generate(prompt, temperature=0.3)
    
    try:
        data = json.loads(response.content)
        
        polarity_map = {"support": 1, "attack": -1, "neutral": 0}
        polarity = polarity_map.get(data.get("polarity", "neutral"), 0)
        
        return CiteResult(
            claim=data.get("claim", ""),
            quote=data.get("quote", ""),
            confidence=float(data.get("confidence", 0.5)),
            relevance=float(data.get("relevance", 0.5)),
            critique=data.get("critique", ""),
            source_id=source_id,
            polarity=polarity,
        )
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract what we can
        return CiteResult(
            claim=response.content[:200],
            confidence=0.5,
            relevance=0.5,
            source_id=source_id,
        )


def extract_claims(
    llm: "BaseLLM",
    proposition: str,
    source_texts: list[tuple[str, str]],  # (source_id, text)
    max_claims: int = 5,
) -> list[CiteResult]:
    """
    Extract multiple claims from multiple sources.
    
    Args:
        llm: LLM instance
        proposition: Proposition to evaluate
        source_texts: List of (source_id, text) tuples
        max_claims: Maximum claims to extract
        
    Returns:
        List of CiteResult objects
    """
    results = []
    
    for source_id, text in source_texts[:max_claims * 2]:
        if len(results) >= max_claims:
            break
        
        result = cite_and_critique(llm, proposition, text, source_id)
        
        # Only include if relevant
        if result.relevance >= 0.3 and result.claim:
            results.append(result)
    
    # Sort by relevance × confidence
    results.sort(key=lambda x: x.relevance * x.confidence, reverse=True)
    
    return results[:max_claims]


def build_evidence_summary(
    claims: list[CiteResult],
    max_length: int = 500,
) -> str:
    """
    Build a summary of evidence claims.
    
    Args:
        claims: List of CiteResult objects
        max_length: Maximum summary length
        
    Returns:
        Formatted summary string
    """
    support = [c for c in claims if c.polarity > 0]
    attack = [c for c in claims if c.polarity < 0]
    neutral = [c for c in claims if c.polarity == 0]
    
    parts = []
    
    if support:
        parts.append("**Supporting Evidence:**")
        for c in support[:3]:
            parts.append(f"- {c.claim[:100]}... [{c.confidence:.0%}]")
    
    if attack:
        parts.append("\n**Contradicting Evidence:**")
        for c in attack[:3]:
            parts.append(f"- {c.claim[:100]}... [{c.confidence:.0%}]")
    
    if neutral:
        parts.append("\n**Related Information:**")
        for c in neutral[:2]:
            parts.append(f"- {c.claim[:100]}...")
    
    summary = "\n".join(parts)
    
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary
