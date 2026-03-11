"""
Specialist Agent for ARGUS.

Domain-specific evidence gatherer that retrieves
and evaluates evidence for propositions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

from pydantic import Field

from argus.core.json_repair import (
    extract_json_object,
    repair_json,
)

from argus.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResponse,
)
from argus.cdag.nodes import Evidence, EvidenceType
from argus.cdag.edges import EdgeType

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.cdag.graph import CDAG
    from argus.knowledge.indexing import HybridIndex

logger = logging.getLogger(__name__)


class SpecialistConfig(AgentConfig):
    """Configuration for Specialist agent."""
    
    name: str = "Specialist"
    role: AgentRole = AgentRole.SPECIALIST
    
    domain: str = Field(
        default="general",
        description="Domain of expertise",
    )
    
    max_evidence_per_query: int = Field(
        default=5,
        ge=1,
        description="Maximum evidence items per query",
    )
    
    min_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to include evidence",
    )


class Specialist(BaseAgent):
    """
    Specialist agent for domain-specific evidence gathering.
    
    The Specialist:
    1. Retrieves relevant evidence from knowledge base
    2. Evaluates evidence relevance and quality
    3. Extracts claims with citations
    4. Attaches evidence to the C-DAG
    
    Example:
        >>> specialist = Specialist(llm=llm, domain="clinical")
        >>> evidence = specialist.gather_evidence(graph, prop_id, index)
        >>> for e in evidence:
        ...     print(f"{e.text[:50]}... [{e.confidence:.2f}]")
    """
    
    def __init__(
        self,
        llm: "BaseLLM",
        config: Optional[SpecialistConfig] = None,
        domain: Optional[str] = None,
    ):
        """
        Initialize Specialist.
        
        Args:
            llm: LLM instance
            config: Specialist configuration
            domain: Domain of expertise override
        """
        config = config or SpecialistConfig()
        if domain:
            config.domain = domain
        super().__init__(llm, config)
    
    @property
    def domain(self) -> str:
        """Get domain of expertise."""
        return getattr(self.config, "domain", "general")
    
    def get_system_prompt(self) -> str:
        """Get Specialist system prompt."""
        return f"""You are a Domain Specialist in the ARGUS debate system.

Your domain of expertise: {self.domain}

Your responsibilities:
1. Evaluate evidence relevance to propositions
2. Extract specific claims supported by citations
3. Assess evidence quality and reliability
4. Identify limitations and caveats

Guidelines:
- Cite specific passages that support claims
- Distinguish between strong and weak evidence
- Note any potential biases or conflicts
- Express confidence levels accurately
- Be skeptical but fair

When extracting claims, use this JSON format:
{{
    "claim": "The specific claim text",
    "confidence": 0.85,
    "polarity": "support" or "attack",
    "quote": "Direct quote from source",
    "reasoning": "Why this evidence is relevant"
}}"""
    
    def act(
        self,
        graph: "CDAG",
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Perform specialist action.
        
        Args:
            graph: The C-DAG graph
            context: Action context
            
        Returns:
            AgentResponse with gathered evidence
        """
        action = context.get("action", "gather")
        prop_id = context.get("proposition_id")
        
        if not prop_id:
            return AgentResponse(
                success=False,
                error="proposition_id required",
            )
        
        if action == "gather":
            index = context.get("index")
            embedding_gen = context.get("embedding_generator")
            
            evidence = self.gather_evidence(
                graph, prop_id, index, embedding_gen
            )
            
            return AgentResponse(
                success=True,
                content=f"Gathered {len(evidence)} evidence items",
                data={"evidence_ids": [e.id for e in evidence]},
            )
        
        elif action == "evaluate":
            chunk_text = context.get("chunk_text", "")
            result = self.evaluate_chunk(graph, prop_id, chunk_text)
            
            return AgentResponse(
                success=True,
                content=result.get("claim", ""),
                data=result,
            )
        
        else:
            return AgentResponse(
                success=False,
                error=f"Unknown action: {action}",
            )
    
    def gather_evidence(
        self,
        graph: "CDAG",
        proposition_id: str,
        index: Optional["HybridIndex"] = None,
        embedding_generator: Any = None,
    ) -> list[Evidence]:
        """
        Gather evidence for a proposition.
        
        Args:
            graph: The C-DAG graph
            proposition_id: Proposition to find evidence for
            index: Hybrid search index
            embedding_generator: Embedding generator
            
        Returns:
            List of Evidence nodes added to graph
        """
        prop = graph.get_proposition(proposition_id)
        if not prop:
            raise ValueError(f"Proposition {proposition_id} not found")
        
        evidence_list: list[Evidence] = []
        config = self.config if isinstance(self.config, SpecialistConfig) else SpecialistConfig()
        
        # If we have an index, use retrieval
        if index and embedding_generator:
            query = prop.text
            query_embedding = embedding_generator.embed_query(query)
            
            results = index.search(
                query,
                query_embedding,
                top_k=config.max_evidence_per_query * 2,
            )
            
            for result in results[:config.max_evidence_per_query]:
                if result.chunk:
                    evaluation = self.evaluate_chunk(
                        graph, proposition_id, result.chunk.text
                    )
                    
                    if evaluation.get("confidence", 0) >= config.min_confidence_threshold:
                        evidence = self._create_evidence(
                            evaluation, result.chunk, result.score
                        )
                        graph.add_evidence(
                            evidence,
                            proposition_id,
                            EdgeType.SUPPORTS if evidence.polarity > 0 else EdgeType.ATTACKS,
                        )
                        evidence_list.append(evidence)
        else:
            # Generate synthetic evidence via LLM
            prompt = f"""You are a domain expert in {self.domain}. Analyze this proposition and provide concrete evidence.

Proposition: {prop.text}

Return ONLY a JSON object — no markdown, no explanation, no extra text.
Keep each "type" description under 30 words. Return 3 items max.

{{"evidence_types":[{{"type":"brief description","polarity":"support","importance":0.8}}]}}"""
            
            response = self.generate(prompt, max_tokens=16384)
            
            # Robust JSON extraction via shared repair module
            data = None
            try:
                data = extract_json_object(response)
            except (ValueError, TypeError):
                pass
            
            # Normalize: accept any key containing a list of evidence items
            evidence_items = None
            if data:
                if "evidence_types" in data:
                    evidence_items = data["evidence_types"]
                elif "evidence" in data:
                    evidence_items = data["evidence"]
                else:
                    # Find the first list value in the dict
                    for v in data.values():
                        if isinstance(v, list):
                            evidence_items = v
                            break

            if evidence_items:
                for et in evidence_items[:5]:
                    # Accept "type", "claim", "description", or "text" as the description field
                    desc = (et.get("type") or et.get("claim") or
                            et.get("description") or et.get("text") or "Evidence needed")
                    # Accept "polarity" or "stance" field
                    pol_str = (et.get("polarity") or et.get("stance") or "support").lower()
                    polarity = -1 if "attack" in pol_str or "against" in pol_str or "counter" in pol_str else 1
                    importance = float(et.get("importance", 0) or et.get("confidence", 0) or 0.6)
                    evidence = Evidence(
                        text=f"[{self.domain}] {desc}",
                        evidence_type=EvidenceType.LITERATURE,
                        polarity=polarity,
                        confidence=importance,
                        quality=importance,
                        relevance=0.8,
                    )
                    graph.add_evidence(
                        evidence,
                        proposition_id,
                        EdgeType.SUPPORTS if polarity > 0 else EdgeType.ATTACKS,
                    )
                    evidence_list.append(evidence)
            else:
                # Fallback: generate basic evidence if JSON parsing fails entirely
                logger.warning(
                    "Specialist [%s] JSON parse failed, generating fallback evidence. Raw: %s",
                    self.domain, response[:200],
                )
                for i, (pol, label) in enumerate([
                    (1, "Supporting evidence from domain literature"),
                    (-1, "Methodological concerns and limitations"),
                    (1, "Corroborating findings from related studies"),
                ]):
                    if i >= config.max_evidence_per_query:
                        break
                    evidence = Evidence(
                        text=f"[{self.domain}] {label} regarding: {prop.text[:80]}",
                        evidence_type=EvidenceType.LITERATURE,
                        polarity=pol,
                        confidence=0.5,
                        quality=0.5,
                        relevance=0.7,
                    )
                    graph.add_evidence(
                        evidence,
                        proposition_id,
                        EdgeType.SUPPORTS if pol > 0 else EdgeType.ATTACKS,
                    )
                    evidence_list.append(evidence)
        
        self.log_action("gather_evidence", {
            "proposition_id": proposition_id,
            "evidence_count": len(evidence_list),
        })
        
        logger.info(
            f"Specialist [{self.domain}] gathered {len(evidence_list)} evidence for {proposition_id}"
        )
        
        return evidence_list
    
    def evaluate_chunk(
        self,
        graph: "CDAG",
        proposition_id: str,
        chunk_text: str,
    ) -> dict[str, Any]:
        """
        Evaluate a chunk for evidence.
        
        Uses Cite & Critique prompting.
        
        Args:
            graph: The C-DAG graph
            proposition_id: Target proposition
            chunk_text: Source chunk text
            
        Returns:
            Evaluation with claim, confidence, polarity
        """
        prop = graph.get_proposition(proposition_id)
        if not prop:
            return {"claim": "", "confidence": 0, "polarity": 0}
        
        prompt = f"""Evaluate this text as evidence for the proposition.

Proposition: {prop.text}

Source Text:
\"\"\"
{chunk_text[:1500]}
\"\"\"

Instructions:
1. Extract any claims relevant to the proposition
2. Cite specific quotes supporting your extraction
3. Critique the quality and relevance
4. Determine if it supports or attacks the proposition

Return JSON:
{{
    "claim": "Extracted claim text",
    "confidence": 0.75,
    "polarity": "support" or "attack" or "neutral",
    "quote": "Direct quote from source",
    "critique": "Limitations or concerns",
    "relevance": 0.8
}}"""
        
        response = self.generate(prompt, max_tokens=8192)
        
        try:
            data = extract_json_object(response)
            polarity_map = {"support": 1, "attack": -1, "neutral": 0}
            data["polarity_int"] = polarity_map.get(data.get("polarity", "neutral"), 0)
            return data
        except (ValueError, TypeError):
            return {
                "claim": response[:200],
                "confidence": 0.5,
                "polarity": "neutral",
                "polarity_int": 0,
                "relevance": 0.5,
            }
    
    def _create_evidence(
        self,
        evaluation: dict[str, Any],
        chunk: Any,
        retrieval_score: float,
    ) -> Evidence:
        """Create Evidence node from evaluation."""
        return Evidence(
            text=evaluation.get("claim", ""),
            evidence_type=EvidenceType.LITERATURE,
            polarity=evaluation.get("polarity_int", 0),
            confidence=evaluation.get("confidence", 0.5),
            relevance=evaluation.get("relevance", retrieval_score),
            quality=0.8,
            chunk_ids=[chunk.id] if hasattr(chunk, "id") else [],
            citations=[{
                "chunk_id": chunk.id if hasattr(chunk, "id") else "",
                "quote": evaluation.get("quote"),
            }],
        )
