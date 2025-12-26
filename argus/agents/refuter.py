"""
Refuter Agent for ARGUS.

Searches for counter-evidence and generates rebuttals
to challenge existing evidence and propositions.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any, TYPE_CHECKING

from pydantic import Field

from argus.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResponse,
)
from argus.cdag.nodes import Evidence, Rebuttal, EvidenceType
from argus.cdag.edges import EdgeType

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.cdag.graph import CDAG

logger = logging.getLogger(__name__)


class RefuterConfig(AgentConfig):
    """Configuration for Refuter agent."""
    
    name: str = "Refuter"
    role: AgentRole = AgentRole.REFUTER
    
    min_rebuttal_strength: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum strength to create rebuttal",
    )
    
    max_rebuttals_per_round: int = Field(
        default=3,
        ge=1,
        description="Maximum rebuttals per round",
    )


class Refuter(BaseAgent):
    """
    Refuter agent for finding counter-evidence.
    
    The Refuter:
    1. Analyzes existing evidence for weaknesses
    2. Searches for contradicting evidence
    3. Generates rebuttals to challenge claims
    4. Ensures balanced debate
    
    Example:
        >>> refuter = Refuter(llm=llm)
        >>> rebuttals = refuter.generate_rebuttals(graph, prop_id)
        >>> for r in rebuttals:
        ...     print(f"Rebuttal: {r.text[:50]}...")
    """
    
    def __init__(
        self,
        llm: "BaseLLM",
        config: Optional[RefuterConfig] = None,
    ):
        """
        Initialize Refuter.
        
        Args:
            llm: LLM instance
            config: Refuter configuration
        """
        super().__init__(llm, config or RefuterConfig())
    
    def get_system_prompt(self) -> str:
        """Get Refuter system prompt."""
        return """You are a Refuter in the ARGUS debate system.

Your role is to ensure rigorous evaluation by finding weaknesses and counter-evidence.

Your responsibilities:
1. Identify logical flaws in arguments
2. Find contradicting evidence
3. Challenge unsupported assumptions
4. Point out methodological limitations
5. Raise alternative explanations

Types of rebuttals:
- METHODOLOGICAL: Flaws in study design, sample size, etc.
- LOGICAL: Logical fallacies, non-sequiturs
- EMPIRICAL: Contradicting data or findings
- CONTEXTUAL: Limited applicability, confounding factors

Guidelines:
- Be rigorous but fair
- Focus on substantive critiques
- Provide specific reasonings
- Acknowledge when evidence is strong

Return rebuttals as JSON:
{
    "rebuttals": [
        {
            "target_id": "evidence_id",
            "type": "methodological/logical/empirical/contextual",
            "content": "The rebuttal text",
            "strength": 0.7,
            "reasoning": "Why this challenges the evidence"
        }
    ]
}"""
    
    def act(
        self,
        graph: "CDAG",
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Perform refuter action.
        
        Args:
            graph: The C-DAG graph
            context: Action context
            
        Returns:
            AgentResponse with rebuttals
        """
        action = context.get("action", "generate_rebuttals")
        prop_id = context.get("proposition_id")
        
        if not prop_id:
            return AgentResponse(
                success=False,
                error="proposition_id required",
            )
        
        if action == "generate_rebuttals":
            rebuttals = self.generate_rebuttals(graph, prop_id)
            
            return AgentResponse(
                success=True,
                content=f"Generated {len(rebuttals)} rebuttals",
                data={"rebuttal_ids": [r.id for r in rebuttals]},
            )
        
        elif action == "find_contradictions":
            contradictions = self.find_contradictions(graph, prop_id)
            
            return AgentResponse(
                success=True,
                content=f"Found {len(contradictions)} contradictions",
                data={"contradictions": contradictions},
            )
        
        else:
            return AgentResponse(
                success=False,
                error=f"Unknown action: {action}",
            )
    
    def generate_rebuttals(
        self,
        graph: "CDAG",
        proposition_id: str,
    ) -> list[Rebuttal]:
        """
        Generate rebuttals for evidence on a proposition.
        
        Args:
            graph: The C-DAG graph
            proposition_id: Proposition to examine
            
        Returns:
            List of Rebuttal nodes added to graph
        """
        prop = graph.get_proposition(proposition_id)
        if not prop:
            raise ValueError(f"Proposition {proposition_id} not found")
        
        # Get existing evidence
        supporting = graph.get_supporting_evidence(proposition_id)
        attacking = graph.get_attacking_evidence(proposition_id)
        
        # Analyze evidence for potential rebuttals
        evidence_summaries = []
        all_evidence = supporting + attacking
        
        for e in all_evidence:
            polarity = "supporting" if e.polarity > 0 else "attacking"
            evidence_summaries.append(f"[{e.id}] ({polarity}): {e.text[:200]}")
        
        if not evidence_summaries:
            return []
        
        prompt = f"""Analyze this evidence for a proposition and generate rebuttals.

Proposition: {prop.text}
Current Posterior: {prop.posterior:.2f}

Evidence:
{chr(10).join(evidence_summaries)}

For each piece of evidence that can be challenged, generate a rebuttal.
Focus on the strongest evidence first.

Return JSON:
{{
    "rebuttals": [
        {{
            "target_id": "evidence_id_here",
            "type": "methodological/logical/empirical/contextual",
            "content": "The specific rebuttal",
            "strength": 0.7,
            "reasoning": "Why this is a valid challenge"
        }}
    ]
}}"""
        
        response = self.generate(prompt)
        
        rebuttals: list[Rebuttal] = []
        config = self.config if isinstance(self.config, RefuterConfig) else RefuterConfig()
        
        try:
            data = json.loads(response)
            rebuttal_data = data.get("rebuttals", [])
            
            for rd in rebuttal_data[:config.max_rebuttals_per_round]:
                strength = rd.get("strength", 0.5)
                if strength < config.min_rebuttal_strength:
                    continue
                
                target_id = rd.get("target_id", "")
                
                # Verify target exists
                target = graph.get_node(target_id)
                if not target:
                    continue
                
                rebuttal = Rebuttal(
                    text=rd.get("content", ""),
                    target_id=target_id,
                    rebuttal_type=rd.get("type", "general"),
                    strength=strength,
                    confidence=strength,
                    metadata={"reasoning": rd.get("reasoning", "")},
                )
                
                graph.add_rebuttal(rebuttal, target_id)
                rebuttals.append(rebuttal)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse refuter response as JSON")
        
        self.log_action("generate_rebuttals", {
            "proposition_id": proposition_id,
            "rebuttal_count": len(rebuttals),
        })
        
        logger.info(f"Generated {len(rebuttals)} rebuttals for {proposition_id}")
        
        return rebuttals
    
    def find_contradictions(
        self,
        graph: "CDAG",
        proposition_id: str,
    ) -> list[dict[str, Any]]:
        """
        Find contradictions between evidence items.
        
        Args:
            graph: The C-DAG graph
            proposition_id: Proposition to examine
            
        Returns:
            List of contradiction pairs
        """
        prop = graph.get_proposition(proposition_id)
        if not prop:
            return []
        
        supporting = graph.get_supporting_evidence(proposition_id)
        attacking = graph.get_attacking_evidence(proposition_id)
        
        if not supporting or not attacking:
            return []
        
        # Prepare summaries for LLM
        support_texts = [f"[{e.id}]: {e.text[:150]}" for e in supporting[:5]]
        attack_texts = [f"[{e.id}]: {e.text[:150]}" for e in attacking[:5]]
        
        prompt = f"""Identify direct contradictions between supporting and attacking evidence.

Proposition: {prop.text}

Supporting Evidence:
{chr(10).join(support_texts)}

Attacking Evidence:
{chr(10).join(attack_texts)}

Find pairs where one directly contradicts the other.

Return JSON:
{{
    "contradictions": [
        {{
            "support_id": "id",
            "attack_id": "id",
            "description": "How they contradict",
            "severity": 0.8
        }}
    ]
}}"""
        
        response = self.generate(prompt)
        
        try:
            data = json.loads(response)
            return data.get("contradictions", [])
        except json.JSONDecodeError:
            return []
