"""
Jury Agent for ARGUS.

Aggregates evidence and renders verdicts on propositions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from pydantic import Field

from argus.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResponse,
)
from argus.cdag.propagation import compute_posterior, PropagationConfig

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.cdag.graph import CDAG

logger = logging.getLogger(__name__)


class JuryConfig(AgentConfig):
    """Configuration for Jury agent."""
    
    name: str = "Jury"
    role: AgentRole = AgentRole.JURY
    
    decision_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Threshold for confident decision",
    )
    
    use_llm_reasoning: bool = Field(
        default=True,
        description="Use LLM to explain verdict",
    )


@dataclass
class Verdict:
    """
    Verdict on a proposition.
    
    Attributes:
        proposition_id: Proposition evaluated
        posterior: Final posterior probability
        label: Verdict label (supported, rejected, undecided)
        confidence: Confidence in verdict
        top_support: Most important supporting evidence
        top_attack: Most important attacking evidence
        reasoning: LLM-generated reasoning
        created_at: Verdict timestamp
        metadata: Additional verdict data
    """
    proposition_id: str
    posterior: float
    label: str
    confidence: float = 0.5
    top_support: list[str] = field(default_factory=list)
    top_attack: list[str] = field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposition_id": self.proposition_id,
            "posterior": self.posterior,
            "label": self.label,
            "confidence": self.confidence,
            "top_support": self.top_support,
            "top_attack": self.top_attack,
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class Jury(BaseAgent):
    """
    Jury agent for aggregating evidence and rendering verdicts.
    
    The Jury:
    1. Aggregates evidence using Bayesian updating
    2. Computes final posteriors
    3. Renders verdicts with explanations
    4. Identifies key evidence on both sides
    
    Example:
        >>> jury = Jury(llm=llm)
        >>> verdict = jury.evaluate(graph, prop_id)
        >>> print(f"Verdict: {verdict.label} (posterior={verdict.posterior:.3f})")
    """
    
    def __init__(
        self,
        llm: "BaseLLM",
        config: Optional[JuryConfig] = None,
    ):
        """
        Initialize Jury.
        
        Args:
            llm: LLM instance
            config: Jury configuration
        """
        super().__init__(llm, config or JuryConfig())
    
    def get_system_prompt(self) -> str:
        """Get Jury system prompt."""
        return """You are a Jury member in the ARGUS debate system.

Your role is to render fair, well-reasoned verdicts on propositions.

Your responsibilities:
1. Weigh supporting and attacking evidence
2. Consider evidence quality and relevance
3. Render a clear verdict with reasoning
4. Identify the most influential evidence

Guidelines:
- Be objective and systematic
- Weight evidence by quality and reliability
- Consider the strength of rebuttals
- Explain your reasoning clearly
- Acknowledge uncertainty when present

Verdict categories:
- SUPPORTED: Strong evidence supports the proposition (posterior > 0.7)
- REJECTED: Strong evidence against the proposition (posterior < 0.3)
- UNDECIDED: Evidence is balanced or insufficient

Return verdicts as JSON:
{
    "verdict": "supported/rejected/undecided",
    "confidence": 0.85,
    "key_support": ["Most important supporting point"],
    "key_attack": ["Most important attacking point"],
    "reasoning": "Explanation of the verdict"
}"""
    
    def act(
        self,
        graph: "CDAG",
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Perform jury action.
        
        Args:
            graph: The C-DAG graph
            context: Action context
            
        Returns:
            AgentResponse with verdict
        """
        action = context.get("action", "evaluate")
        prop_id = context.get("proposition_id")
        
        if action == "evaluate":
            if not prop_id:
                return AgentResponse(
                    success=False,
                    error="proposition_id required for evaluate",
                )
            
            verdict = self.evaluate(graph, prop_id)
            
            return AgentResponse(
                success=True,
                content=verdict.reasoning,
                data={"verdict": verdict.to_dict()},
            )
        
        elif action == "evaluate_all":
            verdicts = self.evaluate_all(graph)
            
            return AgentResponse(
                success=True,
                content=f"Evaluated {len(verdicts)} propositions",
                data={"verdicts": [v.to_dict() for v in verdicts]},
            )
        
        else:
            return AgentResponse(
                success=False,
                error=f"Unknown action: {action}",
            )
    
    def evaluate(
        self,
        graph: "CDAG",
        proposition_id: str,
    ) -> Verdict:
        """
        Evaluate a proposition and render verdict.
        
        Args:
            graph: The C-DAG graph
            proposition_id: Proposition to evaluate
            
        Returns:
            Verdict for the proposition
        """
        prop = graph.get_proposition(proposition_id)
        if not prop:
            raise ValueError(f"Proposition {proposition_id} not found")
        
        # Compute posterior using Bayesian updating
        posterior = compute_posterior(graph, proposition_id)
        
        # Get evidence summary
        supporting = graph.get_supporting_evidence(proposition_id)
        attacking = graph.get_attacking_evidence(proposition_id)
        
        # Sort by effective weight
        supporting.sort(key=lambda e: e.effective_weight, reverse=True)
        attacking.sort(key=lambda e: e.effective_weight, reverse=True)
        
        top_support = [e.text[:100] for e in supporting[:3]]
        top_attack = [e.text[:100] for e in attacking[:3]]
        
        # Determine verdict label
        config = self.config if isinstance(self.config, JuryConfig) else JuryConfig()
        
        if posterior >= config.decision_threshold:
            label = "supported"
            confidence = posterior
        elif posterior <= (1 - config.decision_threshold):
            label = "rejected"
            confidence = 1 - posterior
        else:
            label = "undecided"
            confidence = 1 - 2 * abs(posterior - 0.5)
        
        # Generate reasoning with LLM
        reasoning = ""
        if config.use_llm_reasoning:
            reasoning = self._generate_reasoning(
                prop, posterior, supporting, attacking, label
            )
        
        verdict = Verdict(
            proposition_id=proposition_id,
            posterior=posterior,
            label=label,
            confidence=confidence,
            top_support=top_support,
            top_attack=top_attack,
            reasoning=reasoning,
            metadata={
                "prior": prop.prior,
                "support_count": len(supporting),
                "attack_count": len(attacking),
            },
        )
        
        self.log_action("evaluate", {
            "proposition_id": proposition_id,
            "posterior": posterior,
            "label": label,
        })
        
        logger.info(
            f"Verdict for {proposition_id}: {label} "
            f"(posterior={posterior:.3f})"
        )
        
        return verdict
    
    def evaluate_all(
        self,
        graph: "CDAG",
    ) -> list[Verdict]:
        """
        Evaluate all propositions in the graph.
        
        Args:
            graph: The C-DAG graph
            
        Returns:
            List of Verdict objects
        """
        propositions = graph.get_all_propositions()
        verdicts = []
        
        for prop in propositions:
            verdict = self.evaluate(graph, prop.id)
            verdicts.append(verdict)
        
        return verdicts
    
    def _generate_reasoning(
        self,
        prop: Any,
        posterior: float,
        supporting: list[Any],
        attacking: list[Any],
        label: str,
    ) -> str:
        """Generate LLM reasoning for verdict."""
        support_texts = [f"- {e.text[:150]}" for e in supporting[:5]]
        attack_texts = [f"- {e.text[:150]}" for e in attacking[:5]]
        
        prompt = f"""Explain the verdict for this proposition.

Proposition: {prop.text}
Prior: {prop.prior:.2f}
Posterior: {posterior:.3f}
Verdict: {label.upper()}

Supporting Evidence ({len(supporting)} items):
{chr(10).join(support_texts) if support_texts else "None"}

Attacking Evidence ({len(attacking)} items):
{chr(10).join(attack_texts) if attack_texts else "None"}

Provide a clear, 2-3 sentence explanation of why the verdict was reached."""
        
        return self.generate(prompt)
    
    def compute_disagreement(
        self,
        graph: "CDAG",
        proposition_id: str,
    ) -> float:
        """
        Compute disagreement index for a proposition.
        
        High disagreement = roughly equal support and attack.
        
        Args:
            graph: The C-DAG graph
            proposition_id: Proposition to analyze
            
        Returns:
            Disagreement index (0-1)
        """
        support_score = graph.compute_support_score(proposition_id)
        attack_score = graph.compute_attack_score(proposition_id)
        
        total = support_score + attack_score
        if total == 0:
            return 0.0
        
        # Maximum disagreement when support == attack
        return 4 * (support_score / total) * (attack_score / total)
