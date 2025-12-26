"""
Moderator Agent for ARGUS.

The Moderator coordinates the debate, managing:
    - Agenda creation
    - Agent role assignment
    - Halting criteria
    - Round progression
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from argus.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResponse,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.cdag.graph import CDAG

logger = logging.getLogger(__name__)


class ModeratorConfig(AgentConfig):
    """Configuration for Moderator agent."""
    
    name: str = "Moderator"
    role: AgentRole = AgentRole.MODERATOR
    
    max_rounds: int = Field(
        default=5,
        ge=1,
        description="Maximum debate rounds",
    )
    
    min_evidence_per_round: int = Field(
        default=2,
        ge=1,
        description="Minimum evidence per round",
    )
    
    convergence_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Posterior change threshold for stopping",
    )


@dataclass
class DebateAgenda:
    """
    Agenda for a debate session.
    
    Attributes:
        proposition_id: ID of proposition being debated
        proposition_text: Text of the proposition
        current_round: Current round number
        max_rounds: Maximum rounds
        objectives: Round objectives
        assigned_agents: Agents assigned to roles
        stopping_criteria: Criteria for ending debate
        created_at: Creation timestamp
    """
    proposition_id: str
    proposition_text: str
    current_round: int = 0
    max_rounds: int = 5
    objectives: list[str] = field(default_factory=list)
    assigned_agents: dict[str, str] = field(default_factory=dict)
    stopping_criteria: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposition_id": self.proposition_id,
            "proposition_text": self.proposition_text,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "objectives": self.objectives,
            "assigned_agents": self.assigned_agents,
            "stopping_criteria": self.stopping_criteria,
            "created_at": self.created_at.isoformat(),
        }


class Moderator(BaseAgent):
    """
    Moderator agent for coordinating debates.
    
    The Moderator:
    1. Creates debate agendas for propositions
    2. Assigns specialist agents to relevant domains
    3. Manages round progression
    4. Evaluates stopping criteria
    5. Calls for jury verdict when appropriate
    
    Example:
        >>> moderator = Moderator(llm=llm)
        >>> agenda = moderator.create_agenda(graph, prop_id)
        >>> while not moderator.should_stop(graph, agenda):
        ...     moderator.advance_round(graph, agenda, specialists)
    """
    
    def __init__(
        self,
        llm: "BaseLLM",
        config: Optional[ModeratorConfig] = None,
    ):
        """
        Initialize Moderator.
        
        Args:
            llm: LLM instance
            config: Moderator configuration
        """
        super().__init__(llm, config or ModeratorConfig())
        self._agenda: Optional[DebateAgenda] = None
    
    def get_system_prompt(self) -> str:
        """Get Moderator system prompt."""
        return """You are a Debate Moderator in the ARGUS system.

Your responsibilities:
1. Create structured agendas for evaluating propositions
2. Identify key aspects that require evidence
3. Assign appropriate expertise domains to specialists
4. Monitor debate progress and determine when conclusion is reached
5. Ensure fair consideration of both supporting and opposing evidence

Guidelines:
- Be objective and balanced
- Focus on evidence quality and relevance
- Identify when additional evidence is needed
- Recognize when the debate has reached a stable conclusion
- Structure your responses in clear, actionable formats

Format outputs as JSON when creating agendas or assignments."""
    
    def act(
        self,
        graph: "CDAG",
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Perform moderator action.
        
        Args:
            graph: The C-DAG graph
            context: Action context
            
        Returns:
            AgentResponse with action results
        """
        action = context.get("action", "create_agenda")
        
        if action == "create_agenda":
            prop_id = context.get("proposition_id")
            if not prop_id:
                return AgentResponse(
                    success=False,
                    error="proposition_id required for create_agenda",
                )
            
            agenda = self.create_agenda(graph, prop_id)
            return AgentResponse(
                success=True,
                content=f"Created agenda for {prop_id}",
                data={"agenda": agenda.to_dict()},
            )
        
        elif action == "evaluate_round":
            return self._evaluate_round(graph, context)
        
        elif action == "should_stop":
            stop, reason = self.should_stop(graph)
            return AgentResponse(
                success=True,
                content=reason,
                data={"should_stop": stop, "reason": reason},
            )
        
        else:
            return AgentResponse(
                success=False,
                error=f"Unknown action: {action}",
            )
    
    def create_agenda(
        self,
        graph: "CDAG",
        proposition_id: str,
    ) -> DebateAgenda:
        """
        Create a debate agenda for a proposition.
        
        Args:
            graph: The C-DAG graph
            proposition_id: Proposition to debate
            
        Returns:
            DebateAgenda for the debate
        """
        prop = graph.get_proposition(proposition_id)
        if not prop:
            raise ValueError(f"Proposition {proposition_id} not found")
        
        # Generate objectives using LLM
        prompt = f"""Create a debate agenda for evaluating this proposition:

Proposition: {prop.text}
Prior probability: {prop.prior:.2f}

Identify:
1. 3-5 key aspects that require evidence
2. Relevant domains of expertise needed
3. Potential sources of supporting evidence
4. Potential sources of contradicting evidence

Return as JSON with format:
{{
    "objectives": ["objective1", "objective2", ...],
    "domains": ["domain1", "domain2", ...],
    "support_sources": ["source1", ...],
    "attack_sources": ["source1", ...]
}}"""
        
        response = self.generate(prompt)
        
        # Parse response
        try:
            data = json.loads(response)
            objectives = data.get("objectives", [])
            domains = data.get("domains", [])
        except json.JSONDecodeError:
            objectives = ["Gather supporting evidence", "Gather contradicting evidence", "Evaluate evidence quality"]
            domains = ["general"]
        
        # Create agenda
        config = self.config if isinstance(self.config, ModeratorConfig) else ModeratorConfig()
        
        agenda = DebateAgenda(
            proposition_id=proposition_id,
            proposition_text=prop.text,
            max_rounds=config.max_rounds,
            objectives=objectives,
            assigned_agents={d: f"specialist_{d}" for d in domains},
            stopping_criteria={
                "max_rounds": config.max_rounds,
                "convergence_threshold": config.convergence_threshold,
                "min_evidence": config.min_evidence_per_round,
            },
        )
        
        self._agenda = agenda
        
        self.log_action("create_agenda", {
            "proposition_id": proposition_id,
            "objectives": objectives,
            "domains": domains,
        })
        
        logger.info(
            f"Created agenda for {proposition_id} with {len(objectives)} objectives"
        )
        
        return agenda
    
    def advance_round(
        self,
        graph: "CDAG",
        agenda: Optional[DebateAgenda] = None,
    ) -> dict[str, Any]:
        """
        Advance to the next debate round.
        
        Args:
            graph: The C-DAG graph
            agenda: Debate agenda (uses stored if not provided)
            
        Returns:
            Round summary
        """
        agenda = agenda or self._agenda
        if not agenda:
            raise ValueError("No agenda set. Call create_agenda first.")
        
        agenda.current_round += 1
        
        # Get current state
        prop = graph.get_proposition(agenda.proposition_id)
        support_count = len(graph.get_supporting_evidence(agenda.proposition_id))
        attack_count = len(graph.get_attacking_evidence(agenda.proposition_id))
        
        summary = {
            "round": agenda.current_round,
            "proposition_id": agenda.proposition_id,
            "posterior": prop.posterior if prop else 0.5,
            "support_count": support_count,
            "attack_count": attack_count,
        }
        
        self.log_action("advance_round", summary)
        
        return summary
    
    def should_stop(
        self,
        graph: "CDAG",
        agenda: Optional[DebateAgenda] = None,
    ) -> tuple[bool, str]:
        """
        Check if debate should stop.
        
        Stopping criteria:
        1. Max rounds reached
        2. High confidence (posterior > 0.95 or < 0.05)
        3. Convergence (posterior change < threshold)
        
        Args:
            graph: The C-DAG graph
            agenda: Debate agenda
            
        Returns:
            (should_stop, reason)
        """
        agenda = agenda or self._agenda
        if not agenda:
            return True, "No agenda"
        
        # Check max rounds
        if agenda.current_round >= agenda.max_rounds:
            return True, f"Reached maximum rounds ({agenda.max_rounds})"
        
        # Check proposition posterior
        prop = graph.get_proposition(agenda.proposition_id)
        if prop:
            if prop.posterior > 0.95:
                return True, f"High confidence support ({prop.posterior:.3f})"
            if prop.posterior < 0.05:
                return True, f"High confidence rejection ({prop.posterior:.3f})"
        
        return False, "Continuing"
    
    def _evaluate_round(
        self,
        graph: "CDAG",
        context: dict[str, Any],
    ) -> AgentResponse:
        """Evaluate the current round."""
        agenda = self._agenda
        if not agenda:
            return AgentResponse(
                success=False,
                error="No active agenda",
            )
        
        prop = graph.get_proposition(agenda.proposition_id)
        support = graph.get_supporting_evidence(agenda.proposition_id)
        attack = graph.get_attacking_evidence(agenda.proposition_id)
        
        # Generate evaluation
        prompt = f"""Evaluate the current state of the debate:

Proposition: {agenda.proposition_text}
Current Round: {agenda.current_round} / {agenda.max_rounds}
Posterior: {prop.posterior if prop else 0.5:.3f}
Supporting Evidence: {len(support)} items
Attacking Evidence: {len(attack)} items

Provide:
1. Assessment of evidence balance
2. Recommendation for next steps
3. Any concerns about evidence quality

Return as JSON:
{{
    "assessment": "...",
    "recommendation": "...",
    "concerns": ["...", ...]
}}"""
        
        response = self.generate(prompt)
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = {
                "assessment": response,
                "recommendation": "Continue gathering evidence",
                "concerns": [],
            }
        
        return AgentResponse(
            success=True,
            content=data.get("assessment", ""),
            data=data,
        )
