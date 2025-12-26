"""
Research Debate Chain (RDC) Orchestrator for ARGUS.

Coordinates the full debate workflow including:
    - Agenda creation
    - Agent orchestration
    - Round management
    - Verdict rendering
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from argus.core.llm import get_llm
from argus.cdag.graph import CDAG
from argus.cdag.nodes import Proposition
from argus.cdag.propagation import compute_all_posteriors
from argus.agents.moderator import Moderator, DebateAgenda
from argus.agents.specialist import Specialist
from argus.agents.refuter import Refuter
from argus.agents.jury import Jury, Verdict
from argus.provenance.ledger import ProvenanceLedger, EventType

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    """
    Result of a complete debate.
    
    Attributes:
        proposition_id: Proposition debated
        verdict: Final verdict
        num_rounds: Number of rounds completed
        num_evidence: Total evidence count
        num_rebuttals: Total rebuttal count
        duration_seconds: Total duration
        graph: Final C-DAG state
    """
    proposition_id: str
    verdict: Verdict
    num_rounds: int = 0
    num_evidence: int = 0
    num_rebuttals: int = 0
    duration_seconds: float = 0.0
    graph: Optional[CDAG] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposition_id": self.proposition_id,
            "verdict": self.verdict.to_dict(),
            "num_rounds": self.num_rounds,
            "num_evidence": self.num_evidence,
            "num_rebuttals": self.num_rebuttals,
            "duration_seconds": self.duration_seconds,
        }


class RDCOrchestrator:
    """
    Research Debate Chain orchestrator.
    
    Coordinates multi-agent debate for evaluating propositions.
    Implements Algorithm 4.1: Evidence-Directed Debate Orchestration (EDDO).
    
    Example:
        >>> orchestrator = RDCOrchestrator()
        >>> result = orchestrator.debate(
        ...     "Drug X reduces biomarker Y by >20%",
        ...     retriever=retriever,
        ... )
        >>> print(f"Verdict: {result.verdict.label}")
    """
    
    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        ledger: Optional[ProvenanceLedger] = None,
        max_rounds: int = 5,
    ):
        """
        Initialize orchestrator.
        
        Args:
            llm: LLM instance (creates default if None)
            ledger: Provenance ledger
            max_rounds: Maximum debate rounds
        """
        self.llm = llm or get_llm()
        self.ledger = ledger or ProvenanceLedger()
        self.max_rounds = max_rounds
        
        # Initialize agents
        self.moderator = Moderator(self.llm)
        self.specialist = Specialist(self.llm)
        self.refuter = Refuter(self.llm)
        self.jury = Jury(self.llm)
        
        # Session state
        self._graph: Optional[CDAG] = None
        self._agenda: Optional[DebateAgenda] = None
    
    def debate(
        self,
        proposition_text: str,
        prior: float = 0.5,
        retriever: Optional["HybridRetriever"] = None,
        domain: str = "general",
    ) -> DebateResult:
        """
        Run a complete debate on a proposition.
        
        Args:
            proposition_text: The proposition to evaluate
            prior: Prior probability
            retriever: Hybrid retriever for evidence
            domain: Domain of expertise
            
        Returns:
            DebateResult with verdict and statistics
        """
        start_time = datetime.utcnow()
        
        # Record session start
        self.ledger.record(
            EventType.SESSION_START,
            attributes={"proposition": proposition_text[:100]},
        )
        
        # Initialize graph
        self._graph = CDAG(name="debate")
        
        # Create proposition
        prop = Proposition(
            text=proposition_text,
            prior=prior,
        )
        self._graph.add_proposition(prop)
        
        self.ledger.record(
            EventType.PROPOSITION_ADDED,
            entity_id=prop.id,
            attributes={"text": proposition_text[:100], "prior": prior},
        )
        
        # Create agenda
        self._agenda = self.moderator.create_agenda(self._graph, prop.id)
        
        # Update specialist domain
        self.specialist.config.domain = domain
        
        # Run debate rounds
        round_num = 0
        while round_num < self.max_rounds:
            round_num += 1
            logger.info(f"Starting round {round_num}")
            
            # Specialist gathers evidence
            if retriever:
                evidence = self.specialist.gather_evidence(
                    self._graph,
                    prop.id,
                    retriever._index if hasattr(retriever, '_index') else None,
                    retriever._embedding_gen if hasattr(retriever, '_embedding_gen') else None,
                )
                
                for e in evidence:
                    self.ledger.record(
                        EventType.EVIDENCE_ADDED,
                        agent_id=self.specialist.name,
                        entity_id=e.id,
                        attributes={"polarity": e.polarity},
                    )
            
            # Refuter generates rebuttals
            rebuttals = self.refuter.generate_rebuttals(self._graph, prop.id)
            
            for r in rebuttals:
                self.ledger.record(
                    EventType.REBUTTAL_ADDED,
                    agent_id=self.refuter.name,
                    entity_id=r.id,
                    attributes={"target": r.target_id},
                )
            
            # Update posteriors
            compute_all_posteriors(self._graph)
            
            # Check stopping criteria
            should_stop, reason = self.moderator.should_stop(self._graph)
            if should_stop:
                logger.info(f"Stopping: {reason}")
                break
        
        # Jury renders verdict
        verdict = self.jury.evaluate(self._graph, prop.id)
        
        self.ledger.record(
            EventType.VERDICT_RENDERED,
            agent_id=self.jury.name,
            entity_id=prop.id,
            attributes={"label": verdict.label, "posterior": verdict.posterior},
        )
        
        # Calculate duration
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        self.ledger.record(EventType.SESSION_END)
        
        return DebateResult(
            proposition_id=prop.id,
            verdict=verdict,
            num_rounds=round_num,
            num_evidence=self._graph.num_evidence,
            num_rebuttals=len(self._graph._rebuttals),
            duration_seconds=duration,
            graph=self._graph,
        )
    
    def quick_evaluate(
        self,
        proposition_text: str,
        prior: float = 0.5,
    ) -> Verdict:
        """
        Quick evaluation without full debate.
        
        Uses single LLM call for rapid assessment.
        
        Args:
            proposition_text: Proposition to evaluate
            prior: Prior probability
            
        Returns:
            Verdict object
        """
        prompt = f"""Evaluate this proposition and provide a verdict.

Proposition: {proposition_text}
Prior probability: {prior:.2f}

Consider:
1. General knowledge supporting the proposition
2. Potential counter-evidence
3. Overall plausibility

Return JSON:
{{
    "verdict": "supported/rejected/undecided",
    "posterior": 0.75,
    "reasoning": "Brief explanation"
}}"""
        
        import json
        response = self.llm.generate(prompt, temperature=0.3)
        
        try:
            data = json.loads(response.content)
            
            return Verdict(
                proposition_id="quick_eval",
                posterior=float(data.get("posterior", 0.5)),
                label=data.get("verdict", "undecided"),
                reasoning=data.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            return Verdict(
                proposition_id="quick_eval",
                posterior=0.5,
                label="undecided",
                reasoning=response.content,
            )
    
    @property
    def graph(self) -> Optional[CDAG]:
        """Get current debate graph."""
        return self._graph
    
    @property
    def agenda(self) -> Optional[DebateAgenda]:
        """Get current agenda."""
        return self._agenda
