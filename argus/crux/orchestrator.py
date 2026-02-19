"""
CRUX Orchestrator for ARGUS.

The CRUXOrchestrator wraps the existing RDCOrchestrator and adds
CRUX protocol capabilities:
    - Epistemic Agent Cards for all agents
    - Claim Bundles with uncertainty distributions
    - Dialectical Routing for adversarial engagement
    - Belief Reconciliation Protocol for contradictions
    - Credibility Ledger for statistical trust
    - Epistemic Dead Reckoning for reconnection
    - Challenger Auction for best challenger selection

Example:
    >>> from argus.crux import CRUXOrchestrator, CredibilityLedger
    >>> 
    >>> # Existing ARGUS setup
    >>> orchestrator = RDCOrchestrator(llm=llm, max_rounds=5)
    >>> 
    >>> # Wrap with CRUX layer
    >>> crux = CRUXOrchestrator(
    ...     base=orchestrator,
    ...     contradiction_threshold=0.20,
    ... )
    >>> 
    >>> # Run a CRUX-enabled debate
    >>> result = crux.debate(
    ...     "Treatment X reduces symptoms by more than 20%",
    ...     prior=0.5,
    ... )
    >>> 
    >>> print(result.verdict.label)
    >>> print(result.reconciled_cb.posterior)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from argus.crux.models import (
    generate_crux_id,
    compute_crux_hash,
    CRUXConfig,
    ChallengerBid,
    BRPResolution,
    Polarity,
    ConfidenceDistribution,
    BetaDistribution,
)
from argus.crux.agent_card import (
    EpistemicAgentCard,
    AgentCalibration,
    AgentCapabilities,
    EACRegistry,
)
from argus.crux.claim_bundle import ClaimBundle, ClaimBundleFactory
from argus.crux.routing import DialecticalRouter, DialecticalFitnessScore
from argus.crux.brp import BeliefReconciliationProtocol, BRPSession, ReconciliationResult
from argus.crux.ledger import CredibilityLedger
from argus.crux.edr import EpistemicDeadReckoning, EDRCheckpoint
from argus.crux.auction import ChallengerAuction, AuctionResult

if TYPE_CHECKING:
    from argus.orchestrator import RDCOrchestrator, DebateResult
    from argus.agents.jury import Verdict
    from argus.core.llm.base import BaseLLM
    from argus.retrieval.hybrid import HybridRetriever
    from argus.cdag.graph import CDAG
    from argus.cdag.nodes import Evidence

logger = logging.getLogger(__name__)


class CRUXSessionState(str, Enum):
    """States of a CRUX session."""
    INITIALIZING = "initializing"
    DISCOVERY = "discovery"
    DEBATE = "debate"
    RECONCILIATION = "reconciliation"
    VERDICT = "verdict"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CRUXSessionStats:
    """
    Statistics for a CRUX session.
    
    Attributes:
        num_claim_bundles: Total Claim Bundles emitted
        num_challenges: Number of challenges
        num_brp_sessions: Number of BRP reconciliations
        num_auctions: Number of Challenger Auctions
        num_unchallenged: Number of unchallenged claims
        avg_dfs: Average Dialectical Fitness Score
        total_credibility_updates: Credibility ledger updates
    """
    num_claim_bundles: int = 0
    num_challenges: int = 0
    num_brp_sessions: int = 0
    num_auctions: int = 0
    num_unchallenged: int = 0
    avg_dfs: float = 0.0
    total_credibility_updates: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_claim_bundles": self.num_claim_bundles,
            "num_challenges": self.num_challenges,
            "num_brp_sessions": self.num_brp_sessions,
            "num_auctions": self.num_auctions,
            "num_unchallenged": self.num_unchallenged,
            "avg_dfs": self.avg_dfs,
            "total_credibility_updates": self.total_credibility_updates,
        }


@dataclass
class CRUXSession:
    """
    A CRUX debate session.
    
    Tracks all CRUX-specific data for a debate.
    
    Attributes:
        session_id: Unique session identifier
        proposition_text: Proposition being debated
        prior: Prior probability
        state: Current session state
        claim_bundles: All Claim Bundles emitted
        auctions: All Challenger Auctions
        brp_sessions: All BRP reconciliations
        checkpoints: EDR checkpoints
        stats: Session statistics
        started_at: Session start time
        completed_at: Session completion time
    """
    session_id: str = field(default_factory=lambda: generate_crux_id("session"))
    proposition_text: str = ""
    prior: float = 0.5
    state: CRUXSessionState = CRUXSessionState.INITIALIZING
    
    # CRUX entities
    claim_bundles: list[ClaimBundle] = field(default_factory=list)
    auctions: list[AuctionResult] = field(default_factory=list)
    brp_sessions: list[BRPSession] = field(default_factory=list)
    checkpoints: list[EDRCheckpoint] = field(default_factory=list)
    
    # Tracking
    stats: CRUXSessionStats = field(default_factory=CRUXSessionStats)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Mappings
    cb_by_proposition: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    cb_by_agent: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    
    @property
    def duration_seconds(self) -> float:
        """Session duration in seconds."""
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
    
    def add_claim_bundle(self, cb: ClaimBundle) -> None:
        """Add a Claim Bundle to the session."""
        self.claim_bundles.append(cb)
        self.stats.num_claim_bundles += 1
        
        if cb.proposition_id:
            self.cb_by_proposition[cb.proposition_id].append(cb.cb_id)
        if cb.issuer_agent:
            self.cb_by_agent[cb.issuer_agent].append(cb.cb_id)
    
    def add_auction(self, result: AuctionResult) -> None:
        """Add an auction result to the session."""
        self.auctions.append(result)
        self.stats.num_auctions += 1
        
        if result.unchallenged:
            self.stats.num_unchallenged += 1
        else:
            self.stats.num_challenges += 1
    
    def add_brp_session(self, brp: BRPSession) -> None:
        """Add a BRP session to the session."""
        self.brp_sessions.append(brp)
        self.stats.num_brp_sessions += 1
    
    def get_claim_bundles_for_proposition(self, prop_id: str) -> list[ClaimBundle]:
        """Get all Claim Bundles for a proposition."""
        cb_ids = self.cb_by_proposition.get(prop_id, [])
        return [cb for cb in self.claim_bundles if cb.cb_id in cb_ids]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "proposition_text": self.proposition_text[:100],
            "prior": self.prior,
            "state": self.state.value,
            "stats": self.stats.to_dict(),
            "num_claim_bundles": len(self.claim_bundles),
            "num_auctions": len(self.auctions),
            "num_brp_sessions": len(self.brp_sessions),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class CRUXDebateResult:
    """
    Result of a CRUX-enabled debate.
    
    Extends DebateResult with CRUX-specific data.
    
    Attributes:
        verdict: Final verdict
        reconciled_cb: Final reconciled Claim Bundle
        session: CRUX session data
        proposition_id: Proposition ID
        final_posterior: Final posterior probability
        final_distribution: Final confidence distribution
        challenger_auction: Final auction result (if any)
        num_rounds: Number of debate rounds
        base_result: Underlying RDC result
    """
    verdict: "Verdict"
    reconciled_cb: Optional[ClaimBundle] = None
    session: Optional[CRUXSession] = None
    proposition_id: str = ""
    final_posterior: float = 0.5
    final_distribution: Optional[ConfidenceDistribution] = None
    challenger_auction: Optional[AuctionResult] = None
    num_rounds: int = 0
    base_result: Optional["DebateResult"] = None
    
    @property
    def was_challenged(self) -> bool:
        """Check if the final claim was challenged."""
        return self.challenger_auction is not None and not self.challenger_auction.unchallenged
    
    @property
    def credibility_impact(self) -> dict[str, float]:
        """Get credibility changes for each agent."""
        if not self.session:
            return {}
        # This would be computed from ledger updates
        return {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "verdict": self.verdict.to_dict() if hasattr(self.verdict, 'to_dict') else str(self.verdict),
            "reconciled_cb": self.reconciled_cb.to_dict() if self.reconciled_cb else None,
            "session": self.session.to_dict() if self.session else None,
            "proposition_id": self.proposition_id,
            "final_posterior": self.final_posterior,
            "final_distribution": (
                self.final_distribution.to_dict() 
                if self.final_distribution else None
            ),
            "challenger_auction": (
                self.challenger_auction.to_dict() 
                if self.challenger_auction else None
            ),
            "num_rounds": self.num_rounds,
            "was_challenged": self.was_challenged,
        }


class CRUXOrchestrator:
    """
    CRUX-enabled orchestrator for ARGUS debates.
    
    Wraps the base RDCOrchestrator and adds all seven CRUX primitives:
    1. Epistemic Agent Cards (EAC)
    2. Claim Bundles (CB)
    3. Dialectical Routing (DR)
    4. Belief Reconciliation Protocol (BRP)
    5. Credibility Ledger (CL)
    6. Epistemic Dead Reckoning (EDR)
    7. Challenger Auction (CA)
    
    Example:
        >>> crux = CRUXOrchestrator(
        ...     base=orchestrator,
        ...     contradiction_threshold=0.20,
        ... )
        >>> 
        >>> result = crux.debate(
        ...     "Treatment X reduces symptoms by more than 20%",
        ...     prior=0.5,
        ... )
        >>> print(result.reconciled_cb.posterior)
    """
    
    def __init__(
        self,
        base: Optional["RDCOrchestrator"] = None,
        llm: Optional["BaseLLM"] = None,
        ledger: Optional[CredibilityLedger] = None,
        config: Optional[CRUXConfig] = None,
        contradiction_threshold: float = 0.20,
        auction_timeout_seconds: int = 30,
        enable_edr: bool = True,
    ):
        """
        Initialize CRUX orchestrator.
        
        Args:
            base: Base RDCOrchestrator to wrap
            llm: LLM instance (used if base not provided)
            ledger: Credibility ledger
            config: CRUX configuration
            contradiction_threshold: Threshold for BRP trigger (θ)
            auction_timeout_seconds: Challenger Auction timeout
            enable_edr: Enable Epistemic Dead Reckoning
        """
        # Build config
        if config:
            self.config = config
        else:
            self.config = CRUXConfig(
                contradiction_threshold=contradiction_threshold,
                auction_timeout_seconds=auction_timeout_seconds,
                enable_edr=enable_edr,
            )
        
        # Initialize or wrap base orchestrator
        if base:
            self.base = base
        else:
            # Create base orchestrator
            from argus.orchestrator import RDCOrchestrator
            from argus.core.llm import get_llm
            
            self.base = RDCOrchestrator(
                llm=llm or get_llm(),
            )
        
        # Initialize CRUX components
        self.ledger = ledger or CredibilityLedger()
        self.registry = EACRegistry()
        self.router = DialecticalRouter(registry=self.registry, config=self.config)
        self.brp = BeliefReconciliationProtocol(ledger=self.ledger, config=self.config)
        self.edr = EpistemicDeadReckoning(ledger=self.ledger, brp=self.brp, config=self.config)
        self.auction = ChallengerAuction(
            router=self.router,
            registry=self.registry,
            config=self.config,
        )
        self.cb_factory = ClaimBundleFactory(config=self.config)
        
        # Session tracking
        self._current_session: Optional[CRUXSession] = None
        self._session_history: list[CRUXSession] = []
        
        # Agent card mapping
        self._agent_cards: dict[str, EpistemicAgentCard] = {}
        
        # Initialize agent cards for base agents
        self._setup_agent_cards()
        
        # Callbacks
        self._on_claim_bundle: list[Callable[[ClaimBundle], None]] = []
        self._on_brp_triggered: list[Callable[[BRPSession], None]] = []
        self._on_auction_complete: list[Callable[[AuctionResult], None]] = []
    
    def _setup_agent_cards(self) -> None:
        """Set up Epistemic Agent Cards for base agents."""
        # Specialist agent card
        specialist_card = EpistemicAgentCard(
            agent_id="argus-specialist-001",
            agent_type="Specialist",
            belief_domains=["general", "research", "methodology"],
            calibration=AgentCalibration(
                brier_score=0.15,
                ece=0.05,
                credibility_rating=0.85,
                sample_size=0,
            ),
            capabilities=AgentCapabilities(
                emit_claims=True,
                challenge_claims=False,
                render_verdicts=False,
            ),
        )
        self.registry.register(specialist_card)
        self._agent_cards["specialist"] = specialist_card
        
        # Refuter agent card
        refuter_card = EpistemicAgentCard(
            agent_id="argus-refuter-001",
            agent_type="Refuter",
            belief_domains=["general", "methodology", "statistical"],
            calibration=AgentCalibration(
                brier_score=0.12,
                ece=0.04,
                credibility_rating=0.88,
                sample_size=0,
            ),
            capabilities=AgentCapabilities(
                emit_claims=True,
                challenge_claims=True,
                render_verdicts=False,
            ),
        )
        self.registry.register(refuter_card)
        self._agent_cards["refuter"] = refuter_card
        
        # Jury agent card
        jury_card = EpistemicAgentCard(
            agent_id="argus-jury-001",
            agent_type="Jury",
            belief_domains=["general"],
            calibration=AgentCalibration(
                brier_score=0.10,
                ece=0.03,
                credibility_rating=0.92,
                sample_size=0,
            ),
            capabilities=AgentCapabilities(
                emit_claims=False,
                challenge_claims=False,
                render_verdicts=True,
            ),
        )
        self.registry.register(jury_card)
        self._agent_cards["jury"] = jury_card
    
    def debate(
        self,
        proposition_text: str,
        prior: float = 0.5,
        retriever: Optional["HybridRetriever"] = None,
        domain: str = "general",
    ) -> CRUXDebateResult:
        """
        Run a CRUX-enabled debate on a proposition.
        
        This is the main entry point. It:
        1. Initializes a CRUX session
        2. Runs the base debate
        3. Converts evidence to Claim Bundles
        4. Runs Challenger Auctions
        5. Triggers BRP for contradictions
        6. Updates Credibility Ledger
        7. Returns enhanced result
        
        Args:
            proposition_text: The proposition to evaluate
            prior: Prior probability
            retriever: Hybrid retriever for evidence
            domain: Domain of expertise
            
        Returns:
            CRUXDebateResult with full CRUX data
        """
        # Initialize session
        session = CRUXSession(
            proposition_text=proposition_text,
            prior=prior,
            state=CRUXSessionState.INITIALIZING,
        )
        self._current_session = session
        
        try:
            # Phase 1: Discovery
            session.state = CRUXSessionState.DISCOVERY
            
            # Update agent domains based on proposition
            self._update_agent_domains(proposition_text, domain)
            
            # Phase 2: Run base debate
            session.state = CRUXSessionState.DEBATE
            
            base_result = self.base.debate(
                proposition_text=proposition_text,
                prior=prior,
                retriever=retriever,
                domain=domain,
            )
            
            # Phase 3: Convert to Claim Bundles
            claim_bundles = self._convert_to_claim_bundles(
                base_result,
                proposition_text,
            )
            
            for cb in claim_bundles:
                session.add_claim_bundle(cb)
                self.edr.register_bundle(cb)
                
                # Trigger callbacks
                for callback in self._on_claim_bundle:
                    try:
                        callback(cb)
                    except Exception as e:
                        logger.warning(f"Claim bundle callback error: {e}")
            
            # Phase 4: Run Challenger Auctions for open claims
            for cb in claim_bundles:
                if cb.challenge_open:
                    auction_result = self._run_auction(cb)
                    if auction_result:
                        session.add_auction(auction_result)
            
            # Phase 5: Check for contradictions and run BRP
            session.state = CRUXSessionState.RECONCILIATION
            
            reconciled_cb = self._reconcile_claims(
                claim_bundles,
                base_result.proposition_id,
            )
            
            # Phase 6: Update Credibility Ledger
            self._update_credibility(
                claim_bundles,
                base_result.verdict,
            )
            
            # Phase 7: Prepare result
            session.state = CRUXSessionState.VERDICT
            
            final_posterior = reconciled_cb.posterior if reconciled_cb else prior
            final_distribution = (
                reconciled_cb.confidence_distribution 
                if reconciled_cb else None
            )
            
            # Create checkpoint for EDR
            if self.config.enable_edr:
                checkpoint = self.edr.checkpoint(
                    "session",
                    claim_bundles,
                )
                session.checkpoints.append(checkpoint)
            
            session.state = CRUXSessionState.COMPLETE
            session.completed_at = datetime.utcnow()
            
            result = CRUXDebateResult(
                verdict=base_result.verdict,
                reconciled_cb=reconciled_cb,
                session=session,
                proposition_id=base_result.proposition_id,
                final_posterior=final_posterior,
                final_distribution=final_distribution,
                challenger_auction=session.auctions[-1] if session.auctions else None,
                num_rounds=base_result.num_rounds,
                base_result=base_result,
            )
            
            # Store session
            self._session_history.append(session)
            self._current_session = None
            
            logger.info(
                f"CRUX debate complete: "
                f"posterior={final_posterior:.3f}, "
                f"bundles={len(claim_bundles)}, "
                f"auctions={len(session.auctions)}, "
                f"brp_sessions={len(session.brp_sessions)}"
            )
            
            return result
            
        except Exception as e:
            session.state = CRUXSessionState.ERROR
            session.completed_at = datetime.utcnow()
            logger.error(f"CRUX debate error: {e}")
            raise
    
    def _update_agent_domains(
        self,
        proposition_text: str,
        domain: str,
    ) -> None:
        """Update agent domain beliefs based on proposition."""
        from argus.crux.routing import DomainMatcher
        
        detected_domains = DomainMatcher.extract_domains_from_text(proposition_text)
        
        if domain and domain != "general":
            detected_domains.append(domain)
        
        # Update specialist card with relevant domains
        specialist_card = self._agent_cards.get("specialist")
        if specialist_card:
            specialist_card.belief_domains = list(set(
                specialist_card.belief_domains + detected_domains
            ))
    
    def _convert_to_claim_bundles(
        self,
        result: "DebateResult",
        proposition_text: str,
    ) -> list[ClaimBundle]:
        """
        Convert debate evidence to Claim Bundles.
        
        Args:
            result: Debate result
            proposition_text: Original proposition
            
        Returns:
            List of ClaimBundles
        """
        claim_bundles = []
        
        if result.graph is None:
            return claim_bundles
        
        # Get all evidence nodes from C-DAG
        graph = result.graph
        
        for node in graph.nodes:
            if hasattr(node, 'likelihood_ratio') and hasattr(node, 'confidence'):
                # This is an Evidence node
                evidence = node
                
                # Determine polarity from likelihood ratio
                if evidence.likelihood_ratio > 1.0:
                    polarity = Polarity.SUPPORTS
                elif evidence.likelihood_ratio < 1.0:
                    polarity = Polarity.ATTACKS
                else:
                    polarity = Polarity.NEUTRAL
                
                # Create confidence distribution
                confidence_dist = ConfidenceDistribution.from_mean_std(
                    mean=evidence.confidence,
                    std=0.1,  # Default uncertainty
                )
                
                # Compute posterior estimate
                prior = result.verdict.prior if hasattr(result.verdict, 'prior') else 0.5
                posterior = self._compute_evidence_posterior(
                    prior,
                    evidence.likelihood_ratio,
                    evidence.confidence,
                )
                
                # Determine issuer
                issuer = (
                    "argus-specialist-001" 
                    if polarity == Polarity.SUPPORTS 
                    else "argus-refuter-001"
                )
                
                # Get credibility from registry
                card = self.registry.get_card(issuer)
                issuer_credibility = (
                    card.calibration.credibility_rating 
                    if card else 0.85
                )
                
                cb = self.cb_factory.create(
                    claim_text=evidence.text if hasattr(evidence, 'text') else str(evidence),
                    proposition_id=result.proposition_id,
                    polarity=polarity,
                    prior=prior,
                    posterior=posterior,
                    confidence_distribution=confidence_dist,
                    evidence_refs=[evidence.id if hasattr(evidence, 'id') else ""],
                    issuer_agent=issuer,
                    issuer_credibility=issuer_credibility,
                    challenge_open=True,
                )
                
                claim_bundles.append(cb)
        
        # If no evidence nodes found, create a summary bundle from verdict
        if not claim_bundles and result.verdict:
            verdict = result.verdict
            
            posterior = verdict.posterior if hasattr(verdict, 'posterior') else 0.5
            prior = verdict.prior if hasattr(verdict, 'prior') else 0.5
            
            confidence_dist = ConfidenceDistribution.from_mean_std(
                mean=posterior,
                std=0.15,
            )
            
            cb = self.cb_factory.create(
                claim_text=proposition_text,
                proposition_id=result.proposition_id,
                polarity=Polarity.SUPPORTS if posterior > 0.5 else Polarity.ATTACKS,
                prior=prior,
                posterior=posterior,
                confidence_distribution=confidence_dist,
                issuer_agent="argus-jury-001",
                issuer_credibility=0.92,
                challenge_open=False,  # Verdict is not challengeable
            )
            
            claim_bundles.append(cb)
        
        return claim_bundles
    
    def _compute_evidence_posterior(
        self,
        prior: float,
        likelihood_ratio: float,
        confidence: float,
    ) -> float:
        """
        Compute posterior from evidence.
        
        Uses Bayes' rule with confidence-weighted likelihood ratio.
        
        Args:
            prior: Prior probability
            likelihood_ratio: Evidence likelihood ratio
            confidence: Evidence confidence
            
        Returns:
            Posterior probability
        """
        # Weight LR by confidence
        effective_lr = 1.0 + confidence * (likelihood_ratio - 1.0)
        
        # Bayes' rule
        prior_odds = prior / (1 - prior) if prior < 1 else float('inf')
        posterior_odds = prior_odds * effective_lr
        
        # Convert back to probability
        posterior = posterior_odds / (1 + posterior_odds)
        
        return max(0.001, min(0.999, posterior))
    
    def _run_auction(self, claim_bundle: ClaimBundle) -> Optional[AuctionResult]:
        """
        Run a Challenger Auction for a Claim Bundle.
        
        Args:
            claim_bundle: Open Claim Bundle
            
        Returns:
            AuctionResult or None
        """
        if not claim_bundle.challenge_open:
            return None
        
        # Start auction
        session = self.auction.start(claim_bundle)
        
        # Generate synthetic bids from available challengers
        # In a real system, agents would submit bids asynchronously
        challengers = self._get_potential_challengers(claim_bundle)
        
        for agent_id, dfs_score in challengers:
            bid = ChallengerBid(
                bidder_agent=agent_id,
                cb_id=claim_bundle.cb_id,
                estimated_dfs=dfs_score,
                domain_confidence=0.8,
                adversarial_strategy="counter_evidence",
            )
            self.auction.submit_bid(session.auction_id, bid)
        
        # Close auction
        result = self.auction.close(session.auction_id, force=True)
        
        # Trigger callbacks
        for callback in self._on_auction_complete:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Auction callback error: {e}")
        
        return result
    
    def _get_potential_challengers(
        self,
        claim_bundle: ClaimBundle,
    ) -> list[tuple[str, float]]:
        """
        Get potential challengers for a Claim Bundle.
        
        Args:
            claim_bundle: Target Claim Bundle
            
        Returns:
            List of (agent_id, dfs_score) tuples
        """
        challengers = []
        
        for agent_id, card in self._agent_cards.items():
            # Skip issuer - self-challenge not permitted
            if card.agent_id == claim_bundle.issuer_agent:
                continue
            
            # Skip agents that can't challenge
            if not card.capabilities.challenge_claims:
                continue
            
            # Compute DFS
            dfs = self.router.compute_dfs(card, claim_bundle)
            challengers.append((card.agent_id, dfs.total_score))
        
        # Sort by DFS descending
        challengers.sort(key=lambda x: x[1], reverse=True)
        
        return challengers
    
    def _reconcile_claims(
        self,
        claim_bundles: list[ClaimBundle],
        proposition_id: str,
    ) -> Optional[ClaimBundle]:
        """
        Reconcile potentially contradicting Claim Bundles.
        
        Args:
            claim_bundles: All Claim Bundles
            proposition_id: Proposition ID
            
        Returns:
            Reconciled Claim Bundle or None
        """
        if not claim_bundles:
            return None
        
        # Group by proposition
        prop_bundles = [
            cb for cb in claim_bundles 
            if cb.proposition_id == proposition_id
        ]
        
        if len(prop_bundles) <= 1:
            return prop_bundles[0] if prop_bundles else None
        
        # Check for contradictions
        contradictions = self._find_contradictions(prop_bundles)
        
        if not contradictions:
            # No contradictions - return highest credibility bundle
            return max(prop_bundles, key=lambda cb: cb.issuer_credibility)
        
        # Run BRP for each contradiction
        reconciled_cb = prop_bundles[0]
        
        for cb1_id, cb2_id in contradictions:
            cb1 = next((cb for cb in prop_bundles if cb.cb_id == cb1_id), None)
            cb2 = next((cb for cb in prop_bundles if cb.cb_id == cb2_id), None)
            
            if not cb1 or not cb2:
                continue
            
            # Create BRP session
            brp_result = self.brp.reconcile([cb1, cb2])
            
            # Track BRP session
            if self._current_session and brp_result.session:
                self._current_session.add_brp_session(brp_result.session)
            
            # Trigger callbacks
            for callback in self._on_brp_triggered:
                try:
                    callback(brp_result.session)
                except Exception as e:
                    logger.warning(f"BRP callback error: {e}")
            
            # Use reconciled bundle
            if brp_result.reconciled_bundle:
                reconciled_cb = brp_result.reconciled_bundle
        
        return reconciled_cb
    
    def _find_contradictions(
        self,
        claim_bundles: list[ClaimBundle],
    ) -> list[tuple[str, str]]:
        """
        Find contradicting Claim Bundle pairs.
        
        Args:
            claim_bundles: Claim Bundles to check
            
        Returns:
            List of (cb1_id, cb2_id) contradiction pairs
        """
        contradictions = []
        threshold = self.config.contradiction_threshold
        
        for i, cb1 in enumerate(claim_bundles):
            for cb2 in claim_bundles[i+1:]:
                # Check posterior divergence
                divergence = abs(cb1.posterior - cb2.posterior)
                
                # Check polarity conflict
                polarity_conflict = (
                    cb1.polarity != cb2.polarity and 
                    cb1.polarity != Polarity.NEUTRAL and
                    cb2.polarity != Polarity.NEUTRAL
                )
                
                if divergence > threshold and polarity_conflict:
                    contradictions.append((cb1.cb_id, cb2.cb_id))
        
        return contradictions
    
    def _update_credibility(
        self,
        claim_bundles: list[ClaimBundle],
        verdict: "Verdict",
    ) -> None:
        """
        Update Credibility Ledger based on debate outcome.
        
        Args:
            claim_bundles: All Claim Bundles
            verdict: Final verdict
        """
        if not self._current_session:
            return
        
        final_posterior = verdict.posterior if hasattr(verdict, 'posterior') else 0.5
        
        for cb in claim_bundles:
            if cb.issuer_agent:
                # Compute Brier contribution
                predicted = cb.posterior
                # Use verdict as ground truth approximation
                ground_truth = 1.0 if final_posterior > 0.5 else 0.0
                brier_contribution = (predicted - ground_truth) ** 2
                
                # Record in ledger
                self.ledger.record_prediction(
                    agent_id=cb.issuer_agent,
                    session_id=self._current_session.session_id,
                    proposition_id=cb.proposition_id or "",
                    predicted_posterior=predicted,
                    ground_truth=ground_truth,
                )
                
                self._current_session.stats.total_credibility_updates += 1
    
    # Event registration
    def on_claim_bundle(self, callback: Callable[[ClaimBundle], None]) -> None:
        """Register callback for Claim Bundle events."""
        self._on_claim_bundle.append(callback)
    
    def on_brp_triggered(self, callback: Callable[[BRPSession], None]) -> None:
        """Register callback for BRP trigger events."""
        self._on_brp_triggered.append(callback)
    
    def on_auction_complete(self, callback: Callable[[AuctionResult], None]) -> None:
        """Register callback for auction complete events."""
        self._on_auction_complete.append(callback)
    
    # Session management
    @property
    def current_session(self) -> Optional[CRUXSession]:
        """Get current CRUX session."""
        return self._current_session
    
    @property
    def session_history(self) -> list[CRUXSession]:
        """Get all completed sessions."""
        return self._session_history.copy()
    
    # Agent card management
    def register_agent(self, card: EpistemicAgentCard) -> None:
        """Register an Epistemic Agent Card."""
        self.registry.register(card)
        self._agent_cards[card.agent_id] = card
    
    def get_agent_card(self, agent_id: str) -> Optional[EpistemicAgentCard]:
        """Get agent card by ID."""
        return self.registry.get_card(agent_id)
    
    # CRUX endpoint handlers
    def get_card_endpoint(self, agent_id: str) -> dict[str, Any]:
        """GET /crux/card - Return Epistemic Agent Card."""
        card = self.registry.get_card(agent_id)
        if card:
            return card.to_dict()
        return {"error": "Agent not found"}
    
    def submit_claim_endpoint(self, claim_data: dict[str, Any]) -> dict[str, Any]:
        """POST /crux/claim - Submit a Claim Bundle."""
        try:
            cb = ClaimBundle.from_dict(claim_data)
            
            if self._current_session:
                self._current_session.add_claim_bundle(cb)
            
            self.edr.register_bundle(cb)
            
            return {"cb_id": cb.cb_id, "accepted": True}
        except Exception as e:
            return {"error": str(e), "accepted": False}
    
    def submit_bid_endpoint(self, bid_data: dict[str, Any]) -> dict[str, Any]:
        """POST /crux/bid - Submit a Challenger Bid."""
        try:
            bid = ChallengerBid(
                bidder_agent=bid_data["bidder_agent"],
                cb_id=bid_data["cb_id"],
                estimated_dfs=bid_data.get("estimated_dfs", 0.5),
                domain_confidence=bid_data.get("domain_confidence", 0.5),
                adversarial_strategy=bid_data.get("adversarial_strategy", "general_critique"),
            )
            
            # Find auction for CB
            session = self.auction.get_session_for_cb(bid.cb_id)
            if not session:
                return {"error": "No active auction for CB", "accepted": False}
            
            result = self.auction.submit_bid(session.auction_id, bid)
            
            return result.to_dict()
        except Exception as e:
            return {"error": str(e), "accepted": False}
    
    def get_ledger_endpoint(self, agent_id: str) -> dict[str, Any]:
        """GET /crux/ledger/{agent} - Get Credibility Ledger for agent."""
        entries = self.ledger.get_entries_for_agent(agent_id)
        credibility = self.ledger.get_credibility(agent_id)
        
        return {
            "agent_id": agent_id,
            "credibility_rating": credibility,
            "num_entries": len(entries),
            "entries": [e.to_dict() for e in entries[-10:]],  # Last 10
        }
    
    def sync_endpoint(self, sync_data: dict[str, Any]) -> dict[str, Any]:
        """POST /crux/sync - EDR resync request."""
        if not self.config.enable_edr:
            return {"error": "EDR is disabled"}
        
        result = self.edr.reconnect(
            agent_id=sync_data["agent_id"],
            last_checkpoint_hash=sync_data["checkpoint_hash"],
        )
        
        return result.to_dict()
