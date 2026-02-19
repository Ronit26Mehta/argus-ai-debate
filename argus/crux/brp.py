"""
Belief Reconciliation Protocol (BRP) for CRUX.

When two agents issue contradicting Claim Bundles on the same proposition
(opposite polarity or posterior divergence exceeding threshold θ), the
CRUX orchestrator automatically triggers the Belief Reconciliation Protocol
— a structured mini-debate that merges beliefs into a reconciled posterior.

BRP State Machine:
    IDLE -> TRIGGERED (contradiction detected)
    TRIGGERED -> MINI_DEBATE (orchestrator spawns session)
    MINI_DEBATE -> BAYESIAN_MERGE (agents submit updates)
    BAYESIAN_MERGE -> PROVENANCE_FORK (merge computed)
    PROVENANCE_FORK -> RESOLVED (lineage updated)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING, Union
from dataclasses import dataclass, field
from enum import Enum

from argus.crux.models import (
    generate_crux_id,
    compute_crux_hash,
    BRPState,
    BRPResolution,
    ConfidenceDistribution,
    CRUXConfig,
)
from argus.crux.claim_bundle import ClaimBundle, merge_claim_bundles

if TYPE_CHECKING:
    from argus.crux.agent_card import EpistemicAgentCard, EACRegistry
    from argus.crux.ledger import CredibilityLedger

logger = logging.getLogger(__name__)


class ContradictionType(str, Enum):
    """Types of contradictions between Claim Bundles."""
    POLARITY_CONFLICT = "polarity_conflict"
    POSTERIOR_DIVERGENCE = "posterior_divergence"
    BOTH = "both"


@dataclass
class Contradiction:
    """
    Detected contradiction between two Claim Bundles.
    
    Attributes:
        bundle_a: First Claim Bundle
        bundle_b: Second Claim Bundle
        contradiction_type: Type of contradiction
        posterior_divergence: Absolute difference in posteriors
        detected_at: When contradiction was detected
    """
    bundle_a: ClaimBundle
    bundle_b: ClaimBundle
    contradiction_type: ContradictionType
    posterior_divergence: float = 0.0
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def proposition_id(self) -> str:
        """Get the proposition ID (same for both bundles)."""
        return self.bundle_a.proposition_id
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bundle_a_id": self.bundle_a.cb_id,
            "bundle_b_id": self.bundle_b.cb_id,
            "type": self.contradiction_type.value,
            "posterior_divergence": self.posterior_divergence,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class MiniDebateRound:
    """
    A round in the BRP mini-debate.
    
    Attributes:
        round_num: Round number (1-indexed)
        agent_claims: Claims submitted by each agent
        evidence_refs: New evidence introduced
        distribution_updates: Updated distributions
        timestamp: Round timestamp
    """
    round_num: int
    agent_claims: dict[str, str] = field(default_factory=dict)
    evidence_refs: dict[str, list[str]] = field(default_factory=dict)
    distribution_updates: dict[str, ConfidenceDistribution] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_num": self.round_num,
            "agent_claims": self.agent_claims,
            "evidence_refs": self.evidence_refs,
            "distribution_updates": {
                k: v.to_dict() for k, v in self.distribution_updates.items()
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BRPSession:
    """
    A Belief Reconciliation Protocol session.
    
    Manages state and history of a reconciliation process.
    
    Attributes:
        session_id: Unique session identifier
        contradiction: The contradiction being resolved
        state: Current BRP state
        rounds: Mini-debate rounds
        max_rounds: Maximum allowed rounds
        participant_agents: Agent IDs participating
        reconciled_bundle: Final reconciled Claim Bundle
        resolution: Resolution details
        started_at: Session start time
        completed_at: Session completion time
    """
    session_id: str = field(default_factory=lambda: generate_crux_id("brp"))
    contradiction: Optional[Contradiction] = None
    state: BRPState = BRPState.IDLE
    rounds: list[MiniDebateRound] = field(default_factory=list)
    max_rounds: int = 3
    participant_agents: list[str] = field(default_factory=list)
    reconciled_bundle: Optional[ClaimBundle] = None
    resolution: Optional[BRPResolution] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state not in [BRPState.IDLE, BRPState.RESOLVED]
    
    @property
    def current_round(self) -> int:
        """Get current round number."""
        return len(self.rounds)
    
    def transition(self, new_state: BRPState) -> None:
        """
        Transition to a new state.
        
        Args:
            new_state: New BRP state
        """
        logger.info(f"BRP {self.session_id}: {self.state.value} -> {new_state.value}")
        self.state = new_state
        
        if new_state == BRPState.TRIGGERED:
            self.started_at = datetime.utcnow()
        elif new_state == BRPState.RESOLVED:
            self.completed_at = datetime.utcnow()
    
    def add_round(self, round_data: MiniDebateRound) -> None:
        """Add a mini-debate round."""
        self.rounds.append(round_data)
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "contradiction": (
                self.contradiction.to_dict() if self.contradiction else None
            ),
            "rounds": [r.to_dict() for r in self.rounds],
            "participant_agents": self.participant_agents,
            "reconciled_bundle": (
                self.reconciled_bundle.to_dict() 
                if self.reconciled_bundle else None
            ),
            "resolution": (
                self.resolution.to_dict() if self.resolution else None
            ),
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_seconds": self.duration_seconds,
        }


class ContradictionDetector:
    """
    Detects contradictions between Claim Bundles.
    
    A contradiction occurs when two bundles on the same proposition
    have opposite polarity OR posterior divergence > threshold.
    """
    
    def __init__(self, threshold: float = 0.20):
        """
        Initialize detector.
        
        Args:
            threshold: Posterior divergence threshold (θ)
        """
        self.threshold = threshold
    
    def detect(
        self,
        bundle_a: ClaimBundle,
        bundle_b: ClaimBundle,
    ) -> Optional[Contradiction]:
        """
        Detect contradiction between two bundles.
        
        Args:
            bundle_a: First Claim Bundle
            bundle_b: Second Claim Bundle
            
        Returns:
            Contradiction if detected, None otherwise
        """
        # Must be same proposition
        if bundle_a.proposition_id != bundle_b.proposition_id:
            return None
        
        # Check for polarity conflict
        from argus.crux.models import Polarity
        
        polarity_conflict = (
            bundle_a.polarity != bundle_b.polarity and
            bundle_a.polarity != Polarity.NEUTRAL and
            bundle_b.polarity != Polarity.NEUTRAL
        )
        
        # Check for posterior divergence
        divergence = abs(bundle_a.posterior - bundle_b.posterior)
        posterior_divergence = divergence > self.threshold
        
        if not polarity_conflict and not posterior_divergence:
            return None
        
        # Determine type
        if polarity_conflict and posterior_divergence:
            contradiction_type = ContradictionType.BOTH
        elif polarity_conflict:
            contradiction_type = ContradictionType.POLARITY_CONFLICT
        else:
            contradiction_type = ContradictionType.POSTERIOR_DIVERGENCE
        
        return Contradiction(
            bundle_a=bundle_a,
            bundle_b=bundle_b,
            contradiction_type=contradiction_type,
            posterior_divergence=divergence,
        )
    
    def find_contradictions(
        self,
        bundles: list[ClaimBundle],
    ) -> list[Contradiction]:
        """
        Find all contradictions in a list of bundles.
        
        Args:
            bundles: List of Claim Bundles
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Group by proposition
        by_proposition: dict[str, list[ClaimBundle]] = {}
        for bundle in bundles:
            prop_id = bundle.proposition_id
            if prop_id not in by_proposition:
                by_proposition[prop_id] = []
            by_proposition[prop_id].append(bundle)
        
        # Check pairs within each proposition
        for prop_bundles in by_proposition.values():
            for i, bundle_a in enumerate(prop_bundles):
                for bundle_b in prop_bundles[i + 1:]:
                    contradiction = self.detect(bundle_a, bundle_b)
                    if contradiction:
                        contradictions.append(contradiction)
        
        return contradictions


class BayesianMerger:
    """
    Performs Bayesian merge of contradicting Claim Bundles.
    
    The merge formula:
        merged_posterior = Σ(credibility_i * posterior_i) / Σ(credibility_i)
        merged_distribution = moment-matched Beta from weighted mixture
    """
    
    @staticmethod
    def compute_merged_posterior(
        posteriors: list[float],
        credibilities: list[float],
    ) -> float:
        """
        Compute credibility-weighted merged posterior.
        
        Args:
            posteriors: List of posterior probabilities
            credibilities: List of credibility weights
            
        Returns:
            Merged posterior
        """
        if not posteriors:
            return 0.5
        
        if len(posteriors) != len(credibilities):
            raise ValueError("Posteriors and credibilities must match")
        
        total_weight = sum(credibilities)
        if total_weight == 0:
            return sum(posteriors) / len(posteriors)
        
        weighted_sum = sum(p * c for p, c in zip(posteriors, credibilities))
        return weighted_sum / total_weight
    
    @staticmethod
    def merge_distributions(
        distributions: list[ConfidenceDistribution],
        weights: list[float],
    ) -> ConfidenceDistribution:
        """
        Merge distributions via moment matching.
        
        Computes weighted mean and variance from mixture,
        then fits a Beta distribution.
        
        Args:
            distributions: List of distributions
            weights: Weights for each distribution
            
        Returns:
            Merged ConfidenceDistribution
        """
        if not distributions:
            return ConfidenceDistribution.from_beta(1.0, 1.0)
        
        # Normalize weights
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        
        # Weighted mean
        merged_mean = sum(
            w * d.mean for w, d in zip(norm_weights, distributions)
        )
        
        # Merged variance (within + between)
        within_var = sum(
            w * (d.std ** 2) for w, d in zip(norm_weights, distributions)
        )
        between_var = sum(
            w * ((d.mean - merged_mean) ** 2)
            for w, d in zip(norm_weights, distributions)
        )
        merged_var = within_var + between_var
        merged_std = merged_var ** 0.5
        
        return ConfidenceDistribution.from_mean_std(merged_mean, merged_std)


@dataclass
class ReconciliationResult:
    """
    Result of Belief Reconciliation Protocol.
    
    Attributes:
        success: Whether reconciliation succeeded
        reconciled_bundle: Merged Claim Bundle
        resolution: Resolution details
        session: BRP session data
    """
    success: bool
    reconciled_bundle: Optional[ClaimBundle]
    resolution: Optional[BRPResolution]
    session: BRPSession
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "reconciled_bundle": (
                self.reconciled_bundle.to_dict() 
                if self.reconciled_bundle else None
            ),
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "session_id": self.session.session_id,
            "rounds_completed": self.session.current_round,
            "duration_seconds": self.session.duration_seconds,
        }


class BeliefReconciliationProtocol:
    """
    Belief Reconciliation Protocol (BRP) - CRUX Primitive 4.
    
    Manages the full lifecycle of belief reconciliation when
    contradictions are detected between agents.
    
    Example:
        >>> brp = BeliefReconciliationProtocol(config, registry)
        >>> 
        >>> # Detect contradiction
        >>> contradiction = brp.detect_contradiction(bundle_a, bundle_b)
        >>> if contradiction:
        ...     result = brp.reconcile(contradiction)
        ...     print(f"Merged posterior: {result.reconciled_bundle.posterior}")
    """
    
    def __init__(
        self,
        config: Optional[CRUXConfig] = None,
        registry: Optional["EACRegistry"] = None,
        ledger: Optional["CredibilityLedger"] = None,
    ):
        """
        Initialize BRP.
        
        Args:
            config: CRUX configuration
            registry: EAC registry for agent lookup
            ledger: Credibility ledger for weight lookup
        """
        self.config = config or CRUXConfig()
        self.registry = registry
        self.ledger = ledger
        self.detector = ContradictionDetector(self.config.contradiction_threshold)
        self.merger = BayesianMerger()
        
        # Active sessions
        self._sessions: dict[str, BRPSession] = {}
        
        # History of completed reconciliations
        self._history: list[BRPSession] = []
    
    def detect_contradiction(
        self,
        bundle_a: ClaimBundle,
        bundle_b: ClaimBundle,
    ) -> Optional[Contradiction]:
        """
        Detect contradiction between two bundles.
        
        Args:
            bundle_a: First Claim Bundle
            bundle_b: Second Claim Bundle
            
        Returns:
            Contradiction if detected
        """
        return self.detector.detect(bundle_a, bundle_b)
    
    def find_all_contradictions(
        self,
        bundles: list[ClaimBundle],
    ) -> list[Contradiction]:
        """Find all contradictions in a list of bundles."""
        return self.detector.find_contradictions(bundles)
    
    def start_session(
        self,
        contradiction: Contradiction,
    ) -> BRPSession:
        """
        Start a new BRP session.
        
        Args:
            contradiction: Detected contradiction
            
        Returns:
            New BRPSession
        """
        session = BRPSession(
            contradiction=contradiction,
            max_rounds=self.config.brp_max_rounds,
            participant_agents=[
                contradiction.bundle_a.issuer_agent,
                contradiction.bundle_b.issuer_agent,
            ],
        )
        
        session.transition(BRPState.TRIGGERED)
        self._sessions[session.session_id] = session
        
        logger.info(
            f"Started BRP session {session.session_id} for "
            f"{contradiction.bundle_a.cb_id} vs {contradiction.bundle_b.cb_id}"
        )
        
        return session
    
    def run_mini_debate(
        self,
        session: BRPSession,
        round_claims: Optional[dict[str, str]] = None,
        round_evidence: Optional[dict[str, list[str]]] = None,
        round_distributions: Optional[dict[str, ConfidenceDistribution]] = None,
    ) -> MiniDebateRound:
        """
        Run a mini-debate round.
        
        Args:
            session: Active BRP session
            round_claims: Agent claims for this round
            round_evidence: New evidence refs per agent
            round_distributions: Updated distributions per agent
            
        Returns:
            Completed round data
        """
        session.transition(BRPState.MINI_DEBATE)
        
        round_data = MiniDebateRound(
            round_num=session.current_round + 1,
            agent_claims=round_claims or {},
            evidence_refs=round_evidence or {},
            distribution_updates=round_distributions or {},
        )
        
        session.add_round(round_data)
        
        return round_data
    
    def perform_merge(
        self,
        session: BRPSession,
    ) -> ClaimBundle:
        """
        Perform Bayesian merge of contradicting bundles.
        
        Args:
            session: BRP session with contradiction
            
        Returns:
            Merged ClaimBundle
        """
        session.transition(BRPState.BAYESIAN_MERGE)
        
        contradiction = session.contradiction
        bundles = [contradiction.bundle_a, contradiction.bundle_b]
        
        # Get credibility weights
        credibilities = []
        for bundle in bundles:
            cred = bundle.issuer_credibility
            
            # Try to get updated credibility from ledger
            if self.ledger:
                ledger_cred = self.ledger.get_credibility(bundle.issuer_agent)
                if ledger_cred is not None:
                    cred = ledger_cred
            
            credibilities.append(cred)
        
        # Include any distribution updates from mini-debate
        distributions = []
        for bundle in bundles:
            dist = bundle.confidence_distribution
            
            # Check for updates in rounds
            for round_data in session.rounds:
                if bundle.issuer_agent in round_data.distribution_updates:
                    dist = round_data.distribution_updates[bundle.issuer_agent]
            
            distributions.append(dist)
        
        # Perform merge
        merged_bundle = merge_claim_bundles(bundles, credibilities)
        
        # Update with merged distribution
        merged_dist = self.merger.merge_distributions(distributions, credibilities)
        object.__setattr__(merged_bundle, "confidence_distribution", merged_dist)
        
        return merged_bundle
    
    def complete_session(
        self,
        session: BRPSession,
        merged_bundle: ClaimBundle,
    ) -> BRPResolution:
        """
        Complete a BRP session with provenance fork.
        
        Args:
            session: Active BRP session
            merged_bundle: Merged Claim Bundle
            
        Returns:
            BRPResolution
        """
        session.transition(BRPState.PROVENANCE_FORK)
        
        # Collect ancestor hashes
        ancestor_hashes = [
            session.contradiction.bundle_a.argument_lineage_hash,
            session.contradiction.bundle_b.argument_lineage_hash,
        ]
        
        # Compute new lineage hash (merge commit)
        lineage_content = json.dumps({
            "ancestors": ancestor_hashes,
            "merged_at": datetime.utcnow().isoformat(),
            "session_id": session.session_id,
        }, sort_keys=True)
        new_lineage = compute_crux_hash(lineage_content)
        
        # Create resolution
        resolution = BRPResolution(
            original_cb_ids=[
                session.contradiction.bundle_a.cb_id,
                session.contradiction.bundle_b.cb_id,
            ],
            reconciled_posterior=merged_bundle.posterior,
            reconciled_distribution=merged_bundle.confidence_distribution,
            contributor_agents=session.participant_agents,
            contributor_weights=[
                session.contradiction.bundle_a.issuer_credibility,
                session.contradiction.bundle_b.issuer_credibility,
            ],
            lineage_hash=new_lineage,
            ancestor_hashes=ancestor_hashes,
            rounds_completed=session.current_round,
        )
        
        # Update session
        session.reconciled_bundle = merged_bundle
        session.resolution = resolution
        session.transition(BRPState.RESOLVED)
        
        # Move to history
        del self._sessions[session.session_id]
        self._history.append(session)
        
        logger.info(
            f"Completed BRP session {session.session_id}: "
            f"merged posterior = {merged_bundle.posterior:.3f}"
        )
        
        return resolution
    
    def reconcile(
        self,
        bundles_or_contradiction: Union[Contradiction, list[ClaimBundle]],
        max_rounds: Optional[int] = None,
    ) -> ReconciliationResult:
        """
        Full reconciliation workflow for a contradiction.
        
        Executes the complete BRP state machine:
        IDLE -> TRIGGERED -> MINI_DEBATE -> BAYESIAN_MERGE -> PROVENANCE_FORK -> RESOLVED
        
        Args:
            bundles_or_contradiction: Either a Contradiction or list of 2 ClaimBundles
            max_rounds: Override max rounds
            
        Returns:
            ReconciliationResult
        """
        # Handle list of bundles
        if isinstance(bundles_or_contradiction, list):
            bundles = bundles_or_contradiction
            if len(bundles) < 2:
                return ReconciliationResult(
                    success=False,
                    reconciled_bundle=bundles[0] if bundles else None,
                    resolution=None,
                    session=BRPSession(),
                )
            
            # Detect contradiction between first two bundles
            contradiction = self.detect_contradiction(bundles[0], bundles[1])
            
            if not contradiction:
                # No contradiction - return first bundle
                return ReconciliationResult(
                    success=True,
                    reconciled_bundle=bundles[0],
                    resolution=None,
                    session=BRPSession(),
                )
        else:
            contradiction = bundles_or_contradiction
        
        # Start session
        session = self.start_session(contradiction)
        if max_rounds:
            session.max_rounds = max_rounds
        
        try:
            # Run mini-debate rounds (simplified: single round)
            # In full implementation, would loop with agent input
            self.run_mini_debate(session)
            
            # Perform merge
            merged_bundle = self.perform_merge(session)
            
            # Complete with provenance
            resolution = self.complete_session(session, merged_bundle)
            
            return ReconciliationResult(
                success=True,
                reconciled_bundle=merged_bundle,
                resolution=resolution,
                session=session,
            )
            
        except Exception as e:
            logger.error(f"BRP reconciliation failed: {e}")
            session.transition(BRPState.RESOLVED)
            session.metadata["error"] = str(e)
            
            return ReconciliationResult(
                success=False,
                reconciled_bundle=None,
                resolution=None,
                session=session,
            )
    
    def get_session(self, session_id: str) -> Optional[BRPSession]:
        """Get active session by ID."""
        return self._sessions.get(session_id)
    
    def get_history(
        self,
        proposition_id: Optional[str] = None,
    ) -> list[BRPSession]:
        """
        Get history of completed reconciliations.
        
        Args:
            proposition_id: Filter by proposition
            
        Returns:
            List of completed sessions
        """
        if proposition_id is None:
            return list(self._history)
        
        return [
            s for s in self._history
            if s.contradiction and s.contradiction.proposition_id == proposition_id
        ]
