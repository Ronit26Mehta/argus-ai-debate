"""
Challenger Auction (CA) for CRUX Protocol.

When a Claim Bundle is marked challenge_open: true, CRUX runs a Challenger
Auction to select the best-positioned agent to rebut it. Unlike simple
first-available assignment, the auction ensures the strongest possible
adversarial engagement — a core requirement for ARGUS's epistemic integrity.

Key Features:
    - Bid collection from eligible agents
    - DFS-based bid evaluation
    - Auction timeout management
    - Unchallenged claim marking
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

from argus.crux.models import (
    generate_crux_id,
    ChallengerBid,
    CRUXConfig,
    AuctionState,
)

if TYPE_CHECKING:
    from argus.crux.claim_bundle import ClaimBundle
    from argus.crux.agent_card import EpistemicAgentCard, EACRegistry
    from argus.crux.routing import DialecticalRouter, DialecticalFitnessScore

logger = logging.getLogger(__name__)


class BidRejectionReason(str, Enum):
    """Reasons for rejecting a bid."""
    EXPIRED = "bid_expired"
    INVALID_CB = "invalid_claim_bundle"
    SELF_CHALLENGE = "self_challenge_not_permitted"
    LOW_CREDIBILITY = "credibility_below_floor"
    AUCTION_CLOSED = "auction_already_closed"
    DUPLICATE_BID = "duplicate_bid_from_agent"


@dataclass
class BidResult:
    """
    Result of submitting a bid.
    
    Attributes:
        accepted: Whether bid was accepted
        bid: The submitted bid (if accepted)
        rejection_reason: Reason for rejection (if rejected)
    """
    accepted: bool
    bid: Optional[ChallengerBid] = None
    rejection_reason: Optional[BidRejectionReason] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accepted": self.accepted,
            "bid_id": self.bid.bid_id if self.bid else None,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
        }


@dataclass
class AuctionResult:
    """
    Result of a Challenger Auction.
    
    Attributes:
        auction_id: Unique auction identifier
        cb_id: Claim Bundle that was auctioned
        state: Final auction state
        winner: Winning agent ID (if any)
        winning_bid: Winning bid (if any)
        all_bids: All bids submitted
        dfs_scores: DFS scores for all bidders
        started_at: When auction started
        completed_at: When auction completed
        unchallenged: Whether claim went unchallenged
    """
    auction_id: str = field(default_factory=lambda: generate_crux_id("auction"))
    cb_id: str = ""
    state: AuctionState = AuctionState.CLOSED
    winner: Optional[str] = None
    winning_bid: Optional[ChallengerBid] = None
    all_bids: list[ChallengerBid] = field(default_factory=list)
    dfs_scores: dict[str, float] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    unchallenged: bool = False
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.utcnow()
    
    @property
    def num_bids(self) -> int:
        """Number of bids received."""
        return len(self.all_bids)
    
    @property
    def duration_seconds(self) -> float:
        """Auction duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "auction_id": self.auction_id,
            "cb_id": self.cb_id,
            "state": self.state.value,
            "winner": self.winner,
            "winning_bid": self.winning_bid.to_dict() if self.winning_bid else None,
            "num_bids": self.num_bids,
            "dfs_scores": self.dfs_scores,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "unchallenged": self.unchallenged,
            "duration_seconds": self.duration_seconds,
        }


class BidEvaluator:
    """
    Evaluates bids in a Challenger Auction.
    
    Uses Dialectical Fitness Scores to rank bids and select
    the best challenger.
    
    Example:
        >>> evaluator = BidEvaluator(router, config)
        >>> 
        >>> # Evaluate all bids
        >>> ranked = evaluator.rank_bids(bids, claim_bundle)
        >>> 
        >>> # Select winner
        >>> winner = evaluator.select_winner(ranked)
    """
    
    def __init__(
        self,
        router: Optional["DialecticalRouter"] = None,
        config: Optional[CRUXConfig] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            router: Dialectical router for DFS computation
            config: CRUX configuration
        """
        self.router = router
        self.config = config or CRUXConfig()
    
    def validate_bid(
        self,
        bid: ChallengerBid,
        claim_bundle: "ClaimBundle",
        credibility_floor: Optional[float] = None,
        existing_bidders: Optional[set[str]] = None,
    ) -> tuple[bool, Optional[BidRejectionReason]]:
        """
        Validate a bid.
        
        Args:
            bid: Bid to validate
            claim_bundle: Target Claim Bundle
            credibility_floor: Minimum credibility required
            existing_bidders: Set of agents who already bid
            
        Returns:
            (is_valid, rejection_reason)
        """
        # Check if bid has expired
        if bid.is_expired:
            return False, BidRejectionReason.EXPIRED
        
        # Check for self-challenge
        if bid.bidder_agent == claim_bundle.issuer_agent:
            return False, BidRejectionReason.SELF_CHALLENGE
        
        # Check credibility floor
        if credibility_floor is not None:
            if bid.domain_confidence < credibility_floor:
                return False, BidRejectionReason.LOW_CREDIBILITY
        
        # Check for duplicate bid
        if existing_bidders and bid.bidder_agent in existing_bidders:
            return False, BidRejectionReason.DUPLICATE_BID
        
        return True, None
    
    def compute_bid_score(
        self,
        bid: ChallengerBid,
        claim_bundle: "ClaimBundle",
        agent_card: Optional["EpistemicAgentCard"] = None,
    ) -> float:
        """
        Compute score for a bid.
        
        Combines estimated DFS with domain confidence.
        
        Args:
            bid: Bid to score
            claim_bundle: Target Claim Bundle
            agent_card: Agent's Epistemic Agent Card
            
        Returns:
            Bid score
        """
        # Start with bid's self-reported DFS estimate
        score = bid.estimated_dfs
        
        # If router available, compute actual DFS
        if self.router and agent_card:
            dfs = self.router.compute_dfs(agent_card, claim_bundle)
            # Weight actual DFS more heavily than estimate
            score = 0.7 * dfs.total_score + 0.3 * bid.estimated_dfs
        
        # Apply domain confidence boost
        score *= (0.5 + 0.5 * bid.domain_confidence)
        
        # Apply strategy-based adjustment
        strategy_multipliers = {
            "methodological_critique": 1.1,
            "counter_evidence": 1.15,
            "statistical_analysis": 1.05,
            "domain_expertise": 1.0,
            "general_critique": 0.95,
        }
        multiplier = strategy_multipliers.get(bid.adversarial_strategy, 1.0)
        score *= multiplier
        
        return min(1.0, max(0.0, score))
    
    def rank_bids(
        self,
        bids: list[ChallengerBid],
        claim_bundle: "ClaimBundle",
        agent_cards: Optional[dict[str, "EpistemicAgentCard"]] = None,
    ) -> list[tuple[ChallengerBid, float]]:
        """
        Rank bids by score.
        
        Args:
            bids: Bids to rank
            claim_bundle: Target Claim Bundle
            agent_cards: Agent cards keyed by agent ID
            
        Returns:
            List of (bid, score) sorted by descending score
        """
        scored_bids = []
        
        for bid in bids:
            agent_card = (
                agent_cards.get(bid.bidder_agent) 
                if agent_cards else None
            )
            score = self.compute_bid_score(bid, claim_bundle, agent_card)
            scored_bids.append((bid, score))
        
        # Sort by score descending
        scored_bids.sort(key=lambda x: x[1], reverse=True)
        
        return scored_bids
    
    def select_winner(
        self,
        ranked_bids: list[tuple[ChallengerBid, float]],
    ) -> Optional[ChallengerBid]:
        """
        Select winning bid from ranked list.
        
        Args:
            ranked_bids: Bids ranked by score
            
        Returns:
            Winning bid or None if no valid bids
        """
        if not ranked_bids:
            return None
        
        # Winner is highest scored bid
        return ranked_bids[0][0]


class ChallengerAuction:
    """
    Challenger Auction - CRUX Primitive 7.
    
    Manages the auction process for selecting the best challenger
    for an open Claim Bundle.
    
    Example:
        >>> auction = ChallengerAuction(router, registry, config)
        >>> 
        >>> # Start auction for a claim bundle
        >>> session = auction.start(claim_bundle)
        >>> 
        >>> # Agents submit bids
        >>> result = auction.submit_bid(session_id, bid)
        >>> 
        >>> # Close auction and get result
        >>> result = auction.close(session_id)
        >>> print(f"Winner: {result.winner}")
    """
    
    def __init__(
        self,
        router: Optional["DialecticalRouter"] = None,
        registry: Optional["EACRegistry"] = None,
        config: Optional[CRUXConfig] = None,
    ):
        """
        Initialize auction.
        
        Args:
            router: Dialectical router for DFS
            registry: Agent card registry
            config: CRUX configuration
        """
        self.router = router
        self.registry = registry
        self.config = config or CRUXConfig()
        self.evaluator = BidEvaluator(router, config)
        
        # Active auction sessions
        self._sessions: dict[str, AuctionSession] = {}
        
        # Completed auction results
        self._results: dict[str, AuctionResult] = {}
        
        # Callbacks for auction events
        self._on_bid_received: list[Callable] = []
        self._on_auction_complete: list[Callable] = []
    
    def start(
        self,
        claim_bundle: "ClaimBundle",
        timeout_seconds: Optional[int] = None,
    ) -> "AuctionSession":
        """
        Start a Challenger Auction for a Claim Bundle.
        
        Args:
            claim_bundle: Claim Bundle to auction
            timeout_seconds: Auction timeout (uses config default if None)
            
        Returns:
            AuctionSession
        """
        if not claim_bundle.challenge_open:
            raise ValueError(
                f"Claim Bundle {claim_bundle.cb_id} is not open for challenge"
            )
        
        timeout = timeout_seconds or self.config.auction_timeout_seconds
        
        session = AuctionSession(
            auction_id=generate_crux_id("auction"),
            claim_bundle=claim_bundle,
            timeout_seconds=timeout,
            evaluator=self.evaluator,
            registry=self.registry,
        )
        
        self._sessions[session.auction_id] = session
        
        logger.info(
            f"Started auction {session.auction_id} for CB {claim_bundle.cb_id}, "
            f"timeout={timeout}s"
        )
        
        return session
    
    def get_session(self, auction_id: str) -> Optional["AuctionSession"]:
        """Get auction session by ID."""
        return self._sessions.get(auction_id)
    
    def get_session_for_cb(self, cb_id: str) -> Optional["AuctionSession"]:
        """Get active auction session for a Claim Bundle."""
        for session in self._sessions.values():
            if session.cb_id == cb_id and session.state != AuctionState.CLOSED:
                return session
        return None
    
    def submit_bid(
        self,
        auction_id: str,
        bid: ChallengerBid,
    ) -> BidResult:
        """
        Submit a bid to an auction.
        
        Args:
            auction_id: Auction session ID
            bid: Bid to submit
            
        Returns:
            BidResult indicating success/failure
        """
        session = self._sessions.get(auction_id)
        if not session:
            return BidResult(
                accepted=False,
                rejection_reason=BidRejectionReason.INVALID_CB,
            )
        
        result = session.submit_bid(bid)
        
        if result.accepted:
            for callback in self._on_bid_received:
                try:
                    callback(auction_id, bid)
                except Exception as e:
                    logger.warning(f"Bid callback error: {e}")
        
        return result
    
    def close(
        self,
        auction_id: str,
        force: bool = False,
    ) -> AuctionResult:
        """
        Close an auction and determine winner.
        
        Args:
            auction_id: Auction session ID
            force: Force close even if timeout not reached
            
        Returns:
            AuctionResult
        """
        session = self._sessions.get(auction_id)
        if not session:
            raise ValueError(f"No auction with ID {auction_id}")
        
        result = session.close(force=force)
        
        # Store result
        self._results[auction_id] = result
        
        # Remove from active sessions
        del self._sessions[auction_id]
        
        # Trigger callbacks
        for callback in self._on_auction_complete:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Auction complete callback error: {e}")
        
        logger.info(
            f"Closed auction {auction_id}: "
            f"winner={result.winner}, "
            f"unchallenged={result.unchallenged}, "
            f"bids={result.num_bids}"
        )
        
        return result
    
    def close_if_timeout(self, auction_id: str) -> Optional[AuctionResult]:
        """
        Close auction if timeout reached.
        
        Args:
            auction_id: Auction session ID
            
        Returns:
            AuctionResult if closed, None otherwise
        """
        session = self._sessions.get(auction_id)
        if not session:
            return None
        
        if session.is_expired:
            return self.close(auction_id)
        
        return None
    
    async def run_auction(
        self,
        claim_bundle: "ClaimBundle",
        timeout_seconds: Optional[int] = None,
        bid_source: Optional[Callable] = None,
    ) -> AuctionResult:
        """
        Run a complete auction asynchronously.
        
        Args:
            claim_bundle: Claim Bundle to auction
            timeout_seconds: Auction timeout
            bid_source: Async callable that yields bids
            
        Returns:
            AuctionResult
        """
        session = self.start(claim_bundle, timeout_seconds)
        
        if bid_source:
            # Collect bids from source
            try:
                async for bid in bid_source(claim_bundle):
                    if session.is_expired:
                        break
                    self.submit_bid(session.auction_id, bid)
            except Exception as e:
                logger.error(f"Bid source error: {e}")
        
        # Wait for timeout if not expired
        if not session.is_expired:
            remaining = session.time_remaining
            if remaining > 0:
                await asyncio.sleep(remaining)
        
        return self.close(session.auction_id)
    
    def get_result(self, auction_id: str) -> Optional[AuctionResult]:
        """Get result for a completed auction."""
        return self._results.get(auction_id)
    
    def on_bid_received(self, callback: Callable) -> None:
        """Register callback for bid received events."""
        self._on_bid_received.append(callback)
    
    def on_auction_complete(self, callback: Callable) -> None:
        """Register callback for auction complete events."""
        self._on_auction_complete.append(callback)
    
    @property
    def active_auctions(self) -> list[str]:
        """List of active auction IDs."""
        return list(self._sessions.keys())


class AuctionSession:
    """
    An active auction session.
    
    Manages bid collection and evaluation for a single Claim Bundle.
    """
    
    def __init__(
        self,
        auction_id: str,
        claim_bundle: "ClaimBundle",
        timeout_seconds: int,
        evaluator: BidEvaluator,
        registry: Optional["EACRegistry"] = None,
    ):
        """
        Initialize auction session.
        
        Args:
            auction_id: Unique auction ID
            claim_bundle: Target Claim Bundle
            timeout_seconds: Auction timeout
            evaluator: Bid evaluator
            registry: Agent card registry
        """
        self.auction_id = auction_id
        self.claim_bundle = claim_bundle
        self.timeout_seconds = timeout_seconds
        self.evaluator = evaluator
        self.registry = registry
        
        self.started_at = datetime.utcnow()
        self.expires_at = self.started_at + timedelta(seconds=timeout_seconds)
        
        self._bids: list[ChallengerBid] = []
        self._bidders: set[str] = set()
        self._state = AuctionState.OPEN
        self._dfs_scores: dict[str, float] = {}
    
    @property
    def cb_id(self) -> str:
        """Claim Bundle ID."""
        return self.claim_bundle.cb_id
    
    @property
    def state(self) -> AuctionState:
        """Current auction state."""
        return self._state
    
    @property
    def is_expired(self) -> bool:
        """Check if auction has expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def time_remaining(self) -> float:
        """Time remaining in seconds."""
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0.0, remaining)
    
    @property
    def num_bids(self) -> int:
        """Number of bids received."""
        return len(self._bids)
    
    def submit_bid(self, bid: ChallengerBid) -> BidResult:
        """
        Submit a bid to this auction.
        
        Args:
            bid: Bid to submit
            
        Returns:
            BidResult
        """
        if self._state == AuctionState.CLOSED:
            return BidResult(
                accepted=False,
                rejection_reason=BidRejectionReason.AUCTION_CLOSED,
            )
        
        # Validate bid
        is_valid, reason = self.evaluator.validate_bid(
            bid,
            self.claim_bundle,
            existing_bidders=self._bidders,
        )
        
        if not is_valid:
            logger.debug(
                f"Bid from {bid.bidder_agent} rejected: {reason}"
            )
            return BidResult(accepted=False, rejection_reason=reason)
        
        # Accept bid
        self._bids.append(bid)
        self._bidders.add(bid.bidder_agent)
        self._state = AuctionState.BIDDING
        
        # Compute score for tracking
        agent_card = None
        if self.registry:
            agent_card = self.registry.get_card(bid.bidder_agent)
        
        score = self.evaluator.compute_bid_score(
            bid, self.claim_bundle, agent_card
        )
        self._dfs_scores[bid.bidder_agent] = score
        
        logger.debug(
            f"Accepted bid from {bid.bidder_agent}: "
            f"strategy={bid.adversarial_strategy}, score={score:.3f}"
        )
        
        return BidResult(accepted=True, bid=bid)
    
    def close(self, force: bool = False) -> AuctionResult:
        """
        Close auction and determine winner.
        
        Args:
            force: Force close even if timeout not reached
            
        Returns:
            AuctionResult
        """
        if self._state == AuctionState.CLOSED:
            raise RuntimeError("Auction already closed")
        
        if not force and not self.is_expired:
            raise RuntimeError(
                f"Auction not expired, {self.time_remaining:.1f}s remaining"
            )
        
        self._state = AuctionState.EVALUATING
        
        # Rank bids
        agent_cards = {}
        if self.registry:
            for bidder in self._bidders:
                card = self.registry.get_card(bidder)
                if card:
                    agent_cards[bidder] = card
        
        ranked = self.evaluator.rank_bids(
            self._bids,
            self.claim_bundle,
            agent_cards,
        )
        
        # Select winner
        winner_bid = self.evaluator.select_winner(ranked)
        
        # Determine if unchallenged
        unchallenged = winner_bid is None
        
        self._state = AuctionState.CLOSED if winner_bid else AuctionState.UNCHALLENGED
        
        result = AuctionResult(
            auction_id=self.auction_id,
            cb_id=self.cb_id,
            state=self._state,
            winner=winner_bid.bidder_agent if winner_bid else None,
            winning_bid=winner_bid,
            all_bids=self._bids.copy(),
            dfs_scores=self._dfs_scores.copy(),
            started_at=self.started_at,
            completed_at=datetime.utcnow(),
            unchallenged=unchallenged,
        )
        
        if unchallenged:
            logger.warning(
                f"Claim Bundle {self.cb_id} went UNCHALLENGED "
                f"(notable in provenance)"
            )
        
        return result


def create_bid(
    agent_id: str,
    cb_id: str,
    estimated_dfs: float,
    domain_confidence: float = 0.5,
    strategy: str = "general_critique",
    expires_in_seconds: int = 300,
) -> ChallengerBid:
    """
    Helper function to create a ChallengerBid.
    
    Args:
        agent_id: Bidding agent ID
        cb_id: Target Claim Bundle ID
        estimated_dfs: Estimated Dialectical Fitness Score
        domain_confidence: Confidence in domain expertise
        strategy: Adversarial strategy
        expires_in_seconds: Bid expiry time
        
    Returns:
        ChallengerBid
    """
    return ChallengerBid(
        bidder_agent=agent_id,
        cb_id=cb_id,
        estimated_dfs=estimated_dfs,
        domain_confidence=domain_confidence,
        adversarial_strategy=strategy,
        bid_expires=datetime.utcnow() + timedelta(seconds=expires_in_seconds),
    )
