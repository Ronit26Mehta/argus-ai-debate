"""
CRUX - Claim-Routed Uncertainty eXchange Protocol.

Version 1.0

A novel inter-agent communication protocol for debate-native, Bayesian
multi-agent reasoning systems. CRUX treats epistemic state — beliefs,
uncertainty distributions, argument lineage, and credibility — as
first-class citizens of inter-agent communication.

Seven Core Primitives:
    1. Epistemic Agent Card (EAC) - Agent identity with calibration metadata
    2. Claim Bundle (CB) - Atomic epistemic unit with uncertainty distribution
    3. Dialectical Routing (DR) - Adversarial-aware agent selection
    4. Belief Reconciliation Protocol (BRP) - Merging contradicting claims
    5. Credibility Ledger (CL) - Statistical trust layer
    6. Epistemic Dead Reckoning (EDR) - Reconnection sync protocol
    7. Challenger Auction (CA) - Best challenger selection

Quick Start:
    >>> from argus.crux import CRUXOrchestrator, ClaimBundle, CredibilityLedger
    >>> 
    >>> # Wrap existing ARGUS orchestrator with CRUX
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
    >>> print(result.reconciled_cb.posterior)

Protocol Design:
    - Extends ARGUS C-DAG with confidence distributions
    - Integrates with PROV-O ledger for credibility tracking
    - Supports adversarial routing via Dialectical Fitness Scores
    - Enables belief reconciliation for contradicting claims
"""

__version__ = "1.0.0"
__protocol_name__ = "CRUX"
__full_name__ = "Claim-Routed Uncertainty eXchange Protocol"

# Core models
from argus.crux.models import (
    BetaDistribution,
    ConfidenceDistribution,
    ChallengerBid,
    BRPResolution,
    EDRDelta,
    CRUXConfig,
)

# Epistemic Agent Card
from argus.crux.agent_card import (
    EpistemicAgentCard,
    AgentCalibration,
    AgentCapabilities,
)

# Claim Bundle
from argus.crux.claim_bundle import (
    ClaimBundle,
    ClaimBundleFactory,
    merge_claim_bundles,
)

# Dialectical Routing
from argus.crux.routing import (
    DialecticalRouter,
    DialecticalFitnessScore,
    compute_dfs,
)

# Belief Reconciliation Protocol
from argus.crux.brp import (
    BeliefReconciliationProtocol,
    BRPSession,
    BRPState,
    ReconciliationResult,
)

# Credibility Ledger
from argus.crux.ledger import (
    CredibilityLedger,
    CredibilityEntry,
    CredibilityUpdate,
)

# Epistemic Dead Reckoning
from argus.crux.edr import (
    EpistemicDeadReckoning,
    EDRSynchronizer,
    EDRCheckpoint,
)

# Challenger Auction
from argus.crux.auction import (
    ChallengerAuction,
    AuctionResult,
    BidEvaluator,
)

# CRUX Orchestrator
from argus.crux.orchestrator import (
    CRUXOrchestrator,
    CRUXDebateResult,
    CRUXSession,
)

# Visualization
from argus.crux.visualization import (
    plot_crux_debate_flow,
    plot_credibility_evolution,
    plot_brp_merge,
    plot_dfs_heatmap,
    plot_auction_results,
    create_crux_dashboard,
    export_debate_static,
)

__all__ = [
    # Version info
    "__version__",
    "__protocol_name__",
    "__full_name__",
    # Core models
    "BetaDistribution",
    "ConfidenceDistribution",
    "ChallengerBid",
    "BRPResolution",
    "EDRDelta",
    "CRUXConfig",
    # Agent Card
    "EpistemicAgentCard",
    "AgentCalibration",
    "AgentCapabilities",
    # Claim Bundle
    "ClaimBundle",
    "ClaimBundleFactory",
    "merge_claim_bundles",
    # Routing
    "DialecticalRouter",
    "DialecticalFitnessScore",
    "compute_dfs",
    # BRP
    "BeliefReconciliationProtocol",
    "BRPSession",
    "BRPState",
    "ReconciliationResult",
    # Credibility
    "CredibilityLedger",
    "CredibilityEntry",
    "CredibilityUpdate",
    # EDR
    "EpistemicDeadReckoning",
    "EDRSynchronizer",
    "EDRCheckpoint",
    # Auction
    "ChallengerAuction",
    "AuctionResult",
    "BidEvaluator",
    # Orchestrator
    "CRUXOrchestrator",
    "CRUXDebateResult",
    "CRUXSession",
    # Visualization
    "plot_crux_debate_flow",
    "plot_credibility_evolution",
    "plot_brp_merge",
    "plot_dfs_heatmap",
    "plot_auction_results",
    "create_crux_dashboard",
    "export_debate_static",
]
