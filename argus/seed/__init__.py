"""
SEED — Structured Evidence Extraction & Debate-Priming for ARGUS.

End-to-end pipeline: raw document → extracted propositions →
DebatabilityScore ranking → C-DAG skeleton → DebateReadyBundle.

Example:
    >>> from argus.seed import SEEDOrchestrator, SEEDConfig
    >>> seed = SEEDOrchestrator(llm=get_llm('openai'))
    >>> bundle = seed.from_text(article_text)
    >>> print(bundle.ranked_claims[0].debatability_score)
"""

from argus.seed.claim_miner import ClaimMiner, RawClaim, MiningConfig
from argus.seed.entity_linker import EntityLinker, LinkedEntity, WikidataResolver
from argus.seed.debatability_scorer import DebatabilityScorer, BiPolarityRatio, NoveltyQuotient
from argus.seed.evidence_prepopulator import EvidencePrePopulator, CDAGSkeleton, EvidenceFragment
from argus.seed.bundle import DebateReadyBundle, ScoredClaim, PriorEstimate
from argus.seed.orchestrator import SEEDOrchestrator, SEEDConfig, SEEDResult

__all__ = [
    "ClaimMiner", "RawClaim", "MiningConfig",
    "EntityLinker", "LinkedEntity", "WikidataResolver",
    "DebatabilityScorer", "BiPolarityRatio", "NoveltyQuotient",
    "EvidencePrePopulator", "CDAGSkeleton", "EvidenceFragment",
    "DebateReadyBundle", "ScoredClaim", "PriorEstimate",
    "SEEDOrchestrator", "SEEDConfig", "SEEDResult",
]
