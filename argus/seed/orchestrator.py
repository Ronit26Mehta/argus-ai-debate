"""
SEED Orchestrator — end-to-end document-to-debate pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from argus.seed.claim_miner import ClaimMiner, MiningConfig
from argus.seed.entity_linker import EntityLinker
from argus.seed.debatability_scorer import DebatabilityScorer
from argus.seed.evidence_prepopulator import EvidencePrePopulator
from argus.seed.bundle import DebateReadyBundle, ScoredClaim, PriorEstimate

logger = logging.getLogger(__name__)


@dataclass
class SEEDConfig:
    """Configuration for SEED pipeline."""
    max_claims: int = 20
    top_claims: int = 5
    min_debatability: float = 0.4
    enable_entity_linking: bool = False
    prior_estimation: bool = True


@dataclass
class SEEDResult:
    """Result from SEED extraction."""
    bundle: Optional[DebateReadyBundle] = None
    total_claims_extracted: int = 0
    total_claims_scored: int = 0
    top_debatability: float = 0.0


class SEEDOrchestrator:
    """
    End-to-end document-to-debate pipeline.

    Pipeline stages:
        1. Document Ingestion (text, file, or URL)
        2. Claim Mining (LLM-powered extraction)
        3. Entity Linking (optional)
        4. Debatability Scoring
        5. Evidence Pre-population
        6. Bundle Assembly

    Example:
        >>> seed = SEEDOrchestrator()
        >>> bundle = seed.from_text(article_text)
        >>> for claim in bundle.ranked_claims:
        ...     print(f'{claim.text[:80]}  [{claim.debatability_score:.2f}]')
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[SEEDConfig] = None,
    ):
        self.llm = llm
        self.config = config or SEEDConfig()
        self._miner = ClaimMiner(MiningConfig(max_claims=self.config.max_claims))
        self._linker = EntityLinker(enable_wikidata=self.config.enable_entity_linking)
        self._scorer = DebatabilityScorer(min_debatability=self.config.min_debatability)
        self._populator = EvidencePrePopulator()

    def from_text(
        self,
        text: str,
        source_title: str = "Untitled",
    ) -> DebateReadyBundle:
        """
        Process raw text into a DebateReadyBundle.

        Args:
            text: Raw text content
            source_title: Title for the source

        Returns:
            DebateReadyBundle with ranked scored claims
        """
        logger.info(f"SEED processing text ({len(text)} chars)")

        # Stage 1 + 2: Claim Mining
        raw_claims = self._miner.extract_claims(text, source_title)
        logger.info(f"Stage 2: Extracted {len(raw_claims)} raw claims")

        # Stage 3: Entity Linking
        if self.config.enable_entity_linking:
            for claim in raw_claims:
                linked = self._linker.link_entities(claim.text, claim.entities)
                claim.entities = [le.text for le in linked]

        # Chunk the source text for evidence pre-population
        chunks = self._simple_chunk(text)

        # Stage 4: Debatability Scoring + Stage 5: Evidence Pre-population
        bundle = DebateReadyBundle(source=source_title, source_type="text")

        for raw_claim in raw_claims:
            # Score
            score = self._scorer.score_claim(
                claim_text=raw_claim.text,
                source_chunks=chunks[:5],
                total_chunks=len(chunks),
            )

            if score < self.config.min_debatability:
                continue

            # Build CDAG skeleton
            skeleton = self._populator.build_skeleton(
                proposition=raw_claim.text,
                chunks=chunks,
            )

            # Prior estimation
            prior = PriorEstimate(
                value=skeleton.prior_estimate,
                method="evidence_ratio",
                confidence=score,
            ) if self.config.prior_estimation else None

            scored_claim = ScoredClaim(
                text=raw_claim.text,
                claim_type=raw_claim.claim_type,
                debatability_score=score,
                prior_estimate=prior,
                entities=raw_claim.entities,
                cdag_skeleton=skeleton,
            )
            bundle.add_claim(scored_claim)

        # Limit to top claims
        if len(bundle.ranked_claims) > self.config.top_claims:
            bundle._claims = bundle._claims[:self.config.top_claims]

        logger.info(
            f"SEED complete: {bundle.num_claims} debate-ready claims "
            f"(top score: {bundle.top_claim.debatability_score:.3f})"
            if bundle.top_claim else "SEED complete: no claims extracted"
        )

        return bundle

    def from_file(self, file_path: str) -> DebateReadyBundle:
        """Process a file into a DebateReadyBundle."""
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.from_text(text, source_title=path.name)

    def from_url(self, url: str) -> DebateReadyBundle:
        """Fetch URL content and process into DebateReadyBundle."""
        try:
            import requests
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            return self.from_text(text, source_title=url)
        except ImportError:
            raise ImportError("requests and beautifulsoup4 required for URL processing")

    @staticmethod
    def _simple_chunk(text: str, chunk_size: int = 500) -> list[str]:
        """Simple chunking by character count."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks
