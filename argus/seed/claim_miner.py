"""
Claim Miner — LLM-powered extraction of candidate propositions from raw text.

Identifies declarative claims, quantified assertions, and comparative
statements from unstructured documents.
"""

from __future__ import annotations

import re
import uuid
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MiningConfig:
    """Configuration for claim mining."""
    max_claims: int = 20
    min_claim_length: int = 15
    max_claim_length: int = 300
    include_quantified: bool = True
    include_comparative: bool = True
    include_causal: bool = True
    confidence_threshold: float = 0.3


@dataclass
class RawClaim:
    """
    A raw extracted claim from a document.

    Attributes:
        claim_id: Unique claim identifier
        text: Claim text
        source_text: Surrounding context from source
        claim_type: Type (declarative, quantified, comparative, causal)
        extraction_confidence: LLM confidence in extraction quality
        source_section: Section heading from source document
        char_offset: Character offset in source document
        entities: Extracted entity names
    """
    claim_id: str = field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:10]}")
    text: str = ""
    source_text: str = ""
    claim_type: str = "declarative"
    extraction_confidence: float = 0.5
    source_section: str = ""
    char_offset: int = 0
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "extraction_confidence": round(self.extraction_confidence, 4),
            "source_section": self.source_section,
            "entities": self.entities,
        }


class ClaimMiner:
    """
    Extracts candidate propositions from raw text using pattern matching
    and heuristic analysis.

    Pipeline:
        1. Text segmentation into sentences/paragraphs
        2. Claim pattern detection (quantified, comparative, causal)
        3. Confidence scoring based on linguistic markers
        4. Deduplication and ranking

    Example:
        >>> miner = ClaimMiner()
        >>> claims = miner.extract_claims("The drug reduced mortality by 25%...")
        >>> print(claims[0].text)
    """

    # Quantified claim patterns
    QUANTIFIED_PATTERNS = [
        r'\b(\d+\.?\d*)\s*(%|percent)',
        r'\b(increase|decrease|reduce|improve|decline)\w*\s+by\s+(\d+)',
        r'\b(more|less|fewer|greater|higher|lower)\s+than\s+(\d+)',
        r'\bp\s*[<>]\s*0\.\d+',
        r'\b\d+\s*-\s*fold\b',
    ]

    # Comparative patterns
    COMPARATIVE_PATTERNS = [
        r'\b(compared\s+to|relative\s+to|versus|vs\.?)\b',
        r'\b(outperform|surpass|exceed|inferior|superior)\b',
        r'\b(more|less)\s+effective\b',
        r'\b(better|worse)\s+than\b',
    ]

    # Causal patterns
    CAUSAL_PATTERNS = [
        r'\b(cause|lead\s+to|result\s+in|due\s+to)\b',
        r'\b(because|therefore|consequently|hence)\b',
        r'\b(affect|impact|influence|determine)\w*\b',
        r'\b(associated\s+with|correlated\s+with)\b',
    ]

    # Assertion markers
    ASSERTION_MARKERS = [
        r'\b(show|demonstrate|indicate|suggest|reveal|confirm|prove)\w*\s+that\b',
        r'\b(found|concluded|determined|observed)\s+that\b',
        r'\b(evidence\s+suggests?|data\s+shows?)\b',
        r'\b(is|are|was|were)\s+(effective|ineffective|significant|beneficial)\b',
    ]

    def __init__(self, config: Optional[MiningConfig] = None):
        self.config = config or MiningConfig()

    def extract_claims(
        self,
        text: str,
        source_title: str = "",
    ) -> list[RawClaim]:
        """
        Extract candidate claims from raw text.

        Args:
            text: Raw text content
            source_title: Title of the source document

        Returns:
            List of extracted RawClaim objects, sorted by confidence
        """
        sentences = self._segment_text(text)
        claims: list[RawClaim] = []
        seen_texts: set[str] = set()

        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < self.config.min_claim_length:
                continue
            if len(sentence) > self.config.max_claim_length:
                continue

            claim_type, confidence = self._classify_sentence(sentence)
            if confidence < self.config.confidence_threshold:
                continue

            # Dedup by normalized text
            normalized = sentence.strip().lower()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            entities = self._extract_entities(sentence)

            claim = RawClaim(
                text=sentence.strip(),
                source_text=self._get_context(sentences, i),
                claim_type=claim_type,
                extraction_confidence=confidence,
                source_section=source_title,
                char_offset=text.find(sentence),
                entities=entities,
            )
            claims.append(claim)

        # Sort by confidence and limit
        claims.sort(key=lambda c: c.extraction_confidence, reverse=True)
        claims = claims[:self.config.max_claims]

        logger.info(f"Extracted {len(claims)} claims from {len(sentences)} sentences")
        return claims

    def _segment_text(self, text: str) -> list[str]:
        """Segment text into sentences."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _classify_sentence(self, sentence: str) -> tuple[str, float]:
        """Classify sentence type and compute extraction confidence."""
        confidence = 0.2  # Base confidence

        # Check for quantified claims
        if self.config.include_quantified:
            for pattern in self.QUANTIFIED_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    confidence += 0.25
                    claim_type = "quantified"
                    break
            else:
                claim_type = "declarative"
        else:
            claim_type = "declarative"

        # Check for comparative claims
        if self.config.include_comparative:
            for pattern in self.COMPARATIVE_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    confidence += 0.15
                    if claim_type == "declarative":
                        claim_type = "comparative"
                    break

        # Check for causal claims
        if self.config.include_causal:
            for pattern in self.CAUSAL_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    confidence += 0.15
                    if claim_type == "declarative":
                        claim_type = "causal"
                    break

        # Check for assertion markers
        for pattern in self.ASSERTION_MARKERS:
            if re.search(pattern, sentence, re.IGNORECASE):
                confidence += 0.2
                break

        # Length bonus (medium-length claims are better)
        length = len(sentence.split())
        if 10 <= length <= 30:
            confidence += 0.05

        return claim_type, min(1.0, confidence)

    def _extract_entities(self, sentence: str) -> list[str]:
        """Extract named entities using simple capitalization heuristic."""
        words = sentence.split()
        entities = []
        current_entity = []

        for word in words:
            clean = word.strip('.,;:!?()[]"\'')
            if clean and clean[0].isupper() and not words.index(word) == 0:
                current_entity.append(clean)
            else:
                if current_entity:
                    entity = " ".join(current_entity)
                    if len(entity) > 2:
                        entities.append(entity)
                    current_entity = []

        if current_entity:
            entity = " ".join(current_entity)
            if len(entity) > 2:
                entities.append(entity)

        return entities

    def _get_context(self, sentences: list[str], index: int, window: int = 1) -> str:
        """Get surrounding context for a sentence."""
        start = max(0, index - window)
        end = min(len(sentences), index + window + 1)
        return " ".join(sentences[start:end])
