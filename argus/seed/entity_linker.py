"""
Entity Linker — links entities in claims to knowledge graph nodes.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LinkedEntity:
    """An entity linked to a knowledge base."""
    text: str = ""
    entity_type: str = "unknown"  # person, organization, compound, concept
    wikidata_id: Optional[str] = None
    wikipedia_url: Optional[str] = None
    confidence: float = 0.5
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "entity_type": self.entity_type,
            "wikidata_id": self.wikidata_id, "confidence": round(self.confidence, 4),
        }


class WikidataResolver:
    """Resolves entity mentions to Wikidata IDs using web lookup."""

    def __init__(self, cache_size: int = 1000):
        self._cache: dict[str, Optional[LinkedEntity]] = {}
        self._cache_size = cache_size

    def resolve(self, entity_text: str) -> Optional[LinkedEntity]:
        """Resolve entity to Wikidata. Uses cache."""
        if entity_text in self._cache:
            return self._cache[entity_text]

        linked = self._lookup(entity_text)
        if len(self._cache) < self._cache_size:
            self._cache[entity_text] = linked
        return linked

    def _lookup(self, entity_text: str) -> Optional[LinkedEntity]:
        """Attempt Wikidata lookup via API."""
        try:
            import requests
            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "search": entity_text,
                "language": "en",
                "format": "json",
                "limit": 1,
            }
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            results = data.get("search", [])
            if results:
                r = results[0]
                return LinkedEntity(
                    text=entity_text,
                    wikidata_id=r.get("id"),
                    wikipedia_url=r.get("url"),
                    confidence=0.7,
                    aliases=r.get("aliases", []),
                )
        except Exception as e:
            logger.debug(f"Wikidata lookup failed for '{entity_text}': {e}")
        return None


class EntityLinker:
    """
    Links entities in extracted claims to knowledge graph nodes.

    Resolves abbreviations, acronyms, and domain-specific terms.

    Example:
        >>> linker = EntityLinker()
        >>> linked = linker.link_entities("FDA approved drug X for treatment")
        >>> print(linked[0].wikidata_id)
    """

    # Common abbreviation expansions
    ABBREVIATIONS: dict[str, str] = {
        "FDA": "Food and Drug Administration",
        "WHO": "World Health Organization",
        "NIH": "National Institutes of Health",
        "CDC": "Centers for Disease Control",
        "RCT": "randomized controlled trial",
        "AI": "artificial intelligence",
        "ML": "machine learning",
        "GDP": "gross domestic product",
        "EPA": "Environmental Protection Agency",
        "EMA": "European Medicines Agency",
    }

    def __init__(
        self,
        enable_wikidata: bool = False,
        min_entity_length: int = 2,
    ):
        self.enable_wikidata = enable_wikidata
        self.min_entity_length = min_entity_length
        self._resolver = WikidataResolver() if enable_wikidata else None

    def link_entities(
        self,
        text: str,
        claim_entities: Optional[list[str]] = None,
    ) -> list[LinkedEntity]:
        """
        Link entities found in text.

        Args:
            text: Text containing entities
            claim_entities: Pre-extracted entity names

        Returns:
            List of LinkedEntity objects
        """
        entities = claim_entities or self._extract_entities(text)
        linked: list[LinkedEntity] = []

        for entity_text in entities:
            if len(entity_text) < self.min_entity_length:
                continue

            # Check abbreviations
            expanded = self.ABBREVIATIONS.get(entity_text.upper())

            le = LinkedEntity(
                text=entity_text,
                entity_type=self._guess_type(entity_text),
                confidence=0.5,
                aliases=[expanded] if expanded else [],
            )

            # Wikidata resolution
            if self._resolver:
                resolved = self._resolver.resolve(entity_text)
                if resolved:
                    le.wikidata_id = resolved.wikidata_id
                    le.wikipedia_url = resolved.wikipedia_url
                    le.confidence = resolved.confidence
                    le.aliases.extend(resolved.aliases)

            linked.append(le)

        return linked

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entities using capitalization and pattern heuristics."""
        # Find capitalized multi-word sequences
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        # Find all-caps abbreviations
        abbreviations = re.findall(r'\b[A-Z]{2,6}\b', text)
        return list(set(entities + abbreviations))

    @staticmethod
    def _guess_type(entity_text: str) -> str:
        """Guess entity type from text."""
        if entity_text.isupper() and len(entity_text) <= 6:
            return "abbreviation"
        org_keywords = ["institute", "university", "corporation", "agency"]
        if any(kw in entity_text.lower() for kw in org_keywords):
            return "organization"
        return "concept"
