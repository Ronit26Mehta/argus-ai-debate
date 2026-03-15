"""
Logical Relationship Classifier — classifies relationships between propositions.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class RelationshipType(str, Enum):
    """Types of logical relationships between propositions."""
    NECESSARY = "necessary"       # P(parent) requires P(child) = TRUE
    SUFFICIENT = "sufficient"     # P(child) = TRUE alone guarantees P(parent)
    CONTRIBUTING = "contributing" # P(child) increases P(parent) but doesn't guarantee it
    INDEPENDENT = "independent"   # No logical dependency


class LogicalRelationshipClassifier:
    """
    Classifies logical relationships between parent and child propositions.

    Uses linguistic markers and structural patterns to infer relationships.
    """

    NECESSARY_MARKERS = [
        r'\brequire', r'\bneed', r'\bmust\b', r'\bessential',
        r'\bprerequisite', r'\bif\b.*\bthen\b', r'\bnecessary',
        r'\bdepend', r'\bcondition',
    ]

    SUFFICIENT_MARKERS = [
        r'\bsufficient', r'\benough', r'\bguarantee', r'\bensure',
        r'\bor\b', r'\beither\b', r'\balternative',
    ]

    INDEPENDENT_MARKERS = [
        r'\bunrelated', r'\bseparately\b', r'\bindependent',
        r'\bregardless\b', r'\birrespective\b',
    ]

    def classify(
        self,
        parent_text: str,
        child_text: str,
        context: str = "",
    ) -> RelationshipType:
        """
        Classify the relationship between parent and child.

        Args:
            parent_text: Parent proposition text
            child_text: Child proposition text
            context: Additional context

        Returns:
            RelationshipType
        """
        combined = f"{parent_text} {child_text} {context}".lower()

        for pattern in self.NECESSARY_MARKERS:
            if re.search(pattern, combined):
                return RelationshipType.NECESSARY

        for pattern in self.SUFFICIENT_MARKERS:
            if re.search(pattern, combined):
                return RelationshipType.SUFFICIENT

        for pattern in self.INDEPENDENT_MARKERS:
            if re.search(pattern, combined):
                return RelationshipType.INDEPENDENT

        return RelationshipType.CONTRIBUTING

    def classify_batch(
        self,
        parent_text: str,
        children_texts: list[str],
    ) -> list[RelationshipType]:
        """Classify relationships for multiple children."""
        return [
            self.classify(parent_text, child) for child in children_texts
        ]
