"""
Retrieval and Evidence Engineering for ARGUS.

Provides hybrid retrieval with reranking and 
cite & critique prompting for evidence extraction.
"""

from argus.retrieval.hybrid import (
    HybridRetriever,
    RetrievalResult,
)
from argus.retrieval.reranker import (
    CrossEncoderReranker,
    rerank_results,
)
from argus.retrieval.cite_critique import (
    cite_and_critique,
    extract_claims,
    CiteResult,
)

__all__ = [
    # Hybrid Retrieval
    "HybridRetriever",
    "RetrievalResult",
    # Reranking
    "CrossEncoderReranker",
    "rerank_results",
    # Cite & Critique
    "cite_and_critique",
    "extract_claims",
    "CiteResult",
]
