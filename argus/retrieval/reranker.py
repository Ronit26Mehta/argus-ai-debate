"""
Cross-Encoder Reranking for ARGUS.

Provides neural reranking using cross-encoder models.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result after reranking."""
    id: str
    text: str
    original_score: float
    rerank_score: float
    rank: int


class CrossEncoderReranker:
    """
    Cross-encoder reranker for neural re-ranking.
    
    Uses a cross-encoder model to score query-document pairs
    for more accurate relevance scoring.
    
    Example:
        >>> reranker = CrossEncoderReranker()
        >>> reranked = reranker.rerank(query, documents, top_k=10)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to use
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        self._model: Any = None
    
    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                kwargs = {}
                if self.device:
                    kwargs["device"] = self.device
                
                self._model = CrossEncoder(self.model_name, **kwargs)
                logger.info(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers with CrossEncoder required. "
                    "Install with: pip install sentence-transformers"
                )
    
    def score_pairs(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """
        Score query-document pairs.
        
        Args:
            query: Query text
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        self._load_model()
        
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        
        return [float(s) for s in scores]
    
    def rerank(
        self,
        query: str,
        documents: list[tuple[str, str, float]],  # (id, text, original_score)
        top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """
        Rerank documents for a query.
        
        Args:
            query: Query text
            documents: List of (id, text, original_score) tuples
            top_k: Number of results to return (all if None)
            
        Returns:
            List of RerankResult sorted by rerank score
        """
        if not documents:
            return []
        
        texts = [d[1] for d in documents]
        scores = self.score_pairs(query, texts)
        
        results = [
            RerankResult(
                id=doc[0],
                text=doc[1],
                original_score=doc[2],
                rerank_score=score,
                rank=0,
            )
            for doc, score in zip(documents, scores)
        ]
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(results):
            r.rank = i + 1
        
        if top_k:
            return results[:top_k]
        
        return results


def rerank_results(
    query: str,
    documents: list[tuple[str, str, float]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None,
) -> list[RerankResult]:
    """
    Convenience function for reranking.
    
    Args:
        query: Query text
        documents: List of (id, text, score) tuples
        model_name: Cross-encoder model
        top_k: Number of results
        
    Returns:
        Reranked results
    """
    reranker = CrossEncoderReranker(model_name=model_name)
    return reranker.rerank(query, documents, top_k)
