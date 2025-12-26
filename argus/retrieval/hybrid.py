"""
Hybrid Retriever for ARGUS.

Combines sparse (BM25) and dense (vector) retrieval
with configurable fusion and optional reranking.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from argus.core.models import Chunk
from argus.knowledge.embeddings import EmbeddingGenerator
from argus.knowledge.indexing import HybridIndex, SearchResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Result from hybrid retrieval.
    
    Attributes:
        chunk_id: Source chunk ID
        chunk: Source chunk object
        score: Combined relevance score
        dense_score: Vector similarity score
        sparse_score: BM25 score
        rerank_score: Cross-encoder score if reranked
        rank: Final rank position
    """
    chunk_id: str
    chunk: Optional[Chunk] = None
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: Optional[float] = None
    rank: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.chunk.text[:200] if self.chunk else "",
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rerank_score": self.rerank_score,
            "rank": self.rank,
        }


class HybridRetriever:
    """
    Hybrid retrieval engine combining BM25 and vector search.
    
    Implements the retrieval layer from ARGUS architecture:
    - Dense retrieval via FAISS
    - Sparse retrieval via BM25
    - Reciprocal rank fusion (RRF) or weighted combination
    - Optional cross-encoder reranking
    
    Example:
        >>> retriever = HybridRetriever(embedding_model="all-MiniLM-L6-v2")
        >>> retriever.index_chunks(chunks)
        >>> results = retriever.retrieve("query text", top_k=10)
        >>> for r in results:
        ...     print(f"{r.rank}: {r.chunk.text[:50]}... [{r.score:.3f}]")
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        lambda_param: float = 0.7,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_model: Sentence-transformer model
            lambda_param: Weight for dense vs sparse (0=sparse, 1=dense)
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Cross-encoder model for reranking
        """
        self.embedding_model = embedding_model
        self.lambda_param = lambda_param
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        
        self._embedding_gen: Optional[EmbeddingGenerator] = None
        self._index: Optional[HybridIndex] = None
        self._reranker: Any = None
    
    def _init_components(self) -> None:
        """Lazy initialize components."""
        if self._embedding_gen is None:
            self._embedding_gen = EmbeddingGenerator(model_name=self.embedding_model)
        
        if self._index is None:
            dim = self._embedding_gen.dimension
            self._index = HybridIndex(dimension=dim, lambda_param=self.lambda_param)
    
    def _init_reranker(self) -> None:
        """Lazy initialize reranker."""
        if self._reranker is None and self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.reranker_model)
                logger.info(f"Loaded reranker: {self.reranker_model}")
            except ImportError:
                logger.warning("CrossEncoder not available, reranking disabled")
                self.use_reranker = False
    
    def index_chunks(self, chunks: list[Chunk]) -> int:
        """
        Index chunks for retrieval.
        
        Args:
            chunks: Chunks to index
            
        Returns:
            Number of chunks indexed
        """
        self._init_components()
        
        if not chunks:
            return 0
        
        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self._embedding_gen.embed_texts(texts)
        
        # Add to index
        self._index.add_chunks(chunks, embeddings.tolist())
        
        logger.info(f"Indexed {len(chunks)} chunks")
        return len(chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 50,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            rerank_k: Candidates to fetch for reranking
            
        Returns:
            List of RetrievalResult sorted by score
        """
        self._init_components()
        
        # Generate query embedding
        query_embedding = self._embedding_gen.embed_query(query)
        
        # Hybrid search
        fetch_k = rerank_k if self.use_reranker else top_k
        search_results = self._index.search(query, query_embedding, fetch_k)
        
        # Convert to RetrievalResult
        results = [
            RetrievalResult(
                chunk_id=sr.chunk_id,
                chunk=sr.chunk,
                score=sr.score,
                dense_score=sr.dense_score,
                sparse_score=sr.sparse_score,
            )
            for sr in search_results
        ]
        
        # Rerank if enabled
        if self.use_reranker and len(results) > 0:
            results = self._rerank(query, results)
        
        # Assign ranks and truncate
        for i, r in enumerate(results[:top_k]):
            r.rank = i + 1
        
        return results[:top_k]
    
    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder."""
        self._init_reranker()
        
        if not self._reranker:
            return results
        
        # Prepare pairs
        pairs = [
            [query, r.chunk.text if r.chunk else ""]
            for r in results
        ]
        
        # Score with cross-encoder
        scores = self._reranker.predict(pairs)
        
        # Update scores
        for r, score in zip(results, scores):
            r.rerank_score = float(score)
            r.score = float(score)  # Use rerank score as primary
        
        # Sort by rerank score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def retrieve_with_rrf(
        self,
        query: str,
        top_k: int = 10,
        k_constant: int = 60,
    ) -> list[RetrievalResult]:
        """
        Retrieve using Reciprocal Rank Fusion.
        
        RRF(d) = Σ 1 / (k + rank(d))
        
        Args:
            query: Query text
            top_k: Number of results
            k_constant: RRF constant (default 60)
            
        Returns:
            List of RetrievalResult
        """
        self._init_components()
        
        query_embedding = self._embedding_gen.embed_query(query)
        
        # Get results from both methods
        bm25_results = self._index.bm25_index.search(query, top_k * 2)
        vector_results = self._index.vector_index.search(query_embedding, top_k * 2)
        
        # Build rank mappings
        bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}
        vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}
        
        # Compute RRF scores
        all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        rrf_scores: dict[str, float] = {}
        
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 1 / (k_constant + bm25_ranks[chunk_id])
            if chunk_id in vector_ranks:
                score += 1 / (k_constant + vector_ranks[chunk_id])
            rrf_scores[chunk_id] = score
        
        # Sort and create results
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for i, chunk_id in enumerate(sorted_ids[:top_k]):
            chunk = self._index._chunks.get(chunk_id)
            bm25_r = next((r for r in bm25_results if r.chunk_id == chunk_id), None)
            vec_r = next((r for r in vector_results if r.chunk_id == chunk_id), None)
            
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                chunk=chunk,
                score=rrf_scores[chunk_id],
                dense_score=vec_r.score if vec_r else 0,
                sparse_score=bm25_r.score if bm25_r else 0,
                rank=i + 1,
            ))
        
        return results
    
    def clear(self) -> None:
        """Clear the index."""
        if self._index:
            self._index.clear()
    
    @property
    def num_indexed(self) -> int:
        """Get number of indexed chunks."""
        if self._index:
            return len(self._index._chunks)
        return 0
