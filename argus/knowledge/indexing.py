"""
Hybrid Indexing for ARGUS.

Implements BM25 sparse indexing and FAISS vector indexing
combined into a hybrid retrieval system.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np

from argus.core.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from index search."""
    chunk_id: str
    score: float
    chunk: Optional[Chunk] = None
    dense_score: float = 0.0
    sparse_score: float = 0.0


class BM25Index:
    """
    BM25 sparse text index.
    
    Uses rank-bm25 for sparse retrieval based on term frequency.
    
    Example:
        >>> index = BM25Index()
        >>> index.add_chunks(chunks)
        >>> results = index.search("query text", top_k=10)
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        
        self._bm25: Any = None
        self._chunk_ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._tokenized_corpus: list[list[str]] = []
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization."""
        import re
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add chunks to the index.
        
        Args:
            chunks: List of Chunk objects
        """
        from rank_bm25 import BM25Okapi
        
        for chunk in chunks:
            self._chunk_ids.append(chunk.id)
            self._chunks[chunk.id] = chunk
            self._tokenized_corpus.append(self._tokenize(chunk.text))
        
        # Rebuild index
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )
        
        logger.debug(f"BM25 index now has {len(self._chunk_ids)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Search the index.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        if self._bm25 is None:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self._chunk_ids[idx]
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    score=float(scores[idx]),
                    chunk=self._chunks.get(chunk_id),
                    sparse_score=float(scores[idx]),
                ))
        
        return results
    
    def clear(self) -> None:
        """Clear the index."""
        self._bm25 = None
        self._chunk_ids.clear()
        self._chunks.clear()
        self._tokenized_corpus.clear()


class VectorIndex:
    """
    FAISS vector index for dense retrieval.
    
    Uses FAISS for efficient similarity search on embeddings.
    
    Example:
        >>> index = VectorIndex(dimension=384)
        >>> index.add_vectors(chunk_ids, vectors)
        >>> results = index.search(query_vector, top_k=10)
    """
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
    ):
        """
        Initialize vector index.
        
        Args:
            dimension: Embedding dimension
            index_type: Index type (flat, ivf, hnsw)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        self._index: Any = None
        self._chunk_ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        
        self._init_index()
    
    def _init_index(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )
        
        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dimension, 100,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            self._index = faiss.IndexFlatIP(self.dimension)
    
    def add_vectors(
        self,
        chunk_ids: list[str],
        vectors: list[list[float]],
        chunks: Optional[list[Chunk]] = None,
    ) -> None:
        """
        Add vectors to the index.
        
        Args:
            chunk_ids: List of chunk IDs
            vectors: List of embedding vectors
            chunks: Optional Chunk objects for reference
        """
        import faiss
        
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must have same length")
        
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Normalize for inner product (cosine similarity)
        faiss.normalize_L2(vectors_array)
        
        # Train if needed (IVF)
        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            self._index.train(vectors_array)
        
        # Add to index
        self._index.add(vectors_array)
        
        # Store mappings
        for i, chunk_id in enumerate(chunk_ids):
            self._chunk_ids.append(chunk_id)
            if chunks and i < len(chunks):
                self._chunks[chunk_id] = chunks[i]
        
        logger.debug(f"Vector index now has {len(self._chunk_ids)} vectors")
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Search the index.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        import faiss
        
        if len(self._chunk_ids) == 0:
            return []
        
        query_array = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        distances, indices = self._index.search(query_array, top_k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._chunk_ids):
                chunk_id = self._chunk_ids[idx]
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    chunk=self._chunks.get(chunk_id),
                    dense_score=float(score),
                ))
        
        return results
    
    def clear(self) -> None:
        """Clear the index."""
        self._init_index()
        self._chunk_ids.clear()
        self._chunks.clear()


class HybridIndex:
    """
    Hybrid BM25 + Vector index.
    
    Combines sparse (BM25) and dense (vector) retrieval
    with configurable mixing parameter.
    
    Example:
        >>> index = HybridIndex(dimension=384, lambda_param=0.7)
        >>> index.add_chunks(chunks, embeddings)
        >>> results = index.search(query, query_embedding, top_k=10)
    """
    
    def __init__(
        self,
        dimension: int = 384,
        lambda_param: float = 0.7,
    ):
        """
        Initialize hybrid index.
        
        Args:
            dimension: Embedding dimension
            lambda_param: Mix parameter (0=sparse, 1=dense)
        """
        self.dimension = dimension
        self.lambda_param = lambda_param
        
        self.bm25_index = BM25Index()
        self.vector_index = VectorIndex(dimension=dimension)
        
        self._chunks: dict[str, Chunk] = {}
    
    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Add chunks with their embeddings.
        
        Args:
            chunks: List of Chunk objects
            vectors: List of embedding vectors (same order)
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have same length")
        
        chunk_ids = [c.id for c in chunks]
        
        # Add to both indexes
        self.bm25_index.add_chunks(chunks)
        self.vector_index.add_vectors(chunk_ids, vectors, chunks)
        
        # Store chunks
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        
        logger.debug(f"Hybrid index now has {len(self._chunks)} chunks")
    
    def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        rerank_k: int = 50,
    ) -> list[SearchResult]:
        """
        Hybrid search combining BM25 and vector similarity.
        
        Args:
            query: Query text
            query_vector: Query embedding
            top_k: Number of final results
            rerank_k: Candidates to fetch from each index
            
        Returns:
            List of SearchResult sorted by combined score
        """
        # Get results from both indexes
        bm25_results = self.bm25_index.search(query, rerank_k)
        vector_results = self.vector_index.search(query_vector, rerank_k)
        
        # Normalize scores
        bm25_scores = {r.chunk_id: r.score for r in bm25_results}
        vector_scores = {r.chunk_id: r.score for r in vector_results}
        
        # Min-max normalization
        def normalize(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            min_s = min(scores.values())
            max_s = max(scores.values())
            if max_s == min_s:
                return {k: 1.0 for k in scores}
            return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}
        
        norm_bm25 = normalize(bm25_scores)
        norm_vector = normalize(vector_scores)
        
        # Combine scores
        all_ids = set(norm_bm25.keys()) | set(norm_vector.keys())
        combined_scores: dict[str, tuple[float, float, float]] = {}
        
        for chunk_id in all_ids:
            sparse = norm_bm25.get(chunk_id, 0.0)
            dense = norm_vector.get(chunk_id, 0.0)
            combined = (1 - self.lambda_param) * sparse + self.lambda_param * dense
            combined_scores[chunk_id] = (combined, sparse, dense)
        
        # Sort and return top-k
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x][0],
            reverse=True,
        )[:top_k]
        
        results = []
        for chunk_id in sorted_ids:
            combined, sparse, dense = combined_scores[chunk_id]
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=combined,
                chunk=self._chunks.get(chunk_id),
                dense_score=dense,
                sparse_score=sparse,
            ))
        
        return results
    
    def clear(self) -> None:
        """Clear all indexes."""
        self.bm25_index.clear()
        self.vector_index.clear()
        self._chunks.clear()
