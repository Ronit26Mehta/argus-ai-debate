"""
Embedding Generation for ARGUS.

Provides embedding generation using sentence-transformers
or LLM provider APIs.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, TYPE_CHECKING

import numpy as np

from argus.core.models import Chunk, Embedding

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for chunks.
    
    Uses sentence-transformers for local embedding generation,
    with fallback to LLM provider embeddings.
    
    Example:
        >>> generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        >>> embeddings = generator.embed_chunks(chunks)
        >>> print(f"Dimension: {embeddings[0].dimension}")
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence-transformer model name
            device: Device to use (cpu, cuda, mps)
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        
        self._model: Any = None
        self._dimension: Optional[int] = None
    
    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                kwargs = {}
                if self.device:
                    kwargs["device"] = self.device
                
                self._model = SentenceTransformer(self.model_name, **kwargs)
                self._dimension = self._model.get_sentence_embedding_dimension()
                
                logger.info(
                    f"Loaded embedding model '{self.model_name}' "
                    f"(dim={self._dimension})"
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._dimension or 384
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of shape (n_texts, dimension)
        """
        self._load_model()
        
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as list
        """
        self._load_model()
        
        embedding = self._model.encode(
            query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        
        return embedding.tolist()
    
    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Embedding]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of Embedding objects
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        vectors = self.embed_texts(texts)
        
        # Create Embedding objects
        embeddings = []
        for chunk, vector in zip(chunks, vectors):
            emb = Embedding(
                source_id=chunk.id,
                vector=vector.tolist(),
                model=self.model_name,
            )
            embeddings.append(emb)
        
        logger.debug(
            f"Generated {len(embeddings)} embeddings (dim={self.dimension})"
        )
        
        return embeddings


def generate_embeddings(
    chunks: list[Chunk],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[Embedding]:
    """
    Convenience function to generate embeddings.
    
    Args:
        chunks: List of Chunk objects
        model_name: Embedding model name
        
    Returns:
        List of Embedding objects
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.embed_chunks(chunks)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity (0 to 1 for normalized vectors)
    """
    a = np.array(v1)
    b = np.array(v2)
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


def batch_cosine_similarity(
    query: list[float],
    vectors: list[list[float]],
) -> list[float]:
    """
    Compute cosine similarity between query and multiple vectors.
    
    Args:
        query: Query vector
        vectors: List of vectors to compare
        
    Returns:
        List of similarity scores
    """
    q = np.array(query)
    v = np.array(vectors)
    
    # Normalize query
    q_norm = q / np.linalg.norm(q)
    
    # Normalize vectors
    v_norms = np.linalg.norm(v, axis=1, keepdims=True)
    v_normalized = v / np.maximum(v_norms, 1e-10)
    
    # Dot product
    scores = np.dot(v_normalized, q_norm)
    
    return scores.tolist()
