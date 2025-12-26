"""
Knowledge and Data Layer for ARGUS.

This module provides document ingestion, chunking, embedding,
and hybrid indexing capabilities.
"""

from argus.knowledge.ingestion import (
    DocumentLoader,
    load_document,
    load_pdf,
    load_html,
    load_text,
)
from argus.knowledge.chunking import (
    Chunker,
    chunk_document,
    ChunkingStrategy,
)
from argus.knowledge.embeddings import (
    EmbeddingGenerator,
    generate_embeddings,
)
from argus.knowledge.indexing import (
    HybridIndex,
    BM25Index,
    VectorIndex,
)

__all__ = [
    # Ingestion
    "DocumentLoader",
    "load_document",
    "load_pdf",
    "load_html",
    "load_text",
    # Chunking
    "Chunker",
    "chunk_document",
    "ChunkingStrategy",
    # Embeddings
    "EmbeddingGenerator",
    "generate_embeddings",
    # Indexing
    "HybridIndex",
    "BM25Index",
    "VectorIndex",
]
