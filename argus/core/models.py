"""
Core Pydantic data models for ARGUS.

This module defines the fundamental data structures used throughout the ARGUS system:
    - Document: Raw ingested documents with metadata
    - Chunk: Text segments for embedding and retrieval
    - Embedding: Vector representations of text
    - Claim: Extracted claims with citations
    - NodeBase: Base class for C-DAG nodes

All models are immutable (frozen) and support JSON serialization for provenance logging.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, computed_field


def generate_uuid() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def compute_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of content.
    
    Args:
        content: Text content to hash
        algorithm: Hash algorithm (sha256, sha384, sha512)
        
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


class SourceType(str, Enum):
    """Type of document source."""
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class Document(BaseModel):
    """
    Represents an ingested document with metadata.
    
    Documents are the raw source material that gets chunked and embedded.
    Each document maintains full provenance information for audit.
    
    Attributes:
        id: Unique document identifier
        url: Source URL or file path
        title: Document title
        content: Full text content
        source_type: Type of source (pdf, html, etc.)
        language: ISO 639-1 language code
        checksum: Content hash for deduplication
        metadata: Additional source-specific metadata
        ingested_at: Timestamp of ingestion
        
    Example:
        >>> doc = Document(
        ...     url="https://example.com/paper.pdf",
        ...     title="Research Paper",
        ...     content="Full text...",
        ...     source_type=SourceType.PDF,
        ... )
        >>> print(doc.id)
        'doc_abc123...'
    """
    
    model_config = {"frozen": True}
    
    id: str = Field(
        default_factory=lambda: f"doc_{generate_uuid()[:12]}",
        description="Unique document identifier",
    )
    
    url: str = Field(
        description="Source URL or file path",
    )
    
    title: str = Field(
        default="",
        description="Document title",
    )
    
    content: str = Field(
        description="Full text content of the document",
    )
    
    source_type: SourceType = Field(
        default=SourceType.UNKNOWN,
        description="Type of source document",
    )
    
    language: str = Field(
        default="en",
        description="ISO 639-1 language code",
    )
    
    checksum: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of content for deduplication",
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific metadata",
    )
    
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of ingestion",
    )
    
    @computed_field
    @property
    def content_hash(self) -> str:
        """Compute content hash if not provided."""
        return self.checksum or compute_hash(self.content)
    
    @computed_field
    @property
    def word_count(self) -> int:
        """Count words in document content."""
        return len(self.content.split())
    
    def __hash__(self) -> int:
        """Hash by document ID for set operations."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on document ID."""
        if not isinstance(other, Document):
            return False
        return self.id == other.id


class Chunk(BaseModel):
    """
    Represents a segment of a document for embedding and retrieval.
    
    Chunks are the atomic units for similarity search and evidence extraction.
    Each chunk maintains a reference to its source document and position.
    
    Attributes:
        id: Unique chunk identifier
        doc_id: Parent document ID
        text: Chunk text content
        start_char: Start character position in document
        end_char: End character position in document
        chunk_index: Index of chunk within document
        token_count: Estimated token count
        metadata: Additional chunk-specific metadata
        
    Example:
        >>> chunk = Chunk(
        ...     doc_id="doc_abc123",
        ...     text="This is a chunk of text...",
        ...     start_char=0,
        ...     end_char=500,
        ...     chunk_index=0,
        ... )
    """
    
    model_config = {"frozen": True}
    
    id: str = Field(
        default_factory=lambda: f"chunk_{generate_uuid()[:12]}",
        description="Unique chunk identifier",
    )
    
    doc_id: str = Field(
        description="Parent document ID",
    )
    
    text: str = Field(
        description="Chunk text content",
    )
    
    start_char: int = Field(
        ge=0,
        description="Start character position in document",
    )
    
    end_char: int = Field(
        ge=0,
        description="End character position in document",
    )
    
    chunk_index: int = Field(
        ge=0,
        description="Index of chunk within document",
    )
    
    token_count: Optional[int] = Field(
        default=None,
        description="Estimated token count",
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk-specific metadata",
    )
    
    @computed_field
    @property
    def span(self) -> tuple[int, int]:
        """Character span tuple."""
        return (self.start_char, self.end_char)
    
    @computed_field
    @property
    def length(self) -> int:
        """Character length of chunk."""
        return self.end_char - self.start_char
    
    def __hash__(self) -> int:
        """Hash by chunk ID."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on chunk ID."""
        if not isinstance(other, Chunk):
            return False
        return self.id == other.id


class Embedding(BaseModel):
    """
    Represents a vector embedding for a chunk or query.
    
    Embeddings enable semantic similarity search in the hybrid retrieval system.
    
    Attributes:
        id: Unique embedding identifier
        source_id: ID of the source (chunk_id or query hash)
        vector: Embedding vector as list of floats
        model: Model used to generate embedding
        dimension: Vector dimension
        
    Example:
        >>> embedding = Embedding(
        ...     source_id="chunk_abc123",
        ...     vector=[0.1, 0.2, 0.3, ...],
        ...     model="all-MiniLM-L6-v2",
        ... )
    """
    
    model_config = {"frozen": True}
    
    id: str = Field(
        default_factory=lambda: f"emb_{generate_uuid()[:12]}",
        description="Unique embedding identifier",
    )
    
    source_id: str = Field(
        description="ID of the embedded source",
    )
    
    vector: list[float] = Field(
        description="Embedding vector",
    )
    
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model used for embedding",
    )
    
    @computed_field
    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return len(self.vector)
    
    def __hash__(self) -> int:
        """Hash by embedding ID."""
        return hash(self.id)


class CitationType(str, Enum):
    """Type of citation."""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    REFERENCE = "reference"
    DATA = "data"


class Citation(BaseModel):
    """
    Represents a citation linking a claim to its source.
    
    Citations provide provenance for claims and enable verification.
    
    Attributes:
        doc_id: Source document ID
        chunk_id: Source chunk ID
        quote: Direct quote from source (if applicable)
        start_char: Quote start position in chunk
        end_char: Quote end position in chunk
        citation_type: Type of citation
    """
    
    model_config = {"frozen": True}
    
    doc_id: str = Field(
        description="Source document ID",
    )
    
    chunk_id: str = Field(
        description="Source chunk ID",
    )
    
    quote: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Direct quote from source",
    )
    
    start_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="Quote start position in chunk",
    )
    
    end_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="Quote end position in chunk",
    )
    
    citation_type: CitationType = Field(
        default=CitationType.REFERENCE,
        description="Type of citation",
    )


class Claim(BaseModel):
    """
    Represents an extracted claim with citations and confidence.
    
    Claims are the building blocks for evidence in the C-DAG.
    Each claim must have at least one citation for provenance.
    
    Attributes:
        id: Unique claim identifier
        text: Claim text content
        citations: List of supporting citations
        confidence: Confidence score (0-1)
        self_critique: Self-verification notes
        extracted_at: Timestamp of extraction
        
    Example:
        >>> claim = Claim(
        ...     text="Drug X reduces biomarker Y by 25%",
        ...     citations=[Citation(doc_id="doc_abc", chunk_id="chunk_123")],
        ...     confidence=0.85,
        ... )
    """
    
    model_config = {"frozen": True}
    
    id: str = Field(
        default_factory=lambda: f"claim_{generate_uuid()[:12]}",
        description="Unique claim identifier",
    )
    
    text: str = Field(
        min_length=1,
        description="Claim text content",
    )
    
    citations: list[Citation] = Field(
        default_factory=list,
        min_length=0,  # Allow empty during construction
        description="Supporting citations",
    )
    
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)",
    )
    
    self_critique: Optional[str] = Field(
        default=None,
        description="Self-verification notes from LLM",
    )
    
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of extraction",
    )
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is properly bounded."""
        return max(0.0, min(1.0, v))
    
    def __hash__(self) -> int:
        """Hash by claim ID."""
        return hash(self.id)


class NodeType(str, Enum):
    """Types of nodes in the C-DAG."""
    PROPOSITION = "proposition"
    EVIDENCE = "evidence"
    REBUTTAL = "rebuttal"
    FINDING = "finding"
    ASSUMPTION = "assumption"


class NodeBase(BaseModel):
    """
    Base class for all C-DAG nodes.
    
    Provides common attributes and methods for graph nodes.
    All node types inherit from this base.
    
    Attributes:
        id: Unique node identifier
        node_type: Type of node (proposition, evidence, etc.)
        text: Node content text
        confidence: Confidence score (0-1)
        weight: Node weight for scoring
        created_at: Creation timestamp
        metadata: Additional node-specific metadata
    """
    
    model_config = {"frozen": True}
    
    id: str = Field(
        default_factory=lambda: f"node_{generate_uuid()[:12]}",
        description="Unique node identifier",
    )
    
    node_type: NodeType = Field(
        description="Type of node",
    )
    
    text: str = Field(
        min_length=1,
        description="Node content text",
    )
    
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)",
    )
    
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Node weight for scoring",
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node-specific metadata",
    )
    
    def __hash__(self) -> int:
        """Hash by node ID."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on node ID."""
        if not isinstance(other, NodeBase):
            return False
        return self.id == other.id
    
    def to_prov_dict(self) -> dict[str, Any]:
        """
        Convert node to PROV-O compatible dictionary.
        
        Returns:
            Dictionary with PROV-O entity attributes
        """
        return {
            "prov:type": f"argus:{self.node_type.value}",
            "prov:id": self.id,
            "prov:label": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "argus:confidence": self.confidence,
            "argus:weight": self.weight,
            "prov:generatedAtTime": self.created_at.isoformat(),
        }


class ScoredItem(BaseModel):
    """
    Generic scored item for retrieval results.
    
    Wraps any item with relevance scores from retrieval.
    
    Attributes:
        item_id: ID of the item
        score: Combined relevance score
        dense_score: Dense (embedding) similarity score
        sparse_score: Sparse (BM25) score
        rerank_score: Cross-encoder rerank score
    """
    
    item_id: str = Field(
        description="ID of the scored item",
    )
    
    score: float = Field(
        description="Combined relevance score",
    )
    
    dense_score: Optional[float] = Field(
        default=None,
        description="Dense similarity score",
    )
    
    sparse_score: Optional[float] = Field(
        default=None,
        description="Sparse (BM25) score",
    )
    
    rerank_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder rerank score",
    )


class RetrievalResult(BaseModel):
    """
    Result from hybrid retrieval query.
    
    Contains chunks with relevance scores.
    
    Attributes:
        query: Original query text
        chunks: Retrieved chunks with scores
        total_candidates: Total candidates before filtering
        retrieval_time_ms: Retrieval latency in milliseconds
    """
    
    query: str = Field(
        description="Original query text",
    )
    
    chunks: list[tuple[Chunk, float]] = Field(
        default_factory=list,
        description="Retrieved chunks with scores",
    )
    
    total_candidates: int = Field(
        default=0,
        description="Total candidates before filtering",
    )
    
    retrieval_time_ms: float = Field(
        default=0.0,
        description="Retrieval latency in milliseconds",
    )
