"""
Unit tests for core data models.
"""

import pytest
from datetime import datetime


class TestDocument:
    """Tests for Document model."""
    
    def test_document_creation(self):
        """Test basic document creation."""
        from argus.core.models import Document, SourceType
        
        doc = Document(
            url="https://example.com/paper.pdf",
            title="Test Paper",
            content="This is the content.",
            source_type=SourceType.PDF,
        )
        
        assert doc.title == "Test Paper"
        assert doc.source_type == SourceType.PDF
        assert doc.id is not None
        assert len(doc.id) > 0
    
    def test_document_word_count(self):
        """Test document word count computation."""
        from argus.core.models import Document, SourceType
        
        doc = Document(
            url="test.txt",
            title="Test",
            content="One two three four five",
            source_type=SourceType.TEXT,
        )
        
        assert doc.word_count == 5
    
    def test_document_checksum(self):
        """Test document checksum."""
        from argus.core.models import Document, SourceType
        
        doc = Document(
            url="test.txt",
            title="Test",
            content="Content",
            source_type=SourceType.TEXT,
            checksum="abc123",
        )
        
        assert doc.checksum == "abc123"


class TestChunk:
    """Tests for Chunk model."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        from argus.core.models import Chunk
        
        chunk = Chunk(
            doc_id="doc_123",
            text="This is chunk text.",
            start_char=0,
            end_char=100,
            chunk_index=0,
        )
        
        assert chunk.doc_id == "doc_123"
        assert chunk.chunk_index == 0
        assert chunk.id is not None
    
    def test_chunk_char_count(self):
        """Test chunk character count."""
        from argus.core.models import Chunk
        
        chunk = Chunk(
            doc_id="doc_123",
            text="Hello",
            start_char=0,
            end_char=5,
            chunk_index=0,
        )
        
        assert chunk.char_count == 5


class TestEmbedding:
    """Tests for Embedding model."""
    
    def test_embedding_creation(self):
        """Test embedding creation."""
        from argus.core.models import Embedding
        
        vector = [0.1, 0.2, 0.3, 0.4]
        emb = Embedding(
            source_id="chunk_123",
            vector=vector,
            model="test-model",
        )
        
        assert emb.dimension == 4
        assert emb.vector == vector
    
    def test_embedding_dimension(self):
        """Test embedding dimension computation."""
        from argus.core.models import Embedding
        
        emb = Embedding(
            source_id="test",
            vector=[0.0] * 384,
            model="all-MiniLM-L6-v2",
        )
        
        assert emb.dimension == 384


class TestClaim:
    """Tests for Claim model."""
    
    def test_claim_creation(self):
        """Test claim creation."""
        from argus.core.models import Claim
        
        claim = Claim(
            text="The drug reduces symptoms.",
            confidence=0.85,
        )
        
        assert claim.text == "The drug reduces symptoms."
        assert claim.confidence == 0.85
    
    def test_claim_confidence_bounds(self):
        """Test claim confidence is bounded."""
        from argus.core.models import Claim
        from pydantic import ValidationError
        
        # Valid confidence
        claim = Claim(text="Test", confidence=0.5)
        assert claim.confidence == 0.5
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            Claim(text="Test", confidence=1.5)


class TestCitation:
    """Tests for Citation model."""
    
    def test_citation_creation(self):
        """Test citation creation."""
        from argus.core.models import Citation, CitationType
        
        citation = Citation(
            doc_id="doc_123",
            chunk_id="chunk_456",
            citation_type=CitationType.DIRECT_QUOTE,
            quote="This is the quote.",
        )
        
        assert citation.doc_id == "doc_123"
        assert citation.citation_type == CitationType.DIRECT_QUOTE
