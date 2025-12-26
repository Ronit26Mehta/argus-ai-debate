"""
Unit tests for knowledge layer.
"""

import pytest
from pathlib import Path


class TestDocumentLoader:
    """Tests for document loading."""
    
    def test_load_text_file(self, temp_text_file):
        """Test loading text file."""
        from argus.knowledge.ingestion import DocumentLoader
        
        loader = DocumentLoader()
        doc = loader.load(temp_text_file)
        
        assert doc.title == "test"
        assert "test content" in doc.content.lower()
    
    def test_load_json_file(self, temp_json_file):
        """Test loading JSON file."""
        from argus.knowledge.ingestion import DocumentLoader
        from argus.core.models import SourceType
        
        loader = DocumentLoader()
        doc = loader.load(temp_json_file)
        
        assert doc.source_type == SourceType.JSON
        assert "key" in doc.content
    
    def test_detect_source_type(self):
        """Test source type detection."""
        from argus.knowledge.ingestion import detect_source_type
        from argus.core.models import SourceType
        
        assert detect_source_type("file.pdf") == SourceType.PDF
        assert detect_source_type("file.html") == SourceType.HTML
        assert detect_source_type("file.txt") == SourceType.TEXT
        assert detect_source_type("file.json") == SourceType.JSON


class TestChunker:
    """Tests for document chunking."""
    
    def test_fixed_size_chunking(self, sample_document):
        """Test fixed size chunking."""
        from argus.knowledge.chunking import Chunker, ChunkingStrategy
        
        chunker = Chunker(
            chunk_size=100,
            chunk_overlap=10,
            strategy=ChunkingStrategy.FIXED_SIZE,
        )
        
        chunks = chunker.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all(c.doc_id == sample_document.id for c in chunks)
    
    def test_recursive_chunking(self, sample_document):
        """Test recursive chunking."""
        from argus.knowledge.chunking import Chunker, ChunkingStrategy
        
        chunker = Chunker(
            chunk_size=200,
            strategy=ChunkingStrategy.RECURSIVE,
        )
        
        chunks = chunker.chunk(sample_document)
        
        assert len(chunks) > 0
    
    def test_chunk_indices(self, sample_document):
        """Test chunk indices are sequential."""
        from argus.knowledge.chunking import Chunker
        
        chunker = Chunker(chunk_size=100)
        chunks = chunker.chunk(sample_document)
        
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestEmbeddingGenerator:
    """Tests for embedding generation."""
    
    def test_embed_query(self):
        """Test query embedding."""
        from argus.knowledge.embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        
        embedding = generator.embed_query("Test query")
        
        assert len(embedding) == generator.dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_chunks(self, sample_chunks):
        """Test chunk embedding."""
        from argus.knowledge.embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        embeddings = generator.embed_chunks(sample_chunks[:2])
        
        assert len(embeddings) == 2
        assert embeddings[0].dimension == generator.dimension
    
    def test_cosine_similarity(self):
        """Test cosine similarity."""
        from argus.knowledge.embeddings import cosine_similarity
        
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        
        assert cosine_similarity(v1, v2) == 1.0
        assert cosine_similarity(v1, v3) == 0.0


class TestHybridIndex:
    """Tests for hybrid indexing."""
    
    def test_bm25_index(self, sample_chunks):
        """Test BM25 indexing."""
        from argus.knowledge.indexing import BM25Index
        
        index = BM25Index()
        index.add_chunks(sample_chunks)
        
        results = index.search("chunk test content", top_k=3)
        
        assert len(results) > 0
        assert results[0].score > 0
    
    def test_hybrid_search(self, sample_chunks):
        """Test hybrid search."""
        from argus.knowledge.indexing import HybridIndex
        from argus.knowledge.embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        embeddings = generator.embed_chunks(sample_chunks)
        
        index = HybridIndex(dimension=generator.dimension)
        index.add_chunks(sample_chunks, [e.vector for e in embeddings])
        
        query_emb = generator.embed_query("test content")
        results = index.search("test content", query_emb, top_k=3)
        
        assert len(results) > 0
