"""
Integration tests for ARGUS.

These tests verify that multiple components work together correctly.
"""

import pytest
from unittest.mock import Mock, patch


class TestEndToEndDebate:
    """End-to-end debate tests."""
    
    def test_simple_debate_flow(self, mock_llm_json):
        """Test simple debate from proposition to verdict."""
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType
        from argus.cdag.nodes import EvidenceType
        from argus.cdag.propagation import compute_posterior
        from argus.agents.jury import Jury, JuryConfig
        
        # Create graph
        graph = CDAG()
        
        # Add proposition
        prop = Proposition(
            text="The new drug is effective",
            prior=0.5,
        )
        graph.add_proposition(prop)
        
        # Add supporting evidence
        support1 = Evidence(
            text="Phase 3 trial showed 30% improvement",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.9,
        )
        graph.add_evidence(support1, prop.id, EdgeType.SUPPORTS)
        
        support2 = Evidence(
            text="Biomarker reduction confirmed",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.8,
        )
        graph.add_evidence(support2, prop.id, EdgeType.SUPPORTS)
        
        # Add attacking evidence
        attack1 = Evidence(
            text="Side effects in 15% of patients",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=-1,
            confidence=0.7,
        )
        graph.add_evidence(attack1, prop.id, EdgeType.ATTACKS)
        
        # Compute posterior
        posterior = compute_posterior(graph, prop.id)
        
        # Verify posterior updated
        assert posterior != prop.prior
        assert 0 < posterior < 1
        
        # Jury verdict
        config = JuryConfig(use_llm_reasoning=False)
        jury = Jury(llm=mock_llm_json, config=config)
        verdict = jury.evaluate(graph, prop.id)
        
        assert verdict.label in ["supported", "rejected", "undecided"]
        assert verdict.num_support == 2
        assert verdict.num_attack == 1
    
    def test_debate_with_rebuttals(self, mock_llm, cdag_with_evidence):
        """Test debate with rebuttals."""
        from argus.cdag import Rebuttal
        from argus.cdag.propagation import compute_posterior
        
        graph, prop, support, attack = cdag_with_evidence
        
        # Initial posterior
        initial_posterior = compute_posterior(graph, prop.id)
        
        # Add rebuttal to supporting evidence
        rebuttal = Rebuttal(
            text="The study had methodological flaws",
            target_id=support.id,
            rebuttal_type="methodological",
            strength=0.8,
            confidence=0.75,
        )
        graph.add_rebuttal(rebuttal, support.id)
        
        # Posterior should potentially change
        new_posterior = compute_posterior(graph, prop.id)
        
        # Graph should have rebuttal
        assert len(graph._rebuttals) == 1


class TestKnowledgePipeline:
    """Tests for knowledge pipeline."""
    
    def test_ingest_and_index(self, temp_text_file):
        """Test document ingestion and indexing."""
        from argus.knowledge.ingestion import DocumentLoader
        from argus.knowledge.chunking import Chunker
        from argus.knowledge.embeddings import EmbeddingGenerator
        from argus.knowledge.indexing import HybridIndex
        
        # Load document
        loader = DocumentLoader()
        doc = loader.load(temp_text_file)
        
        # Chunk document
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        
        assert len(chunks) > 0
        
        # Generate embeddings
        embedder = EmbeddingGenerator()
        embeddings = embedder.embed_chunks(chunks)
        
        assert len(embeddings) == len(chunks)
        
        # Create index
        index = HybridIndex(dimension=embedder.dimension)
        index.add_chunks(chunks, [e.vector for e in embeddings])
        
        # Search
        query_emb = embedder.embed_query("test content")
        results = index.search("test content", query_emb, top_k=5)
        
        assert len(results) > 0


class TestRetrieverIntegration:
    """Tests for retriever integration."""
    
    def test_hybrid_retriever(self, temp_text_file):
        """Test hybrid retriever."""
        from argus.knowledge.ingestion import DocumentLoader
        from argus.knowledge.chunking import Chunker
        from argus.retrieval.hybrid import HybridRetriever
        
        # Load and chunk
        loader = DocumentLoader()
        doc = loader.load(temp_text_file)
        
        chunker = Chunker(chunk_size=50)
        chunks = chunker.chunk(doc)
        
        # Create retriever and index
        retriever = HybridRetriever()
        retriever.index_chunks(chunks)
        
        # Retrieve
        results = retriever.retrieve("test", top_k=3)
        
        assert len(results) > 0
        assert results[0].rank == 1


class TestProvenanceIntegration:
    """Tests for provenance integration."""
    
    def test_ledger_with_debate(self, mock_llm, temp_ledger):
        """Test ledger records debate events."""
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType
        from argus.cdag.nodes import EvidenceType
        from argus.provenance import EventType
        
        # Record session
        temp_ledger.record(EventType.SESSION_START)
        
        # Create graph
        graph = CDAG()
        prop = Proposition(text="Test", prior=0.5)
        graph.add_proposition(prop)
        
        temp_ledger.record(
            EventType.PROPOSITION_ADDED,
            entity_id=prop.id,
            attributes={"text": prop.text},
        )
        
        # Add evidence
        evidence = Evidence(
            text="Test evidence",
            evidence_type=EvidenceType.LITERATURE,
            polarity=1,
        )
        graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)
        
        temp_ledger.record(
            EventType.EVIDENCE_ADDED,
            entity_id=evidence.id,
        )
        
        temp_ledger.record(EventType.SESSION_END)
        
        # Verify ledger
        assert len(temp_ledger) == 4
        
        is_valid, _ = temp_ledger.verify_integrity()
        assert is_valid
        
        # Query events
        prop_events = temp_ledger.query(entity_id=prop.id)
        assert len(prop_events) == 1


class TestConfigIntegration:
    """Tests for configuration integration."""
    
    def test_config_affects_components(self):
        """Test config values propagate to components."""
        from argus.core.config import ArgusConfig, ChunkingConfig
        
        config = ArgusConfig(
            chunking=ChunkingConfig(
                chunk_size=256,
                chunk_overlap=32,
            )
        )
        
        assert config.chunking.chunk_size == 256
        assert config.chunking.chunk_overlap == 32
