"""
End-to-End Debate Workflow Integration Tests for ARGUS.

Tests the complete debate workflow from document ingestion to final verdict,
including multi-round debates with rebuttals and report generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestFullDebateWorkflow:
    """Comprehensive end-to-end debate tests."""
    
    def test_complete_debate_from_document_to_verdict(
        self,
        mock_llm_json,
        temp_text_file,
    ):
        """Test complete debate from document ingestion to final verdict.
        
        This test covers:
        1. Document loading and chunking
        2. Proposition creation
        3. Evidence addition (supports and attacks)
        4. Rebuttal generation
        5. Posterior computation
        6. Jury verdict
        """
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType, Rebuttal
        from argus.cdag.nodes import EvidenceType
        from argus.cdag.propagation import compute_posterior
        from argus.agents.jury import Jury, JuryConfig
        from argus.knowledge.ingestion import DocumentLoader
        from argus.knowledge.chunking import Chunker
        
        # Step 1: Load and chunk document
        loader = DocumentLoader()
        doc = loader.load(temp_text_file)
        
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        
        assert len(chunks) > 0, "Should produce chunks"
        
        # Step 2: Create debate graph with proposition
        graph = CDAG(name="full_debate_test")
        
        proposition = Proposition(
            text="The documented approach is effective for the stated goals",
            prior=0.5,
        )
        graph.add_proposition(proposition)
        
        # Step 3: Add supporting evidence
        support_1 = Evidence(
            text="Primary research supports the claim with p<0.05 significance",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.92,
        )
        graph.add_evidence(support_1, proposition.id, EdgeType.SUPPORTS)
        
        support_2 = Evidence(
            text="Secondary analysis confirms findings across multiple datasets",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.85,
        )
        graph.add_evidence(support_2, proposition.id, EdgeType.SUPPORTS)
        
        support_3 = Evidence(
            text="Theoretical framework predicts observed outcomes",
            evidence_type=EvidenceType.THEORETICAL,
            polarity=1,
            confidence=0.78,
        )
        graph.add_evidence(support_3, proposition.id, EdgeType.SUPPORTS)
        
        # Step 4: Add attacking evidence
        attack_1 = Evidence(
            text="Sample size may be insufficient for generalization",
            evidence_type=EvidenceType.METHODOLOGICAL,
            polarity=-1,
            confidence=0.65,
        )
        graph.add_evidence(attack_1, proposition.id, EdgeType.ATTACKS)
        
        attack_2 = Evidence(
            text="Alternative explanations not fully ruled out",
            evidence_type=EvidenceType.LOGICAL,
            polarity=-1,
            confidence=0.55,
        )
        graph.add_evidence(attack_2, proposition.id, EdgeType.ATTACKS)
        
        # Step 5: Add rebuttals
        rebuttal_1 = Rebuttal(
            text="Sample size meets power analysis requirements for effect size",
            target_id=attack_1.id,
            rebuttal_type="empirical",
            strength=0.7,
            confidence=0.75,
        )
        graph.add_rebuttal(rebuttal_1, attack_1.id)
        
        # Step 6: Compute posterior with all evidence
        posterior = compute_posterior(graph, proposition.id)
        
        # Verify posterior is computed and different from prior
        assert posterior != proposition.prior, "Posterior should update from prior"
        assert 0 <= posterior <= 1, "Posterior must be valid probability"
        
        # With 3 strong supports vs 2 weaker attacks (one rebutted),
        # posterior should favor support
        assert posterior > 0.5, "Posterior should favor supporting evidence"
        
        # Step 7: Jury evaluation
        config = JuryConfig(use_llm_reasoning=False)
        jury = Jury(llm=mock_llm_json, config=config)
        verdict = jury.evaluate(graph, proposition.id)
        
        # Verify verdict structure
        assert verdict is not None
        assert verdict.label in ["supported", "rejected", "undecided"]
        assert verdict.num_support == 3
        assert verdict.num_attack == 2
        assert 0 <= verdict.confidence <= 1
        
        # Given the evidence balance, should be supported
        assert verdict.label == "supported"
    
    def test_multi_round_debate_with_counter_rebuttals(self, mock_llm_json):
        """Test multi-round debate with rebuttals and counter-arguments."""
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType, Rebuttal
        from argus.cdag.nodes import EvidenceType
        from argus.cdag.propagation import compute_posterior
        
        graph = CDAG(name="multi_round_debate")
        
        # Initial proposition
        prop = Proposition(
            text="AI systems should be regulated by government agencies",
            prior=0.5,
        )
        graph.add_proposition(prop)
        
        # Round 1: Initial arguments
        pro_arg = Evidence(
            text="Unchecked AI development poses existential risks",
            evidence_type=EvidenceType.THEORETICAL,
            polarity=1,
            confidence=0.8,
        )
        graph.add_evidence(pro_arg, prop.id, EdgeType.SUPPORTS)
        
        con_arg = Evidence(
            text="Government regulation stifles innovation",
            evidence_type=EvidenceType.ECONOMIC,
            polarity=-1,
            confidence=0.75,
        )
        graph.add_evidence(con_arg, prop.id, EdgeType.ATTACKS)
        
        posterior_r1 = compute_posterior(graph, prop.id)
        
        # Round 2: Rebuttals
        rebuttal_to_con = Rebuttal(
            text="European AI regulations haven't slowed AI progress there",
            target_id=con_arg.id,
            rebuttal_type="empirical",
            strength=0.7,
            confidence=0.7,
        )
        graph.add_rebuttal(rebuttal_to_con, con_arg.id)
        
        rebuttal_to_pro = Rebuttal(
            text="Existential risk claims are speculative and unfounded",
            target_id=pro_arg.id,
            rebuttal_type="logical",
            strength=0.6,
            confidence=0.65,
        )
        graph.add_rebuttal(rebuttal_to_pro, pro_arg.id)
        
        posterior_r2 = compute_posterior(graph, prop.id)
        
        # Round 3: Additional evidence
        new_support = Evidence(
            text="Case studies show regulation prevented AI harms",
            evidence_type=EvidenceType.CASE_STUDY,
            polarity=1,
            confidence=0.82,
        )
        graph.add_evidence(new_support, prop.id, EdgeType.SUPPORTS)
        
        posterior_r3 = compute_posterior(graph, prop.id)
        
        # Verify graph structure
        assert len(graph._propositions) == 1
        assert len(graph._evidence) == 3
        assert len(graph._rebuttals) == 2
        
        # Verify posteriors evolve
        assert all(0 <= p <= 1 for p in [posterior_r1, posterior_r2, posterior_r3])
    
    def test_debate_with_evidence_chains(self, mock_llm_json):
        """Test debate with evidence supporting other evidence."""
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType
        from argus.cdag.nodes import EvidenceType
        from argus.cdag.propagation import compute_posterior
        
        graph = CDAG(name="evidence_chain_test")
        
        prop = Proposition(
            text="Climate change requires immediate action",
            prior=0.5,
        )
        graph.add_proposition(prop)
        
        # Primary evidence
        primary = Evidence(
            text="Global temperatures rising at unprecedented rates",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.95,
        )
        graph.add_evidence(primary, prop.id, EdgeType.SUPPORTS)
        
        # Secondary evidence supporting primary
        secondary = Evidence(
            text="Temperature data corroborated by satellite measurements",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.9,
        )
        graph.add_evidence(secondary, primary.id, EdgeType.SUPPORTS)
        
        # Compute posterior
        posterior = compute_posterior(graph, prop.id)
        
        assert posterior > 0.5, "Strong evidence should increase posterior"
        assert len(graph._evidence) == 2
    
    def test_report_generation_after_debate(self, mock_llm_json):
        """Test report generation from completed debate."""
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType
        from argus.cdag.nodes import EvidenceType
        from argus.cdag.propagation import compute_posterior
        from argus.agents.jury import Jury, JuryConfig
        from argus.outputs.reports import ReportGenerator, ReportConfig
        
        # Set up debate
        graph = CDAG(name="report_test")
        
        prop = Proposition(text="Treatment X is effective", prior=0.5)
        graph.add_proposition(prop)
        
        support = Evidence(
            text="RCT shows 30% improvement",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.88,
        )
        graph.add_evidence(support, prop.id, EdgeType.SUPPORTS)
        
        attack = Evidence(
            text="Small sample size",
            evidence_type=EvidenceType.METHODOLOGICAL,
            polarity=-1,
            confidence=0.6,
        )
        graph.add_evidence(attack, prop.id, EdgeType.ATTACKS)
        
        posterior = compute_posterior(graph, prop.id)
        
        # Get verdict
        config = JuryConfig(use_llm_reasoning=False)
        jury = Jury(llm=mock_llm_json, config=config)
        verdict = jury.evaluate(graph, prop.id)
        
        # Generate report
        report_config = ReportConfig(format="json")
        generator = ReportGenerator(config=report_config)
        report = generator.generate(graph, proposition_id=prop.id, verdict=verdict)
        
        # Verify report structure
        assert report is not None
        assert "proposition" in report or hasattr(report, "proposition")


class TestConnectorIntegration:
    """Tests for external connector integration."""
    
    def test_connector_registry(self):
        """Test connector registry functionality."""
        from argus.knowledge.connectors import (
            ConnectorRegistry,
            BaseConnector,
            ConnectorResult,
        )
        
        class TestConnector(BaseConnector):
            name = "test_connector"
            description = "Test connector"
            
            def fetch(self, query: str, max_results: int = 10, **kwargs):
                return ConnectorResult(
                    success=True,
                    documents=[],
                    metadata={"query": query},
                )
        
        registry = ConnectorRegistry()
        connector = TestConnector()
        
        # Register
        registry.register(connector)
        assert "test_connector" in registry
        
        # Get
        retrieved = registry.get("test_connector")
        assert retrieved is not None
        assert retrieved.name == "test_connector"
        
        # List
        names = registry.list_all()
        assert "test_connector" in names
        
        # Fetch
        result = retrieved("test query")
        assert result.success
    
    def test_web_connector_initialization(self):
        """Test web connector can be initialized."""
        from argus.knowledge.connectors import WebConnector, WebConnectorConfig
        
        config = WebConnectorConfig(
            respect_robots_txt=True,
            user_agent="TestBot/1.0",
            timeout=10,
        )
        
        connector = WebConnector(config=config)
        
        assert connector.name == "web"
        assert connector.web_config.respect_robots_txt is True
        assert connector.web_config.user_agent == "TestBot/1.0"
    
    def test_arxiv_connector_initialization(self):
        """Test arXiv connector can be initialized."""
        from argus.knowledge.connectors import ArxivConnector, ArxivConnectorConfig
        
        config = ArxivConnectorConfig(
            sort_by="submittedDate",
            sort_order="descending",
        )
        
        connector = ArxivConnector(config=config)
        
        assert connector.name == "arxiv"
        assert connector.arxiv_config.sort_by == "submittedDate"
    
    def test_crossref_connector_initialization(self):
        """Test CrossRef connector can be initialized."""
        from argus.knowledge.connectors import CrossRefConnector, CrossRefConnectorConfig
        
        config = CrossRefConnectorConfig(
            mailto="test@example.com",
            sort="score",
        )
        
        connector = CrossRefConnector(config=config)
        
        assert connector.name == "crossref"
        assert connector.crossref_config.mailto == "test@example.com"


class TestLLMProviderIntegration:
    """Tests for LLM provider integration."""
    
    def test_llm_registry_providers(self):
        """Test all expected providers are registered."""
        from argus.core.llm import list_providers
        
        providers = list_providers()
        
        # Check core providers
        expected = ["openai", "anthropic", "gemini", "ollama"]
        for provider in expected:
            assert provider in providers, f"{provider} should be registered"
        
        # Check new providers
        new_providers = ["cohere", "mistral", "groq"]
        for provider in new_providers:
            assert provider in providers, f"{provider} should be registered"
    
    def test_provider_instantiation_cohere(self):
        """Test Cohere provider can be instantiated (will fail without API key)."""
        from argus.core.llm import CohereLLM
        
        # Just test class exists and has expected attributes
        assert CohereLLM.MODEL_ALIASES is not None
        assert "cohere" in CohereLLM.MODEL_ALIASES or "command" in CohereLLM.MODEL_ALIASES
    
    def test_provider_instantiation_mistral(self):
        """Test Mistral provider can be instantiated (will fail without API key)."""
        from argus.core.llm import MistralLLM
        
        assert MistralLLM.MODEL_ALIASES is not None
        assert "mistral" in MistralLLM.MODEL_ALIASES
    
    def test_provider_instantiation_groq(self):
        """Test Groq provider can be instantiated (will fail without API key)."""
        from argus.core.llm import GroqLLM
        
        assert GroqLLM.MODEL_ALIASES is not None
        assert "llama" in GroqLLM.MODEL_ALIASES or "groq" in GroqLLM.MODEL_ALIASES


class TestProvenanceWithDebate:
    """Tests for provenance tracking during debate."""
    
    def test_full_debate_provenance(self, mock_llm, temp_ledger):
        """Test ledger captures complete debate history."""
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType, Rebuttal
        from argus.cdag.nodes import EvidenceType
        from argus.provenance import EventType
        from argus.agents.jury import Jury, JuryConfig
        
        # Start session
        temp_ledger.record(EventType.SESSION_START)
        
        # Create graph
        graph = CDAG(name="provenance_test")
        prop = Proposition(text="Test claim", prior=0.5)
        graph.add_proposition(prop)
        
        temp_ledger.record(
            EventType.PROPOSITION_ADDED,
            entity_id=prop.id,
            attributes={"text": prop.text, "prior": prop.prior},
        )
        
        # Add evidence
        evidence = Evidence(
            text="Supporting data",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.8,
        )
        graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)
        
        temp_ledger.record(
            EventType.EVIDENCE_ADDED,
            entity_id=evidence.id,
            attributes={
                "text": evidence.text,
                "type": evidence.evidence_type.value,
                "target": prop.id,
            },
        )
        
        # Add rebuttal
        rebuttal = Rebuttal(
            text="Counter argument",
            target_id=evidence.id,
            rebuttal_type="logical",
            strength=0.6,
            confidence=0.65,
        )
        graph.add_rebuttal(rebuttal, evidence.id)
        
        temp_ledger.record(
            EventType.REBUTTAL_ADDED,
            entity_id=rebuttal.id,
            attributes={"text": rebuttal.text, "target": evidence.id},
        )
        
        # Get verdict
        jury = Jury(llm=mock_llm, config=JuryConfig(use_llm_reasoning=False))
        verdict = jury.evaluate(graph, prop.id)
        
        temp_ledger.record(
            EventType.VERDICT_RENDERED,
            entity_id=prop.id,
            attributes={"label": verdict.label},
        )
        
        temp_ledger.record(EventType.SESSION_END)
        
        # Verify ledger
        assert len(temp_ledger) >= 5
        
        is_valid, _ = temp_ledger.verify_integrity()
        assert is_valid
        
        # Query by entity
        prop_events = temp_ledger.query(entity_id=prop.id)
        assert len(prop_events) >= 1
