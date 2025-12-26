"""
Unit tests for C-DAG module.
"""

import pytest


class TestProposition:
    """Tests for Proposition node."""
    
    def test_proposition_creation(self):
        """Test basic proposition creation."""
        from argus.cdag.nodes import Proposition
        
        prop = Proposition(
            text="The treatment is effective",
            prior=0.5,
        )
        
        assert prop.text == "The treatment is effective"
        assert prop.prior == 0.5
        assert prop.posterior == 0.5
        assert prop.id.startswith("prop_")
    
    def test_proposition_log_odds(self):
        """Test log-odds computation."""
        from argus.cdag.nodes import Proposition
        
        prop = Proposition(text="Test", prior=0.8)
        
        # log(0.8 / 0.2) ≈ 1.386
        assert prop.log_odds_prior > 1.0
    
    def test_proposition_update_posterior(self):
        """Test posterior update."""
        from argus.cdag.nodes import Proposition, NodeStatus
        
        prop = Proposition(text="Test", prior=0.5)
        
        prop.update_posterior(0.9)
        
        assert prop.posterior == 0.9
        assert prop.status == NodeStatus.ENDORSED


class TestEvidence:
    """Tests for Evidence node."""
    
    def test_evidence_creation(self):
        """Test evidence creation."""
        from argus.cdag.nodes import Evidence, EvidenceType
        
        evidence = Evidence(
            text="Study showed 25% improvement",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.85,
        )
        
        assert evidence.polarity == 1
        assert evidence.confidence == 0.85
        assert evidence.is_supporting
    
    def test_evidence_effective_weight(self):
        """Test effective weight computation."""
        from argus.cdag.nodes import Evidence, EvidenceType
        
        evidence = Evidence(
            text="Test",
            evidence_type=EvidenceType.LITERATURE,
            polarity=1,
            confidence=0.8,
            relevance=0.9,
            quality=0.7,
            weight=1.0,
        )
        
        # 0.8 * 0.9 * 0.7 * 1.0 = 0.504
        assert abs(evidence.effective_weight - 0.504) < 0.001
    
    def test_evidence_add_citation(self):
        """Test adding citation."""
        from argus.cdag.nodes import Evidence, EvidenceType
        
        evidence = Evidence(
            text="Test",
            evidence_type=EvidenceType.LITERATURE,
            polarity=1,
        )
        
        evidence.add_citation("doc_123", "chunk_456", quote="Test quote")
        
        assert len(evidence.citations) == 1
        assert evidence.citations[0]["doc_id"] == "doc_123"


class TestRebuttal:
    """Tests for Rebuttal node."""
    
    def test_rebuttal_creation(self):
        """Test rebuttal creation."""
        from argus.cdag.nodes import Rebuttal
        
        rebuttal = Rebuttal(
            text="The sample size was too small",
            target_id="evid_123",
            rebuttal_type="methodological",
            strength=0.7,
        )
        
        assert rebuttal.target_id == "evid_123"
        assert rebuttal.strength == 0.7


class TestEdge:
    """Tests for Edge."""
    
    def test_edge_creation(self):
        """Test edge creation."""
        from argus.cdag.edges import Edge, EdgeType, EdgePolarity
        
        edge = Edge(
            source_id="evid_123",
            target_id="prop_456",
            edge_type=EdgeType.SUPPORTS,
            weight=0.8,
        )
        
        assert edge.source_id == "evid_123"
        assert edge.polarity == EdgePolarity.POSITIVE
    
    def test_edge_signed_weight(self):
        """Test signed weight computation."""
        from argus.cdag.edges import Edge, EdgeType, EdgePolarity
        
        support_edge = Edge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.SUPPORTS,
            polarity=EdgePolarity.POSITIVE,
            weight=0.8,
        )
        
        attack_edge = Edge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.ATTACKS,
            polarity=EdgePolarity.NEGATIVE,
            weight=0.8,
        )
        
        assert support_edge.signed_weight > 0
        assert attack_edge.signed_weight < 0


class TestCDAG:
    """Tests for C-DAG graph."""
    
    def test_cdag_creation(self):
        """Test C-DAG creation."""
        from argus.cdag import CDAG
        
        graph = CDAG(name="test")
        
        assert graph.name == "test"
        assert graph.num_nodes == 0
        assert graph.num_edges == 0
    
    def test_add_proposition(self):
        """Test adding proposition."""
        from argus.cdag import CDAG, Proposition
        
        graph = CDAG()
        prop = Proposition(text="Test", prior=0.5)
        
        graph.add_proposition(prop)
        
        assert graph.num_propositions == 1
        assert graph.get_proposition(prop.id) == prop
    
    def test_add_evidence(self):
        """Test adding evidence."""
        from argus.cdag import CDAG, Proposition, Evidence, EdgeType
        from argus.cdag.nodes import EvidenceType
        
        graph = CDAG()
        prop = Proposition(text="Test", prior=0.5)
        graph.add_proposition(prop)
        
        evidence = Evidence(
            text="Supporting evidence",
            evidence_type=EvidenceType.EMPIRICAL,
            polarity=1,
            confidence=0.8,
        )
        
        graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)
        
        assert graph.num_evidence == 1
        assert len(graph.get_supporting_evidence(prop.id)) == 1
    
    def test_compute_support_score(self, cdag_with_evidence):
        """Test support score computation."""
        graph, prop, support, attack = cdag_with_evidence
        
        score = graph.compute_support_score(prop.id)
        
        assert score > 0
    
    def test_compute_net_influence(self, cdag_with_evidence):
        """Test net influence computation."""
        graph, prop, support, attack = cdag_with_evidence
        
        net = graph.compute_net_influence(prop.id)
        
        # Support is stronger than attack
        assert net != 0


class TestPropagation:
    """Tests for influence propagation."""
    
    def test_compute_posterior(self, cdag_with_evidence):
        """Test posterior computation."""
        from argus.cdag.propagation import compute_posterior
        
        graph, prop, _, _ = cdag_with_evidence
        
        posterior = compute_posterior(graph, prop.id)
        
        assert 0 <= posterior <= 1
        assert posterior != prop.prior  # Should have changed
    
    def test_propagate_influence(self, cdag_with_evidence):
        """Test influence propagation."""
        from argus.cdag.propagation import propagate_influence
        
        graph, prop, _, _ = cdag_with_evidence
        
        scores = propagate_influence(graph)
        
        assert prop.id in scores
        assert 0 <= scores[prop.id] <= 1
