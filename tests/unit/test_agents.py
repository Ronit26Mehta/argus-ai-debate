"""
Unit tests for agents.
"""

import pytest
from unittest.mock import Mock, patch


class TestBaseAgent:
    """Tests for base agent."""
    
    def test_agent_config(self):
        """Test agent configuration."""
        from argus.agents.base import AgentConfig, AgentRole
        
        config = AgentConfig(
            name="TestAgent",
            role=AgentRole.SPECIALIST,
            temperature=0.5,
        )
        
        assert config.name == "TestAgent"
        assert config.role == AgentRole.SPECIALIST
    
    def test_agent_response(self):
        """Test agent response."""
        from argus.agents.base import AgentResponse
        
        response = AgentResponse(
            success=True,
            content="Test response",
            data={"key": "value"},
        )
        
        assert response.success
        assert not response.failed


class TestModerator:
    """Tests for Moderator agent."""
    
    def test_moderator_creation(self, mock_llm):
        """Test moderator creation."""
        from argus.agents.moderator import Moderator
        
        moderator = Moderator(llm=mock_llm)
        
        assert moderator.name == "Moderator"
    
    def test_create_agenda(self, mock_llm_json, cdag_with_proposition):
        """Test agenda creation."""
        from argus.agents.moderator import Moderator
        
        graph, prop = cdag_with_proposition
        
        moderator = Moderator(llm=mock_llm_json)
        agenda = moderator.create_agenda(graph, prop.id)
        
        assert agenda.proposition_id == prop.id
        assert agenda.proposition_text == prop.text
    
    def test_should_stop_max_rounds(self, mock_llm, cdag_with_proposition):
        """Test stopping at max rounds."""
        from argus.agents.moderator import Moderator
        
        graph, prop = cdag_with_proposition
        
        moderator = Moderator(llm=mock_llm)
        agenda = moderator.create_agenda(graph, prop.id)
        
        # Advance to max
        for _ in range(agenda.max_rounds):
            agenda.current_round += 1
        
        should_stop, reason = moderator.should_stop(graph, agenda)
        
        assert should_stop
        assert "maximum" in reason.lower()


class TestSpecialist:
    """Tests for Specialist agent."""
    
    def test_specialist_creation(self, mock_llm):
        """Test specialist creation."""
        from argus.agents.specialist import Specialist
        
        specialist = Specialist(llm=mock_llm, domain="clinical")
        
        assert specialist.domain == "clinical"
    
    def test_evaluate_chunk(self, mock_llm_json, cdag_with_proposition):
        """Test chunk evaluation."""
        from argus.agents.specialist import Specialist
        
        graph, prop = cdag_with_proposition
        
        specialist = Specialist(llm=mock_llm_json)
        result = specialist.evaluate_chunk(
            graph,
            prop.id,
            "This is test evidence text that supports the claim.",
        )
        
        assert "claim" in result
        assert "confidence" in result


class TestRefuter:
    """Tests for Refuter agent."""
    
    def test_refuter_creation(self, mock_llm):
        """Test refuter creation."""
        from argus.agents.refuter import Refuter
        
        refuter = Refuter(llm=mock_llm)
        
        assert refuter.name == "Refuter"
    
    def test_generate_rebuttals(self, mock_llm_json, cdag_with_evidence):
        """Test rebuttal generation."""
        from argus.agents.refuter import Refuter
        
        graph, prop, support, attack = cdag_with_evidence
        
        refuter = Refuter(llm=mock_llm_json)
        rebuttals = refuter.generate_rebuttals(graph, prop.id)
        
        # May be empty if JSON parsing fails
        assert isinstance(rebuttals, list)


class TestJury:
    """Tests for Jury agent."""
    
    def test_jury_creation(self, mock_llm):
        """Test jury creation."""
        from argus.agents.jury import Jury
        
        jury = Jury(llm=mock_llm)
        
        assert jury.name == "Jury"
    
    def test_evaluate(self, mock_llm, cdag_with_evidence):
        """Test proposition evaluation."""
        from argus.agents.jury import Jury, JuryConfig
        
        graph, prop, _, _ = cdag_with_evidence
        
        config = JuryConfig(use_llm_reasoning=False)
        jury = Jury(llm=mock_llm, config=config)
        
        verdict = jury.evaluate(graph, prop.id)
        
        assert verdict.proposition_id == prop.id
        assert 0 <= verdict.posterior <= 1
        assert verdict.label in ["supported", "rejected", "undecided"]
    
    def test_compute_disagreement(self, mock_llm, cdag_with_evidence):
        """Test disagreement computation."""
        from argus.agents.jury import Jury
        
        graph, prop, _, _ = cdag_with_evidence
        
        jury = Jury(llm=mock_llm)
        disagreement = jury.compute_disagreement(graph, prop.id)
        
        assert 0 <= disagreement <= 1
