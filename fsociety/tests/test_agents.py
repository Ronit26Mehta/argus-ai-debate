"""Tests for fsociety agents — all 13 agents instantiate correctly."""

import pytest

# Guard: only run if argus is available
pytest.importorskip("argus")


class TestAllAgentsInstantiate:
    """Verify all 13 agents can be imported and have correct metadata."""

    def _make_mock_llm(self):
        """Create a minimal mock LLM."""
        from unittest.mock import MagicMock
        llm = MagicMock()
        llm.generate.return_value = '{"findings": []}'
        return llm

    def test_elliot(self):
        from fsociety.agents.elliot import ElliotAgent
        agent = ElliotAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "ELLIOT"
        assert "Reconnaissance" in agent.VAPT_DOMAIN

    def test_mrrobot(self):
        from fsociety.agents.mrrobot import MrRobotAgent
        agent = MrRobotAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "MR.ROBOT"

    def test_darlene(self):
        from fsociety.agents.darlene import DarleneAgent
        agent = DarleneAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "DARLENE"

    def test_whiterose(self):
        from fsociety.agents.whiterose import WhiteroseAgent
        agent = WhiteroseAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "WHITEROSE"

    def test_irving(self):
        from fsociety.agents.irving import IrvingAgent
        agent = IrvingAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "IRVING"

    def test_romero(self):
        from fsociety.agents.romero import RomeroAgent
        agent = RomeroAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "ROMERO"

    def test_mobley(self):
        from fsociety.agents.mobley import MobleyAgent
        agent = MobleyAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "MOBLEY"

    def test_trenton(self):
        from fsociety.agents.trenton import TrentonAgent
        agent = TrentonAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "TRENTON"

    def test_tyrell(self):
        from fsociety.agents.tyrell import TyrellAgent
        agent = TyrellAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "TYRELL"

    def test_angela(self):
        from fsociety.agents.angela import AngelaAgent
        agent = AngelaAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "ANGELA"

    def test_dom(self):
        from fsociety.agents.dom import DomAgent
        agent = DomAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "DOM"

    def test_leon(self):
        from fsociety.agents.leon import LeonAgent
        agent = LeonAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "LEON"

    def test_cisco(self):
        from fsociety.agents.cisco import CiscoAgent
        agent = CiscoAgent(llm=self._make_mock_llm())
        assert agent.PERSONA_NAME == "CISCO"

    def test_all_agents_list(self):
        from fsociety.agents import ALL_AGENTS, TIER1_CORE, TIER2_SPECIALIST, TIER3_OUTPUT
        assert len(ALL_AGENTS) == 13
        assert len(TIER1_CORE) == 5
        assert len(TIER2_SPECIALIST) == 6
        assert len(TIER3_OUTPUT) == 2
