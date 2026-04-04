"""Tests for fsociety configuration module."""

import pytest
from fsociety.config import FsocietyConfig, LLMConfig, ScanConfig, get_config


class TestFsocietyConfig:
    """Test FsocietyConfig."""

    def test_default_config(self):
        cfg = FsocietyConfig()
        assert cfg.llm.base_url == "http://localhost:8080"
        assert cfg.llm.model_name == "local-model"
        assert cfg.llm.api_key == "not-needed"
        assert cfg.scan.max_debate_rounds == 6
        assert cfg.output_dir == "./fsociety_reports"

    def test_llm_config(self):
        llm = LLMConfig(model_name="qwen2.5", temperature=0.7)
        assert llm.model_name == "qwen2.5"
        assert llm.temperature == 0.7
        assert llm.base_url == "http://localhost:8080"

    def test_session_id_generation(self):
        cfg = FsocietyConfig()
        sid = cfg.generate_session_id()
        assert sid.startswith("fs-")
        assert cfg.session_id == sid

    def test_get_config_helper(self):
        cfg = get_config()
        assert isinstance(cfg, FsocietyConfig)

    def test_custom_agents(self):
        scan = ScanConfig(agents=["elliot", "darlene"])
        assert len(scan.agents) == 2
        assert "elliot" in scan.agents
