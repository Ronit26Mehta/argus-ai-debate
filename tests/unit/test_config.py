"""
Unit tests for core configuration module.
"""

import pytest
import os
from unittest.mock import patch


class TestArgusConfig:
    """Tests for ArgusConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from argus.core.config import ArgusConfig
        
        config = ArgusConfig()
        
        assert config.temperature >= 0.0
        assert config.temperature <= 2.0
        assert config.max_tokens > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        from argus.core.config import ArgusConfig
        from pydantic import ValidationError
        
        # Valid config
        config = ArgusConfig(temperature=0.5, max_tokens=1000)
        assert config.temperature == 0.5
        
        # Invalid temperature
        with pytest.raises(ValidationError):
            ArgusConfig(temperature=3.0)
    
    def test_get_config_singleton(self):
        """Test get_config returns consistent config."""
        from argus.core.config import get_config, _config_instance
        
        config1 = get_config()
        config2 = get_config()
        
        # Should return same instance
        assert config1 is config2
    
    def test_model_for_provider(self):
        """Test get_model_for_provider method."""
        from argus.core.config import ArgusConfig
        
        config = ArgusConfig(default_model="gpt-4")
        
        model = config.get_model_for_provider("openai")
        assert model is not None


class TestLLMConfig:
    """Tests for LLM configuration."""
    
    def test_llm_config_defaults(self):
        """Test LLM config default values."""
        from argus.core.config import LLMConfig
        
        config = LLMConfig()
        
        assert config.ollama_host == "http://localhost:11434"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_llm_config_from_env(self):
        """Test LLM config from environment variables."""
        from argus.core.config import LLMConfig
        
        config = LLMConfig()
        
        assert config.openai_api_key == "test-key"


class TestRetrievalConfig:
    """Tests for retrieval configuration."""
    
    def test_retrieval_config_defaults(self):
        """Test retrieval config defaults."""
        from argus.core.config import RetrievalConfig
        
        config = RetrievalConfig()
        
        assert config.top_k > 0
        assert 0 <= config.lambda_param <= 1
