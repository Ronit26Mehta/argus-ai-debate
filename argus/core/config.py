"""
ARGUS Configuration Management.

This module provides centralized configuration for the ARGUS system using Pydantic Settings.
Configuration can be set via environment variables, .env files, or programmatically.

Environment Variables:
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic/Claude API key
    GOOGLE_API_KEY: Google Gemini API key
    OLLAMA_HOST: Ollama server host (default: http://localhost:11434)
    
    ARGUS_DEFAULT_PROVIDER: Default LLM provider (openai, anthropic, gemini, ollama)
    ARGUS_DEFAULT_MODEL: Default model name
    ARGUS_EMBEDDING_MODEL: Embedding model name (default: all-MiniLM-L6-v2)
    ARGUS_TEMPERATURE: Default temperature for LLM generation
    ARGUS_MAX_TOKENS: Default max tokens for LLM generation
    
    ARGUS_CHUNK_SIZE: Target chunk size in tokens (default: 512)
    ARGUS_CHUNK_OVERLAP: Chunk overlap in tokens (default: 50)
    
    ARGUS_RETRIEVAL_TOP_K: Number of documents to retrieve (default: 10)
    ARGUS_RETRIEVAL_LAMBDA: Hybrid retrieval mix (0=sparse, 1=dense, default: 0.7)
    
    ARGUS_LOG_LEVEL: Logging level (default: INFO)
    ARGUS_PROVENANCE_ENABLED: Enable provenance tracking (default: true)

Example:
    >>> from argus.core.config import get_config
    >>> config = get_config()
    >>> print(config.default_provider)
    'openai'
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Literal, Any
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure module logger
logger = logging.getLogger(__name__)


class LLMProviderConfig(BaseSettings):
    """
    Configuration for LLM providers.
    
    Manages API keys and endpoints for all supported LLM providers.
    Keys can be set via environment variables or programmatically.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Core Provider API keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    mistral_api_key: Optional[str] = Field(default=None, description="Mistral API key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    
    # OpenAI-compatible providers
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API key")
    xai_api_key: Optional[str] = Field(default=None, description="xAI (Grok) API key")
    perplexity_api_key: Optional[str] = Field(default=None, description="Perplexity API key")
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA API key")
    together_api_key: Optional[str] = Field(default=None, description="Together AI API key")
    fireworks_api_key: Optional[str] = Field(default=None, description="Fireworks AI API key")
    
    # Cloud providers
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    
    # Enterprise providers
    watsonx_api_key: Optional[str] = Field(default=None, description="IBM watsonx API key")
    watsonx_project_id: Optional[str] = Field(default=None, description="IBM watsonx project ID")
    databricks_token: Optional[str] = Field(default=None, description="Databricks token")
    databricks_host: Optional[str] = Field(default=None, description="Databricks host URL")
    sambanova_api_key: Optional[str] = Field(default=None, description="SambaNova API key")
    cerebras_api_key: Optional[str] = Field(default=None, description="Cerebras API key")
    
    # Utility/self-hosted
    cloudflare_api_token: Optional[str] = Field(default=None, description="Cloudflare API token")
    cloudflare_account_id: Optional[str] = Field(default=None, description="Cloudflare account ID")
    replicate_api_token: Optional[str] = Field(default=None, description="Replicate API token")
    
    # Local/self-hosted
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server host")
    vllm_base_url: Optional[str] = Field(default=None, description="vLLM server URL")
    llamacpp_model_path: Optional[str] = Field(default=None, description="Llama.cpp model path")
    
    def has_provider(self, provider: str) -> bool:
        """Check if a provider is configured with valid credentials."""
        provider = provider.lower()
        provider_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.google_api_key,
            "google": self.google_api_key,
            "cohere": self.cohere_api_key,
            "mistral": self.mistral_api_key,
            "groq": self.groq_api_key,
            "deepseek": self.deepseek_api_key,
            "xai": self.xai_api_key,
            "grok": self.xai_api_key,
            "perplexity": self.perplexity_api_key,
            "nvidia": self.nvidia_api_key,
            "together": self.together_api_key,
            "fireworks": self.fireworks_api_key,
            "azure": self.azure_openai_api_key,
            "azure_openai": self.azure_openai_api_key,
            "huggingface": self.huggingface_api_key,
            "hf": self.huggingface_api_key,
            "watsonx": self.watsonx_api_key,
            "ibm": self.watsonx_api_key,
            "databricks": self.databricks_token,
            "sambanova": self.sambanova_api_key,
            "cerebras": self.cerebras_api_key,
            "cloudflare": self.cloudflare_api_token,
            "replicate": self.replicate_api_token,
            "ollama": self.ollama_host,  # Always available if host set
            "vllm": self.vllm_base_url or True,  # Local, may not need key
            "llamacpp": self.llamacpp_model_path or True,  # Local
            "litellm": True,  # Proxy, uses underlying provider keys
        }
        return bool(provider_keys.get(provider, False))
    

    def get_available_providers(self) -> list[str]:
        """Get list of providers with valid credentials."""
        all_providers = [
            "openai", "anthropic", "gemini", "ollama", "cohere", "mistral", "groq",
            "deepseek", "xai", "perplexity", "nvidia", "together", "fireworks",
            "azure", "huggingface", "watsonx", "databricks", "sambanova", "cerebras",
            "cloudflare", "replicate", "vllm", "llamacpp", "litellm"
        ]
        return [p for p in all_providers if self.has_provider(p)]


class EmbeddingProviderConfig(BaseSettings):
    """
    Configuration for embedding providers.
    
    Manages API keys for all supported embedding providers.
    Many providers share keys with LLM providers (OpenAI, Cohere, etc.)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Core embedding API keys (some shared with LLM providers)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    huggingface_api_key: Optional[str] = Field(default=None, alias="hf_token", description="HuggingFace API token")
    
    # Specialized embedding providers
    voyage_api_key: Optional[str] = Field(default=None, description="Voyage AI API key")
    jina_api_key: Optional[str] = Field(default=None, description="Jina AI API key")
    nomic_api_key: Optional[str] = Field(default=None, description="Nomic AI API key")
    mistral_api_key: Optional[str] = Field(default=None, description="Mistral AI API key")
    
    # Cloud/Enterprise
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA NIM API key")
    together_api_key: Optional[str] = Field(default=None, description="Together AI API key")
    fireworks_api_key: Optional[str] = Field(default=None, description="Fireworks AI API key")
    
    # Local options
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server host")
    
    # Default embedding settings
    default_embedding_provider: str = Field(
        default="sentence_transformers",
        description="Default embedding provider",
    )
    default_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Default embedding model",
    )
    embedding_batch_size: int = Field(default=32, ge=1, le=1000)
    embedding_normalize: bool = Field(default=True)
    
    def has_provider(self, provider: str) -> bool:
        """Check if an embedding provider is configured."""
        provider = provider.lower()
        provider_keys = {
            "openai": self.openai_api_key,
            "cohere": self.cohere_api_key,
            "google": self.google_api_key,
            "gemini": self.google_api_key,
            "huggingface": self.huggingface_api_key,
            "hf": self.huggingface_api_key,
            "voyage": self.voyage_api_key,
            "jina": self.jina_api_key,
            "nomic": self.nomic_api_key,
            "mistral": self.mistral_api_key,
            "azure": self.azure_openai_api_key,
            "nvidia": self.nvidia_api_key,
            "together": self.together_api_key,
            "fireworks": self.fireworks_api_key,
            "ollama": self.ollama_host,  # Always available
            "sentence_transformers": True,  # Local, always available
            "fastembed": True,  # Local, always available
            "bedrock": True,  # Uses AWS credentials
        }
        return bool(provider_keys.get(provider, False))
    
    def get_available_providers(self) -> list[str]:
        """Get list of configured embedding providers."""
        all_providers = [
            "sentence_transformers", "fastembed", "ollama",  # Local
            "openai", "cohere", "google", "huggingface", "voyage",
            "jina", "nomic", "mistral", "azure", "nvidia",
            "together", "fireworks", "bedrock"
        ]
        return [p for p in all_providers if self.has_provider(p)]


class RetrievalConfig(BaseSettings):
    """
    Configuration for document retrieval and evidence engineering.
    
    Controls hybrid retrieval parameters, reranking, and evidence extraction.
    
    Attributes:
        top_k: Number of documents to retrieve per query
        lambda_param: Hybrid mix (0=BM25 only, 1=dense only)
        rerank_top_k: Number of documents to pass to reranker
        use_reranking: Whether to apply cross-encoder reranking
        diversity_weight: MMR diversity weight (0=relevance only)
    """
    
    model_config = SettingsConfigDict(
        env_prefix="ARGUS_RETRIEVAL_",
        env_file=".env",
        extra="ignore",
    )
    
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of documents to retrieve",
    )
    
    lambda_param: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="lambda",
        description="Hybrid retrieval mix (0=sparse, 1=dense)",
    )
    
    rerank_top_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Candidates for reranking",
    )
    
    use_reranking: bool = Field(
        default=True,
        description="Enable cross-encoder reranking",
    )
    
    diversity_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="MMR diversity weight",
    )


class ChunkingConfig(BaseSettings):
    """
    Configuration for document chunking.
    
    Controls how documents are split into chunks for embedding and retrieval.
    
    Attributes:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between consecutive chunks
        min_chunk_size: Minimum chunk size (smaller merged with previous)
        max_chunk_size: Maximum chunk size (larger split)
    """
    
    model_config = SettingsConfigDict(
        env_prefix="ARGUS_CHUNK_",
        env_file=".env",
        extra="ignore",
    )
    
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        alias="size",
        description="Target chunk size in tokens",
    )
    
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        alias="overlap",
        description="Chunk overlap in tokens",
    )
    
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        description="Minimum chunk size",
    )
    
    max_chunk_size: int = Field(
        default=1000,
        le=4000,
        description="Maximum chunk size",
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: Any) -> int:
        """Ensure overlap is less than chunk size."""
        # Note: We can't access other fields in field_validator easily
        # Cross-field validation happens in model_validator
        return v
    
    @model_validator(mode="after")
    def validate_chunk_params(self) -> "ChunkingConfig":
        """Validate chunk parameters are consistent."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size >= self.chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class CalibrationConfig(BaseSettings):
    """
    Configuration for confidence calibration.
    
    Controls temperature scaling and calibration metrics computation.
    
    Attributes:
        initial_temperature: Starting temperature for scaling
        num_bins: Number of bins for ECE calculation
        enable_temperature_scaling: Whether to apply temperature scaling
    """
    
    model_config = SettingsConfigDict(
        env_prefix="ARGUS_CALIBRATION_",
        env_file=".env",
        extra="ignore",
    )
    
    initial_temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Initial calibration temperature",
    )
    
    num_bins: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of bins for ECE",
    )
    
    enable_temperature_scaling: bool = Field(
        default=True,
        description="Enable temperature scaling calibration",
    )


class ProvenanceConfig(BaseSettings):
    """
    Configuration for provenance tracking.
    
    Controls the PROV-O compatible ledger for audit and reproducibility.
    
    Attributes:
        enabled: Whether provenance tracking is active
        ledger_path: Path to store provenance ledger
        hash_algorithm: Algorithm for integrity hashing
        enable_signing: Whether to cryptographically sign events
    """
    
    model_config = SettingsConfigDict(
        env_prefix="ARGUS_PROVENANCE_",
        env_file=".env",
        extra="ignore",
    )
    
    enabled: bool = Field(
        default=True,
        description="Enable provenance tracking",
    )
    
    ledger_path: Optional[Path] = Field(
        default=None,
        description="Path to provenance ledger file",
    )
    
    hash_algorithm: Literal["sha256", "sha384", "sha512"] = Field(
        default="sha256",
        description="Hash algorithm for integrity",
    )
    
    enable_signing: bool = Field(
        default=False,
        description="Enable cryptographic signing",
    )


class ArgusConfig(BaseSettings):
    """
    Main ARGUS configuration.
    
    Central configuration hub that aggregates all sub-configurations.
    Can be loaded from environment variables or .env file.
    
    Example:
        >>> config = ArgusConfig()
        >>> print(config.default_provider)
        'openai'
        >>> print(config.llm.has_provider('openai'))
        True
    
    Attributes:
        default_provider: Default LLM provider to use
        default_model: Default model name
        embedding_model: Model for embedding generation
        temperature: Default LLM temperature
        max_tokens: Default max tokens for generation
        log_level: Logging level
        llm: LLM provider configuration
        retrieval: Retrieval configuration
        chunking: Chunking configuration
        calibration: Calibration configuration
        provenance: Provenance configuration
    """
    
    model_config = SettingsConfigDict(
        env_prefix="ARGUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Default LLM settings
    default_provider: Literal["openai", "anthropic", "gemini", "ollama"] = Field(
        default="openai",
        description="Default LLM provider",
    )
    
    default_model: str = Field(
        default="gpt-4",
        description="Default model name",
    )
    
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for generation",
    )
    
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Default max tokens for generation",
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    
    # Sub-configurations
    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description="LLM provider configuration",
    )
    
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="Retrieval configuration",
    )
    
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Chunking configuration",
    )
    
    calibration: CalibrationConfig = Field(
        default_factory=CalibrationConfig,
        description="Calibration configuration",
    )
    
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig,
        description="Provenance configuration",
    )
    
    @model_validator(mode="after")
    def validate_config(self) -> "ArgusConfig":
        """Validate overall configuration consistency."""
        # Warn if default provider has no credentials
        if not self.llm.has_provider(self.default_provider):
            logger.warning(
                f"Default provider '{self.default_provider}' has no credentials configured. "
                f"Available providers: {self.llm.get_available_providers()}"
            )
        return self
    
    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.info(f"ARGUS logging configured at level {self.log_level}")
    
    def get_model_for_provider(self, provider: Optional[str] = None) -> str:
        """
        Get appropriate model name for a provider.
        
        Args:
            provider: Provider name, or None for default
            
        Returns:
            Model name appropriate for the provider
        """
        provider = provider or self.default_provider
        
        # Default models per provider
        default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-1.5-pro",
            "ollama": "llama3.2",
        }
        
        # If using default model and it matches openai default, adapt to provider
        if self.default_model == "gpt-4" and provider != "openai":
            return default_models.get(provider, self.default_model)
        
        return self.default_model


# Global configuration instance (lazy-loaded)
_config: Optional[ArgusConfig] = None


@lru_cache(maxsize=1)
def get_config() -> ArgusConfig:
    """
    Get the global ARGUS configuration.
    
    Loads configuration from environment variables and .env file.
    Configuration is cached after first load.
    
    Returns:
        ArgusConfig instance
        
    Example:
        >>> config = get_config()
        >>> print(config.default_provider)
        'openai'
    """
    global _config
    if _config is None:
        _config = ArgusConfig()
        _config.configure_logging()
    return _config


def reset_config() -> None:
    """
    Reset the global configuration.
    
    Clears the cached configuration, forcing reload on next get_config() call.
    Useful for testing or dynamic configuration changes.
    """
    global _config
    _config = None
    get_config.cache_clear()


def set_config(config: ArgusConfig) -> None:
    """
    Set the global configuration explicitly.
    
    Args:
        config: ArgusConfig instance to use
        
    Example:
        >>> custom_config = ArgusConfig(default_provider="anthropic")
        >>> set_config(custom_config)
    """
    global _config
    _config = config
    get_config.cache_clear()
