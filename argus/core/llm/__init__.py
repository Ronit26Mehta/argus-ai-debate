"""
LLM Provider abstraction layer for ARGUS.

This module provides a unified interface for 27+ LLM providers:
    - OpenAI (GPT-4, GPT-4o, o1)
    - Anthropic (Claude 3.5, Claude 3)
    - Google (Gemini Pro, Gemini Flash)
    - Azure OpenAI, AWS Bedrock, Vertex AI
    - Groq, Mistral, Cohere, DeepSeek, xAI (Grok)
    - Together, Fireworks, NVIDIA, Perplexity
    - HuggingFace, IBM watsonx, Databricks, Snowflake
    - SambaNova, Cerebras, LiteLLM, Cloudflare
    - Replicate, vLLM, Llama.cpp, Ollama
"""

from argus.core.llm.base import (
    BaseLLM,
    LLMResponse,
    LLMConfig,
    LLMUsage,
    Message,
    MessageRole,
    EmbeddingModel,
)
from argus.core.llm.registry import (
    get_llm,
    register_provider,
    list_providers,
    LLMRegistry,
)

# Core providers
from argus.core.llm.openai import OpenAILLM
from argus.core.llm.anthropic import AnthropicLLM
from argus.core.llm.gemini import GeminiLLM
from argus.core.llm.ollama import OllamaLLM
from argus.core.llm.cohere import CohereLLM
from argus.core.llm.mistral import MistralLLM
from argus.core.llm.groq import GroqLLM

# OpenAI-compatible providers
from argus.core.llm.deepseek import DeepSeekLLM
from argus.core.llm.xai import XaiLLM
from argus.core.llm.perplexity import PerplexityLLM
from argus.core.llm.nvidia import NvidiaLLM
from argus.core.llm.together import TogetherLLM
from argus.core.llm.fireworks import FireworksLLM

# Cloud providers
from argus.core.llm.bedrock import BedrockLLM
from argus.core.llm.azure_openai import AzureOpenAILLM
from argus.core.llm.vertex import VertexAILLM
from argus.core.llm.huggingface import HuggingFaceLLM

# Enterprise providers
from argus.core.llm.watsonx import WatsonxLLM
from argus.core.llm.databricks import DatabricksLLM
from argus.core.llm.snowflake import SnowflakeLLM
from argus.core.llm.sambanova import SambanovaLLM
from argus.core.llm.cerebras import CerebrasLLM

# Utility/self-hosted providers
from argus.core.llm.litellm import LiteLLMLLM
from argus.core.llm.cloudflare import CloudflareLLM
from argus.core.llm.replicate import ReplicateLLM
from argus.core.llm.vllm import VllmLLM
from argus.core.llm.llamacpp import LlamaCppLLM

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMResponse",
    "LLMConfig",
    "LLMUsage",
    "Message",
    "MessageRole",
    "EmbeddingModel",
    # Registry
    "get_llm",
    "register_provider",
    "list_providers",
    "LLMRegistry",
    # Core Providers
    "OpenAILLM",
    "AnthropicLLM",
    "GeminiLLM",
    "OllamaLLM",
    "CohereLLM",
    "MistralLLM",
    "GroqLLM",
    # OpenAI-compatible
    "DeepSeekLLM",
    "XaiLLM",
    "PerplexityLLM",
    "NvidiaLLM",
    "TogetherLLM",
    "FireworksLLM",
    # Cloud
    "BedrockLLM",
    "AzureOpenAILLM",
    "VertexAILLM",
    "HuggingFaceLLM",
    # Enterprise
    "WatsonxLLM",
    "DatabricksLLM",
    "SnowflakeLLM",
    "SambanovaLLM",
    "CerebrasLLM",
    # Utility/Self-hosted
    "LiteLLMLLM",
    "CloudflareLLM",
    "ReplicateLLM",
    "VllmLLM",
    "LlamaCppLLM",
]
