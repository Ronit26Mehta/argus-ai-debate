"""
LLM Provider abstraction layer for ARGUS.

This module provides a unified interface for multiple LLM providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude 3.5, Claude 3)
    - Google (Gemini Pro, Gemini Ultra)
    - Ollama (Local models)

Uses LiteLLM as unified interface with direct SDK fallbacks for reliability.
"""

from argus.core.llm.base import (
    BaseLLM,
    LLMResponse,
    LLMConfig,
    Message,
    MessageRole,
)
from argus.core.llm.registry import (
    get_llm,
    register_provider,
    list_providers,
    LLMRegistry,
)
from argus.core.llm.openai import OpenAILLM
from argus.core.llm.anthropic import AnthropicLLM
from argus.core.llm.gemini import GeminiLLM
from argus.core.llm.ollama import OllamaLLM
from argus.core.llm.cohere import CohereLLM
from argus.core.llm.mistral import MistralLLM
from argus.core.llm.groq import GroqLLM

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMResponse",
    "LLMConfig",
    "Message",
    "MessageRole",
    # Registry
    "get_llm",
    "register_provider",
    "list_providers",
    "LLMRegistry",
    # Providers
    "OpenAILLM",
    "AnthropicLLM",
    "GeminiLLM",
    "OllamaLLM",
    "CohereLLM",
    "MistralLLM",
    "GroqLLM",
]
