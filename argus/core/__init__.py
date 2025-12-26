"""
Core module for ARGUS configuration, models, and LLM provider abstraction.

This module provides:
    - ArgusConfig: Central configuration management with environment variable support
    - Base data models: Document, Chunk, Embedding, Claim
    - LLM provider abstraction: Unified interface for OpenAI, Anthropic, Gemini, Ollama
"""

from argus.core.config import ArgusConfig, get_config
from argus.core.models import (
    Document,
    Chunk,
    Embedding,
    Claim,
    NodeBase,
)

__all__ = [
    "ArgusConfig",
    "get_config",
    "Document",
    "Chunk",
    "Embedding",
    "Claim",
    "NodeBase",
]
