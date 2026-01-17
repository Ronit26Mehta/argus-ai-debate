"""
ARGUS Memory Systems Configuration.

Configuration for short-term, long-term, and semantic cache memory.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory."""
    BUFFER = "buffer"          # Full conversation buffer
    WINDOW = "window"          # Sliding window
    SUMMARY = "summary"        # Summarized memory
    ENTITY = "entity"          # Entity tracking
    VECTOR = "vector"          # Vector store backed
    SEMANTIC = "semantic"      # Semantic/facts memory
    EPISODIC = "episodic"      # Past interactions


class StorageBackend(str, Enum):
    """Storage backend options."""
    MEMORY = "memory"       # In-memory (volatile)
    SQLITE = "sqlite"       # SQLite persistence
    FAISS = "faiss"         # FAISS vector store
    FILESYSTEM = "filesystem"  # File-based JSON


class ShortTermConfig(BaseModel):
    """Configuration for short-term memory."""
    memory_type: MemoryType = Field(default=MemoryType.BUFFER)
    max_messages: int = Field(default=100, ge=1, le=10000, description="Max messages to store")
    window_size: int = Field(default=10, ge=1, le=100, description="Window size for sliding window")
    summary_max_tokens: int = Field(default=500, ge=100, le=2000, description="Max tokens for summary")
    include_system_messages: bool = Field(default=False)


class LongTermConfig(BaseModel):
    """Configuration for long-term memory."""
    enabled: bool = Field(default=True)
    storage_backend: StorageBackend = Field(default=StorageBackend.FAISS)
    persistence_path: Optional[str] = Field(default=None, description="Path for persistent storage")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    max_memories: int = Field(default=10000, ge=100)
    namespace: str = Field(default="default", description="Memory namespace for multi-tenant")


class SemanticCacheConfig(BaseModel):
    """Configuration for semantic cache."""
    enabled: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_entries: int = Field(default=1000, ge=10)
    ttl_seconds: int = Field(default=3600, ge=0, description="TTL for cache entries (0=no expiry)")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


class MemoryConfig(BaseModel):
    """Main memory configuration."""
    enabled: bool = Field(default=True)
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    long_term: LongTermConfig = Field(default_factory=LongTermConfig)
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    auto_persist: bool = Field(default=True, description="Auto-save to persistence")
    persist_on_shutdown: bool = Field(default=True)


def get_default_memory_config() -> MemoryConfig:
    """Get default memory configuration."""
    return MemoryConfig()
