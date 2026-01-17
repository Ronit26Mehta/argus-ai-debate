"""
ARGUS Memory Systems Module.

Short-term and long-term memory with semantic caching for
maintaining context across agent interactions.

Example:
    >>> from argus.memory import ConversationBufferMemory, VectorStoreMemory, SemanticCache
    >>> 
    >>> # Short-term conversation memory
    >>> memory = ConversationBufferMemory(max_messages=100)
    >>> memory.add("user", "What is ARGUS?")
    >>> 
    >>> # Long-term vector memory
    >>> ltm = VectorStoreMemory(persistence_path="./memories")
    >>> ltm.add("ARGUS is a debate-driven AI framework")
    >>> 
    >>> # Semantic cache
    >>> cache = SemanticCache(similarity_threshold=0.85)
"""

from argus.memory.config import (
    MemoryConfig,
    MemoryType,
    StorageBackend,
    ShortTermConfig,
    LongTermConfig,
    SemanticCacheConfig,
    get_default_memory_config,
)

from argus.memory.short_term import (
    Message,
    BaseShortTermMemory,
    ConversationBufferMemory,
    ConversationWindowMemory,
    ConversationSummaryMemory,
    EntityMemory,
    ShortTermMemoryManager,
)

from argus.memory.long_term import (
    MemoryEntry,
    BaseLongTermMemory,
    VectorStoreMemory,
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
)

from argus.memory.semantic_cache import (
    CacheEntry,
    SemanticCache,
)

from argus.memory.store import (
    MemoryStore,
    InMemoryStore,
    SQLiteStore,
    FAISSStore,
    FileSystemStore,
)

__all__ = [
    # Config
    "MemoryConfig",
    "MemoryType",
    "StorageBackend",
    "ShortTermConfig",
    "LongTermConfig",
    "SemanticCacheConfig",
    "get_default_memory_config",
    # Short-term
    "Message",
    "BaseShortTermMemory",
    "ConversationBufferMemory",
    "ConversationWindowMemory",
    "ConversationSummaryMemory",
    "EntityMemory",
    "ShortTermMemoryManager",
    # Long-term
    "MemoryEntry",
    "BaseLongTermMemory",
    "VectorStoreMemory",
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
    # Cache
    "CacheEntry",
    "SemanticCache",
    # Store
    "MemoryStore",
    "InMemoryStore",
    "SQLiteStore",
    "FAISSStore",
    "FileSystemStore",
]
