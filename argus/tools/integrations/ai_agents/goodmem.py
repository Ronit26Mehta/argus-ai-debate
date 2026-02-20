"""
GoodMem Tool for ARGUS.

Add persistent semantic memory to agents across conversations.
Provides long-term memory storage and retrieval with semantic search.
"""

from __future__ import annotations

import os
import uuid
import json
import logging
import hashlib
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    memory_id: str
    agent_id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    importance: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
        }


@dataclass
class MemorySpace:
    """A namespace for organizing agent memories."""
    space_id: str
    agent_id: str
    name: str
    memories: list[MemoryEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_memory(self, memory: MemoryEntry) -> None:
        self.memories.append(memory)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "agent_id": self.agent_id,
            "name": self.name,
            "memory_count": len(self.memories),
            "created_at": self.created_at.isoformat(),
        }


class GoodMemTool(BaseTool):
    """
    GoodMem - Persistent semantic memory for AI agents.
    
    Features:
    - Store and retrieve memories with semantic search
    - Memory importance scoring and decay
    - Cross-conversation memory persistence
    - Memory namespaces/spaces for organization
    - Automatic memory consolidation
    
    Example:
        >>> tool = GoodMemTool()
        >>> result = tool(action="store", agent_id="agent1", 
        ...               content="User prefers dark mode")
        >>> result = tool(action="search", agent_id="agent1",
        ...               query="user preferences")
    """
    
    name = "goodmem"
    description = "Add persistent semantic memory to agents across conversations"
    category = ToolCategory.EXTERNAL_API
    version = "1.0.0"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        enable_persistence: bool = True,
        embedding_model: str = "sentence-transformers",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("GOODMEM_API_KEY")
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".argus" / "goodmem"
        self.enable_persistence = enable_persistence
        self.embedding_model = embedding_model
        
        # In-memory storage
        self._spaces: dict[str, MemorySpace] = {}
        self._embedder = None
        
        # Load persisted memories if enabled
        if self.enable_persistence:
            self._load_memories()
        
        logger.debug(f"GoodMem initialized (persistence={self.enable_persistence})")
    
    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed, using hash-based similarity")
                self._embedder = False
        return self._embedder
    
    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for text."""
        embedder = self._get_embedder()
        if embedder and embedder is not False:
            return embedder.encode(text).tolist()
        # Fallback: hash-based pseudo-embedding
        hash_val = hashlib.md5(text.encode()).hexdigest()
        return [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _load_memories(self) -> None:
        """Load memories from persistent storage."""
        if not self.storage_path.exists():
            return
        
        try:
            for space_file in self.storage_path.glob("*.json"):
                with open(space_file, "r") as f:
                    data = json.load(f)
                
                space = MemorySpace(
                    space_id=data["space_id"],
                    agent_id=data["agent_id"],
                    name=data["name"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                )
                
                for m in data.get("memories", []):
                    memory = MemoryEntry(
                        memory_id=m["memory_id"],
                        agent_id=m["agent_id"],
                        content=m["content"],
                        embedding=m.get("embedding"),
                        metadata=m.get("metadata", {}),
                        tags=m.get("tags", []),
                        created_at=datetime.fromisoformat(m["created_at"]),
                        accessed_at=datetime.fromisoformat(m["accessed_at"]),
                        access_count=m.get("access_count", 0),
                        importance=m.get("importance", 0.5),
                    )
                    space.add_memory(memory)
                
                self._spaces[space.space_id] = space
            
            logger.info(f"Loaded {len(self._spaces)} memory spaces")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def _save_memories(self) -> None:
        """Save memories to persistent storage."""
        if not self.enable_persistence:
            return
        
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            for space in self._spaces.values():
                space_file = self.storage_path / f"{space.space_id}.json"
                data = {
                    **space.to_dict(),
                    "memories": [
                        {**m.to_dict(), "embedding": m.embedding}
                        for m in space.memories
                    ],
                }
                with open(space_file, "w") as f:
                    json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self._spaces)} memory spaces")
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def execute(
        self,
        action: str = "search",
        agent_id: Optional[str] = None,
        space_name: str = "default",
        content: Optional[str] = None,
        query: Optional[str] = None,
        memory_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
        limit: int = 10,
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute GoodMem operations.
        
        Args:
            action: Operation to perform
            agent_id: Agent identifier
            space_name: Memory space name
            content: Memory content to store
            query: Search query
            memory_id: Specific memory ID
            tags: Memory tags
            metadata: Additional metadata
            importance: Memory importance score (0-1)
            limit: Max results
            threshold: Similarity threshold for search
            
        Returns:
            ToolResult with operation result
        """
        actions = {
            "store": self._store_memory,
            "search": self._search_memories,
            "recall": self._recall_memory,
            "forget": self._forget_memory,
            "list": self._list_memories,
            "create_space": self._create_space,
            "list_spaces": self._list_spaces,
            "consolidate": self._consolidate_memories,
            "get_stats": self._get_stats,
            "update_importance": self._update_importance,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            result = actions[action](
                agent_id=agent_id,
                space_name=space_name,
                content=content,
                query=query,
                memory_id=memory_id,
                tags=tags or [],
                metadata=metadata or {},
                importance=importance,
                limit=limit,
                threshold=threshold,
                **kwargs,
            )
            
            # Auto-save after modifications
            if action in ["store", "forget", "create_space", "consolidate", "update_importance"]:
                self._save_memories()
            
            return result
        except Exception as e:
            logger.error(f"GoodMem error: {e}")
            return ToolResult.from_error(f"GoodMem error: {e}")
    
    def _get_or_create_space(self, agent_id: str, space_name: str) -> MemorySpace:
        """Get or create a memory space."""
        space_key = f"{agent_id}:{space_name}"
        
        if space_key not in self._spaces:
            space = MemorySpace(
                space_id=str(uuid.uuid4())[:8],
                agent_id=agent_id,
                name=space_name,
            )
            self._spaces[space_key] = space
        
        return self._spaces[space_key]
    
    def _store_memory(
        self,
        agent_id: Optional[str] = None,
        space_name: str = "default",
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
        **kwargs,
    ) -> ToolResult:
        """Store a new memory."""
        if not agent_id:
            return ToolResult.from_error("agent_id is required")
        if not content:
            return ToolResult.from_error("content is required")
        
        space = self._get_or_create_space(agent_id, space_name)
        
        # Compute embedding
        embedding = self._compute_embedding(content)
        
        memory = MemoryEntry(
            memory_id=str(uuid.uuid4())[:12],
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            tags=tags or [],
            metadata=metadata or {},
            importance=min(1.0, max(0.0, importance)),
        )
        
        space.add_memory(memory)
        
        logger.info(f"Stored memory {memory.memory_id} for agent {agent_id}")
        
        return ToolResult.from_data({
            "memory_id": memory.memory_id,
            "agent_id": agent_id,
            "space": space_name,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
        })
    
    def _search_memories(
        self,
        agent_id: Optional[str] = None,
        space_name: str = "default",
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        threshold: float = 0.3,
        **kwargs,
    ) -> ToolResult:
        """Search memories using semantic similarity."""
        if not agent_id:
            return ToolResult.from_error("agent_id is required")
        if not query:
            return ToolResult.from_error("query is required")
        
        space_key = f"{agent_id}:{space_name}"
        if space_key not in self._spaces:
            return ToolResult.from_data({"memories": [], "count": 0})
        
        space = self._spaces[space_key]
        query_embedding = self._compute_embedding(query)
        
        # Score memories
        scored = []
        for memory in space.memories:
            # Tag filter
            if tags and not any(t in memory.tags for t in tags):
                continue
            
            # Compute similarity
            similarity = self._cosine_similarity(query_embedding, memory.embedding or [])
            
            if similarity >= threshold:
                scored.append((memory, similarity))
        
        # Sort by similarity and importance
        scored.sort(key=lambda x: x[1] * (1 + x[0].importance), reverse=True)
        scored = scored[:limit]
        
        # Update access stats
        for memory, _ in scored:
            memory.accessed_at = datetime.utcnow()
            memory.access_count += 1
        
        results = [
            {**m.to_dict(), "similarity": s}
            for m, s in scored
        ]
        
        return ToolResult.from_data({
            "query": query,
            "memories": results,
            "count": len(results),
        })
    
    def _recall_memory(
        self,
        memory_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Recall a specific memory by ID."""
        if not memory_id:
            return ToolResult.from_error("memory_id is required")
        
        for space in self._spaces.values():
            if agent_id and space.agent_id != agent_id:
                continue
            
            for memory in space.memories:
                if memory.memory_id == memory_id:
                    memory.accessed_at = datetime.utcnow()
                    memory.access_count += 1
                    return ToolResult.from_data({"memory": memory.to_dict()})
        
        return ToolResult.from_error(f"Memory {memory_id} not found")
    
    def _forget_memory(
        self,
        memory_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Forget (delete) a memory."""
        if not memory_id:
            return ToolResult.from_error("memory_id is required")
        
        for space in self._spaces.values():
            if agent_id and space.agent_id != agent_id:
                continue
            
            for i, memory in enumerate(space.memories):
                if memory.memory_id == memory_id:
                    del space.memories[i]
                    return ToolResult.from_data({"message": f"Memory {memory_id} forgotten"})
        
        return ToolResult.from_error(f"Memory {memory_id} not found")
    
    def _list_memories(
        self,
        agent_id: Optional[str] = None,
        space_name: str = "default",
        limit: int = 20,
        **kwargs,
    ) -> ToolResult:
        """List memories in a space."""
        if not agent_id:
            return ToolResult.from_error("agent_id is required")
        
        space_key = f"{agent_id}:{space_name}"
        if space_key not in self._spaces:
            return ToolResult.from_data({"memories": [], "count": 0})
        
        space = self._spaces[space_key]
        memories = space.memories[-limit:]
        
        return ToolResult.from_data({
            "agent_id": agent_id,
            "space": space_name,
            "memories": [m.to_dict() for m in memories],
            "count": len(memories),
            "total": len(space.memories),
        })
    
    def _create_space(
        self,
        agent_id: Optional[str] = None,
        space_name: str = "default",
        **kwargs,
    ) -> ToolResult:
        """Create a new memory space."""
        if not agent_id:
            return ToolResult.from_error("agent_id is required")
        
        space = self._get_or_create_space(agent_id, space_name)
        
        return ToolResult.from_data({
            "space_id": space.space_id,
            "agent_id": agent_id,
            "name": space_name,
        })
    
    def _list_spaces(
        self,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List memory spaces."""
        spaces = [
            s.to_dict()
            for s in self._spaces.values()
            if not agent_id or s.agent_id == agent_id
        ]
        
        return ToolResult.from_data({
            "spaces": spaces,
            "count": len(spaces),
        })
    
    def _consolidate_memories(
        self,
        agent_id: Optional[str] = None,
        space_name: str = "default",
        similarity_threshold: float = 0.85,
        **kwargs,
    ) -> ToolResult:
        """Consolidate similar memories to reduce redundancy."""
        if not agent_id:
            return ToolResult.from_error("agent_id is required")
        
        space_key = f"{agent_id}:{space_name}"
        if space_key not in self._spaces:
            return ToolResult.from_data({"consolidated": 0})
        
        space = self._spaces[space_key]
        original_count = len(space.memories)
        
        # Find and merge similar memories
        to_remove = set()
        for i, m1 in enumerate(space.memories):
            if i in to_remove:
                continue
            for j, m2 in enumerate(space.memories[i + 1:], i + 1):
                if j in to_remove:
                    continue
                
                similarity = self._cosine_similarity(
                    m1.embedding or [],
                    m2.embedding or [],
                )
                
                if similarity >= similarity_threshold:
                    # Merge into the more important/accessed memory
                    if m1.importance > m2.importance or m1.access_count > m2.access_count:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        # Remove consolidated memories
        space.memories = [m for i, m in enumerate(space.memories) if i not in to_remove]
        
        consolidated = len(to_remove)
        
        return ToolResult.from_data({
            "original_count": original_count,
            "final_count": len(space.memories),
            "consolidated": consolidated,
        })
    
    def _get_stats(
        self,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get memory statistics."""
        if agent_id:
            spaces = [s for s in self._spaces.values() if s.agent_id == agent_id]
        else:
            spaces = list(self._spaces.values())
        
        total_memories = sum(len(s.memories) for s in spaces)
        total_tags = set()
        avg_importance = 0.0
        
        for space in spaces:
            for m in space.memories:
                total_tags.update(m.tags)
                avg_importance += m.importance
        
        if total_memories > 0:
            avg_importance /= total_memories
        
        return ToolResult.from_data({
            "total_spaces": len(spaces),
            "total_memories": total_memories,
            "unique_tags": list(total_tags),
            "avg_importance": avg_importance,
        })
    
    def _update_importance(
        self,
        memory_id: Optional[str] = None,
        importance: float = 0.5,
        **kwargs,
    ) -> ToolResult:
        """Update memory importance score."""
        if not memory_id:
            return ToolResult.from_error("memory_id is required")
        
        for space in self._spaces.values():
            for memory in space.memories:
                if memory.memory_id == memory_id:
                    memory.importance = min(1.0, max(0.0, importance))
                    return ToolResult.from_data({
                        "memory_id": memory_id,
                        "importance": memory.importance,
                    })
        
        return ToolResult.from_error(f"Memory {memory_id} not found")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "search", "recall", "forget", "list",
                                "create_space", "list_spaces", "consolidate",
                                "get_stats", "update_importance"],
                    },
                    "agent_id": {"type": "string"},
                    "space_name": {"type": "string", "default": "default"},
                    "content": {"type": "string"},
                    "query": {"type": "string"},
                    "memory_id": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "limit": {"type": "integer"},
                    "threshold": {"type": "number"},
                },
                "required": ["action"],
            },
        }
