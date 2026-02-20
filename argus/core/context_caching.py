"""
Context Caching Module for ARGUS.

Intelligent caching for LLM context, conversations, and intermediate results.
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from functools import wraps
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached entry with metadata."""
    key: str
    value: T
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def touch(self):
        """Update access time and count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    oldest_entry_age: float = 0
    newest_entry_age: float = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        pass
    
    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        """Set entry."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry by key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Clear all entries. Returns count of cleared entries."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get total number of entries."""
        pass


class MemoryBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_entries: int = 1000, max_size_bytes: int = 100 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_size = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
            return entry
    
    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            # Remove old entry if exists
            if entry.key in self._cache:
                old_entry = self._cache[entry.key]
                self._total_size -= old_entry.size_bytes
            
            # Evict if needed
            while (
                self._cache and 
                (len(self._cache) >= self.max_entries or 
                 self._total_size + entry.size_bytes > self.max_size_bytes)
            ):
                # Remove oldest (first) entry
                oldest_key = next(iter(self._cache))
                oldest = self._cache.pop(oldest_key)
                self._total_size -= oldest.size_bytes
            
            self._cache[entry.key] = entry
            self._total_size += entry.size_bytes
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._total_size -= entry.size_bytes
                return True
            return False
    
    def exists(self, key: str) -> bool:
        return key in self._cache
    
    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._total_size = 0
            return count
    
    def keys(self) -> List[str]:
        return list(self._cache.keys())
    
    def size(self) -> int:
        return len(self._cache)


class FileBackend(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: str = ".argus_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "index.json"
        self._lock = threading.RLock()
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
    
    def _load_index(self):
        """Load cache index from file."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {}
    
    def _save_index(self):
        """Save cache index to file."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)
    
    def _get_entry_path(self, key: str) -> Path:
        """Get file path for entry."""
        # Use hash for filename to handle special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            if key not in self._index:
                return None
            
            entry_path = self._get_entry_path(key)
            if not entry_path.exists():
                del self._index[key]
                self._save_index()
                return None
            
            try:
                with open(entry_path, "rb") as f:
                    entry = pickle.load(f)
                entry.touch()
                
                # Update index with access info
                self._index[key]["access_count"] = entry.access_count
                self._index[key]["last_accessed"] = entry.last_accessed.isoformat()
                self._save_index()
                
                return entry
            except (pickle.PickleError, IOError):
                return None
    
    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            entry_path = self._get_entry_path(entry.key)
            
            try:
                with open(entry_path, "wb") as f:
                    pickle.dump(entry, f)
                
                self._index[entry.key] = {
                    "path": str(entry_path),
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                    "size_bytes": entry.size_bytes,
                    "access_count": entry.access_count,
                }
                self._save_index()
            except (pickle.PickleError, IOError) as e:
                logger.error(f"Failed to cache entry: {e}")
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key not in self._index:
                return False
            
            entry_path = self._get_entry_path(key)
            if entry_path.exists():
                entry_path.unlink()
            
            del self._index[key]
            self._save_index()
            return True
    
    def exists(self, key: str) -> bool:
        return key in self._index
    
    def clear(self) -> int:
        with self._lock:
            count = len(self._index)
            
            for key in list(self._index.keys()):
                entry_path = self._get_entry_path(key)
                if entry_path.exists():
                    entry_path.unlink()
            
            self._index = {}
            self._save_index()
            return count
    
    def keys(self) -> List[str]:
        return list(self._index.keys())
    
    def size(self) -> int:
        return len(self._index)


class RedisBackend(CacheBackend):
    """Redis cache backend."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "argus:",
    ):
        self.prefix = prefix
        
        try:
            import redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
            )
        except ImportError:
            raise ImportError("redis package required: pip install redis")
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        data = self._redis.get(self._make_key(key))
        if data is None:
            return None
        
        entry = pickle.loads(data)
        entry.touch()
        
        # Update in Redis
        ttl = self._redis.ttl(self._make_key(key))
        self._redis.set(
            self._make_key(key),
            pickle.dumps(entry),
            ex=ttl if ttl > 0 else None,
        )
        
        return entry
    
    def set(self, entry: CacheEntry) -> None:
        data = pickle.dumps(entry)
        
        ttl = None
        if entry.expires_at:
            ttl = int((entry.expires_at - datetime.utcnow()).total_seconds())
            if ttl <= 0:
                return  # Already expired
        
        self._redis.set(self._make_key(entry.key), data, ex=ttl)
    
    def delete(self, key: str) -> bool:
        return self._redis.delete(self._make_key(key)) > 0
    
    def exists(self, key: str) -> bool:
        return self._redis.exists(self._make_key(key)) > 0
    
    def clear(self) -> int:
        keys = self._redis.keys(f"{self.prefix}*")
        if keys:
            return self._redis.delete(*keys)
        return 0
    
    def keys(self) -> List[str]:
        redis_keys = self._redis.keys(f"{self.prefix}*")
        return [k.decode().replace(self.prefix, "", 1) for k in redis_keys]
    
    def size(self) -> int:
        return len(self._redis.keys(f"{self.prefix}*"))


class ContextCache:
    """
    Main context caching interface.
    
    Provides intelligent caching for LLM contexts, conversations,
    and intermediate computation results.
    
    Example:
        >>> cache = ContextCache()
        >>> cache.set("conversation:123", messages, ttl=3600)
        >>> messages = cache.get("conversation:123")
        >>> 
        >>> @cache.cached(ttl=600)
        >>> def expensive_computation(x):
        >>>     return compute(x)
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: Optional[int] = None,
        namespace: str = "",
    ):
        self.backend = backend or MemoryBackend()
        self.default_ttl = default_ttl
        self.namespace = namespace
        
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
        logger.debug(f"Context cache initialized (namespace={namespace})")
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        if self.namespace:
            return f"{self.namespace}:{key}"
        return key
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except (pickle.PickleError, TypeError):
            return 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cached value.
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        full_key = self._make_key(key)
        entry = self.backend.get(full_key)
        
        if entry is None:
            with self._lock:
                self._stats.misses += 1
            return default
        
        if entry.is_expired:
            self.delete(key)
            with self._lock:
                self._stats.misses += 1
                self._stats.expirations += 1
            return default
        
        with self._lock:
            self._stats.hits += 1
        
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiration)
            metadata: Optional metadata to store with entry
        """
        full_key = self._make_key(key)
        
        effective_ttl = ttl if ttl is not None else self.default_ttl
        
        now = datetime.utcnow()
        expires_at = None
        if effective_ttl is not None:
            expires_at = now + timedelta(seconds=effective_ttl)
        
        entry = CacheEntry(
            key=full_key,
            value=value,
            created_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
            size_bytes=self._estimate_size(value),
        )
        
        self.backend.set(entry)
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        full_key = self._make_key(key)
        return self.backend.delete(full_key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        full_key = self._make_key(key)
        entry = self.backend.get(full_key)
        
        if entry is None:
            return False
        
        if entry.is_expired:
            self.delete(key)
            return False
        
        return True
    
    def clear(self) -> int:
        """Clear all cached entries."""
        count = self.backend.clear()
        
        with self._lock:
            self._stats.evictions += count
        
        return count
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                total_entries=self.backend.size(),
            )
        return stats
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable[..., str]] = None,
    ) -> Callable:
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time-to-live in seconds
            key_func: Custom function to generate cache key
        
        Example:
            >>> @cache.cached(ttl=300)
            >>> def get_user(user_id):
            >>>     return fetch_user(user_id)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key from function name and arguments
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = ":".join(key_parts)
                
                # Check cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """
        Get cached value or compute and cache if missing.
        
        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time-to-live in seconds
        
        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl=ttl)
        return value


class ConversationCache(ContextCache):
    """
    Specialized cache for conversation/chat history.
    
    Provides efficient storage and retrieval of conversation
    messages with automatic cleanup and size management.
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        max_messages_per_conversation: int = 100,
        default_ttl: int = 3600,
    ):
        super().__init__(backend, default_ttl, namespace="conversation")
        self.max_messages_per_conversation = max_messages_per_conversation
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        return self.get(conversation_id, default=[])
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to a conversation."""
        messages = self.get_messages(conversation_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        messages.append(message)
        
        # Trim to max messages
        if len(messages) > self.max_messages_per_conversation:
            messages = messages[-self.max_messages_per_conversation:]
        
        self.set(conversation_id, messages)
    
    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a user message."""
        self.add_message(conversation_id, "user", content, metadata)
    
    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an assistant message."""
        self.add_message(conversation_id, "assistant", content, metadata)
    
    def add_system_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a system message."""
        self.add_message(conversation_id, "system", content, metadata)
    
    def get_recent_messages(
        self,
        conversation_id: str,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        messages = self.get_messages(conversation_id)
        return messages[-count:] if count < len(messages) else messages
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear all messages from a conversation."""
        return self.delete(conversation_id)
    
    def summarize_and_truncate(
        self,
        conversation_id: str,
        summary: str,
        keep_recent: int = 5,
    ) -> None:
        """
        Replace old messages with a summary.
        
        Useful for managing context window limits.
        """
        messages = self.get_messages(conversation_id)
        
        if len(messages) <= keep_recent:
            return
        
        # Keep system messages and recent messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        recent_messages = messages[-keep_recent:]
        
        # Create summary message
        summary_message = {
            "role": "system",
            "content": f"[Conversation Summary]: {summary}",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"is_summary": True},
        }
        
        # Rebuild messages
        new_messages = system_messages + [summary_message] + recent_messages
        self.set(conversation_id, new_messages)


class EmbeddingCache(ContextCache):
    """
    Specialized cache for embeddings.
    
    Efficiently caches text embeddings to avoid
    redundant embedding computations.
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: int = 86400,  # 24 hours
    ):
        super().__init__(backend, default_ttl, namespace="embedding")
    
    def _make_embedding_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{model}:{text_hash}"
    
    def get_embedding(
        self,
        text: str,
        model: str,
    ) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._make_embedding_key(text, model)
        return self.get(key)
    
    def set_embedding(
        self,
        text: str,
        model: str,
        embedding: List[float],
    ) -> None:
        """Cache an embedding."""
        key = self._make_embedding_key(text, model)
        self.set(key, embedding)
    
    def get_or_compute(
        self,
        text: str,
        model: str,
        compute_func: Callable[[str], List[float]],
    ) -> List[float]:
        """Get cached embedding or compute if not cached."""
        embedding = self.get_embedding(text, model)
        
        if embedding is not None:
            return embedding
        
        embedding = compute_func(text)
        self.set_embedding(text, model, embedding)
        
        return embedding
    
    def batch_get_or_compute(
        self,
        texts: List[str],
        model: str,
        compute_func: Callable[[List[str]], List[List[float]]],
    ) -> List[List[float]]:
        """
        Get cached embeddings or compute missing ones in batch.
        
        Optimizes by batching uncached texts for single computation.
        """
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            embedding = self.get_embedding(text, model)
            if embedding is not None:
                results[i] = embedding
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Compute missing embeddings in batch
        if uncached_texts:
            computed = compute_func(uncached_texts)
            
            for idx, embedding in zip(uncached_indices, computed):
                results[idx] = embedding
                self.set_embedding(texts[idx], model, embedding)
        
        return results


class LLMResponseCache(ContextCache):
    """
    Specialized cache for LLM responses.
    
    Caches LLM responses based on prompt and parameters
    to avoid redundant API calls.
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: int = 3600,
    ):
        super().__init__(backend, default_ttl, namespace="llm_response")
    
    def _make_response_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
        **kwargs,
    ) -> str:
        """Create cache key from prompt and parameters."""
        # Only cache deterministic responses (low temperature)
        if temperature > 0.1:
            # Include random component to differentiate
            import random
            return f"nocache:{random.randint(0, 1000000)}"
        
        # Create deterministic key
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            **{k: v for k, v in sorted(kwargs.items()) if v is not None},
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:32]
        
        return f"{model}:{key_hash}"
    
    def get_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Get cached LLM response."""
        key = self._make_response_key(prompt, model, temperature, **kwargs)
        
        if key.startswith("nocache:"):
            return None
        
        return self.get(key)
    
    def set_response(
        self,
        prompt: str,
        model: str,
        response: Dict[str, Any],
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        """Cache an LLM response."""
        key = self._make_response_key(prompt, model, temperature, **kwargs)
        
        if key.startswith("nocache:"):
            return
        
        self.set(key, response)
    
    def cached_completion(
        self,
        model: str,
        temperature: float = 0.0,
    ) -> Callable:
        """
        Decorator for caching LLM completion calls.
        
        Example:
            >>> @llm_cache.cached_completion(model="gpt-4", temperature=0)
            >>> def generate(prompt):
            >>>     return client.chat.completions.create(...)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(prompt: str, **kwargs):
                # Check cache
                cached = self.get_response(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    **kwargs,
                )
                
                if cached is not None:
                    logger.debug(f"LLM cache hit for model {model}")
                    return cached
                
                # Call function
                response = func(prompt, **kwargs)
                
                # Cache response
                self.set_response(
                    prompt=prompt,
                    model=model,
                    response=response,
                    temperature=temperature,
                    **kwargs,
                )
                
                return response
            
            return wrapper
        return decorator


# Factory functions

def create_cache(
    backend_type: str = "memory",
    **kwargs,
) -> ContextCache:
    """
    Create a context cache with specified backend.
    
    Args:
        backend_type: "memory", "file", or "redis"
        **kwargs: Backend-specific configuration
    
    Returns:
        Configured ContextCache
    """
    if backend_type == "memory":
        backend = MemoryBackend(
            max_entries=kwargs.get("max_entries", 1000),
            max_size_bytes=kwargs.get("max_size_bytes", 100 * 1024 * 1024),
        )
    elif backend_type == "file":
        backend = FileBackend(
            cache_dir=kwargs.get("cache_dir", ".argus_cache"),
        )
    elif backend_type == "redis":
        backend = RedisBackend(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6379),
            db=kwargs.get("db", 0),
            password=kwargs.get("password"),
            prefix=kwargs.get("prefix", "argus:"),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return ContextCache(
        backend=backend,
        default_ttl=kwargs.get("default_ttl"),
        namespace=kwargs.get("namespace", ""),
    )


__all__ = [
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    "MemoryBackend",
    "FileBackend",
    "RedisBackend",
    "ContextCache",
    "ConversationCache",
    "EmbeddingCache",
    "LLMResponseCache",
    "create_cache",
]
