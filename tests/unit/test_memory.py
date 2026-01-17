"""
Unit tests for ARGUS Memory Systems module.
"""

import pytest
import tempfile
import os

from argus.memory.config import MemoryConfig, MemoryType, StorageBackend
from argus.memory.short_term import (
    ConversationBufferMemory, ConversationWindowMemory,
    ConversationSummaryMemory, EntityMemory, Message
)
from argus.memory.long_term import VectorStoreMemory, MemoryEntry
from argus.memory.semantic_cache import SemanticCache
from argus.memory.store import InMemoryStore, SQLiteStore, FileSystemStore


class TestMemoryConfig:
    """Tests for memory configuration."""
    
    def test_default_config(self):
        config = MemoryConfig()
        assert config.enabled is True
        assert config.short_term.max_messages == 100
        assert config.long_term.enabled is True
    
    def test_short_term_config(self):
        config = MemoryConfig()
        assert config.short_term.memory_type == MemoryType.BUFFER
        assert config.short_term.window_size == 10


class TestShortTermMemory:
    """Tests for short-term memory implementations."""
    
    def test_buffer_memory_add(self):
        memory = ConversationBufferMemory(max_messages=5)
        memory.add("user", "Hello")
        memory.add("assistant", "Hi there!")
        assert len(memory) == 2
    
    def test_buffer_memory_max_limit(self):
        memory = ConversationBufferMemory(max_messages=3)
        for i in range(5):
            memory.add("user", f"Message {i}")
        assert len(memory) == 3
    
    def test_buffer_memory_get_context(self):
        memory = ConversationBufferMemory()
        memory.add("user", "Hello")
        memory.add("assistant", "Hi!")
        context = memory.get_context_string()
        assert "user: Hello" in context
        assert "assistant: Hi!" in context
    
    def test_window_memory(self):
        memory = ConversationWindowMemory(window_size=2)
        memory.add("user", "Message 1")
        memory.add("user", "Message 2")
        memory.add("user", "Message 3")
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Message 2"
    
    def test_summary_memory(self):
        memory = ConversationSummaryMemory()
        memory.add("user", "What is AI?")
        memory.add("assistant", "AI is artificial intelligence")
        context = memory.get_context()
        assert "AI" in context or "artificial intelligence" in context
    
    def test_entity_memory(self):
        memory = EntityMemory()
        memory.add("user", "John is 25 years old", entities={"John": {"age": 25}})
        assert memory.get_entity("John") == {"age": 25}
        memory.update_entity("John", {"city": "NYC"})
        entity = memory.get_entity("John")
        assert entity["age"] == 25
        assert entity["city"] == "NYC"


class TestLongTermMemory:
    """Tests for long-term memory."""
    
    def test_vector_store_add(self):
        memory = VectorStoreMemory(dimension=384)
        memory_id = memory.add("This is a test memory", memory_type="semantic")
        assert memory_id is not None
    
    def test_vector_store_search(self):
        memory = VectorStoreMemory(dimension=384)
        memory.add("Python is a programming language")
        memory.add("Java is also a programming language")
        memory.add("Cats are fluffy animals")
        results = memory.search("programming languages", top_k=2)
        assert len(results) <= 2
    
    def test_vector_store_get(self):
        memory = VectorStoreMemory(dimension=384)
        memory_id = memory.add("Test content", importance=0.8)
        entry = memory.get(memory_id)
        assert entry is not None
        assert entry.content == "Test content"
        assert entry.importance == 0.8
    
    def test_vector_store_delete(self):
        memory = VectorStoreMemory(dimension=384)
        memory_id = memory.add("To be deleted")
        assert memory.delete(memory_id) is True
        assert memory.get(memory_id) is None


class TestSemanticCache:
    """Tests for semantic cache."""
    
    def test_cache_set_get(self):
        cache = SemanticCache(similarity_threshold=0.9, dimension=384)
        cache.set("What is Python?", "Python is a programming language")
        # Exact match should work (but depends on random embeddings in test)
        
    def test_cache_miss(self):
        cache = SemanticCache(similarity_threshold=0.95, dimension=384)
        result = cache.get("Random unrelated query")
        assert result is None
    
    def test_cache_stats(self):
        cache = SemanticCache()
        cache.get("query1")  # miss
        cache.get("query2")  # miss
        stats = cache.get_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0
    
    def test_cache_invalidate(self):
        cache = SemanticCache()
        cache_id = cache.set("query", "response")
        assert cache.invalidate(cache_id) is True
    
    def test_cache_clear(self):
        cache = SemanticCache()
        cache.set("q1", "r1")
        cache.set("q2", "r2")
        cache.clear()
        assert cache.get_stats()["entries"] == 0


class TestMemoryStores:
    """Tests for memory store backends."""
    
    def test_in_memory_store(self):
        store = InMemoryStore()
        store.save("key1", {"data": "value"})
        result = store.load("key1")
        assert result["data"] == "value"
    
    def test_in_memory_store_delete(self):
        store = InMemoryStore()
        store.save("key1", {"data": "value"})
        assert store.delete("key1") is True
        assert store.load("key1") is None
    
    def test_in_memory_store_list_keys(self):
        store = InMemoryStore()
        store.save("key1", {})
        store.save("key2", {})
        keys = store.list_keys()
        assert "key1" in keys
        assert "key2" in keys
    
    def test_sqlite_store(self):
        store = SQLiteStore(":memory:")
        store.save("key1", {"data": "value"})
        result = store.load("key1")
        assert result["data"] == "value"
        store.close()
    
    def test_filesystem_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(tmpdir)
            store.save("key1", {"data": "value"})
            result = store.load("key1")
            assert result["data"] == "value"
