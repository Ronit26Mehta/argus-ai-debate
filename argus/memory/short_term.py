"""
ARGUS Short-Term Memory.

Conversation memory for maintaining context within sessions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, List

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, 
                "timestamp": self.timestamp.isoformat(), "metadata": self.metadata}


class BaseShortTermMemory(ABC):
    """Abstract base for short-term memory."""
    
    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self._messages: List[Message] = []
    
    @abstractmethod
    def add(self, role: str, content: str, **kwargs: Any) -> None:
        pass
    
    @abstractmethod
    def get_messages(self) -> List[Message]:
        pass
    
    def clear(self) -> None:
        self._messages.clear()
    
    def __len__(self) -> int:
        return len(self._messages)


class ConversationBufferMemory(BaseShortTermMemory):
    """Full conversation history buffer."""
    
    def add(self, role: str, content: str, **kwargs: Any) -> None:
        msg = Message(role=role, content=content, metadata=kwargs)
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
    
    def get_messages(self) -> List[Message]:
        return self._messages.copy()
    
    def get_context_string(self) -> str:
        """Get messages as formatted string."""
        return "\n".join(f"{m.role}: {m.content}" for m in self._messages)


class ConversationWindowMemory(BaseShortTermMemory):
    """Sliding window memory - keeps last K messages."""
    
    def __init__(self, window_size: int = 10, max_messages: int = 100):
        super().__init__(max_messages)
        self.window_size = window_size
    
    def add(self, role: str, content: str, **kwargs: Any) -> None:
        msg = Message(role=role, content=content, metadata=kwargs)
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
    
    def get_messages(self) -> List[Message]:
        return self._messages[-self.window_size:]
    
    def get_full_history(self) -> List[Message]:
        return self._messages.copy()


class ConversationSummaryMemory(BaseShortTermMemory):
    """Memory that maintains a running summary."""
    
    def __init__(self, max_messages: int = 100, summary_threshold: int = 10,
                 summarizer: Optional[Any] = None):
        super().__init__(max_messages)
        self.summary_threshold = summary_threshold
        self.summarizer = summarizer
        self._summary: str = ""
        self._unsummarized: List[Message] = []
    
    def add(self, role: str, content: str, **kwargs: Any) -> None:
        msg = Message(role=role, content=content, metadata=kwargs)
        self._messages.append(msg)
        self._unsummarized.append(msg)
        if len(self._unsummarized) >= self.summary_threshold:
            self._update_summary()
    
    def _update_summary(self) -> None:
        if not self._unsummarized:
            return
        text = "\n".join(f"{m.role}: {m.content}" for m in self._unsummarized)
        if self.summarizer:
            try:
                new_summary = self.summarizer(self._summary, text)
                self._summary = new_summary
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
                self._summary += f"\n{text}"
        else:
            self._summary += f"\n{text}"
        self._unsummarized.clear()
    
    def get_messages(self) -> List[Message]:
        return self._unsummarized.copy()
    
    def get_summary(self) -> str:
        return self._summary.strip()
    
    def get_context(self) -> str:
        recent = "\n".join(f"{m.role}: {m.content}" for m in self._unsummarized)
        if self._summary:
            return f"Summary: {self._summary}\n\nRecent:\n{recent}"
        return recent


class EntityMemory(BaseShortTermMemory):
    """Memory that tracks entities mentioned in conversation."""
    
    def __init__(self, max_messages: int = 100):
        super().__init__(max_messages)
        self._entities: dict[str, dict[str, Any]] = {}
    
    def add(self, role: str, content: str, entities: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        msg = Message(role=role, content=content, metadata=kwargs)
        self._messages.append(msg)
        if entities:
            for name, info in entities.items():
                if name in self._entities:
                    self._entities[name].update(info)
                else:
                    self._entities[name] = info
    
    def get_messages(self) -> List[Message]:
        return self._messages.copy()
    
    def get_entity(self, name: str) -> Optional[dict[str, Any]]:
        return self._entities.get(name)
    
    def get_all_entities(self) -> dict[str, dict[str, Any]]:
        return self._entities.copy()
    
    def update_entity(self, name: str, info: dict[str, Any]) -> None:
        if name in self._entities:
            self._entities[name].update(info)
        else:
            self._entities[name] = info


class ShortTermMemoryManager:
    """Manager for short-term memory with multiple strategies."""
    
    def __init__(self, memory_type: str = "buffer", **kwargs: Any):
        self.memory_type = memory_type
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(**kwargs)
        elif memory_type == "window":
            self.memory = ConversationWindowMemory(**kwargs)
        elif memory_type == "summary":
            self.memory = ConversationSummaryMemory(**kwargs)
        elif memory_type == "entity":
            self.memory = EntityMemory(**kwargs)
        else:
            self.memory = ConversationBufferMemory(**kwargs)
    
    def add(self, role: str, content: str, **kwargs: Any) -> None:
        self.memory.add(role, content, **kwargs)
    
    def get_messages(self) -> List[Message]:
        return self.memory.get_messages()
    
    def clear(self) -> None:
        self.memory.clear()
