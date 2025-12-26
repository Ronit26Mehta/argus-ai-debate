"""
Base LLM interface for ARGUS.

Defines the abstract interface that all LLM providers must implement,
along with common data structures for requests and responses.
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, Iterator, AsyncIterator
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class Message(BaseModel):
    """
    A single message in a conversation.
    
    Attributes:
        role: Role of the message sender
        content: Message content text
        name: Optional name for function/tool messages
        tool_calls: Optional tool call requests (for assistant messages)
        tool_call_id: Optional tool call ID (for tool response messages)
    """
    
    role: MessageRole = Field(
        description="Role of the message sender",
    )
    
    content: str = Field(
        description="Message content text",
    )
    
    name: Optional[str] = Field(
        default=None,
        description="Optional name for function/tool messages",
    )
    
    tool_calls: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Tool call requests",
    )
    
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call ID for tool responses",
    )
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to provider-agnostic dictionary."""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class LLMConfig(BaseModel):
    """
    Configuration for LLM generation.
    
    Attributes:
        model: Model identifier
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling probability
        top_k: Top-k sampling (if supported)
        stop: Stop sequences
        presence_penalty: Presence penalty (-2 to 2)
        frequency_penalty: Frequency penalty (-2 to 2)
        seed: Random seed for reproducibility
        timeout: Request timeout in seconds
    """
    
    model: str = Field(
        description="Model identifier",
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens to generate",
    )
    
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k sampling",
    )
    
    stop: Optional[list[str]] = Field(
        default=None,
        description="Stop sequences",
    )
    
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty",
    )
    
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty",
    )
    
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="Request timeout in seconds",
    )


@dataclass
class LLMUsage:
    """Token usage statistics from LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """Add usage statistics."""
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    """
    Response from an LLM generation call.
    
    Attributes:
        content: Generated text content
        model: Model that generated the response
        provider: Provider name
        usage: Token usage statistics
        finish_reason: Reason for stopping generation
        latency_ms: Response latency in milliseconds
        raw_response: Original provider response (for debugging)
    """
    content: str
    model: str
    provider: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return bool(self.content) and self.finish_reason != "error"


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers (OpenAI, Anthropic, Gemini, Ollama) implement this interface.
    Supports both synchronous and streaming generation.
    
    Example:
        >>> llm = get_llm("openai", model="gpt-4")
        >>> response = llm.generate("What is 2+2?")
        >>> print(response.content)
        '4'
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize the LLM provider.
        
        Args:
            model: Model identifier
            api_key: API key (if required)
            temperature: Default temperature
            max_tokens: Default max tokens
            **kwargs: Additional provider-specific options
        """
        self.model = model
        self.api_key = api_key
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.extra_config = kwargs
        self._client: Any = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str | list[Message],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt string or list of messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            stop: Stop sequences
            **kwargs: Additional provider-specific options
            
        Returns:
            LLMResponse with generated content and metadata
        """
        pass
    
    @abstractmethod
    def stream(
        self,
        prompt: str | list[Message],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream generated tokens from the LLM.
        
        Args:
            prompt: User prompt string or list of messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            stop: Stop sequences
            **kwargs: Additional options
            
        Yields:
            Generated tokens as strings
        """
        pass
    
    def embed(
        self,
        texts: str | list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings for text(s).
        
        Default implementation raises NotImplementedError.
        Providers that support embeddings should override this.
        
        Args:
            texts: Single text or list of texts to embed
            **kwargs: Additional options
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError(
            f"{self.provider_name} does not support embeddings via this interface. "
            "Use sentence-transformers or a dedicated embedding provider."
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Default implementation uses tiktoken for OpenAI-compatible tokenization.
        Providers may override for more accurate counting.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        try:
            import tiktoken
            # Use cl100k_base as default (GPT-4 compatible)
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4
    
    def _prepare_messages(
        self,
        prompt: str | list[Message],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare messages for API call.
        
        Args:
            prompt: User prompt or message list
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages: list[dict[str, Any]] = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Handle prompt
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            for msg in prompt:
                messages.append(msg.to_dict())
        
        return messages
    
    def _measure_latency(self, start_time: float) -> float:
        """Calculate latency in milliseconds."""
        return (time.time() - start_time) * 1000
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model='{self.model}')"


class EmbeddingModel:
    """
    Dedicated embedding model wrapper.
    
    Uses sentence-transformers for local embedding generation.
    Supports caching and batching for efficiency.
    
    Example:
        >>> embedder = EmbeddingModel("all-MiniLM-L6-v2")
        >>> vectors = embedder.embed(["Hello world", "Goodbye world"])
        >>> print(len(vectors[0]))  # 384 dimensions
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Sentence-transformer model name
            device: Device to use (cpu, cuda, mps)
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model: Any = None
        self._dimension: Optional[int] = None
    
    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                kwargs = {}
                if self.device:
                    kwargs["device"] = self.device
                
                self._model = SentenceTransformer(self.model_name, **kwargs)
                self._dimension = self._model.get_sentence_embedding_dimension()
                
                logger.info(
                    f"Loaded embedding model '{self.model_name}' "
                    f"(dim={self._dimension}, device={self._model.device})"
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._dimension or 384
    
    def embed(
        self,
        texts: str | list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query (single text).
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed(query)[0]
    
    def embed_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            batch_size: Batch size
            
        Returns:
            List of embedding vectors
        """
        return self.embed(documents, batch_size=batch_size)
