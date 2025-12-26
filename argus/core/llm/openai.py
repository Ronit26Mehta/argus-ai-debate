"""
OpenAI LLM provider implementation.

Supports GPT-4, GPT-4 Turbo, GPT-3.5, and OpenAI embedding models.
Uses the official openai Python SDK.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import (
    BaseLLM,
    LLMResponse,
    LLMUsage,
    Message,
)

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM provider.
    
    Supports GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, and other OpenAI models.
    Uses the official openai SDK with retry logic.
    
    Example:
        >>> from argus.core.llm import OpenAILLM
        >>> llm = OpenAILLM(model="gpt-4", api_key="sk-...")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response.content)
    
    Supported Models:
        - gpt-4, gpt-4-turbo, gpt-4-turbo-preview
        - gpt-4o, gpt-4o-mini
        - gpt-3.5-turbo, gpt-3.5-turbo-16k
        - o1-preview, o1-mini (reasoning models)
    """
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        "gpt4": "gpt-4",
        "gpt4o": "gpt-4o",
        "gpt4-turbo": "gpt-4-turbo",
        "gpt35": "gpt-3.5-turbo",
        "gpt-3.5": "gpt-3.5-turbo",
    }
    
    # Embedding models
    EMBEDDING_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI LLM provider.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o")
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
            temperature: Default sampling temperature
            max_tokens: Default max tokens
            organization: OpenAI organization ID
            base_url: Custom API base URL (for proxies)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional options
        """
        # Resolve model aliases
        resolved_model = self.MODEL_ALIASES.get(model, model)
        
        super().__init__(
            model=resolved_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize client
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"
    
    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        
        # Build client kwargs
        client_kwargs: dict[str, Any] = {}
        
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        # If no key provided, openai SDK will look for OPENAI_API_KEY env var
        
        if self.organization:
            client_kwargs["organization"] = self.organization
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        client_kwargs["timeout"] = self.timeout
        client_kwargs["max_retries"] = self.max_retries
        
        self._client = OpenAI(**client_kwargs)
        logger.debug(f"Initialized OpenAI client for model '{self.model}'")
    
    def generate(
        self,
        prompt: str | list[Message],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        response_format: Optional[dict[str, str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from OpenAI.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            response_format: Response format (e.g., {"type": "json_object"})
            seed: Random seed for reproducibility
            **kwargs: Additional OpenAI-specific options
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages
        messages = self._prepare_messages(prompt, system_prompt)
        
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        
        # Add optional parameters
        if stop:
            request_kwargs["stop"] = stop
        
        if response_format:
            request_kwargs["response_format"] = response_format
        
        if seed is not None:
            request_kwargs["seed"] = seed
        
        # Handle reasoning models (o1-*) which don't support temperature
        if self.model.startswith("o1-"):
            request_kwargs.pop("temperature", None)
            request_kwargs.pop("max_tokens", None)
            if "max_completion_tokens" not in kwargs:
                request_kwargs["max_completion_tokens"] = max_tokens or self.default_max_tokens
        
        # Add any extra kwargs
        request_kwargs.update(kwargs)
        
        try:
            # Make API call
            response = self._client.chat.completions.create(**request_kwargs)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Build usage
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=choice.finish_reason,
                latency_ms=self._measure_latency(start_time),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                finish_reason="error",
                latency_ms=self._measure_latency(start_time),
                raw_response={"error": str(e)},
            )
    
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
        Stream generated tokens from OpenAI.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            stop: Stop sequences
            **kwargs: Additional options
            
        Yields:
            Generated tokens as strings
        """
        # Prepare messages
        messages = self._prepare_messages(prompt, system_prompt)
        
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stream": True,
        }
        
        if stop:
            request_kwargs["stop"] = stop
        
        request_kwargs.update(kwargs)
        
        try:
            # Create streaming response
            stream = self._client.chat.completions.create(**request_kwargs)
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def embed(
        self,
        texts: str | list[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings using OpenAI.
        
        Args:
            texts: Text(s) to embed
            model: Embedding model (default: text-embedding-3-small)
            **kwargs: Additional options
            
        Returns:
            List of embedding vectors
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Use default embedding model
        embed_model = model or "text-embedding-3-small"
        
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=embed_model,
                **kwargs,
            )
            
            # Sort by index and extract embeddings
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [e.embedding for e in embeddings]
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken for accurate OpenAI tokenization.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        try:
            import tiktoken
            
            # Get encoding for model
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to cl100k_base for newer models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
            
        except ImportError:
            # Fallback to base implementation
            return super().count_tokens(text)
