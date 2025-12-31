"""
Cohere LLM provider implementation.

Supports Cohere Command R, Command R+, and other models.
Uses the official cohere Python SDK.
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


class CohereLLM(BaseLLM):
    """
    Cohere LLM provider.
    
    Supports Command R, Command R+, and other Cohere models.
    Uses the official cohere SDK with chat endpoint.
    
    Example:
        >>> from argus.core.llm import CohereLLM
        >>> llm = CohereLLM(model="command-r-plus", api_key="...")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response.content)
    
    Supported Models:
        - command-r-plus (recommended)
        - command-r
        - command
        - command-light
    """
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        "cohere": "command-r-plus",
        "command": "command-r",
        "command-plus": "command-r-plus",
        "c4ai": "command-r-plus",
    }
    
    def __init__(
        self,
        model: str = "command-r-plus",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize Cohere LLM provider.
        
        Args:
            model: Model name (e.g., "command-r-plus")
            api_key: Cohere API key (or COHERE_API_KEY env var)
            temperature: Default sampling temperature
            max_tokens: Default max tokens
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
        
        # Initialize client
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "cohere"
    
    def _init_client(self) -> None:
        """Initialize the Cohere client."""
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere package is required. Install with: pip install cohere"
            )
        
        # Create client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        # If no key, cohere SDK will look for COHERE_API_KEY env var
        
        self._client = cohere.ClientV2(**client_kwargs)
        logger.debug(f"Initialized Cohere client for model '{self.model}'")
    
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
        Generate a response from Cohere.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            **kwargs: Additional Cohere-specific options
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages
        messages = self._prepare_cohere_messages(prompt, system_prompt)
        
        try:
            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            }
            
            if stop:
                request_kwargs["stop_sequences"] = stop
            
            # Add any extra kwargs
            request_kwargs.update(kwargs)
            
            # Make API call
            response = self._client.chat(**request_kwargs)
            
            # Extract response content
            content = ""
            if response.message and response.message.content:
                for block in response.message.content:
                    if hasattr(block, "text"):
                        content += block.text
            
            # Build usage
            usage = LLMUsage()
            if hasattr(response, "usage") and response.usage:
                usage = LLMUsage(
                    prompt_tokens=getattr(response.usage.tokens, "input_tokens", 0),
                    completion_tokens=getattr(response.usage.tokens, "output_tokens", 0),
                    total_tokens=(
                        getattr(response.usage.tokens, "input_tokens", 0) +
                        getattr(response.usage.tokens, "output_tokens", 0)
                    ),
                )
            
            # Determine finish reason
            finish_reason = "stop"
            if hasattr(response, "finish_reason"):
                finish_reason = str(response.finish_reason).lower()
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=self._measure_latency(start_time),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Cohere generation failed: {e}")
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
        Stream generated tokens from Cohere.
        
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
        messages = self._prepare_cohere_messages(prompt, system_prompt)
        
        try:
            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            }
            
            if stop:
                request_kwargs["stop_sequences"] = stop
            
            request_kwargs.update(kwargs)
            
            # Create streaming response
            for event in self._client.chat_stream(**request_kwargs):
                if hasattr(event, "type") and event.type == "content-delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "message"):
                        if event.delta.message.content:
                            for block in event.delta.message.content:
                                if hasattr(block, "text"):
                                    yield block.text
                    
        except Exception as e:
            logger.error(f"Cohere streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def _prepare_cohere_messages(
        self,
        prompt: str | list[Message],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare messages in Cohere format.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            
        Returns:
            Message list for Cohere API
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        if isinstance(prompt, str):
            messages.append({
                "role": "user",
                "content": prompt,
            })
        elif isinstance(prompt, list):
            for msg in prompt:
                role = msg.role.value
                # Map roles to Cohere format
                if role == "assistant":
                    role = "assistant"
                elif role in ("user", "human"):
                    role = "user"
                elif role == "system":
                    role = "system"
                else:
                    role = "user"
                
                messages.append({
                    "role": role,
                    "content": msg.content,
                })
        
        return messages
    
    def embed(
        self,
        texts: str | list[str],
        model: Optional[str] = None,
        input_type: str = "search_document",
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings using Cohere.
        
        Args:
            texts: Text(s) to embed
            model: Embedding model (default: embed-english-v3.0)
            input_type: search_document, search_query, classification, clustering
            **kwargs: Additional options
            
        Returns:
            List of embedding vectors
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        embed_model = model or "embed-english-v3.0"
        
        try:
            response = self._client.embed(
                texts=texts,
                model=embed_model,
                input_type=input_type,
                embedding_types=["float"],
                **kwargs,
            )
            
            # Extract float embeddings
            if hasattr(response, "embeddings") and response.embeddings:
                if hasattr(response.embeddings, "float_"):
                    return response.embeddings.float_
                elif hasattr(response.embeddings, "float"):
                    return response.embeddings.float
            
            return []
            
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Cohere's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        try:
            response = self._client.tokenize(
                text=text,
                model=self.model,
            )
            return len(response.tokens)
        except Exception:
            # Fallback to base implementation
            return super().count_tokens(text)
