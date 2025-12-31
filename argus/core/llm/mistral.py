"""
Mistral AI LLM provider implementation.

Supports Mistral Large, Medium, Small, Codestral, and other models.
Uses the official mistralai Python SDK.
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


class MistralLLM(BaseLLM):
    """
    Mistral AI LLM provider.
    
    Supports Mistral Large, Medium, Small, Codestral, and other models.
    Uses the official mistralai SDK.
    
    Example:
        >>> from argus.core.llm import MistralLLM
        >>> llm = MistralLLM(model="mistral-large-latest", api_key="...")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response.content)
    
    Supported Models:
        - mistral-large-latest (recommended)
        - mistral-medium-latest
        - mistral-small-latest
        - codestral-latest (code-focused)
        - open-mistral-7b
        - open-mixtral-8x7b
        - open-mixtral-8x22b
    """
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        "mistral": "mistral-large-latest",
        "mistral-large": "mistral-large-latest",
        "mistral-medium": "mistral-medium-latest",
        "mistral-small": "mistral-small-latest",
        "codestral": "codestral-latest",
        "mixtral": "open-mixtral-8x7b",
        "mixtral-8x22b": "open-mixtral-8x22b",
    }
    
    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize Mistral LLM provider.
        
        Args:
            model: Model name (e.g., "mistral-large-latest")
            api_key: Mistral API key (or MISTRAL_API_KEY env var)
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
        return "mistral"
    
    def _init_client(self) -> None:
        """Initialize the Mistral client."""
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "mistralai package is required. Install with: pip install mistralai"
            )
        
        # Create client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        # If no key, SDK will look for MISTRAL_API_KEY env var
        
        self._client = Mistral(**client_kwargs)
        logger.debug(f"Initialized Mistral client for model '{self.model}'")
    
    def generate(
        self,
        prompt: str | list[Message],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        safe_prompt: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from Mistral.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            safe_prompt: Enable Mistral's safety mode
            **kwargs: Additional Mistral-specific options
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages
        messages = self._prepare_mistral_messages(prompt, system_prompt)
        
        try:
            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            }
            
            if stop:
                request_kwargs["stop"] = stop
            
            if safe_prompt:
                request_kwargs["safe_prompt"] = True
            
            # Add any extra kwargs
            request_kwargs.update(kwargs)
            
            # Make API call
            response = self._client.chat.complete(**request_kwargs)
            
            # Extract response content
            content = ""
            finish_reason = "stop"
            
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content
                if choice.finish_reason:
                    finish_reason = str(choice.finish_reason).lower()
            
            # Build usage
            usage = LLMUsage()
            if response.usage:
                usage = LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0,
                )
            
            return LLMResponse(
                content=content,
                model=response.model or self.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=self._measure_latency(start_time),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Mistral generation failed: {e}")
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
        Stream generated tokens from Mistral.
        
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
        messages = self._prepare_mistral_messages(prompt, system_prompt)
        
        try:
            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            }
            
            if stop:
                request_kwargs["stop"] = stop
            
            request_kwargs.update(kwargs)
            
            # Create streaming response
            stream = self._client.chat.stream(**request_kwargs)
            
            for event in stream:
                if event.data.choices:
                    delta = event.data.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                    
        except Exception as e:
            logger.error(f"Mistral streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def _prepare_mistral_messages(
        self,
        prompt: str | list[Message],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare messages in Mistral format.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            
        Returns:
            Message list for Mistral API
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
                # Mistral uses standard OpenAI-like roles
                if role not in ("system", "user", "assistant"):
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
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings using Mistral.
        
        Args:
            texts: Text(s) to embed
            model: Embedding model (default: mistral-embed)
            **kwargs: Additional options
            
        Returns:
            List of embedding vectors
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        embed_model = model or "mistral-embed"
        
        try:
            response = self._client.embeddings.create(
                model=embed_model,
                inputs=texts,
                **kwargs,
            )
            
            # Extract embeddings
            if response.data:
                return [item.embedding for item in response.data]
            
            return []
            
        except Exception as e:
            logger.error(f"Mistral embedding failed: {e}")
            raise
