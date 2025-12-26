"""
Anthropic (Claude) LLM provider implementation.

Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and other Anthropic models.
Uses the official anthropic Python SDK.
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


class AnthropicLLM(BaseLLM):
    """
    Anthropic (Claude) LLM provider.
    
    Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and other models.
    Uses the official anthropic SDK with retry logic.
    
    Example:
        >>> from argus.core.llm import AnthropicLLM
        >>> llm = AnthropicLLM(model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response.content)
    
    Supported Models:
        - claude-3-5-sonnet-20241022 (latest Sonnet)
        - claude-3-opus-20240229
        - claude-3-sonnet-20240229
        - claude-3-haiku-20240307
    """
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        "claude": "claude-3-5-sonnet-20241022",
        "claude-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5": "claude-3-5-sonnet-20241022",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-opus": "claude-3-opus-20240229",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-haiku": "claude-3-haiku-20240307",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }
    
    # Default max tokens per model
    MODEL_MAX_TOKENS = {
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
    }
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize Anthropic LLM provider.
        
        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
            temperature: Default sampling temperature
            max_tokens: Default max tokens
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
        
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize client
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"
    
    def _init_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )
        
        # Build client kwargs
        client_kwargs: dict[str, Any] = {}
        
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        # If no key provided, anthropic SDK will look for ANTHROPIC_API_KEY env var
        
        client_kwargs["timeout"] = self.timeout
        client_kwargs["max_retries"] = self.max_retries
        
        self._client = Anthropic(**client_kwargs)
        logger.debug(f"Initialized Anthropic client for model '{self.model}'")
    
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
        Generate a response from Anthropic.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            **kwargs: Additional Anthropic-specific options
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages (Anthropic format: no system in messages)
        messages = self._prepare_anthropic_messages(prompt)
        
        # Get max tokens (Anthropic requires this)
        actual_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": actual_max_tokens,
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_kwargs["system"] = system_prompt
        
        # Temperature (only if not 0, Anthropic doesn't like temp=0)
        temp = temperature if temperature is not None else self.default_temperature
        if temp > 0:
            request_kwargs["temperature"] = temp
        
        if stop:
            request_kwargs["stop_sequences"] = stop
        
        # Add any extra kwargs
        request_kwargs.update(kwargs)
        
        try:
            # Make API call
            response = self._client.messages.create(**request_kwargs)
            
            # Extract content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
            
            # Build usage
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage else 0
                ),
            )
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=response.stop_reason,
                latency_ms=self._measure_latency(start_time),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
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
        Stream generated tokens from Anthropic.
        
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
        messages = self._prepare_anthropic_messages(prompt)
        
        actual_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": actual_max_tokens,
        }
        
        if system_prompt:
            request_kwargs["system"] = system_prompt
        
        temp = temperature if temperature is not None else self.default_temperature
        if temp > 0:
            request_kwargs["temperature"] = temp
        
        if stop:
            request_kwargs["stop_sequences"] = stop
        
        request_kwargs.update(kwargs)
        
        try:
            # Create streaming response
            with self._client.messages.stream(**request_kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def _prepare_anthropic_messages(
        self,
        prompt: str | list[Message],
    ) -> list[dict[str, Any]]:
        """
        Prepare messages in Anthropic format.
        
        Anthropic requires alternating user/assistant messages,
        with system prompt passed separately.
        
        Args:
            prompt: User prompt or message list
            
        Returns:
            List of message dictionaries
        """
        messages: list[dict[str, Any]] = []
        
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            for msg in prompt:
                # Skip system messages (handled separately in Anthropic)
                if msg.role.value == "system":
                    continue
                
                # Map roles
                role = msg.role.value
                if role == "assistant":
                    role = "assistant"
                else:
                    role = "user"
                
                messages.append({"role": role, "content": msg.content})
        
        return messages
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Anthropic's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        try:
            # Anthropic provides a count_tokens method
            return self._client.count_tokens(text)
        except Exception:
            # Fallback to base implementation
            return super().count_tokens(text)
