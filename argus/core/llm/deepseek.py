"""
DeepSeek LLM provider implementation.

Supports DeepSeek models with reasoning token capture.
Uses OpenAI-compatible API.
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


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM provider.
    
    Supports DeepSeek-V3, DeepSeek-Coder, and reasoning models.
    Uses OpenAI-compatible API with reasoning token support.
    
    Example:
        >>> from argus.core.llm import DeepSeekLLM
        >>> llm = DeepSeekLLM(model="deepseek-chat", api_key="sk-...")
        >>> response = llm.generate("Solve this math problem step by step")
    
    Supported Models:
        - deepseek-chat (DeepSeek-V3)
        - deepseek-coder
        - deepseek-reasoner (with reasoning tokens)
    """
    
    BASE_URL = "https://api.deepseek.com/v1"
    
    MODEL_ALIASES = {
        "deepseek": "deepseek-chat",
        "deepseek-v3": "deepseek-chat",
        "coder": "deepseek-coder",
    }
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        resolved_model = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved_model, api_key, temperature, max_tokens, **kwargs)
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "deepseek"
    
    def _init_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        import os
        api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)
        logger.debug(f"Initialized DeepSeek client for '{self.model}'")
    
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
        start_time = time.time()
        messages = self._prepare_messages(prompt, system_prompt)
        
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if stop:
            request_kwargs["stop"] = stop
        request_kwargs.update(kwargs)
        
        try:
            response = self._client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            
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
            logger.error(f"DeepSeek generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
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
        messages = self._prepare_messages(prompt, system_prompt)
        request_kwargs = {
            "model": self.model, "messages": messages, "stream": True,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if stop:
            request_kwargs["stop"] = stop
        
        try:
            stream = self._client.chat.completions.create(**request_kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"DeepSeek streaming failed: {e}")
            yield f"[Error: {e}]"
