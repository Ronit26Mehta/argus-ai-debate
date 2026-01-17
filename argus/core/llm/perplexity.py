"""
Perplexity LLM provider implementation.

Supports search-grounded AI responses via OpenAI-compatible API.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class PerplexityLLM(BaseLLM):
    """
    Perplexity LLM provider with search grounding.
    
    Example:
        >>> llm = PerplexityLLM(model="llama-3.1-sonar-large-128k-online")
        >>> response = llm.generate("What are the latest AI news?")
    
    Models:
        - llama-3.1-sonar-small-128k-online
        - llama-3.1-sonar-large-128k-online
        - llama-3.1-sonar-huge-128k-online
    """
    
    BASE_URL = "https://api.perplexity.ai"
    
    MODEL_ALIASES = {
        "sonar": "llama-3.1-sonar-large-128k-online",
        "sonar-small": "llama-3.1-sonar-small-128k-online",
        "sonar-huge": "llama-3.1-sonar-huge-128k-online",
    }
    
    def __init__(
        self,
        model: str = "llama-3.1-sonar-large-128k-online",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "perplexity"
    
    def _init_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required")
        import os
        api_key = self.api_key or os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        logger.debug(f"Initialized Perplexity client for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, return_citations: bool = False,
                 **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        messages = self._prepare_messages(prompt, system_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens, stop=stop, **kwargs
            )
            choice = response.choices[0]
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )
            return LLMResponse(content=choice.message.content or "", model=response.model,
                             provider=self.provider_name, usage=usage,
                             finish_reason=choice.finish_reason,
                             latency_ms=self._measure_latency(start_time), raw_response=response)
        except Exception as e:
            logger.error(f"Perplexity generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        messages = self._prepare_messages(prompt, system_prompt)
        try:
            stream = self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens, stop=stop
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[Error: {e}]"
