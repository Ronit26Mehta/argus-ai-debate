"""
Cerebras LLM provider implementation.

Ultra-fast inference on Cerebras hardware.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class CerebrasLLM(BaseLLM):
    """
    Cerebras LLM provider - fastest inference.
    
    Example:
        >>> llm = CerebrasLLM(model="llama3.1-70b")
        >>> response = llm.generate("Explain wafer-scale computing")
    
    Models: llama3.1-70b, llama3.1-8b, llama-3.3-70b
    """
    
    BASE_URL = "https://api.cerebras.ai/v1"
    
    MODEL_ALIASES = {
        "llama-70b": "llama3.1-70b",
        "llama-8b": "llama3.1-8b",
    }
    
    def __init__(
        self,
        model: str = "llama3.1-70b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "cerebras"
    
    def _init_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required")
        import os
        api_key = self.api_key or os.getenv("CEREBRAS_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        logger.debug(f"Initialized Cerebras client for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
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
                             latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"Cerebras generation failed: {e}")
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
