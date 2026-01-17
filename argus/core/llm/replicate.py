"""
Replicate LLM provider implementation.

Supports models hosted on Replicate.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class ReplicateLLM(BaseLLM):
    """
    Replicate LLM provider.
    
    Example:
        >>> llm = ReplicateLLM(model="meta/meta-llama-3.1-405b-instruct")
        >>> response = llm.generate("Explain ML deployment")
    
    Models: meta/meta-llama-*, mistralai/*, meta/llama-*
    """
    
    MODEL_ALIASES = {
        "llama-3.1-405b": "meta/meta-llama-3.1-405b-instruct",
        "llama-3.1-70b": "meta/meta-llama-3.1-70b-instruct",
        "llama-3.1-8b": "meta/meta-llama-3.1-8b-instruct",
    }
    
    def __init__(
        self,
        model: str = "meta/meta-llama-3.1-405b-instruct",
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
        return "replicate"
    
    def _init_client(self) -> None:
        try:
            import replicate
            self._replicate = replicate
        except ImportError:
            raise ImportError("replicate required. Install: pip install replicate")
        import os
        if self.api_key:
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
        logger.debug(f"Initialized Replicate for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        
        if isinstance(prompt, list):
            text = "\n".join(f'{m.role.value}: {m.content}' for m in prompt)
        else:
            text = prompt
        
        input_data = {
            "prompt": text,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature,
        }
        if system_prompt:
            input_data["system_prompt"] = system_prompt
        
        try:
            output = self._replicate.run(self.model, input=input_data)
            content = "".join(output) if hasattr(output, '__iter__') else str(output)
            return LLMResponse(content=content, model=self.model, provider=self.provider_name,
                             finish_reason="stop", latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"Replicate generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        if isinstance(prompt, list):
            text = "\n".join(f'{m.role.value}: {m.content}' for m in prompt)
        else:
            text = prompt
        
        input_data = {
            "prompt": text,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature,
        }
        if system_prompt:
            input_data["system_prompt"] = system_prompt
        
        try:
            for event in self._replicate.stream(self.model, input=input_data):
                yield str(event)
        except Exception as e:
            yield f"[Error: {e}]"
