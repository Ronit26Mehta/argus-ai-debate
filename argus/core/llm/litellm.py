"""
LiteLLM proxy LLM provider implementation.

Universal proxy supporting 100+ providers through a single interface.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class LiteLLMLLM(BaseLLM):
    """
    LiteLLM universal proxy provider.
    
    Example:
        >>> llm = LiteLLMLLM(model="gpt-4")  # Uses OpenAI
        >>> llm = LiteLLMLLM(model="claude-3-5-sonnet-20241022")  # Uses Anthropic
        >>> llm = LiteLLMLLM(model="ollama/llama3")  # Uses Ollama
    
    Supports: OpenAI, Anthropic, Azure, AWS Bedrock, Vertex AI, HuggingFace, Ollama, and 100+ more
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)
        self.api_base = api_base
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "litellm"
    
    def _init_client(self) -> None:
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError("litellm required. Install: pip install litellm")
        
        import os
        if self.api_key:
            # Set appropriate API key based on model prefix
            if "anthropic" in self.model or "claude" in self.model:
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            elif "openai" in self.model or "gpt" in self.model:
                os.environ["OPENAI_API_KEY"] = self.api_key
        
        logger.debug(f"Initialized LiteLLM for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        messages = self._prepare_messages(prompt, system_prompt)
        
        try:
            response = self._litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                stop=stop,
                api_base=self.api_base,
                **kwargs
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
            logger.error(f"LiteLLM generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        messages = self._prepare_messages(prompt, system_prompt)
        try:
            response = self._litellm.completion(
                model=self.model, messages=messages, stream=True,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens, stop=stop,
                api_base=self.api_base
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[Error: {e}]"
    
    def embed(self, texts: str | list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embed_model = model or "text-embedding-3-small"
        response = self._litellm.embedding(model=embed_model, input=texts, **kwargs)
        return [e["embedding"] for e in response.data]
