"""
HuggingFace Inference API LLM provider implementation.

Supports HuggingFace hosted models and Inference Endpoints.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """
    HuggingFace Inference API LLM provider.
    
    Example:
        >>> llm = HuggingFaceLLM(model="meta-llama/Llama-3.1-70B-Instruct")
        >>> response = llm.generate("Explain transformers")
    
    Models: meta-llama/*, mistralai/*, Qwen/*, microsoft/*
    """
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-70B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        endpoint_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)
        self.endpoint_url = endpoint_url
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    def _init_client(self) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
        import os
        token = self.api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self._client = InferenceClient(model=self.endpoint_url or self.model, token=token)
        logger.debug(f"Initialized HuggingFace client for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        messages = self._prepare_messages(prompt, system_prompt)
        
        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                stop=stop,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )
            return LLMResponse(content=content, model=self.model, provider=self.provider_name,
                             usage=usage, finish_reason=choice.finish_reason,
                             latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        messages = self._prepare_messages(prompt, system_prompt)
        try:
            for chunk in self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                stream=True,
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[Error: {e}]"
    
    def embed(self, texts: str | list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embed_model = model or "sentence-transformers/all-MiniLM-L6-v2"
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=embed_model, token=self.api_key)
            return [client.feature_extraction(t) for t in texts]
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            raise
