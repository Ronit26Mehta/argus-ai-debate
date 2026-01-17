"""
Azure OpenAI LLM provider implementation.

Supports Azure-hosted GPT models.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class AzureOpenAILLM(BaseLLM):
    """
    Azure OpenAI LLM provider.
    
    Example:
        >>> llm = AzureOpenAILLM(
        ...     model="gpt-4",
        ...     azure_endpoint="https://your-resource.openai.azure.com",
        ...     api_key="your-key",
        ...     api_version="2024-02-15-preview"
        ... )
        >>> response = llm.generate("Explain Azure services")
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        azure_deployment: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.azure_deployment = azure_deployment or model
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "azure_openai"
    
    def _init_client(self) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package required")
        import os
        self._client = AzureOpenAI(
            api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=self.api_version,
        )
        logger.debug(f"Initialized Azure OpenAI for deployment '{self.azure_deployment}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        messages = self._prepare_messages(prompt, system_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self.azure_deployment, messages=messages,
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
            logger.error(f"Azure OpenAI generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        messages = self._prepare_messages(prompt, system_prompt)
        try:
            stream = self._client.chat.completions.create(
                model=self.azure_deployment, messages=messages, stream=True,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens, stop=stop
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[Error: {e}]"
    
    def embed(self, texts: str | list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embed_model = model or "text-embedding-ada-002"
        response = self._client.embeddings.create(input=texts, model=embed_model, **kwargs)
        return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
