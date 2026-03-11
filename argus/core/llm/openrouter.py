"""
OpenRouter LLM provider implementation.

Routes to 100+ models (DeepSeek, Llama, Mistral, GPT, Claude, etc.)
via https://openrouter.ai using an OpenAI-compatible API.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class OpenRouterLLM(BaseLLM):
    """
    OpenRouter LLM provider.

    Uses the OpenAI-compatible API at https://openrouter.ai/api/v1.

    Example:
        >>> llm = OpenRouterLLM(model="deepseek/deepseek-chat-v3-0324:free")
        >>> response = llm.generate("Explain quantum computing")

    Models: deepseek/*, meta-llama/*, mistralai/*, openai/*, anthropic/*
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    MODEL_ALIASES = {
        "deepseek-small": "deepseek/deepseek-chat-v3-0324:free",
        "deepseek-chat": "deepseek/deepseek-chat-v3-0324:free",
        "deepseek-r1": "deepseek/deepseek-r1:free",
        "deepseek-v3": "deepseek/deepseek-chat-v3-0324:free",
        "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
        "gemma-3-4b": "google/gemma-3-4b-it:free",
    }

    def __init__(
        self,
        model: str = "deepseek/deepseek-chat-v3-0324:free",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self._init_client()

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        import os

        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/Ronit26Mehta/argus-ai-debate",
                "X-Title": "ARGUS-AI",
            },
        )
        logger.debug(f"Initialized OpenRouter client for '{self.model}'")

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
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                stop=stop,
                **kwargs,
            )
            choice = response.choices[0]
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=choice.finish_reason,
                latency_ms=self._measure_latency(start_time),
            )
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                finish_reason="error",
                latency_ms=self._measure_latency(start_time),
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
        messages = self._prepare_messages(prompt, system_prompt)
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                stop=stop,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[Error: {e}]"

    def embed(
        self, texts: str | list[str], model: Optional[str] = None, **kwargs: Any
    ) -> list[list[float]]:
        raise NotImplementedError("OpenRouter does not support embeddings directly.")
