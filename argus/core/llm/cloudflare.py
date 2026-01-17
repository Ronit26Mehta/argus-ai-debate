"""
Cloudflare Workers AI LLM provider implementation.

Supports edge inference via Cloudflare Workers AI.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class CloudflareLLM(BaseLLM):
    """
    Cloudflare Workers AI LLM provider - edge inference.
    
    Example:
        >>> llm = CloudflareLLM(model="@cf/meta/llama-3.1-70b-instruct", account_id="...")
        >>> response = llm.generate("Explain edge computing")
    
    Models: @cf/meta/llama-*, @cf/mistral/*, @cf/qwen/*
    """
    
    MODEL_ALIASES = {
        "llama-70b": "@cf/meta/llama-3.1-70b-instruct",
        "llama-8b": "@cf/meta/llama-3.1-8b-instruct",
        "mistral-7b": "@cf/mistral/mistral-7b-instruct-v0.1",
    }
    
    def __init__(
        self,
        model: str = "@cf/meta/llama-3.1-70b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        account_id: Optional[str] = None,
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self.account_id = account_id
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "cloudflare"
    
    def _init_client(self) -> None:
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("requests required")
        import os
        self._api_token = self.api_key or os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("CF_API_TOKEN")
        self._account_id = self.account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID") or os.getenv("CF_ACCOUNT_ID")
        self._base_url = f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/run"
        logger.debug(f"Initialized Cloudflare Workers AI for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        messages = self._prepare_messages(prompt, system_prompt)
        
        headers = {"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"}
        payload = {
            "messages": messages,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        
        try:
            response = self._requests.post(f"{self._base_url}/{self.model}", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                content = result.get("result", {}).get("response", "")
            else:
                content = ""
                logger.error(f"Cloudflare error: {result.get('errors')}")
            
            return LLMResponse(content=content, model=self.model, provider=self.provider_name,
                             finish_reason="stop", latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"Cloudflare generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        messages = self._prepare_messages(prompt, system_prompt)
        headers = {"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"}
        payload = {"messages": messages, "stream": True, "max_tokens": max_tokens or self.default_max_tokens}
        
        try:
            response = self._requests.post(f"{self._base_url}/{self.model}", headers=headers, json=payload, stream=True)
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data:"):
                        import json
                        data = json.loads(line[5:].strip())
                        if "response" in data:
                            yield data["response"]
        except Exception as e:
            yield f"[Error: {e}]"
