"""
IBM watsonx.ai LLM provider implementation.

Supports IBM Granite and hosted foundation models.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class WatsonxLLM(BaseLLM):
    """
    IBM watsonx.ai LLM provider.
    
    Example:
        >>> llm = WatsonxLLM(model="ibm/granite-3-8b-instruct", project_id="...")
        >>> response = llm.generate("Explain enterprise AI")
    
    Models: ibm/granite-*, meta-llama/*, mistralai/*
    """
    
    MODEL_ALIASES = {
        "granite-8b": "ibm/granite-3-8b-instruct",
        "granite-34b": "ibm/granite-34b-code-instruct",
    }
    
    def __init__(
        self,
        model: str = "ibm/granite-3-8b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        project_id: Optional[str] = None,
        url: str = "https://us-south.ml.cloud.ibm.com",
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self.project_id = project_id
        self.url = url
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "watsonx"
    
    def _init_client(self) -> None:
        try:
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai import Credentials
        except ImportError:
            raise ImportError("ibm-watsonx-ai required. Install: pip install ibm-watsonx-ai")
        import os
        api_key = self.api_key or os.getenv("WATSONX_API_KEY") or os.getenv("IBM_API_KEY")
        project_id = self.project_id or os.getenv("WATSONX_PROJECT_ID")
        
        credentials = Credentials(url=self.url, api_key=api_key)
        self._client = ModelInference(model_id=self.model, credentials=credentials, project_id=project_id)
        logger.debug(f"Initialized watsonx.ai for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        
        if isinstance(prompt, list):
            text = "\n".join(f'{m.role.value}: {m.content}' for m in prompt)
        else:
            text = prompt
        if system_prompt:
            text = f"{system_prompt}\n\n{text}"
        
        params = {
            "max_new_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature,
        }
        if stop:
            params["stop_sequences"] = stop
        
        try:
            response = self._client.generate(prompt=text, params=params)
            content = response.get("results", [{}])[0].get("generated_text", "")
            usage = LLMUsage(
                prompt_tokens=response.get("results", [{}])[0].get("input_token_count", 0),
                completion_tokens=response.get("results", [{}])[0].get("generated_token_count", 0),
                total_tokens=response.get("results", [{}])[0].get("input_token_count", 0) +
                            response.get("results", [{}])[0].get("generated_token_count", 0),
            )
            return LLMResponse(content=content, model=self.model, provider=self.provider_name,
                             usage=usage, finish_reason="stop",
                             latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"watsonx.ai generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        if isinstance(prompt, list):
            text = "\n".join(f'{m.role.value}: {m.content}' for m in prompt)
        else:
            text = prompt
        if system_prompt:
            text = f"{system_prompt}\n\n{text}"
        
        params = {"max_new_tokens": max_tokens or self.default_max_tokens,
                 "temperature": temperature or self.default_temperature}
        
        try:
            for chunk in self._client.generate_text_stream(prompt=text, params=params):
                yield chunk
        except Exception as e:
            yield f"[Error: {e}]"
