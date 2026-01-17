"""
Google Vertex AI LLM provider implementation.

Supports Gemini and other models via Vertex AI.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class VertexAILLM(BaseLLM):
    """
    Google Vertex AI LLM provider.
    
    Example:
        >>> llm = VertexAILLM(model="gemini-1.5-pro", project="my-project")
        >>> response = llm.generate("Explain GCP services")
    
    Models: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
    """
    
    MODEL_ALIASES = {
        "gemini-pro": "gemini-1.5-pro",
        "gemini-flash": "gemini-1.5-flash",
    }
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        project: Optional[str] = None,
        location: str = "us-central1",
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self.project = project
        self.location = location
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "vertex_ai"
    
    def _init_client(self) -> None:
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise ImportError("google-cloud-aiplatform required. Install: pip install google-cloud-aiplatform")
        import os
        project = self.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT")
        vertexai.init(project=project, location=self.location)
        self._model = GenerativeModel(self.model)
        logger.debug(f"Initialized Vertex AI for '{self.model}'")
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        from vertexai.generative_models import GenerationConfig
        
        # Build prompt
        if isinstance(prompt, list):
            text = "\n".join(m.content for m in prompt)
        else:
            text = prompt
        if system_prompt:
            text = f"{system_prompt}\n\n{text}"
        
        config = GenerationConfig(
            temperature=temperature or self.default_temperature,
            max_output_tokens=max_tokens or self.default_max_tokens,
            stop_sequences=stop,
        )
        
        try:
            response = self._model.generate_content(text, generation_config=config)
            content = response.text if response.text else ""
            usage = LLMUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                completion_tokens=response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                total_tokens=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
            )
            return LLMResponse(content=content, model=self.model, provider=self.provider_name,
                             usage=usage, finish_reason="stop",
                             latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        from vertexai.generative_models import GenerationConfig
        
        if isinstance(prompt, list):
            text = "\n".join(m.content for m in prompt)
        else:
            text = prompt
        if system_prompt:
            text = f"{system_prompt}\n\n{text}"
        
        config = GenerationConfig(
            temperature=temperature or self.default_temperature,
            max_output_tokens=max_tokens or self.default_max_tokens,
        )
        
        try:
            for chunk in self._model.generate_content(text, generation_config=config, stream=True):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"[Error: {e}]"
