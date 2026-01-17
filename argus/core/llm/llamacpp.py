"""
Llama.cpp local LLM provider implementation.

Supports local GGUF models via llama-cpp-python.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class LlamaCppLLM(BaseLLM):
    """
    Llama.cpp local LLM provider.
    
    Example:
        >>> llm = LlamaCppLLM(model_path="/path/to/llama-3.1-8b.Q4_K_M.gguf")
        >>> response = llm.generate("Explain quantization")
    
    Note: Requires a GGUF model file. Uses llama-cpp-python bindings.
    """
    
    def __init__(
        self,
        model: str = "llama",  # Display name
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "llamacpp"
    
    def _init_client(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python required. Install: pip install llama-cpp-python")
        import os
        model_path = self.model_path or os.getenv("LLAMACPP_MODEL_PATH")
        if not model_path:
            raise ValueError("model_path required for LlamaCppLLM")
        
        kwargs = {
            "model_path": model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.n_threads:
            kwargs["n_threads"] = self.n_threads
        
        self._llm = Llama(**kwargs)
        logger.debug(f"Initialized Llama.cpp with '{model_path}'")
    
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
        
        try:
            output = self._llm(
                text,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                stop=stop or [],
                echo=False,
            )
            content = output["choices"][0]["text"]
            usage = LLMUsage(
                prompt_tokens=output.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=output.get("usage", {}).get("completion_tokens", 0),
                total_tokens=output.get("usage", {}).get("total_tokens", 0),
            )
            return LLMResponse(content=content.strip(), model=self.model,
                             provider=self.provider_name, usage=usage, finish_reason="stop",
                             latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"Llama.cpp generation failed: {e}")
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
        
        try:
            for output in self._llm(
                text,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                stop=stop or [],
                stream=True,
            ):
                token = output["choices"][0]["text"]
                yield token
        except Exception as e:
            yield f"[Error: {e}]"
