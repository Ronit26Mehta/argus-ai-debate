"""
Ollama LLM provider implementation.

Supports local LLM models via Ollama server.
Enables fully offline operation with models like Llama, Mistral, etc.
"""

from __future__ import annotations

import time
import logging
import json
from typing import Optional, Any, Iterator

import httpx

from argus.core.llm.base import (
    BaseLLM,
    LLMResponse,
    LLMUsage,
    Message,
)

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    Ollama LLM provider for local models.
    
    Connects to a local Ollama server to run models like Llama 3, Mistral,
    CodeLlama, etc. Supports both generate and chat endpoints.
    
    Example:
        >>> from argus.core.llm import OllamaLLM
        >>> llm = OllamaLLM(model="llama3.2", host="http://localhost:11434")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response.content)
    
    Supported Models:
        - llama3.2, llama3.1, llama3
        - mistral, mixtral
        - codellama
        - phi3, phi
        - gemma2
        - Any model available via `ollama pull`
    """
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        "llama": "llama3.2",
        "llama3": "llama3.2",
        "mistral": "mistral",
        "codellama": "codellama",
        "phi": "phi3",
    }
    
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        """
        Initialize Ollama LLM provider.
        
        Args:
            model: Model name (e.g., "llama3.2", "mistral")
            host: Ollama server URL
            temperature: Default sampling temperature
            max_tokens: Default max tokens
            timeout: Request timeout in seconds
            **kwargs: Additional options
        """
        # Resolve model aliases
        resolved_model = self.MODEL_ALIASES.get(model, model)
        
        super().__init__(
            model=resolved_model,
            api_key=None,  # Ollama doesn't need API key
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        self.host = host.rstrip("/")
        self.timeout = timeout
        
        # Initialize client
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "ollama"
    
    def _init_client(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.Client(
            base_url=self.host,
            timeout=self.timeout,
        )
        
        # Check if server is reachable
        try:
            response = self._client.get("/api/tags")
            if response.status_code == 200:
                available_models = [m["name"] for m in response.json().get("models", [])]
                if available_models:
                    logger.debug(f"Ollama server connected. Models: {available_models[:5]}...")
                else:
                    logger.warning("Ollama server connected but no models available. Run: ollama pull <model>")
        except httpx.ConnectError:
            logger.warning(
                f"Could not connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )
    
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
        """
        Generate a response from Ollama.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            **kwargs: Additional Ollama-specific options
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages for chat endpoint
        messages = self._prepare_ollama_messages(prompt, system_prompt)
        
        # Build request body
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.default_temperature,
                "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
            },
        }
        
        if stop:
            request_body["options"]["stop"] = stop
        
        # Add any extra options
        if kwargs:
            request_body["options"].update(kwargs)
        
        try:
            # Make API call
            response = self._client.post(
                "/api/chat",
                json=request_body,
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract content
            content = data.get("message", {}).get("content", "")
            
            # Build usage
            usage = LLMUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            )
            
            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                provider=self.provider_name,
                usage=usage,
                finish_reason=data.get("done_reason", "stop"),
                latency_ms=self._measure_latency(start_time),
                raw_response=data,
            )
            
        except httpx.ConnectError as e:
            logger.error(f"Ollama connection failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                finish_reason="error",
                latency_ms=self._measure_latency(start_time),
                raw_response={"error": f"Connection failed: {e}. Is Ollama running?"},
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                finish_reason="error",
                latency_ms=self._measure_latency(start_time),
                raw_response={"error": str(e)},
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
        """
        Stream generated tokens from Ollama.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            stop: Stop sequences
            **kwargs: Additional options
            
        Yields:
            Generated tokens as strings
        """
        # Prepare messages
        messages = self._prepare_ollama_messages(prompt, system_prompt)
        
        # Build request body
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.default_temperature,
                "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
            },
        }
        
        if stop:
            request_body["options"]["stop"] = stop
        
        if kwargs:
            request_body["options"].update(kwargs)
        
        try:
            # Create streaming request
            with self._client.stream(
                "POST",
                "/api/chat",
                json=request_body,
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ConnectError as e:
            logger.error(f"Ollama streaming connection failed: {e}")
            yield f"[Error: Connection failed. Is Ollama running?]"
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def _prepare_ollama_messages(
        self,
        prompt: str | list[Message],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Prepare messages in Ollama format.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            
        Returns:
            List of message dictionaries
        """
        messages: list[dict[str, str]] = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            for msg in prompt:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        return messages
    
    def embed(
        self,
        texts: str | list[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings using Ollama.
        
        Args:
            texts: Text(s) to embed
            model: Embedding model (default: uses main model)
            **kwargs: Additional options
            
        Returns:
            List of embedding vectors
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        embed_model = model or self.model
        
        try:
            embeddings = []
            for text in texts:
                response = self._client.post(
                    "/api/embeddings",
                    json={
                        "model": embed_model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embedding", []))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise
    
    def list_models(self) -> list[str]:
        """
        List available models on the Ollama server.
        
        Returns:
            List of model names
        """
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful
        """
        try:
            # This is a streaming endpoint
            with self._client.stream(
                "POST",
                "/api/pull",
                json={"name": model_name},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if "success" in status.lower():
                            logger.info(f"Successfully pulled model: {model_name}")
                            return True
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
