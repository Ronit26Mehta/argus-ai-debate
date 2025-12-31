"""
Groq LLM provider implementation.

Supports Llama, Mixtral, Gemma, and other models running on Groq LPU.
Uses the official groq Python SDK with OpenAI-compatible interface.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import (
    BaseLLM,
    LLMResponse,
    LLMUsage,
    Message,
)

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """
    Groq LLM provider.
    
    Supports Llama, Mixtral, Gemma, and other models on Groq's LPU infrastructure.
    Uses the official groq SDK with OpenAI-compatible interface.
    
    Example:
        >>> from argus.core.llm import GroqLLM
        >>> llm = GroqLLM(model="llama-3.1-70b-versatile", api_key="...")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response.content)
    
    Supported Models:
        - llama-3.1-70b-versatile (recommended)
        - llama-3.1-8b-instant
        - llama-3.2-90b-vision-preview
        - mixtral-8x7b-32768
        - gemma2-9b-it
        - whisper-large-v3 (audio transcription)
    """
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        "groq": "llama-3.1-70b-versatile",
        "llama": "llama-3.1-70b-versatile",
        "llama-70b": "llama-3.1-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "llama3": "llama-3.1-70b-versatile",
        "mixtral": "mixtral-8x7b-32768",
        "gemma": "gemma2-9b-it",
    }
    
    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize Groq LLM provider.
        
        Args:
            model: Model name (e.g., "llama-3.1-70b-versatile")
            api_key: Groq API key (or GROQ_API_KEY env var)
            temperature: Default sampling temperature
            max_tokens: Default max tokens
            **kwargs: Additional options
        """
        # Resolve model aliases
        resolved_model = self.MODEL_ALIASES.get(model, model)
        
        super().__init__(
            model=resolved_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Initialize client
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "groq"
    
    def _init_client(self) -> None:
        """Initialize the Groq client."""
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package is required. Install with: pip install groq"
            )
        
        # Create client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        # If no key, SDK will look for GROQ_API_KEY env var
        
        self._client = Groq(**client_kwargs)
        logger.debug(f"Initialized Groq client for model '{self.model}'")
    
    def generate(
        self,
        prompt: str | list[Message],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        response_format: Optional[dict[str, str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from Groq.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            response_format: Response format (e.g., {"type": "json_object"})
            seed: Random seed for reproducibility
            **kwargs: Additional Groq-specific options
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare messages
        messages = self._prepare_messages(prompt, system_prompt)
        
        try:
            # Build request kwargs (OpenAI-compatible)
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            }
            
            if stop:
                request_kwargs["stop"] = stop
            
            if response_format:
                request_kwargs["response_format"] = response_format
            
            if seed is not None:
                request_kwargs["seed"] = seed
            
            # Add any extra kwargs
            request_kwargs.update(kwargs)
            
            # Make API call
            response = self._client.chat.completions.create(**request_kwargs)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Build usage
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=choice.finish_reason,
                latency_ms=self._measure_latency(start_time),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
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
        Stream generated tokens from Groq.
        
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
        messages = self._prepare_messages(prompt, system_prompt)
        
        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
                "stream": True,
            }
            
            if stop:
                request_kwargs["stop"] = stop
            
            request_kwargs.update(kwargs)
            
            # Create streaming response
            stream = self._client.chat.completions.create(**request_kwargs)
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def transcribe(
        self,
        audio_file: str | bytes,
        model: str = "whisper-large-v3",
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Transcribe audio using Groq's Whisper.
        
        Args:
            audio_file: Path to audio file or audio bytes
            model: Whisper model to use
            language: Optional language code (ISO-639-1)
            **kwargs: Additional options
            
        Returns:
            Transcribed text
        """
        try:
            # Handle file path vs bytes
            if isinstance(audio_file, str):
                with open(audio_file, "rb") as f:
                    audio_data = f.read()
                filename = audio_file.split("/")[-1].split("\\")[-1]
            else:
                audio_data = audio_file
                filename = "audio.wav"
            
            # Create file tuple for the API
            file_tuple = (filename, audio_data)
            
            request_kwargs = {
                "file": file_tuple,
                "model": model,
            }
            
            if language:
                request_kwargs["language"] = language
            
            request_kwargs.update(kwargs)
            
            response = self._client.audio.transcriptions.create(**request_kwargs)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Groq transcription failed: {e}")
            raise
    
    def list_models(self) -> list[dict[str, Any]]:
        """
        List available models on Groq.
        
        Returns:
            List of model information dicts
        """
        try:
            response = self._client.models.list()
            return [
                {
                    "id": model.id,
                    "owned_by": model.owned_by,
                    "created": model.created,
                }
                for model in response.data
            ]
        except Exception as e:
            logger.error(f"Failed to list Groq models: {e}")
            return []
