"""
Google Gemini LLM provider implementation.

Supports Gemini Pro, Gemini Ultra, and other Google AI models.
Uses the official google-generativeai Python SDK.
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


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM provider.
    
    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini Pro, and other models.
    Uses the official google-generativeai SDK.
    
    Example:
        >>> from argus.core.llm import GeminiLLM
        >>> llm = GeminiLLM(model="gemini-1.5-pro", api_key="...")
        >>> response = llm.generate("Explain quantum computing")
        >>> print(response.content)
    
    Supported Models:
        - gemini-1.5-pro, gemini-1.5-pro-latest
        - gemini-1.5-flash, gemini-1.5-flash-latest
        - gemini-pro (legacy)
        - gemini-pro-vision (legacy, multimodal)
    """
    
    # Model aliases for convenience
    MODEL_ALIASES = {
        "gemini": "gemini-1.5-pro",
        "gemini-pro": "gemini-1.5-pro",
        "gemini-1.5": "gemini-1.5-pro",
        "gemini-flash": "gemini-1.5-flash",
        "gemini-1.5-flash": "gemini-1.5-flash",
    }
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize Gemini LLM provider.
        
        Args:
            model: Model name (e.g., "gemini-1.5-pro")
            api_key: Google API key (or GOOGLE_API_KEY env var)
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
        return "gemini"
    
    def _init_client(self) -> None:
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            )
        
        # Configure API key
        if self.api_key:
            genai.configure(api_key=self.api_key)
        # If no key provided, will look for GOOGLE_API_KEY env var
        
        # Create model instance
        self._genai = genai
        self._client = genai.GenerativeModel(self.model)
        
        logger.debug(f"Initialized Gemini client for model '{self.model}'")
    
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
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            **kwargs: Additional Gemini-specific options
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        # Prepare content
        content = self._prepare_gemini_content(prompt, system_prompt)
        
        # Build generation config
        generation_config = self._genai.GenerationConfig(
            temperature=temperature if temperature is not None else self.default_temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            stop_sequences=stop if stop else None,
        )
        
        try:
            # Make API call
            response = self._client.generate_content(
                content,
                generation_config=generation_config,
                **kwargs,
            )
            
            # Extract response text
            content_text = ""
            if response.parts:
                content_text = "".join(part.text for part in response.parts if hasattr(part, "text"))
            elif hasattr(response, "text"):
                content_text = response.text
            
            # Build usage (Gemini provides this in usage_metadata)
            usage = LLMUsage()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = LLMUsage(
                    prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                    completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
                    total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
                )
            
            # Determine finish reason
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason.name).lower()
            
            return LLMResponse(
                content=content_text,
                model=self.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=self._measure_latency(start_time),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
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
        Stream generated tokens from Gemini.
        
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
        # Prepare content
        content = self._prepare_gemini_content(prompt, system_prompt)
        
        # Build generation config
        generation_config = self._genai.GenerationConfig(
            temperature=temperature if temperature is not None else self.default_temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            stop_sequences=stop if stop else None,
        )
        
        try:
            # Create streaming response
            response = self._client.generate_content(
                content,
                generation_config=generation_config,
                stream=True,
                **kwargs,
            )
            
            for chunk in response:
                if chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, "text"):
                            yield part.text
                            
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def _prepare_gemini_content(
        self,
        prompt: str | list[Message],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare content in Gemini format.
        
        Args:
            prompt: User prompt or message list
            system_prompt: System prompt
            
        Returns:
            Content list for Gemini API
        """
        content_parts = []
        
        # Add system prompt as first user message if provided
        if system_prompt:
            content_parts.append({
                "role": "user",
                "parts": [{"text": f"System: {system_prompt}"}]
            })
            content_parts.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        
        if isinstance(prompt, str):
            content_parts.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
        elif isinstance(prompt, list):
            for msg in prompt:
                # Map roles
                role = "user" if msg.role.value in ["user", "system"] else "model"
                content_parts.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        return content_parts
    
    def embed(
        self,
        texts: str | list[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings using Gemini.
        
        Args:
            texts: Text(s) to embed
            model: Embedding model (default: embedding-001)
            **kwargs: Additional options
            
        Returns:
            List of embedding vectors
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        embed_model = model or "models/embedding-001"
        
        try:
            embeddings = []
            for text in texts:
                result = self._genai.embed_content(
                    model=embed_model,
                    content=text,
                    **kwargs,
                )
                embeddings.append(result["embedding"])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Gemini's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        try:
            result = self._client.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fallback to base implementation
            return super().count_tokens(text)
