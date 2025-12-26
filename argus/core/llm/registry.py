"""
LLM Provider Registry.

Manages registration and instantiation of LLM providers.
Provides a unified entry point for getting LLM instances.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, Type

from argus.core.config import get_config
from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class LLMRegistry:
    """
    Registry for LLM providers.
    
    Maintains a mapping of provider names to their implementation classes.
    Supports dynamic registration of custom providers.
    
    Example:
        >>> from argus.core.llm import LLMRegistry, get_llm
        >>> 
        >>> # Get default provider
        >>> llm = get_llm()
        >>> 
        >>> # Get specific provider
        >>> llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
        >>> 
        >>> # List available providers
        >>> providers = LLMRegistry.list_providers()
    """
    
    _providers: dict[str, Type[BaseLLM]] = {}
    _instances: dict[str, BaseLLM] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """
        Register an LLM provider.
        
        Args:
            name: Provider name (e.g., "openai", "anthropic")
            provider_class: Provider implementation class
        """
        cls._providers[name.lower()] = provider_class
        logger.debug(f"Registered LLM provider: {name}")
    
    @classmethod
    def get(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        **kwargs: Any,
    ) -> BaseLLM:
        """
        Get an LLM provider instance.
        
        Args:
            provider: Provider name (default from config)
            model: Model name (default from config or provider default)
            api_key: API key (default from config)
            cache: Whether to cache and reuse instances
            **kwargs: Additional provider-specific options
            
        Returns:
            BaseLLM instance
            
        Raises:
            ValueError: If provider is not registered
        """
        config = get_config()
        
        # Use default provider if not specified
        provider = (provider or config.default_provider).lower()
        
        # Get API key from config if not provided
        if api_key is None:
            if provider == "openai":
                api_key = config.llm.openai_api_key
            elif provider == "anthropic":
                api_key = config.llm.anthropic_api_key
            elif provider == "gemini":
                api_key = config.llm.google_api_key
        
        # Get model from config if not provided
        if model is None:
            model = config.get_model_for_provider(provider)
        
        # Check cache
        cache_key = f"{provider}:{model}"
        if cache and cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Get provider class
        if provider not in cls._providers:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available providers: {list(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider]
        
        # Build kwargs
        init_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **kwargs,
        }
        
        # Add API key if available
        if api_key:
            init_kwargs["api_key"] = api_key
        
        # Add provider-specific config
        if provider == "ollama":
            init_kwargs["host"] = config.llm.ollama_host
        
        # Instantiate provider
        instance = provider_class(**init_kwargs)
        
        # Cache if requested
        if cache:
            cls._instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List registered provider names.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def has_provider(cls, name: str) -> bool:
        """
        Check if a provider is registered.
        
        Args:
            name: Provider name
            
        Returns:
            True if provider is registered
        """
        return name.lower() in cls._providers
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the instance cache."""
        cls._instances.clear()


# Register default providers
def _register_default_providers() -> None:
    """Register the built-in LLM providers."""
    from argus.core.llm.openai import OpenAILLM
    from argus.core.llm.anthropic import AnthropicLLM
    from argus.core.llm.gemini import GeminiLLM
    from argus.core.llm.ollama import OllamaLLM
    
    LLMRegistry.register("openai", OpenAILLM)
    LLMRegistry.register("anthropic", AnthropicLLM)
    LLMRegistry.register("gemini", GeminiLLM)
    LLMRegistry.register("ollama", OllamaLLM)


# Auto-register on import
_register_default_providers()


# Convenience functions
def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLM:
    """
    Get an LLM provider instance.
    
    Convenience function that delegates to LLMRegistry.get().
    
    Args:
        provider: Provider name (default from ARGUS_DEFAULT_PROVIDER)
        model: Model name (default from ARGUS_DEFAULT_MODEL)
        **kwargs: Additional options
        
    Returns:
        BaseLLM instance
        
    Example:
        >>> from argus.core.llm import get_llm
        >>> 
        >>> # Use default provider (from config)
        >>> llm = get_llm()
        >>> 
        >>> # Use specific provider and model
        >>> llm = get_llm("openai", model="gpt-4o")
        >>> 
        >>> # Generate response
        >>> response = llm.generate("What is 2+2?")
        >>> print(response.content)
    """
    return LLMRegistry.get(provider=provider, model=model, **kwargs)


def register_provider(name: str, provider_class: Type[BaseLLM]) -> None:
    """
    Register a custom LLM provider.
    
    Args:
        name: Provider name
        provider_class: Provider implementation class
        
    Example:
        >>> from argus.core.llm import register_provider, BaseLLM
        >>> 
        >>> class MyCustomLLM(BaseLLM):
        ...     # Implementation
        ...     pass
        >>> 
        >>> register_provider("custom", MyCustomLLM)
    """
    LLMRegistry.register(name, provider_class)


def list_providers() -> list[str]:
    """
    List available LLM providers.
    
    Returns:
        List of provider names
        
    Example:
        >>> from argus.core.llm import list_providers
        >>> print(list_providers())
        ['openai', 'anthropic', 'gemini', 'ollama']
    """
    return LLMRegistry.list_providers()
