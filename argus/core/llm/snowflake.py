"""
Snowflake Cortex LLM provider implementation.

Supports Snowflake Cortex LLM functions.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class SnowflakeLLM(BaseLLM):
    """
    Snowflake Cortex LLM provider.
    
    Example:
        >>> llm = SnowflakeLLM(model="llama3.1-70b", account="...")
        >>> response = llm.generate("Explain data warehousing")
    
    Models: llama3.1-70b, llama3.1-8b, mistral-large, snowflake-arctic
    """
    
    MODEL_ALIASES = {
        "llama-70b": "llama3.1-70b",
        "llama-8b": "llama3.1-8b",
        "arctic": "snowflake-arctic",
    }
    
    def __init__(
        self,
        model: str = "llama3.1-70b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: str = "PUBLIC",
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "snowflake"
    
    def _init_client(self) -> None:
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError("snowflake-connector-python required. Install: pip install snowflake-connector-python")
        import os
        self._conn = snowflake.connector.connect(
            account=self.account or os.getenv("SNOWFLAKE_ACCOUNT"),
            user=self.user or os.getenv("SNOWFLAKE_USER"),
            password=self.password or os.getenv("SNOWFLAKE_PASSWORD"),
            warehouse=self.warehouse or os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=self.database or os.getenv("SNOWFLAKE_DATABASE"),
            schema=self.schema,
        )
        logger.debug(f"Initialized Snowflake Cortex for '{self.model}'")
    
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
        
        # Escape quotes
        text = text.replace("'", "''")
        
        sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{self.model}',
            '{text}',
            {{'temperature': {temperature or self.default_temperature}, 
              'max_tokens': {max_tokens or self.default_max_tokens}}}
        )
        """
        
        try:
            cursor = self._conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchone()
            content = result[0] if result else ""
            cursor.close()
            
            return LLMResponse(content=content, model=self.model, provider=self.provider_name,
                             finish_reason="stop", latency_ms=self._measure_latency(start_time))
        except Exception as e:
            logger.error(f"Snowflake Cortex generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        # Snowflake Cortex doesn't support streaming via SQL
        response = self.generate(prompt, system_prompt=system_prompt, temperature=temperature,
                                max_tokens=max_tokens, stop=stop, **kwargs)
        yield response.content
