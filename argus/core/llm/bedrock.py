"""
AWS Bedrock LLM provider implementation.

Supports Claude, Llama, Titan, and other models via AWS Bedrock.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Optional, Any, Iterator

from argus.core.llm.base import BaseLLM, LLMResponse, LLMUsage, Message

logger = logging.getLogger(__name__)


class BedrockLLM(BaseLLM):
    """
    AWS Bedrock LLM provider.
    
    Example:
        >>> llm = BedrockLLM(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
        >>> response = llm.generate("Explain cloud computing")
    
    Models:
        - anthropic.claude-3-5-sonnet-*, anthropic.claude-3-haiku-*
        - meta.llama3-*, meta.llama3-1-*
        - amazon.titan-text-*, amazon.nova-*
        - mistral.mistral-*, mistral.mixtral-*
    """
    
    MODEL_ALIASES = {
        "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "llama-3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
        "titan": "amazon.titan-text-premier-v1:0",
        "nova-pro": "amazon.nova-pro-v1:0",
    }
    
    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
        **kwargs: Any,
    ):
        resolved = self.MODEL_ALIASES.get(model, model)
        super().__init__(resolved, api_key, temperature, max_tokens, **kwargs)
        self.region = region
        self.profile_name = profile_name
        self._init_client()
    
    @property
    def provider_name(self) -> str:
        return "bedrock"
    
    def _init_client(self) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 package required. Install: pip install boto3")
        
        session_kwargs = {}
        if self.profile_name:
            session_kwargs["profile_name"] = self.profile_name
        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime", region_name=self.region)
        logger.debug(f"Initialized Bedrock client for '{self.model}' in {self.region}")
    
    def _get_model_family(self) -> str:
        if "anthropic" in self.model:
            return "anthropic"
        elif "meta" in self.model:
            return "meta"
        elif "amazon" in self.model:
            return "amazon"
        elif "mistral" in self.model:
            return "mistral"
        return "unknown"
    
    def generate(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse:
        start_time = time.time()
        messages = self._prepare_messages(prompt, system_prompt)
        family = self._get_model_family()
        
        try:
            if family == "anthropic":
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens or self.default_max_tokens,
                    "messages": [{"role": m["role"], "content": m["content"]} 
                                for m in messages if m["role"] != "system"],
                    "temperature": temperature or self.default_temperature,
                }
                sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                if sys_msg:
                    body["system"] = sys_msg
            else:
                # Generic format for other models
                prompt_text = "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)
                body = {
                    "prompt": prompt_text,
                    "max_gen_len": max_tokens or self.default_max_tokens,
                    "temperature": temperature or self.default_temperature,
                }
            
            response = self._client.invoke_model(
                modelId=self.model, body=json.dumps(body), contentType="application/json"
            )
            result = json.loads(response["body"].read())
            
            if family == "anthropic":
                content = result.get("content", [{}])[0].get("text", "")
                usage = LLMUsage(
                    prompt_tokens=result.get("usage", {}).get("input_tokens", 0),
                    completion_tokens=result.get("usage", {}).get("output_tokens", 0),
                    total_tokens=result.get("usage", {}).get("input_tokens", 0) +
                                result.get("usage", {}).get("output_tokens", 0),
                )
            else:
                content = result.get("generation", result.get("completions", [{}])[0].get("data", {}).get("text", ""))
                usage = LLMUsage()
            
            return LLMResponse(content=content, model=self.model, provider=self.provider_name,
                             usage=usage, finish_reason="stop",
                             latency_ms=self._measure_latency(start_time), raw_response=result)
        except Exception as e:
            logger.error(f"Bedrock generation failed: {e}")
            return LLMResponse(content="", model=self.model, provider=self.provider_name,
                             finish_reason="error", latency_ms=self._measure_latency(start_time))
    
    def stream(self, prompt: str | list[Message], *, system_prompt: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]:
        messages = self._prepare_messages(prompt, system_prompt)
        family = self._get_model_family()
        
        try:
            if family == "anthropic":
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens or self.default_max_tokens,
                    "messages": [{"role": m["role"], "content": m["content"]} 
                                for m in messages if m["role"] != "system"],
                }
            else:
                prompt_text = "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)
                body = {"prompt": prompt_text, "max_gen_len": max_tokens or self.default_max_tokens}
            
            response = self._client.invoke_model_with_response_stream(
                modelId=self.model, body=json.dumps(body)
            )
            for event in response.get("body", []):
                chunk = json.loads(event.get("chunk", {}).get("bytes", b"{}"))
                if family == "anthropic" and chunk.get("type") == "content_block_delta":
                    yield chunk.get("delta", {}).get("text", "")
                elif "generation" in chunk:
                    yield chunk["generation"]
        except Exception as e:
            yield f"[Error: {e}]"
