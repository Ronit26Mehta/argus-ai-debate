"""
fsociety configuration module.

Uses OpenAI-compatible client pointed at a local LLM server (localhost:8080)
following the same pattern as ARGUS Aristotle.
"""

from __future__ import annotations

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for the LLM backend (OpenAI-compatible local server)."""

    base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL of the OpenAI-compatible server",
    )
    model_name: str = Field(
        default="local-model",
        description="Model name / ID sent to the server",
    )
    api_key: str = Field(
        default="not-needed",
        description="API key (local servers typically don't require one)",
    )
    max_tokens: int = Field(default=4096, description="Max tokens per response")
    temperature: float = Field(default=0.3, description="Sampling temperature")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retries on failure")


class ScanConfig(BaseModel):
    """Configuration for scan operations."""

    depth: str = Field(
        default="surface",
        description="Scan depth: 'surface' (fast) or 'deep' (full git history + deps)",
    )
    max_debate_rounds: int = Field(default=6, description="Maximum debate rounds")
    posterior_threshold: float = Field(
        default=0.85,
        description="Posterior threshold for P0 finding escalation",
    )
    agents: list[str] = Field(
        default_factory=lambda: ["elliot", "mrrobot", "darlene", "whiterose", "irving"],
        description="Active agents (Tier 1 by default)",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model for code vectors",
    )


class FsocietyConfig(BaseSettings):
    """Main fsociety configuration."""

    model_config = {"env_prefix": "FSOCIETY_", "env_nested_delimiter": "__"}

    # LLM
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Scan defaults
    scan: ScanConfig = Field(default_factory=ScanConfig)

    # Output
    output_dir: str = Field(
        default="./fsociety_reports",
        description="Root output directory for reports",
    )

    # Session
    session_id: str = Field(
        default="",
        description="Current session ID (auto-generated if empty)",
    )

    def get_llm(self):
        """Create an OpenAILLM instance pointed at the local server.

        Uses the ARGUS OpenAILLM with base_url override — same pattern
        as the Aristotle module in the ARGUS codebase.
        """
        from argus.core.llm.openai import OpenAILLM

        base_url = self.llm.base_url.rstrip("/")
        return OpenAILLM(
            model=self.llm.model_name,
            base_url=f"{base_url}/v1",
            api_key=self.llm.api_key,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            timeout=self.llm.timeout,
            max_retries=self.llm.max_retries,
        )

    def generate_session_id(self) -> str:
        """Generate a new session ID."""
        short_uuid = uuid.uuid4().hex[:8]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_id = f"fs-{short_uuid}-{ts}"
        return self.session_id

    def get_session_dir(self, target_name: str) -> Path:
        """Get/create session output directory."""
        if not self.session_id:
            self.generate_session_id()
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in target_name)
        path = Path(self.output_dir) / safe_name / self.session_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_config() -> FsocietyConfig:
    """Get the default fsociety configuration."""
    return FsocietyConfig()
