"""
ARGUS Durable Execution Configuration.

Configuration for checkpointing and workflow persistence.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CheckpointerType(str, Enum):
    """Checkpointer storage backend types."""
    MEMORY = "memory"         # In-memory (volatile)
    SQLITE = "sqlite"         # SQLite database
    FILESYSTEM = "filesystem" # File-based JSON


class RetryPolicy(BaseModel):
    """Retry policy for failed tasks."""
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay: float = Field(default=1.0, ge=0.1)
    max_delay: float = Field(default=60.0, ge=1.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class DurableConfig(BaseModel):
    """Configuration for durable execution."""
    enabled: bool = Field(default=True)
    checkpointer_type: CheckpointerType = Field(default=CheckpointerType.MEMORY)
    persistence_path: Optional[str] = Field(default=None)
    checkpoint_interval: int = Field(default=1, ge=1, description="Steps between auto-checkpoints")
    auto_checkpoint: bool = Field(default=True)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    preserve_on_error: bool = Field(default=True)
    max_checkpoints: int = Field(default=100, ge=1)
    compress_checkpoints: bool = Field(default=False)


def get_default_durable_config() -> DurableConfig:
    return DurableConfig()
