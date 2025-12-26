"""
Base Agent Interface for ARGUS.

Defines the abstract agent interface and common utilities
for all agent roles in the debate system.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.cdag.graph import CDAG

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent roles in the debate system."""
    MODERATOR = "moderator"
    SPECIALIST = "specialist"
    REFUTER = "refuter"
    JURY = "jury"
    OBSERVER = "observer"


class AgentConfig(BaseModel):
    """
    Base configuration for agents.
    
    Attributes:
        name: Agent name
        role: Agent role
        temperature: LLM temperature
        max_tokens: Max tokens for generation
        system_prompt: Custom system prompt
    """
    name: str = Field(
        default="Agent",
        description="Agent name",
    )
    
    role: AgentRole = Field(
        default=AgentRole.SPECIALIST,
        description="Agent role",
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature",
    )
    
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Max tokens for generation",
    )
    
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt",
    )


@dataclass
class AgentResponse:
    """
    Response from an agent action.
    
    Attributes:
        success: Whether the action succeeded
        content: Response content
        data: Structured data from the response
        usage: Token usage statistics
        latency_ms: Response latency
        error: Error message if failed
    """
    success: bool = True
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    
    @property
    def failed(self) -> bool:
        """Check if response failed."""
        return not self.success


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality for agent initialization,
    LLM interaction, and response handling.
    
    Subclasses must implement:
        - act(): Perform the agent's main action
        - get_system_prompt(): Return role-specific system prompt
    """
    
    def __init__(
        self,
        llm: "BaseLLM",
        config: Optional[AgentConfig] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize agent.
        
        Args:
            llm: LLM instance for generation
            config: Agent configuration
            name: Optional agent name override
        """
        self.llm = llm
        self.config = config or AgentConfig()
        
        if name:
            self.config.name = name
        
        self._history: list[dict[str, Any]] = []
        self._action_count = 0
        
        logger.debug(
            f"Initialized {self.config.role.value} agent: {self.config.name}"
        )
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name
    
    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return self.config.role
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the agent's system prompt.
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def act(
        self,
        graph: "CDAG",
        context: dict[str, Any],
    ) -> AgentResponse:
        """
        Perform the agent's main action.
        
        Args:
            graph: The C-DAG graph
            context: Action context
            
        Returns:
            AgentResponse with results
        """
        pass
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Override system prompt
            **kwargs: Additional generation options
            
        Returns:
            Generated text
        """
        system = system_prompt or self.config.system_prompt or self.get_system_prompt()
        
        response = self.llm.generate(
            prompt,
            system_prompt=system,
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            **kwargs,
        )
        
        self._action_count += 1
        
        return response.content
    
    def log_action(
        self,
        action: str,
        details: dict[str, Any],
    ) -> None:
        """
        Log an action to history.
        
        Args:
            action: Action name
            details: Action details
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "agent": self.name,
            "role": self.role.value,
            **details,
        }
        self._history.append(entry)
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get action history."""
        return self._history.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', role={self.role.value})"
