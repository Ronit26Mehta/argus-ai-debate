"""
Multi-Agent Orchestration for ARGUS.

This module provides the agent roles for debate orchestration:
    - Moderator: Plans and coordinates debate
    - Specialist: Domain-specific evidence gathering
    - Refuter: Finds counter-evidence and rebuttals
    - Jury: Aggregates evidence and renders verdicts
"""

from argus.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResponse,
)
from argus.agents.moderator import (
    Moderator,
    ModeratorConfig,
    DebateAgenda,
)
from argus.agents.specialist import (
    Specialist,
    SpecialistConfig,
)
from argus.agents.refuter import (
    Refuter,
    RefuterConfig,
)
from argus.agents.jury import (
    Jury,
    JuryConfig,
    Verdict,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    "AgentRole",
    "AgentResponse",
    # Moderator
    "Moderator",
    "ModeratorConfig",
    "DebateAgenda",
    # Specialist
    "Specialist",
    "SpecialistConfig",
    # Refuter
    "Refuter",
    "RefuterConfig",
    # Jury
    "Jury",
    "JuryConfig",
    "Verdict",
]
