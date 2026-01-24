"""
Agent Status Widget - Display for debate agents (Moderator, Specialist, Refuter, Jury).

Features:
- Visual representation of agent states
- Activity indicators
- Current action display
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static
from textual.reactive import reactive
from dataclasses import dataclass
from enum import Enum


class AgentState(Enum):
    """Agent state enumeration."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Agent information container."""
    name: str
    role: str
    icon: str
    state: AgentState = AgentState.IDLE
    action: str = ""
    count: int = 0


class AgentCard(Vertical):
    """Individual agent status card."""
    
    DEFAULT_CSS = """
    AgentCard {
        width: 1fr;
        height: 6;
        background: #0d0d0d;
        border: double #665200;
        padding: 0 1;
        margin: 0 1 0 0;
    }
    
    AgentCard.--idle {
        border: double #4d3800;
    }
    
    AgentCard.--working {
        border: double #ffcc33;
    }
    
    AgentCard.--complete {
        border: double #33ff33;
    }
    
    AgentCard.--error {
        border: double #ff3333;
    }
    
    AgentCard #agent-header {
        width: 100%;
        height: 1;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    AgentCard #agent-icon {
        width: 3;
        color: #b833ff;
    }
    
    AgentCard #agent-name {
        width: 1fr;
        color: #ffcc33;
        text-style: bold;
    }
    
    AgentCard #agent-state {
        width: auto;
        color: #806000;
    }
    
    AgentCard #agent-role {
        width: 100%;
        height: 1;
        color: #ff8c00;
    }
    
    AgentCard #agent-action {
        width: 100%;
        height: 1;
        color: #ffb000;
    }
    
    AgentCard #agent-count {
        width: 100%;
        height: 1;
        color: #806000;
    }
    """
    
    def __init__(
        self,
        agent_info: AgentInfo,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.agent_info = agent_info
        self.add_class(f"--{agent_info.state.value}")
    
    def compose(self) -> ComposeResult:
        """Compose the agent card layout."""
        state_indicators = {
            AgentState.IDLE: "○",
            AgentState.WORKING: "◐",
            AgentState.COMPLETE: "●",
            AgentState.ERROR: "✖",
        }
        
        with Horizontal(id="agent-header"):
            yield Static(self.agent_info.icon, id="agent-icon")
            yield Static(self.agent_info.name, id="agent-name")
            yield Static(
                state_indicators[self.agent_info.state],
                id="agent-state"
            )
        
        yield Static(self.agent_info.role, id="agent-role")
        yield Static(
            self.agent_info.action or "Waiting...",
            id="agent-action"
        )
        yield Static(
            f"Items: {self.agent_info.count}",
            id="agent-count"
        )
    
    def update_state(self, state: AgentState) -> None:
        """Update the agent state."""
        # Remove old state class
        self.remove_class(f"--{self.agent_info.state.value}")
        # Update state
        self.agent_info.state = state
        # Add new state class
        self.add_class(f"--{state.value}")
        # Update indicator
        state_indicators = {
            AgentState.IDLE: "○",
            AgentState.WORKING: "◐",
            AgentState.COMPLETE: "●",
            AgentState.ERROR: "✖",
        }
        try:
            state_widget = self.query_one("#agent-state", Static)
            state_widget.update(state_indicators[state])
        except Exception:
            pass
    
    def update_action(self, action: str) -> None:
        """Update the current action."""
        self.agent_info.action = action
        try:
            action_widget = self.query_one("#agent-action", Static)
            action_widget.update(action or "Waiting...")
        except Exception:
            pass
    
    def update_count(self, count: int) -> None:
        """Update the item count."""
        self.agent_info.count = count
        try:
            count_widget = self.query_one("#agent-count", Static)
            count_widget.update(f"Items: {count}")
        except Exception:
            pass


class AgentStatusWidget(Vertical):
    """Widget displaying all debate agent statuses."""
    
    DEFAULT_CSS = """
    AgentStatusWidget {
        width: 100%;
        height: 8;
        padding: 1 0;
    }
    
    AgentStatusWidget #agents-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    """
    
    # Default agents
    DEFAULT_AGENTS = [
        AgentInfo("Moderator", "Orchestration", "🎯"),
        AgentInfo("Specialist", "Evidence Gathering", "🔍"),
        AgentInfo("Refuter", "Challenge Generation", "⚔️"),
        AgentInfo("Jury", "Verdict Rendering", "⚖️"),
    ]
    
    def __init__(
        self,
        agents: list[AgentInfo] | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.agents = agents or self.DEFAULT_AGENTS.copy()
    
    def compose(self) -> ComposeResult:
        """Compose the agents status layout."""
        with Horizontal(id="agents-container"):
            for i, agent in enumerate(self.agents):
                yield AgentCard(
                    agent_info=agent,
                    id=f"agent-{agent.name.lower()}",
                )
    
    def get_agent_card(self, agent_name: str) -> AgentCard | None:
        """Get an agent card by name."""
        try:
            return self.query_one(f"#agent-{agent_name.lower()}", AgentCard)
        except Exception:
            return None
    
    def set_agent_state(self, agent_name: str, state: AgentState) -> None:
        """Set an agent's state."""
        card = self.get_agent_card(agent_name)
        if card:
            card.update_state(state)
    
    def set_agent_action(self, agent_name: str, action: str) -> None:
        """Set an agent's current action."""
        card = self.get_agent_card(agent_name)
        if card:
            card.update_action(action)
    
    def set_agent_count(self, agent_name: str, count: int) -> None:
        """Set an agent's item count."""
        card = self.get_agent_card(agent_name)
        if card:
            card.update_count(count)
    
    def reset_all(self) -> None:
        """Reset all agents to idle state."""
        for agent in self.agents:
            self.set_agent_state(agent.name, AgentState.IDLE)
            self.set_agent_action(agent.name, "")
            self.set_agent_count(agent.name, 0)
