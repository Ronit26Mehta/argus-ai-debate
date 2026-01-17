# ARGUS Agents Module

## Overview

The `agents/` module implements specialized AI agents for the multi-agent debate system.

## Agent Types

| Agent | Role | Description |
|-------|------|-------------|
| `ProponentAgent` | Advocate | Argues in favor of a position |
| `OpponentAgent` | Critic | Challenges and counters arguments |
| `JudgeAgent` | Arbiter | Evaluates arguments and scores |
| `ResearchAgent` | Researcher | Gathers evidence and facts |
| `SynthesisAgent` | Synthesizer | Combines insights into conclusions |

## Usage

```python
from argus.agents import (
    ProponentAgent,
    OpponentAgent,
    JudgeAgent,
    AgentConfig,
)

# Configure agent
config = AgentConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2000,
)

# Create proponent
proponent = ProponentAgent(
    topic="Renewable energy is essential for climate action",
    config=config,
)

# Generate argument
argument = proponent.generate_argument(
    context="Current global warming trends...",
    previous_arguments=[],
)
print(argument.claim)
print(argument.evidence)
print(argument.reasoning)

# Create opponent
opponent = OpponentAgent(topic=proponent.topic, config=config)
rebuttal = opponent.generate_rebuttal(argument)

# Create judge
judge = JudgeAgent(config=config)
evaluation = judge.evaluate(
    proponent_argument=argument,
    opponent_argument=rebuttal,
)
print(f"Winner: {evaluation.winner}")
print(f"Score: {evaluation.score}")
print(f"Reasoning: {evaluation.reasoning}")
```

## Agent Configuration

```python
from argus.agents import AgentConfig

config = AgentConfig(
    model="gpt-4o",              # LLM model
    provider="openai",           # LLM provider
    temperature=0.7,             # Creativity
    max_tokens=2000,             # Response length
    system_prompt="You are...",  # Custom system prompt
    enable_tools=True,           # Allow tool use
    tools=["duckduckgo", "arxiv"],  # Specific tools
)
```

## Custom Agents

```python
from argus.agents import BaseAgent, AgentConfig

class MyCustomAgent(BaseAgent):
    """Custom agent implementation."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.role = "custom"
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Add custom logic
        enhanced_prompt = f"[Custom Context] {prompt}"
        return self.llm.generate(enhanced_prompt).content

# Use
agent = MyCustomAgent(config)
response = agent.generate("Analyze this data...")
```

## Agent Memory

```python
from argus.agents import ProponentAgent, AgentMemory

# Create agent with memory
agent = ProponentAgent(
    topic="AI Safety",
    memory=AgentMemory(max_history=10),
)

# Agent remembers previous interactions
response1 = agent.generate_argument(context="...")
response2 = agent.generate_argument(context="...")  # Has memory of response1
```
