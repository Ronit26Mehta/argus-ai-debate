# ARGUS Debate Module

## Overview

The `debate/` module provides the debate orchestration system for structured multi-agent argumentation.

## Components

| Component | Description |
|-----------|-------------|
| `DebateOrchestrator` | Main coordinator for debates |
| `DebateConfig` | Debate configuration |
| `DebateResult` | Final debate results |
| `RoundResult` | Per-round results |
| `ArgumentGraph` | Tracks argument relationships |

## Quick Start

```python
from argus.debate import DebateOrchestrator, DebateConfig

# Configure debate
config = DebateConfig(
    topic="Should AI development be regulated by governments?",
    num_rounds=3,
    proponent_model="gpt-4o",
    opponent_model="gpt-4o",
    judge_model="gpt-4o",
    enable_evidence=True,
    evidence_sources=["arxiv", "wikipedia"],
)

# Create orchestrator
orchestrator = DebateOrchestrator(config)

# Run debate
result = orchestrator.run()

# Access results
print(f"Topic: {result.topic}")
print(f"Winner: {result.winner}")
print(f"Final Score: {result.final_score}")
print(f"Confidence: {result.confidence}")

# Detailed breakdown
for i, round_result in enumerate(result.rounds):
    print(f"\n=== Round {i+1} ===")
    print(f"Proponent: {round_result.proponent_score}")
    print(f"Opponent: {round_result.opponent_score}")
    print(f"Judge notes: {round_result.judge_notes}")
```

## Configuration Options

```python
from argus.debate import DebateConfig

config = DebateConfig(
    # Topic
    topic="Is remote work better than office work?",
    
    # Rounds
    num_rounds=3,
    time_per_round=None,  # Optional time limit
    
    # Models
    proponent_model="gpt-4o",
    opponent_model="gpt-4o-mini",
    judge_model="gpt-4o",
    
    # Model alternatives
    provider="openai",  # Default provider for all
    
    # Evidence
    enable_evidence=True,
    evidence_sources=["arxiv", "wikipedia", "duckduckgo"],
    max_evidence_per_round=5,
    
    # Scoring
    scoring_criteria=["logic", "evidence", "persuasion"],
    require_citations=True,
    
    # Output
    verbose=True,
    save_transcript=True,
    output_format="markdown",
)
```

## Debate Modes

### Standard Debate

```python
from argus.debate import DebateOrchestrator, DebateConfig

config = DebateConfig(
    topic="Should nuclear energy be expanded?",
    num_rounds=3,
)
result = DebateOrchestrator(config).run()
```

### Research Debate

```python
config = DebateConfig(
    topic="What is the optimal treatment for condition X?",
    mode="research",
    enable_evidence=True,
    evidence_sources=["pubmed", "arxiv", "semantic_scholar"],
    require_citations=True,
)
result = DebateOrchestrator(config).run()
```

### Quick Debate

```python
config = DebateConfig(
    topic="Is Python better than JavaScript?",
    num_rounds=1,
    proponent_model="gpt-4o-mini",
    opponent_model="gpt-4o-mini",
)
result = DebateOrchestrator(config).run()
```

## Accessing Arguments

```python
result = orchestrator.run()

# Get all proponent arguments
for round_result in result.rounds:
    arg = round_result.proponent_argument
    print(f"Claim: {arg.claim}")
    print(f"Evidence: {arg.evidence}")
    print(f"Reasoning: {arg.reasoning}")
    print(f"Sources: {arg.sources}")

# Export argument graph
graph = result.get_argument_graph()
graph.export("debate_graph.json")
```

## Streaming Debate

```python
orchestrator = DebateOrchestrator(config)

# Stream debate progress
for event in orchestrator.stream():
    if event.type == "round_start":
        print(f"\n=== Round {event.round} ===")
    elif event.type == "proponent_speaking":
        print(f"Proponent: {event.content}", end="")
    elif event.type == "opponent_speaking":
        print(f"Opponent: {event.content}", end="")
    elif event.type == "judge_verdict":
        print(f"Judge: {event.verdict}")
```

## Custom Scoring

```python
from argus.debate import DebateConfig, ScoringConfig

scoring = ScoringConfig(
    criteria={
        "logic": 0.3,
        "evidence_quality": 0.3,
        "persuasion": 0.2,
        "clarity": 0.2,
    },
    min_score=0,
    max_score=10,
)

config = DebateConfig(
    topic="...",
    scoring=scoring,
)
```
