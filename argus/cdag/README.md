# ARGUS CDAG Module

## Overview

The **Conceptual Debate Argumentation Graph (C-DAG)** is the core data structure for representing debates as directed graphs with propositions, evidence, and rebuttals.

## Components

| File | Description |
|------|-------------|
| `graph.py` | Main CDAG graph class |
| `nodes.py` | Node types (Proposition, Evidence, Rebuttal, Finding) |
| `edges.py` | Edge types (SUPPORTS, ATTACKS, REBUTS, REFINES) |
| `propagation.py` | Bayesian belief propagation algorithms |

## Quick Start

```python
from argus.cdag import CDAG, Proposition, Evidence, EdgeType
from argus.cdag.nodes import EvidenceType
from argus.cdag.propagation import compute_posterior

# Create graph
graph = CDAG(name="my_debate")

# Add proposition
prop = Proposition(
    text="AI will transform healthcare",
    prior=0.5,
    domain="technology",
)
graph.add_proposition(prop)

# Add supporting evidence
evidence = Evidence(
    text="FDA approved 521 AI medical devices in 2023",
    evidence_type=EvidenceType.EMPIRICAL,
    polarity=1,  # +1 supports, -1 attacks
    confidence=0.9,
    relevance=0.85,
)
graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)

# Compute posterior probability
posterior = compute_posterior(graph, prop.id)
print(f"Posterior: {posterior:.3f}")
```

## Node Types

```python
from argus.cdag.nodes import (
    Proposition,    # Main claim to evaluate
    Evidence,       # Supporting/attacking information
    Rebuttal,       # Challenge to evidence
    Finding,        # Intermediate conclusion
    Assumption,     # Underlying premise
)

# Evidence types
from argus.cdag.nodes import EvidenceType
EvidenceType.EMPIRICAL     # Data, studies, measurements
EvidenceType.EXPERT        # Expert testimony
EvidenceType.THEORETICAL   # Logical/theoretical argument
EvidenceType.ANECDOTAL     # Case studies, examples
```

## Edge Types

```python
from argus.cdag.edges import EdgeType

EdgeType.SUPPORTS   # +1 polarity, evidence supports claim
EdgeType.ATTACKS    # -1 polarity, evidence challenges claim
EdgeType.REBUTS     # -1 polarity, rebuttal targets evidence
EdgeType.REFINES    #  0 polarity, clarification/specification
```

## Belief Propagation

```python
from argus.cdag.propagation import (
    compute_posterior,
    propagate_beliefs,
    log_odds_update,
)

# Compute posterior using log-odds Bayesian updating
posterior = compute_posterior(graph, proposition_id)

# Propagate beliefs through entire graph
posteriors = propagate_beliefs(graph, max_iterations=10)

# Manual log-odds update
new_odds = log_odds_update(
    prior_odds=1.0,
    likelihood_ratio=2.5,
    confidence=0.8,
)
```

## Graph Operations

```python
# Get all evidence for a proposition
evidence_list = graph.get_evidence(prop.id)

# Get rebuttals targeting evidence
rebuttals = graph.get_rebuttals(evidence.id)

# Export to JSON
graph.save("debate.json")
graph = CDAG.load("debate.json")

# Visualize (requires networkx, matplotlib)
graph.visualize("cdag_plot.png")
```
