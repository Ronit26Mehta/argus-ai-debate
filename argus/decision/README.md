# ARGUS Decision Module

## Overview

The `decision/` module provides decision-theoretic planning using **Value of Information (VoI)** and **Expected Information Gain (EIG)** to prioritize evidence gathering.

## Components

| File | Description |
|------|-------------|
| `bayesian.py` | Bayesian inference utilities |
| `calibration.py` | Confidence calibration methods |
| `eig.py` | Expected Information Gain calculation |
| `planner.py` | VoI-based action planning |

## Quick Start

```python
from argus.decision import VoIPlanner, EIGCalculator
from argus import get_llm

llm = get_llm("openai", model="gpt-4o")

# Create VoI planner
planner = VoIPlanner(llm=llm, n_samples=1000)

# Define possible experiments/actions
experiments = [
    {"name": "clinical_trial", "cost": 100, "expected_impact": 0.8},
    {"name": "literature_review", "cost": 10, "expected_impact": 0.3},
    {"name": "expert_interview", "cost": 25, "expected_impact": 0.5},
]

# Rank by Expected Information Gain
ranked = planner.rank_by_eig(experiments, current_belief=0.5)
for exp in ranked:
    print(f"{exp['name']}: EIG={exp['eig']:.3f}")

# Select optimal set under budget constraint
optimal = planner.select_under_budget(experiments, budget=50)
print(f"Selected: {[e['name'] for e in optimal]}")
```

## Expected Information Gain (EIG)

```python
from argus.decision.eig import (
    compute_eig,
    entropy,
    conditional_entropy,
)

# Compute entropy of belief distribution
h = entropy([0.5, 0.5])  # Maximum entropy

# EIG = H(belief) - E[H(belief|experiment)]
eig = compute_eig(
    prior_belief=0.5,
    experiment_outcomes=[0.2, 0.8],
    outcome_probs=[0.4, 0.6],
)
```

## Calibration

```python
from argus.decision.calibration import (
    brier_score,
    expected_calibration_error,
    temperature_scale,
)

# Compute Brier Score (lower is better)
predictions = [0.8, 0.6, 0.9, 0.3]
outcomes = [1, 1, 1, 0]
bs = brier_score(predictions, outcomes)

# Expected Calibration Error
ece = expected_calibration_error(predictions, outcomes, n_bins=10)

# Temperature scaling for confidence adjustment
calibrated = temperature_scale(predictions, temperature=1.5)
```

## Bayesian Inference

```python
from argus.decision.bayesian import (
    bayes_update,
    log_odds_to_prob,
    prob_to_log_odds,
)

# Bayesian update
posterior = bayes_update(
    prior=0.5,
    likelihood_given_true=0.9,
    likelihood_given_false=0.2,
)

# Log-odds conversion
log_odds = prob_to_log_odds(0.75)  # 1.099
prob = log_odds_to_prob(1.099)     # 0.75
```

## Planning Strategies

```python
from argus.decision.planner import (
    VoIPlanner,
    GreedyPlanner,
    BudgetedPlanner,
)

# Greedy selection by EIG
planner = GreedyPlanner(llm)
actions = planner.select(experiments, n=3)

# Budget-constrained optimization
planner = BudgetedPlanner(llm, budget=100)
actions = planner.optimize(experiments)
```
