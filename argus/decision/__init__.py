"""
Decision and Experimentation module for ARGUS.

This module implements:
    - Bayesian posterior updates
    - Expected Information Gain (EIG) estimation
    - VoI-based experiment planning
    - Calibration metrics (Brier, ECE)
"""

from argus.decision.bayesian import (
    BayesianUpdater,
    log_odds,
    from_log_odds,
    update_posterior,
)
from argus.decision.eig import (
    EIGEstimator,
    estimate_eig,
    ActionCandidate,
)
from argus.decision.planner import (
    VoIPlanner,
    ExperimentQueue,
    PlannerConfig,
)
from argus.decision.calibration import (
    CalibrationMetrics,
    compute_brier_score,
    compute_ece,
    temperature_scaling,
    ReliabilityDiagram,
)

__all__ = [
    # Bayesian
    "BayesianUpdater",
    "log_odds",
    "from_log_odds",
    "update_posterior",
    # EIG
    "EIGEstimator",
    "estimate_eig",
    "ActionCandidate",
    # Planner
    "VoIPlanner",
    "ExperimentQueue",
    "PlannerConfig",
    # Calibration
    "CalibrationMetrics",
    "compute_brier_score",
    "compute_ece",
    "temperature_scaling",
    "ReliabilityDiagram",
]
