"""
CHRONOS — Temporal Belief Drift Engine for ARGUS.

Introduces time as a first-class dimension of belief. Evidence nodes carry
decay functions that reduce their effective confidence as they age. The
Bayesian posterior becomes a TemporalPosterior — a time-indexed series with
credible intervals. A BeliefDriftDetector identifies inflection points and
traces them back to specific evidence clusters.

Key innovations:
    - Evidence Half-Life Decay (type-specific exponential decay)
    - PELT-based Belief Drift Detection on posterior series
    - Causal Attribution tracing C-DAG evidence to inflection points

Example:
    >>> from argus.chronos import ChronosOrchestrator, EvidenceHalfLifeRegistry
    >>>
    >>> registry = EvidenceHalfLifeRegistry()
    >>> orchestrator = ChronosOrchestrator(
    ...     base=rdc_orchestrator,
    ...     half_life_registry=registry,
    ... )
    >>> result = orchestrator.debate(
    ...     'Drug X reduces mortality by >15%',
    ...     prior=0.5,
    ...     evidence_timestamps={'ev-001': '2020-03-01'},
    ... )
    >>> print(result.temporal_posterior.at('2025-01'))
"""

from argus.chronos.temporal_cdag import (
    TemporalCDAG,
    TemporalEvidence,
    DecayFunction,
    EvidenceCategory,
)
from argus.chronos.half_life_registry import (
    EvidenceHalfLifeRegistry,
    HalfLifeConfig,
)
from argus.chronos.temporal_posterior import (
    TemporalPosterior,
    CredibleBand,
    PosteriorSnapshot,
)
from argus.chronos.drift_detector import (
    BeliefDriftDetector,
    InflectionPoint,
    CausalAttribution,
    BeliefDriftReport,
)
from argus.chronos.orchestrator import (
    ChronosOrchestrator,
    ChronosConfig,
    ChronosResult,
)
from argus.chronos.visualization import (
    plot_temporal_posterior,
    plot_drift_timeline,
)

__all__ = [
    "TemporalCDAG",
    "TemporalEvidence",
    "DecayFunction",
    "EvidenceCategory",
    "EvidenceHalfLifeRegistry",
    "HalfLifeConfig",
    "TemporalPosterior",
    "CredibleBand",
    "PosteriorSnapshot",
    "BeliefDriftDetector",
    "InflectionPoint",
    "CausalAttribution",
    "BeliefDriftReport",
    "ChronosOrchestrator",
    "ChronosConfig",
    "ChronosResult",
    "plot_temporal_posterior",
    "plot_drift_timeline",
]
