"""
CHRONOS Orchestrator — wraps RDCOrchestrator with temporal C-DAG.

Adds temporal belief tracking to the standard debate flow without
modifying any existing APIs.

Example:
    >>> orchestrator = ChronosOrchestrator(
    ...     base=rdc_orchestrator,
    ...     half_life_registry=EvidenceHalfLifeRegistry(),
    ... )
    >>> result = orchestrator.debate(
    ...     'Drug X reduces mortality by >15%',
    ...     prior=0.5,
    ...     evidence_timestamps={'ev-001': '2020-03-01'},
    ... )
    >>> print(result.temporal_posterior.at('2025-01'))
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from argus.chronos.temporal_cdag import (
    TemporalCDAG,
    TemporalEvidence,
    EvidenceCategory,
)
from argus.chronos.half_life_registry import EvidenceHalfLifeRegistry
from argus.chronos.temporal_posterior import TemporalPosterior
from argus.chronos.drift_detector import (
    BeliefDriftDetector,
    BeliefDriftReport,
)

if TYPE_CHECKING:
    from argus.orchestrator import RDCOrchestrator, DebateResult
    from argus.agents.jury import Verdict

logger = logging.getLogger(__name__)


@dataclass
class ChronosConfig:
    """
    Configuration for CHRONOS temporal debate.

    Attributes:
        temporal_resolution: Time resolution for posterior series
        lookback_years: How far back to compute temporal posterior
        drift_min_magnitude: Minimum posterior change for inflection detection
        drift_window_size: Window size for drift analysis
        enable_drift_detection: Whether to run drift detector
        default_evidence_category: Default category for unclassified evidence
    """
    temporal_resolution: str = "month"
    lookback_years: float = 5.0
    drift_min_magnitude: float = 0.05
    drift_window_size: int = 3
    enable_drift_detection: bool = True
    default_evidence_category: EvidenceCategory = EvidenceCategory.LITERATURE


@dataclass
class ChronosResult:
    """
    Result from a CHRONOS temporal debate.

    Contains:
        - The standard debate result (verdict, etc.)
        - Temporal posterior series
        - Drift analysis report
        - Temporal C-DAG

    Attributes:
        base_result: Standard DebateResult from wrapped orchestrator
        temporal_posterior: Time-indexed posterior series
        drift_report: Belief drift analysis report
        temporal_cdag: The temporal C-DAG used
        config: Configuration used
    """
    base_result: Any = None
    temporal_posterior: Optional[TemporalPosterior] = None
    drift_report: Optional[BeliefDriftReport] = None
    temporal_cdag: Optional[TemporalCDAG] = None
    config: Optional[ChronosConfig] = None

    @property
    def verdict(self) -> Any:
        """Access base verdict."""
        return self.base_result.verdict if self.base_result else None

    @property
    def proposition_id(self) -> str:
        return self.base_result.proposition_id if self.base_result else ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.base_result:
            result["base_result"] = self.base_result.to_dict()
        if self.temporal_posterior:
            result["temporal_posterior"] = self.temporal_posterior.to_dict()
        if self.drift_report:
            result["drift_report"] = self.drift_report.to_dict()
        return result


class ChronosOrchestrator:
    """
    Temporal Belief Drift Orchestrator.

    Wraps an existing RDCOrchestrator with temporal C-DAG capabilities.
    Adds evidence half-life decay, temporal posterior tracking, and
    belief drift detection.

    Example:
        >>> from argus.chronos import ChronosOrchestrator, EvidenceHalfLifeRegistry
        >>>
        >>> registry = EvidenceHalfLifeRegistry(empirical_years=5.0)
        >>> orchestrator = ChronosOrchestrator(
        ...     base=rdc_orchestrator,
        ...     half_life_registry=registry,
        ...     temporal_resolution='month',
        ...     lookback_years=5,
        ... )
        >>> result = orchestrator.debate(
        ...     'Drug X reduces mortality by >15%',
        ...     prior=0.5,
        ...     evidence_timestamps={'ev-001': '2020-03-01'},
        ... )
    """

    def __init__(
        self,
        base: Optional[Any] = None,
        half_life_registry: Optional[EvidenceHalfLifeRegistry] = None,
        config: Optional[ChronosConfig] = None,
        **kwargs: Any,
    ):
        self.base = base
        self.half_life_registry = half_life_registry or EvidenceHalfLifeRegistry()
        self.config = config or ChronosConfig(**{
            k: v for k, v in kwargs.items()
            if k in ChronosConfig.__dataclass_fields__
        })
        self._drift_detector = BeliefDriftDetector(
            min_magnitude=self.config.drift_min_magnitude,
            window_size=self.config.drift_window_size,
        )

    def debate(
        self,
        proposition: str,
        prior: float = 0.5,
        evidence_timestamps: Optional[dict[str, str]] = None,
        evidence_categories: Optional[dict[str, str]] = None,
        temporal_evidence: Optional[list[TemporalEvidence]] = None,
        **kwargs: Any,
    ) -> ChronosResult:
        """
        Run a temporal debate.

        Executes the base debate, then builds a temporal posterior by
        computing the posterior at regular intervals considering evidence
        decay.

        Args:
            proposition: Proposition text
            prior: Prior probability
            evidence_timestamps: Mapping of evidence IDs to ISO timestamps
            evidence_categories: Mapping of evidence IDs to category names
            temporal_evidence: Pre-built temporal evidence nodes
            **kwargs: Passed to base orchestrator

        Returns:
            ChronosResult with temporal posterior and drift analysis
        """
        evidence_timestamps = evidence_timestamps or {}
        evidence_categories = evidence_categories or {}

        # Build temporal C-DAG
        tcdag = TemporalCDAG(
            proposition_text=proposition,
            prior=prior,
        )

        # Register decay functions from registry
        for config in self.half_life_registry.all_configs:
            tcdag.register_decay(config.category, config.half_life_hours)

        # Add pre-built temporal evidence if provided
        if temporal_evidence:
            for ev in temporal_evidence:
                tcdag.add_evidence(ev)
        else:
            # Create temporal evidence from timestamps
            for ev_id, ts_str in evidence_timestamps.items():
                try:
                    ts = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    ts = datetime.utcnow()

                cat_str = evidence_categories.get(
                    ev_id,
                    self.config.default_evidence_category.value,
                )
                try:
                    category = EvidenceCategory(cat_str)
                except ValueError:
                    category = self.config.default_evidence_category

                ev = TemporalEvidence(
                    evidence_id=ev_id,
                    text=f"Evidence {ev_id}",
                    confidence=0.7,
                    polarity=1,
                    category=category,
                    timestamp=ts,
                )
                tcdag.add_evidence(ev)

        # Run base debate if available
        base_result = None
        if self.base is not None:
            try:
                base_result = self.base.debate(proposition, prior=prior, **kwargs)
            except Exception as e:
                logger.warning(f"Base debate failed: {e}")

        # Compute temporal posterior
        now = datetime.utcnow()
        from datetime import timedelta
        lookback_hours = self.config.lookback_years * 365.25 * 24.0
        start_time = now - timedelta(hours=lookback_hours)

        series = tcdag.compute_posterior_series(
            start_time=start_time,
            end_time=now,
            resolution=self.config.temporal_resolution,
        )

        temporal_posterior = TemporalPosterior(
            proposition_id=tcdag.proposition_id,
            proposition_text=proposition,
            prior=prior,
        )

        for time_point, posterior_value in series:
            # Count evidence active at this time
            n_evidence = sum(
                1 for ev in tcdag.evidence if ev.timestamp <= time_point
            )
            temporal_posterior.add_point(
                time=time_point,
                posterior=posterior_value,
                num_evidence=n_evidence,
            )

        # Run drift detection
        drift_report = None
        if self.config.enable_drift_detection and len(series) >= 3:
            drift_report = self._drift_detector.generate_drift_report(
                temporal_posterior, tcdag,
            )

        result = ChronosResult(
            base_result=base_result,
            temporal_posterior=temporal_posterior,
            drift_report=drift_report,
            temporal_cdag=tcdag,
            config=self.config,
        )

        logger.info(
            f"CHRONOS debate complete: {len(series)} temporal points, "
            f"trend={temporal_posterior.trend_direction}"
        )

        return result

    def __or__(self, other: Any) -> Any:
        """Support pipe operator for chaining orchestrators."""
        return other
