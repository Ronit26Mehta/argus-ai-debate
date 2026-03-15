"""
Debate Probes — lightweight instrumentation for debate stages.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

from argus.pulse.metrics import MetricStore

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result from a single probe execution."""
    probe_name: str = ""
    stage: str = ""
    latency_ms: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    success: bool = True
    error_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LatencyTracker:
    """Context manager for tracking operation latency."""

    def __init__(self, store: MetricStore, metric_name: str = "latency_ms"):
        self.store = store
        self.metric_name = metric_name
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "LatencyTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.store.histogram(self.metric_name).observe(self.elapsed_ms)


class DebateProbe:
    """
    Lightweight probe embedded inside debate stages.

    Records latency, token usage, success/failure, and custom metrics
    for each stage of the debate pipeline.

    Example:
        >>> probe = DebateProbe(store)
        >>> with probe.track("evidence_extraction"):
        ...     results = extract_evidence(...)
        >>> probe.record_tokens("evidence_extraction", 500, 200)
    """

    STAGES = [
        "document_ingestion", "evidence_extraction", "claim_generation",
        "agent_deliberation", "cdag_construction", "propagation",
        "verdict_synthesis", "visualization",
    ]

    def __init__(self, store: Optional[MetricStore] = None):
        self.store = store or MetricStore()
        self._results: list[ProbeResult] = []

    @contextmanager
    def track(self, stage: str):
        """Track a debate stage execution."""
        start = time.perf_counter()
        result = ProbeResult(probe_name="debate", stage=stage)
        self.store.counter(f"stage_{stage}_total").inc()

        try:
            yield result
            result.success = True
        except Exception as e:
            result.success = False
            result.error_type = type(e).__name__
            self.store.counter(f"stage_{stage}_errors").inc()
            raise
        finally:
            result.latency_ms = (time.perf_counter() - start) * 1000
            self.store.histogram(f"stage_{stage}_latency_ms").observe(result.latency_ms)
            self._results.append(result)

    def record_tokens(
        self,
        stage: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage."""
        self.store.counter("total_input_tokens").inc(input_tokens)
        self.store.counter("total_output_tokens").inc(output_tokens)
        self.store.histogram(f"stage_{stage}_input_tokens").observe(input_tokens)
        self.store.histogram(f"stage_{stage}_output_tokens").observe(output_tokens)

    def record_accuracy(self, correct: bool) -> None:
        """Record prediction accuracy."""
        self.store.counter("predictions_total").inc()
        if correct:
            self.store.counter("predictions_correct").inc()

    @property
    def results(self) -> list[ProbeResult]:
        return self._results.copy()

    def stage_summary(self) -> dict[str, dict[str, Any]]:
        """Summarize metrics by stage."""
        summary: dict[str, dict[str, Any]] = {}
        for stage in self.STAGES:
            hist = self.store.histogram(f"stage_{stage}_latency_ms")
            if hist.count > 0:
                summary[stage] = hist.to_dict()
        return summary
