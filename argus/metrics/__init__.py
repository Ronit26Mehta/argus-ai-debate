"""
ARGUS Metrics and Traces Module.

Provides observability for the ARGUS system:
    - Execution metrics collection
    - Performance tracing
    - Event logging

Example:
    >>> from argus.metrics import MetricsCollector, Tracer
    >>> 
    >>> # Collect metrics
    >>> collector = MetricsCollector()
    >>> collector.record("debate_duration", 5.2, tags={"proposition": "p1"})
    >>> 
    >>> # Create traces
    >>> with Tracer.span("debate_round") as span:
    ...     span.set_attribute("round", 1)
    ...     # Do work
"""

from argus.metrics.collector import (
    MetricsCollector,
    MetricType,
    Metric,
    get_default_collector,
    record_metric,
    record_counter,
    record_gauge,
    record_histogram,
)
from argus.metrics.traces import (
    Tracer,
    Span,
    SpanContext,
    TraceConfig,
    get_tracer,
)

__all__ = [
    # Collector
    "MetricsCollector",
    "MetricType",
    "Metric",
    "get_default_collector",
    "record_metric",
    "record_counter",
    "record_gauge",
    "record_histogram",
    # Traces
    "Tracer",
    "Span",
    "SpanContext",
    "TraceConfig",
    "get_tracer",
]
