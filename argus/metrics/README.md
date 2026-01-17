# ARGUS Metrics Module

## Overview

The `metrics/` module provides observability and performance tracking for debate workflows through metrics collection and tracing.

## Components

| File | Description |
|------|-------------|
| `collector.py` | Metrics collection and aggregation |
| `traces.py` | Trace/span tracking for debugging |

## Quick Start

```python
from argus.metrics import MetricsCollector, Timer

# Create collector
collector = MetricsCollector()

# Track debate metrics
collector.increment("debates_started")
collector.observe("evidence_count", 15)
collector.observe("round_duration_seconds", 12.5)

# Get summary
summary = collector.get_summary()
print(summary)
```

## Metrics Collector

```python
from argus.metrics import MetricsCollector

collector = MetricsCollector()

# Counter (increments only)
collector.increment("api_calls")
collector.increment("api_calls", delta=5)

# Gauge (current value)
collector.set_gauge("active_debates", 3)

# Histogram (observations)
collector.observe("response_time", 0.234)
collector.observe("response_time", 0.156)

# Get statistics
stats = collector.get_histogram_stats("response_time")
print(f"Mean: {stats['mean']:.3f}, P99: {stats['p99']:.3f}")

# Export metrics
metrics = collector.export()  # Dict format
collector.to_prometheus()     # Prometheus format
```

## Tracing

```python
from argus.metrics import Tracer, Span

tracer = Tracer()

# Create trace for a debate
with tracer.start_span("debate") as root:
    root.set_attribute("topic", "AI safety")
    
    with tracer.start_span("evidence_gathering", parent=root) as span:
        span.set_attribute("sources", ["arxiv", "wikipedia"])
        # ... gather evidence
        span.set_attribute("evidence_count", 12)
    
    with tracer.start_span("verdict_rendering", parent=root) as span:
        # ... render verdict
        span.set_attribute("verdict", "SUPPORTED")

# Export trace
trace = tracer.export()
print(f"Total duration: {trace.duration_ms}ms")
print(f"Spans: {len(trace.spans)}")
```

## Timer Utility

```python
from argus.metrics import Timer

# Context manager
with Timer() as t:
    # ... do work
print(f"Duration: {t.elapsed:.3f}s")

# Decorator
from argus.metrics import timed

@timed("my_function")
def my_function():
    pass  # Duration automatically logged
```

## Integration with Debates

```python
from argus import RDCOrchestrator
from argus.metrics import MetricsCollector

collector = MetricsCollector()

orchestrator = RDCOrchestrator(
    llm=llm,
    metrics_collector=collector,
)

result = orchestrator.debate("Is X true?")

# Access metrics
print(collector.get_summary())
# {
#   "debate_rounds": 3,
#   "evidence_gathered": 15,
#   "rebuttals_generated": 7,
#   "total_duration_seconds": 45.2,
#   "api_calls": 12,
# }
```
