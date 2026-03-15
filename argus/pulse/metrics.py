"""
Metric Store — lightweight metric collection and aggregation.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Base metric data point."""
    name: str = ""
    value: float = 0.0
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class Counter:
    """Monotonically increasing counter."""

    def __init__(self, name: str = ""):
        self.name = name
        self._value: float = 0.0
        self._history: list[tuple[float, float]] = []

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount
        self._history.append((time.time(), self._value))

    @property
    def value(self) -> float:
        return self._value

    @property
    def rate(self) -> float:
        """Events per second over last minute."""
        now = time.time()
        recent = [(t, v) for t, v in self._history if now - t < 60]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1][0] - recent[0][0]
        dv = recent[-1][1] - recent[0][1]
        return dv / max(dt, 0.001)


class Histogram:
    """Records value distribution with percentile computation."""

    def __init__(self, name: str = ""):
        self.name = name
        self._values: list[float] = []
        self._timestamps: list[float] = []

    def observe(self, value: float) -> None:
        self._values.append(value)
        self._timestamps.append(time.time())

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self._values)) if self._values else 0.0

    def percentile(self, p: float) -> float:
        return float(np.percentile(self._values, p)) if self._values else 0.0

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def values(self) -> list[float]:
        return self._values.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "count": self.count,
            "mean": round(self.mean, 4), "p50": round(self.p50, 4),
            "p95": round(self.p95, 4), "p99": round(self.p99, 4),
        }


class Gauge:
    """Instantaneous value gauge."""

    def __init__(self, name: str = "", initial: float = 0.0):
        self.name = name
        self._value: float = initial
        self._history: list[tuple[float, float]] = [(time.time(), initial)]

    def set(self, value: float) -> None:
        self._value = value
        self._history.append((time.time(), value))

    def inc(self, amount: float = 1.0) -> None:
        self.set(self._value + amount)

    def dec(self, amount: float = 1.0) -> None:
        self.set(self._value - amount)

    @property
    def value(self) -> float:
        return self._value

    @property
    def history(self) -> list[tuple[float, float]]:
        return self._history.copy()


class MetricStore:
    """
    Central metric store for PULSE.

    Manages counters, histograms, and gauges for operational monitoring.

    Example:
        >>> store = MetricStore()
        >>> store.counter("debates_total").inc()
        >>> store.histogram("debate_latency_ms").observe(1234.5)
        >>> store.gauge("active_agents").set(3)
    """

    def __init__(self):
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name)
        return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name)
        return self._histograms[name]

    def gauge(self, name: str) -> Gauge:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name)
        return self._gauges[name]

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": {k: v.value for k, v in self._counters.items()},
            "histograms": {k: v.to_dict() for k, v in self._histograms.items()},
            "gauges": {k: v.value for k, v in self._gauges.items()},
        }

    def reset(self) -> None:
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()
