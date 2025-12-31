"""
Metrics Collector for ARGUS.

Collects and aggregates metrics from ARGUS operations including:
    - Debate execution times
    - Evidence counts
    - LLM usage
    - Cache hit rates

Example:
    >>> collector = MetricsCollector()
    >>> 
    >>> # Record metrics
    >>> collector.record_counter("debates_completed", 1)
    >>> collector.record_histogram("debate_duration", 5.2)
    >>> 
    >>> # Get stats
    >>> stats = collector.get_stats()
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """A recorded metric value."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


class MetricsCollector:
    """Collects and aggregates metrics.
    
    Thread-safe collector for recording various types of metrics.
    
    Example:
        >>> collector = MetricsCollector()
        >>> 
        >>> # Record a counter (cumulative)
        >>> collector.record_counter("api_calls", 1)
        >>> 
        >>> # Record a gauge (point-in-time)
        >>> collector.record_gauge("active_debates", 5)
        >>> 
        >>> # Record a histogram (distribution)
        >>> collector.record_histogram("response_time_ms", 150.5)
    """
    
    def __init__(self, max_history: int = 10000):
        """Initialize collector.
        
        Args:
            max_history: Maximum metrics to keep in history
        """
        self.max_history = max_history
        self._lock = RLock()
        
        # Storage
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._history: list[Metric] = []
        
        # Tags
        self._default_tags: dict[str, str] = {}
        
        logger.debug("Initialized MetricsCollector")
    
    def set_default_tags(self, tags: dict[str, str]) -> None:
        """Set default tags for all metrics.
        
        Args:
            tags: Dict of tag key-value pairs
        """
        with self._lock:
            self._default_tags.update(tags)
    
    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags
        """
        with self._lock:
            # Merge tags
            all_tags = {**self._default_tags, **(tags or {})}
            
            # Create metric
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                tags=all_tags,
            )
            
            # Update storage based on type
            if metric_type == MetricType.COUNTER:
                self._counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self._gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                self._histograms[name].append(value)
                # Limit histogram size
                if len(self._histograms[name]) > 10000:
                    self._histograms[name] = self._histograms[name][-10000:]
            
            # Add to history
            self._history.append(metric)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
    
    def record_counter(
        self,
        name: str,
        value: float = 1,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a counter increment.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        self.record(name, value, MetricType.COUNTER, tags)
    
    def record_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a gauge value.
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags
        """
        self.record(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation.
        
        Args:
            name: Histogram name
            value: Observed value
            tags: Optional tags
        """
        self.record(name, value, MetricType.HISTOGRAM, tags)
    
    def get_counter(self, name: str) -> float:
        """Get current counter value.
        
        Args:
            name: Counter name
            
        Returns:
            Counter value
        """
        with self._lock:
            return self._counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value.
        
        Args:
            name: Gauge name
            
        Returns:
            Gauge value or None
        """
        with self._lock:
            return self._gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics.
        
        Args:
            name: Histogram name
            
        Returns:
            Dict with count, min, max, mean, p50, p95, p99
        """
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return {}
            
            import statistics
            sorted_values = sorted(values)
            count = len(values)
            
            return {
                "count": count,
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p50": sorted_values[int(count * 0.50)] if count > 0 else 0,
                "p95": sorted_values[int(count * 0.95)] if count >= 20 else sorted_values[-1],
                "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1],
            }
    
    def get_stats(self) -> dict[str, Any]:
        """Get all metrics statistics.
        
        Returns:
            Dict with all counters, gauges, and histogram stats
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name)
                    for name in self._histograms
                },
                "total_recorded": len(self._history),
            }
    
    def get_history(
        self,
        name: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Metric]:
        """Get metric history.
        
        Args:
            name: Filter by metric name
            metric_type: Filter by type
            since: Filter by timestamp
            limit: Maximum records
            
        Returns:
            List of Metric objects
        """
        with self._lock:
            result = self._history
            
            if name:
                result = [m for m in result if m.name == name]
            if metric_type:
                result = [m for m in result if m.metric_type == metric_type]
            if since:
                result = [m for m in result if m.timestamp >= since]
            
            return result[-limit:]
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._history.clear()
            logger.info("Reset all metrics")
    
    def export(self) -> list[dict[str, Any]]:
        """Export all metrics as dicts.
        
        Returns:
            List of metric dicts
        """
        with self._lock:
            return [m.to_dict() for m in self._history]


# =============================================================================
# Global Default Collector
# =============================================================================

_default_collector: Optional[MetricsCollector] = None
_collector_lock = RLock()


def get_default_collector() -> MetricsCollector:
    """Get the default metrics collector.
    
    Returns:
        MetricsCollector instance
    """
    global _default_collector
    
    with _collector_lock:
        if _default_collector is None:
            _default_collector = MetricsCollector()
        return _default_collector


def record_metric(
    name: str,
    value: float,
    metric_type: MetricType = MetricType.GAUGE,
    tags: Optional[dict[str, str]] = None,
) -> None:
    """Record a metric in the default collector."""
    get_default_collector().record(name, value, metric_type, tags)


def record_counter(
    name: str,
    value: float = 1,
    tags: Optional[dict[str, str]] = None,
) -> None:
    """Record a counter in the default collector."""
    get_default_collector().record_counter(name, value, tags)


def record_gauge(
    name: str,
    value: float,
    tags: Optional[dict[str, str]] = None,
) -> None:
    """Record a gauge in the default collector."""
    get_default_collector().record_gauge(name, value, tags)


def record_histogram(
    name: str,
    value: float,
    tags: Optional[dict[str, str]] = None,
) -> None:
    """Record a histogram in the default collector."""
    get_default_collector().record_histogram(name, value, tags)
