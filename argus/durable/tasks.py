"""
ARGUS Idempotent Tasks.

Task wrappers for ensuring idempotent execution during replay.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable, Dict

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from a task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    executed_at: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "executed_at": self.executed_at.isoformat(),
            "duration_ms": self.duration_ms,
        }


class TaskRegistry:
    """Registry for tracking executed tasks to prevent re-execution."""
    
    def __init__(self):
        self._executed: Dict[str, TaskResult] = {}
        self._pending: set[str] = set()
    
    def _compute_task_id(self, name: str, args: tuple, kwargs: dict) -> str:
        """Compute deterministic task ID from name and arguments."""
        content = json.dumps({"name": name, "args": list(args), "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_executed(self, task_id: str) -> bool:
        return task_id in self._executed
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        return self._executed.get(task_id)
    
    def record(self, task_id: str, result: TaskResult) -> None:
        self._executed[task_id] = result
        self._pending.discard(task_id)
    
    def mark_pending(self, task_id: str) -> None:
        self._pending.add(task_id)
    
    def is_pending(self, task_id: str) -> bool:
        return task_id in self._pending
    
    def clear(self) -> None:
        self._executed.clear()
        self._pending.clear()
    
    def get_all_results(self) -> Dict[str, TaskResult]:
        return self._executed.copy()


# Global registry
_global_registry = TaskRegistry()


def get_task_registry() -> TaskRegistry:
    return _global_registry


def idempotent_task(
    registry: Optional[TaskRegistry] = None,
    cache_result: bool = True,
) -> Callable:
    """Decorator to make a function idempotent for workflow replay.
    
    When a workflow resumes, previously executed tasks return cached results
    instead of re-executing, ensuring consistency.
    
    Example:
        >>> @idempotent_task()
        ... def call_external_api(query: str) -> dict:
        ...     # This will only execute once per unique query
        ...     return api.call(query)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            reg = registry or _global_registry
            task_id = reg._compute_task_id(func.__name__, args, kwargs)
            
            # Check if already executed
            if cache_result and reg.is_executed(task_id):
                cached = reg.get_result(task_id)
                logger.debug(f"Returning cached result for task {task_id}")
                if cached.success:
                    return cached.result
                else:
                    raise RuntimeError(cached.error)
            
            # Execute the task
            reg.mark_pending(task_id)
            import time
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                task_result = TaskResult(
                    task_id=task_id, success=True,
                    result=result, duration_ms=duration
                )
                reg.record(task_id, task_result)
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                task_result = TaskResult(
                    task_id=task_id, success=False,
                    error=str(e), duration_ms=duration
                )
                reg.record(task_id, task_result)
                raise
        
        return wrapper
    return decorator


class TaskExecutor:
    """Execute tasks with idempotency and retry support."""
    
    def __init__(self, registry: Optional[TaskRegistry] = None, max_retries: int = 3):
        self.registry = registry or _global_registry
        self.max_retries = max_retries
    
    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function as an idempotent task."""
        task_id = self.registry._compute_task_id(func.__name__, args, kwargs)
        
        if self.registry.is_executed(task_id):
            cached = self.registry.get_result(task_id)
            if cached.success:
                return cached.result
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                import time
                start = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                task_result = TaskResult(
                    task_id=task_id, success=True,
                    result=result, duration_ms=duration
                )
                self.registry.record(task_id, task_result)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Task {task_id} failed (attempt {attempt + 1}): {e}")
        
        task_result = TaskResult(task_id=task_id, success=False, error=str(last_error))
        self.registry.record(task_id, task_result)
        raise last_error
