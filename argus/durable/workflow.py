"""
ARGUS Durable Workflow.

Workflow orchestration with checkpointing and resume capability.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable, List, TYPE_CHECKING

from argus.durable.config import DurableConfig
from argus.durable.checkpointer import BaseCheckpointer, MemoryCheckpointer, Checkpoint
from argus.durable.state import DebateState, StateManager
from argus.durable.tasks import TaskRegistry, get_task_registry

if TYPE_CHECKING:
    from argus.orchestrator import RDCOrchestrator

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowRun:
    """A single workflow execution run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    checkpoints_created: int = 0


class DurableWorkflow:
    """Durable workflow with checkpointing and resume capability.
    
    Example:
        >>> workflow = DurableWorkflow(thread_id="debate-123")
        >>> 
        >>> # Run with automatic checkpointing
        >>> async def my_workflow():
        ...     result1 = await workflow.step("step1", do_something)
        ...     await workflow.checkpoint()
        ...     result2 = await workflow.step("step2", do_more)
        ...     return result2
        >>> 
        >>> result = workflow.run(my_workflow)
        >>> 
        >>> # Resume from checkpoint if interrupted
        >>> workflow.resume()
    """
    
    def __init__(
        self,
        thread_id: Optional[str] = None,
        config: Optional[DurableConfig] = None,
        checkpointer: Optional[BaseCheckpointer] = None,
    ):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = config or DurableConfig()
        self.checkpointer = checkpointer or MemoryCheckpointer()
        self.state_manager = StateManager()
        self.task_registry = TaskRegistry()
        self._current_run: Optional[WorkflowRun] = None
        self._step_count = 0
    
    def step(self, name: str, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a workflow step with automatic tracking."""
        self._step_count += 1
        if self._current_run:
            self._current_run.current_step = self._step_count
        
        task_id = self.task_registry._compute_task_id(name, args, kwargs)
        if self.task_registry.is_executed(task_id):
            cached = self.task_registry.get_result(task_id)
            if cached and cached.success:
                logger.debug(f"Returning cached result for step '{name}'")
                return cached.result
        
        logger.debug(f"Executing step '{name}' (step {self._step_count})")
        result = func(*args, **kwargs)
        
        if self.config.auto_checkpoint and self._step_count % self.config.checkpoint_interval == 0:
            self.checkpoint(f"Auto-checkpoint after step '{name}'")
        
        return result
    
    def checkpoint(self, description: str = "") -> str:
        """Create a checkpoint of current state."""
        state = {
            "step_count": self._step_count,
            "task_results": {k: v.to_dict() for k, v in self.task_registry.get_all_results().items()},
            "debate_state": self.state_manager.get_state().to_dict() if self.state_manager.get_state() else None,
            "description": description,
        }
        checkpoint_id = self.checkpointer.save(
            self.thread_id, state, step=self._step_count, description=description
        )
        if self._current_run:
            self._current_run.checkpoints_created += 1
        logger.info(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def resume(self, checkpoint_id: Optional[str] = None) -> bool:
        """Resume workflow from checkpoint."""
        checkpoint = self.checkpointer.load(self.thread_id, checkpoint_id)
        if not checkpoint:
            logger.warning(f"No checkpoint found for thread {self.thread_id}")
            return False
        
        self._step_count = checkpoint.state.get("step_count", 0)
        task_results = checkpoint.state.get("task_results", {})
        for task_id, result_dict in task_results.items():
            from argus.durable.tasks import TaskResult
            result = TaskResult(
                task_id=result_dict["task_id"],
                success=result_dict["success"],
                result=result_dict.get("result"),
                error=result_dict.get("error"),
            )
            self.task_registry.record(task_id, result)
        
        debate_state = checkpoint.state.get("debate_state")
        if debate_state:
            self.state_manager._current_state = DebateState.from_dict(debate_state)
        
        if self._current_run:
            self._current_run.status = WorkflowStatus.RUNNING
        
        logger.info(f"Resumed from checkpoint {checkpoint.checkpoint_id} at step {self._step_count}")
        return True
    
    def rollback(self, steps: int = 1) -> bool:
        """Rollback to a previous checkpoint."""
        checkpoints = self.checkpointer.list_checkpoints(self.thread_id, limit=steps + 1)
        if len(checkpoints) <= steps:
            logger.warning("Not enough checkpoints to rollback")
            return False
        target = checkpoints[steps]
        return self.resume(target.checkpoint_id)
    
    def start_run(self, total_steps: int = 0) -> WorkflowRun:
        """Start a new workflow run."""
        self._current_run = WorkflowRun(
            thread_id=self.thread_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow(),
            total_steps=total_steps,
        )
        self._step_count = 0
        logger.info(f"Started workflow run: {self._current_run.run_id}")
        return self._current_run
    
    def complete_run(self, error: Optional[str] = None) -> WorkflowRun:
        """Complete the current workflow run."""
        if not self._current_run:
            raise RuntimeError("No active run")
        self._current_run.completed_at = datetime.utcnow()
        self._current_run.status = WorkflowStatus.COMPLETED if not error else WorkflowStatus.FAILED
        self._current_run.error = error
        logger.info(f"Completed workflow run: {self._current_run.run_id} ({self._current_run.status.value})")
        return self._current_run
    
    def get_current_run(self) -> Optional[WorkflowRun]:
        return self._current_run
    
    def list_checkpoints(self, limit: int = 10) -> List[Checkpoint]:
        return self.checkpointer.list_checkpoints(self.thread_id, limit)


class DurableDebateWorkflow(DurableWorkflow):
    """Durable workflow specialized for debate orchestration."""
    
    def __init__(self, orchestrator: Optional["RDCOrchestrator"] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._orchestrator = orchestrator
    
    def initialize_debate(self, proposition: str, prior: float = 0.5) -> DebateState:
        """Initialize a new debate."""
        import uuid
        state = self.state_manager.initialize(
            debate_id=str(uuid.uuid4()),
            proposition=proposition,
        )
        state.metadata["prior"] = prior
        self.checkpoint("Debate initialized")
        return state
    
    def run_round(self, round_num: int) -> dict[str, Any]:
        """Execute a debate round as a workflow step."""
        def execute_round() -> dict[str, Any]:
            state = self.state_manager.get_state()
            if not state:
                raise RuntimeError("No debate state")
            self.state_manager.update(current_round=round_num)
            return {"round": round_num, "status": "completed"}
        return self.step(f"round_{round_num}", execute_round)
