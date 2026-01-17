"""
ARGUS Durable Execution Module.

State checkpointing, workflow persistence, and resume capability
for reliable long-running debate workflows.

Example:
    >>> from argus.durable import DurableWorkflow, SQLiteCheckpointer
    >>> 
    >>> # Create durable workflow
    >>> checkpointer = SQLiteCheckpointer("debates.db")
    >>> workflow = DurableWorkflow(thread_id="debate-123", checkpointer=checkpointer)
    >>> 
    >>> # Execute with automatic checkpointing
    >>> workflow.start_run()
    >>> result = workflow.step("evidence_collection", collect_evidence)
    >>> workflow.checkpoint("After evidence")
    >>> 
    >>> # Resume if interrupted
    >>> workflow.resume()
"""

from argus.durable.config import (
    DurableConfig,
    CheckpointerType,
    RetryPolicy,
    get_default_durable_config,
)

from argus.durable.checkpointer import (
    Checkpoint,
    BaseCheckpointer,
    MemoryCheckpointer,
    SQLiteCheckpointer,
    FileSystemCheckpointer,
)

from argus.durable.state import (
    StateSnapshot,
    DebateState,
    StateManager,
    serialize_state,
    deserialize_state,
    serialize_graph,
)

from argus.durable.tasks import (
    TaskResult,
    TaskRegistry,
    TaskExecutor,
    idempotent_task,
    get_task_registry,
)

from argus.durable.workflow import (
    WorkflowStatus,
    WorkflowRun,
    DurableWorkflow,
    DurableDebateWorkflow,
)

__all__ = [
    # Config
    "DurableConfig",
    "CheckpointerType",
    "RetryPolicy",
    "get_default_durable_config",
    # Checkpointer
    "Checkpoint",
    "BaseCheckpointer",
    "MemoryCheckpointer",
    "SQLiteCheckpointer",
    "FileSystemCheckpointer",
    # State
    "StateSnapshot",
    "DebateState",
    "StateManager",
    "serialize_state",
    "deserialize_state",
    "serialize_graph",
    # Tasks
    "TaskResult",
    "TaskRegistry",
    "TaskExecutor",
    "idempotent_task",
    "get_task_registry",
    # Workflow
    "WorkflowStatus",
    "WorkflowRun",
    "DurableWorkflow",
    "DurableDebateWorkflow",
]
