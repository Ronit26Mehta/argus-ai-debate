"""
Unit tests for ARGUS Durable Execution module.
"""

import pytest
import tempfile
import os

from argus.durable.config import DurableConfig, CheckpointerType, RetryPolicy
from argus.durable.checkpointer import (
    MemoryCheckpointer, SQLiteCheckpointer, FileSystemCheckpointer, Checkpoint
)
from argus.durable.state import DebateState, StateManager, serialize_state, deserialize_state
from argus.durable.tasks import TaskRegistry, idempotent_task, TaskExecutor, TaskResult
from argus.durable.workflow import DurableWorkflow, WorkflowStatus


class TestDurableConfig:
    """Tests for durable execution configuration."""
    
    def test_default_config(self):
        config = DurableConfig()
        assert config.enabled is True
        assert config.checkpointer_type == CheckpointerType.MEMORY
        assert config.auto_checkpoint is True
    
    def test_retry_policy(self):
        policy = RetryPolicy(max_retries=5, initial_delay=2.0)
        assert policy.max_retries == 5
        assert policy.initial_delay == 2.0


class TestCheckpointers:
    """Tests for checkpointer implementations."""
    
    def test_memory_checkpointer_save_load(self):
        cp = MemoryCheckpointer()
        cp_id = cp.save("thread-1", {"step": 1, "data": "value"}, step=1)
        checkpoint = cp.load("thread-1")
        assert checkpoint is not None
        assert checkpoint.state["step"] == 1
    
    def test_memory_checkpointer_latest(self):
        cp = MemoryCheckpointer()
        cp.save("thread-1", {"step": 1}, step=1)
        cp.save("thread-1", {"step": 2}, step=2)
        latest = cp.load("thread-1")
        assert latest.state["step"] == 2
    
    def test_memory_checkpointer_list(self):
        cp = MemoryCheckpointer()
        cp.save("thread-1", {"step": 1}, step=1)
        cp.save("thread-1", {"step": 2}, step=2)
        checkpoints = cp.list_checkpoints("thread-1")
        assert len(checkpoints) == 2
    
    def test_memory_checkpointer_delete(self):
        cp = MemoryCheckpointer()
        cp_id = cp.save("thread-1", {"step": 1})
        assert cp.delete(cp_id) is True
        assert cp.load("thread-1", cp_id) is None
    
    def test_sqlite_checkpointer(self):
        cp = SQLiteCheckpointer(":memory:")
        cp_id = cp.save("thread-1", {"data": "test"}, step=1)
        checkpoint = cp.load("thread-1")
        assert checkpoint is not None
        assert checkpoint.state["data"] == "test"
    
    def test_filesystem_checkpointer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = FileSystemCheckpointer(tmpdir)
            cp_id = cp.save("thread-1", {"data": "test"}, step=1)
            checkpoint = cp.load("thread-1")
            assert checkpoint is not None
            assert checkpoint.state["data"] == "test"


class TestDebateState:
    """Tests for debate state."""
    
    def test_debate_state_creation(self):
        state = DebateState(debate_id="d-1", proposition="Test prop")
        assert state.debate_id == "d-1"
        assert state.proposition == "Test prop"
        assert state.current_round == 0
    
    def test_serialize_deserialize(self):
        state = DebateState(debate_id="d-1", proposition="Test", current_round=3)
        json_str = serialize_state(state)
        restored = deserialize_state(json_str)
        assert restored.debate_id == "d-1"
        assert restored.current_round == 3
    
    def test_state_manager(self):
        manager = StateManager()
        state = manager.initialize("d-1", "Test proposition")
        assert state.debate_id == "d-1"
        manager.update(current_round=2)
        assert manager.get_state().current_round == 2
    
    def test_state_snapshots(self):
        manager = StateManager()
        manager.initialize("d-1", "Test")
        snap1 = manager.snapshot("Initial")
        manager.update(current_round=1)
        snap2 = manager.snapshot("After round 1")
        assert snap1.version == 1
        assert snap2.version == 2


class TestIdempotentTasks:
    """Tests for idempotent task execution."""
    
    def test_task_registry(self):
        registry = TaskRegistry()
        result = TaskResult(task_id="t-1", success=True, result=42)
        registry.record("t-1", result)
        assert registry.is_executed("t-1") is True
        assert registry.get_result("t-1").result == 42
    
    def test_idempotent_decorator(self):
        registry = TaskRegistry()
        call_count = 0
        
        @idempotent_task(registry=registry)
        def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_operation(5)
        result2 = expensive_operation(5)  # Should return cached
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once
    
    def test_task_executor(self):
        registry = TaskRegistry()
        executor = TaskExecutor(registry=registry)
        
        def simple_task(x: int) -> int:
            return x + 1
        
        result = executor.execute(simple_task, 10)
        assert result == 11


class TestDurableWorkflow:
    """Tests for durable workflow."""
    
    def test_workflow_initialization(self):
        workflow = DurableWorkflow(thread_id="test-thread")
        assert workflow.thread_id == "test-thread"
    
    def test_workflow_step(self):
        workflow = DurableWorkflow()
        result = workflow.step("step1", lambda: "done")
        assert result == "done"
    
    def test_workflow_checkpoint(self):
        workflow = DurableWorkflow()
        workflow.start_run()
        workflow.step("step1", lambda: 1)
        cp_id = workflow.checkpoint("After step 1")
        assert cp_id is not None
        checkpoints = workflow.list_checkpoints()
        assert len(checkpoints) >= 1
    
    def test_workflow_resume(self):
        cp = MemoryCheckpointer()
        workflow1 = DurableWorkflow(thread_id="t-1", checkpointer=cp)
        workflow1.start_run()
        workflow1.step("step1", lambda: "result1")
        workflow1.checkpoint()
        
        # Create new workflow with same thread
        workflow2 = DurableWorkflow(thread_id="t-1", checkpointer=cp)
        resumed = workflow2.resume()
        assert resumed is True
    
    def test_workflow_run(self):
        workflow = DurableWorkflow()
        run = workflow.start_run(total_steps=3)
        assert run.status == WorkflowStatus.RUNNING
        workflow.complete_run()
        assert workflow.get_current_run().status == WorkflowStatus.COMPLETED
