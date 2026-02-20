"""
Freeplay Tool for ARGUS.

Build, optimize, and evaluate AI agents with end-to-end observability.
"""

from __future__ import annotations

import os
import uuid
import logging
import time
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with version tracking."""
    template_id: str
    name: str
    template: str
    version: int = 1
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def render(self, **kwargs) -> str:
        """Render template with variables."""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "template": self.template,
            "version": self.version,
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Experiment:
    """An A/B test experiment for prompts."""
    experiment_id: str
    name: str
    variants: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_result(self, variant_id: str, score: float) -> None:
        self.results[variant_id].append(score)
    
    def get_winner(self) -> Optional[str]:
        if not self.results:
            return None
        return max(
            self.results.keys(),
            key=lambda k: sum(self.results[k]) / len(self.results[k]) if self.results[k] else 0
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "variants": self.variants,
            "results": {k: {"scores": v, "avg": sum(v)/len(v) if v else 0} for k, v in self.results.items()},
            "status": self.status,
            "winner": self.get_winner(),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EvaluationRun:
    """An evaluation run for measuring agent performance."""
    run_id: str
    name: str
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "test_case_count": len(self.test_cases),
            "completed_count": len(self.results),
            "metrics": self.metrics,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class FreeplayTool(BaseTool):
    """
    Freeplay - Build, optimize, and evaluate AI agents.
    
    Features:
    - Prompt management and versioning
    - A/B testing and experimentation
    - Evaluation and testing framework
    - Performance analytics
    - End-to-end observability
    
    Example:
        >>> tool = FreeplayTool()
        >>> result = tool(action="create_template", name="greeting",
        ...               template="Hello {name}!")
        >>> result = tool(action="create_experiment", name="test1",
        ...               variants=[...])
        >>> result = tool(action="run_evaluation", test_cases=[...])
    """
    
    name = "freeplay"
    description = "Build, optimize, and evaluate AI agents with end-to-end observability"
    category = ToolCategory.EXTERNAL_API
    version = "1.0.0"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("FREEPLAY_API_KEY")
        self.project_id = project_id or os.getenv("FREEPLAY_PROJECT_ID")
        
        # Local storage
        self._templates: dict[str, PromptTemplate] = {}
        self._experiments: dict[str, Experiment] = {}
        self._evaluations: dict[str, EvaluationRun] = {}
        self._traces: list[dict] = []
        
        logger.debug("Freeplay initialized")
    
    def execute(
        self,
        action: str = "list_templates",
        name: Optional[str] = None,
        template: Optional[str] = None,
        template_id: Optional[str] = None,
        variables: Optional[dict] = None,
        variants: Optional[list] = None,
        experiment_id: Optional[str] = None,
        test_cases: Optional[list] = None,
        run_id: Optional[str] = None,
        score: Optional[float] = None,
        variant_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Freeplay operations."""
        actions = {
            # Template management
            "create_template": self._create_template,
            "update_template": self._update_template,
            "render_template": self._render_template,
            "list_templates": self._list_templates,
            "get_template": self._get_template,
            # Experiments
            "create_experiment": self._create_experiment,
            "record_result": self._record_result,
            "get_experiment": self._get_experiment,
            "list_experiments": self._list_experiments,
            # Evaluation
            "create_evaluation": self._create_evaluation,
            "run_evaluation": self._run_evaluation,
            "get_evaluation": self._get_evaluation,
            "list_evaluations": self._list_evaluations,
            # Tracing
            "log_trace": self._log_trace,
            "get_traces": self._get_traces,
            # Analytics
            "get_analytics": self._get_analytics,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                name=name,
                template=template,
                template_id=template_id,
                variables=variables or {},
                variants=variants or [],
                experiment_id=experiment_id,
                test_cases=test_cases or [],
                run_id=run_id,
                score=score,
                variant_id=variant_id,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Freeplay error: {e}")
            return ToolResult.from_error(f"Freeplay error: {e}")
    
    def _create_template(
        self,
        name: Optional[str] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new prompt template."""
        if not name:
            return ToolResult.from_error("name is required")
        if not template:
            return ToolResult.from_error("template is required")
        
        # Extract variables from template (e.g., {name})
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        
        template_id = str(uuid.uuid4())[:8]
        
        tmpl = PromptTemplate(
            template_id=template_id,
            name=name,
            template=template,
            variables=variables,
        )
        
        self._templates[template_id] = tmpl
        
        return ToolResult.from_data({
            "template_id": template_id,
            "name": name,
            "variables": variables,
        })
    
    def _update_template(
        self,
        template_id: Optional[str] = None,
        template: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update an existing template (creates new version)."""
        if not template_id or template_id not in self._templates:
            return ToolResult.from_error("Invalid template_id")
        if not template:
            return ToolResult.from_error("template is required")
        
        tmpl = self._templates[template_id]
        
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        
        tmpl.template = template
        tmpl.variables = variables
        tmpl.version += 1
        
        return ToolResult.from_data({
            "template_id": template_id,
            "name": tmpl.name,
            "version": tmpl.version,
            "variables": variables,
        })
    
    def _render_template(
        self,
        template_id: Optional[str] = None,
        variables: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Render a template with variables."""
        if not template_id or template_id not in self._templates:
            return ToolResult.from_error("Invalid template_id")
        
        tmpl = self._templates[template_id]
        rendered = tmpl.render(**(variables or {}))
        
        return ToolResult.from_data({
            "template_id": template_id,
            "rendered": rendered,
            "missing_vars": [v for v in tmpl.variables if v not in (variables or {})],
        })
    
    def _list_templates(self, **kwargs) -> ToolResult:
        """List all templates."""
        templates = [t.to_dict() for t in self._templates.values()]
        return ToolResult.from_data({"templates": templates, "count": len(templates)})
    
    def _get_template(
        self,
        template_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get a specific template."""
        if not template_id or template_id not in self._templates:
            return ToolResult.from_error("Invalid template_id")
        
        return ToolResult.from_data({"template": self._templates[template_id].to_dict()})
    
    def _create_experiment(
        self,
        name: Optional[str] = None,
        variants: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new A/B experiment."""
        if not name:
            return ToolResult.from_error("name is required")
        if not variants:
            return ToolResult.from_error("variants is required (list of variant configs)")
        
        experiment_id = str(uuid.uuid4())[:8]
        
        # Add IDs to variants if not present
        for i, v in enumerate(variants):
            if "id" not in v:
                v["id"] = f"variant_{i}"
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            variants=variants,
            status="active",
        )
        
        self._experiments[experiment_id] = experiment
        
        return ToolResult.from_data({
            "experiment_id": experiment_id,
            "name": name,
            "variant_count": len(variants),
        })
    
    def _record_result(
        self,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        score: Optional[float] = None,
        **kwargs,
    ) -> ToolResult:
        """Record a result for an experiment variant."""
        if not experiment_id or experiment_id not in self._experiments:
            return ToolResult.from_error("Invalid experiment_id")
        if not variant_id:
            return ToolResult.from_error("variant_id is required")
        if score is None:
            return ToolResult.from_error("score is required")
        
        experiment = self._experiments[experiment_id]
        experiment.add_result(variant_id, score)
        
        return ToolResult.from_data({
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "score": score,
            "total_results": sum(len(v) for v in experiment.results.values()),
        })
    
    def _get_experiment(
        self,
        experiment_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get experiment details and results."""
        if not experiment_id or experiment_id not in self._experiments:
            return ToolResult.from_error("Invalid experiment_id")
        
        return ToolResult.from_data({"experiment": self._experiments[experiment_id].to_dict()})
    
    def _list_experiments(self, **kwargs) -> ToolResult:
        """List all experiments."""
        experiments = [e.to_dict() for e in self._experiments.values()]
        return ToolResult.from_data({"experiments": experiments, "count": len(experiments)})
    
    def _create_evaluation(
        self,
        name: Optional[str] = None,
        test_cases: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new evaluation run."""
        if not name:
            return ToolResult.from_error("name is required")
        
        run_id = str(uuid.uuid4())[:8]
        
        evaluation = EvaluationRun(
            run_id=run_id,
            name=name,
            test_cases=test_cases or [],
        )
        
        self._evaluations[run_id] = evaluation
        
        return ToolResult.from_data({
            "run_id": run_id,
            "name": name,
            "test_case_count": len(test_cases or []),
        })
    
    def _run_evaluation(
        self,
        run_id: Optional[str] = None,
        evaluator_fn: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Run an evaluation (simulated)."""
        if not run_id or run_id not in self._evaluations:
            return ToolResult.from_error("Invalid run_id")
        
        evaluation = self._evaluations[run_id]
        evaluation.status = "running"
        evaluation.started_at = datetime.utcnow()
        
        # Simulate evaluation
        import random
        for tc in evaluation.test_cases:
            result = {
                "test_case": tc,
                "passed": random.random() > 0.3,
                "score": random.uniform(0.5, 1.0),
                "latency_ms": random.uniform(100, 500),
            }
            evaluation.results.append(result)
        
        # Calculate metrics
        if evaluation.results:
            evaluation.metrics = {
                "pass_rate": sum(1 for r in evaluation.results if r["passed"]) / len(evaluation.results),
                "avg_score": sum(r["score"] for r in evaluation.results) / len(evaluation.results),
                "avg_latency_ms": sum(r["latency_ms"] for r in evaluation.results) / len(evaluation.results),
            }
        
        evaluation.status = "completed"
        evaluation.completed_at = datetime.utcnow()
        
        return ToolResult.from_data({
            "run_id": run_id,
            "status": evaluation.status,
            "metrics": evaluation.metrics,
            "result_count": len(evaluation.results),
        })
    
    def _get_evaluation(
        self,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get evaluation details."""
        if not run_id or run_id not in self._evaluations:
            return ToolResult.from_error("Invalid run_id")
        
        return ToolResult.from_data({"evaluation": self._evaluations[run_id].to_dict()})
    
    def _list_evaluations(self, **kwargs) -> ToolResult:
        """List all evaluations."""
        evaluations = [e.to_dict() for e in self._evaluations.values()]
        return ToolResult.from_data({"evaluations": evaluations, "count": len(evaluations)})
    
    def _log_trace(
        self,
        trace_data: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Log a trace event."""
        if not trace_data:
            trace_data = kwargs
        
        trace = {
            "trace_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.utcnow().isoformat(),
            **trace_data,
        }
        
        self._traces.append(trace)
        
        return ToolResult.from_data({"trace_id": trace["trace_id"]})
    
    def _get_traces(
        self,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Get recent traces."""
        traces = self._traces[-limit:]
        return ToolResult.from_data({"traces": traces, "count": len(traces)})
    
    def _get_analytics(self, **kwargs) -> ToolResult:
        """Get analytics summary."""
        return ToolResult.from_data({
            "total_templates": len(self._templates),
            "total_experiments": len(self._experiments),
            "active_experiments": len([e for e in self._experiments.values() if e.status == "active"]),
            "total_evaluations": len(self._evaluations),
            "completed_evaluations": len([e for e in self._evaluations.values() if e.status == "completed"]),
            "total_traces": len(self._traces),
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "create_template", "update_template", "render_template",
                            "list_templates", "get_template",
                            "create_experiment", "record_result", "get_experiment",
                            "list_experiments",
                            "create_evaluation", "run_evaluation", "get_evaluation",
                            "list_evaluations",
                            "log_trace", "get_traces", "get_analytics",
                        ],
                    },
                    "name": {"type": "string"},
                    "template": {"type": "string"},
                    "template_id": {"type": "string"},
                    "variables": {"type": "object"},
                    "variants": {"type": "array"},
                    "experiment_id": {"type": "string"},
                    "test_cases": {"type": "array"},
                    "run_id": {"type": "string"},
                    "score": {"type": "number"},
                    "variant_id": {"type": "string"},
                },
                "required": ["action"],
            },
        }
