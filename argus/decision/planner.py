"""
VoI-based Experiment Planner.

Implements Algorithm 4.3: EIG-First Experiment Planner.
Prioritizes experiments by expected information gain with cost constraints.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from heapq import heappush, heappop

import numpy as np

from argus.decision.eig import (
    ActionCandidate,
    EIGEstimator,
    estimate_eig,
    rank_actions_by_eig,
)

if TYPE_CHECKING:
    from argus.cdag.graph import CDAG

logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    """
    Configuration for VoI planner.
    
    Attributes:
        cost_weight: Weight for cost penalty (λ in algorithm)
        budget: Total budget constraint
        risk_aversion: Risk aversion parameter (0-1)
        min_eig: Minimum EIG to consider action
        max_queue_size: Maximum experiments in queue
        lookahead_steps: Number of lookahead steps (1 or 2)
    """
    cost_weight: float = 0.1
    budget: Optional[float] = None
    risk_aversion: float = 0.0
    min_eig: float = 0.01
    max_queue_size: int = 10
    lookahead_steps: int = 1


@dataclass
class QueuedExperiment:
    """An experiment in the priority queue."""
    action: ActionCandidate
    priority: float
    added_at: int = 0
    
    def __lt__(self, other: "QueuedExperiment") -> bool:
        # Higher priority = first in queue (negate for heapq)
        return self.priority > other.priority


class ExperimentQueue:
    """
    Priority queue for experiments.
    
    Maintains a sorted queue of experiments by VoI/utility.
    Supports budget constraints and dynamic re-prioritization.
    """
    
    def __init__(
        self,
        max_size: int = 10,
        budget: Optional[float] = None,
    ):
        """
        Initialize experiment queue.
        
        Args:
            max_size: Maximum queue size
            budget: Optional budget constraint
        """
        self.max_size = max_size
        self.budget = budget
        self._heap: list[QueuedExperiment] = []
        self._counter = 0
        self._total_cost = 0.0
    
    def push(
        self,
        action: ActionCandidate,
        priority: Optional[float] = None,
    ) -> bool:
        """
        Add an experiment to the queue.
        
        Args:
            action: Action candidate
            priority: Priority (default: action.utility)
            
        Returns:
            True if added, False if full or over budget
        """
        priority = priority if priority is not None else action.utility
        
        # Check budget
        if self.budget is not None:
            if self._total_cost + action.cost > self.budget:
                logger.debug(f"Action {action.id} exceeds budget")
                return False
        
        # Check max size
        if len(self._heap) >= self.max_size:
            # Check if this beats the lowest priority
            if self._heap:
                lowest = min(self._heap, key=lambda x: x.priority)
                if priority > lowest.priority:
                    self.remove(lowest.action.id)
                else:
                    return False
        
        # Add to queue
        entry = QueuedExperiment(
            action=action,
            priority=priority,
            added_at=self._counter,
        )
        heappush(self._heap, entry)
        self._counter += 1
        self._total_cost += action.cost
        
        return True
    
    def pop(self) -> Optional[ActionCandidate]:
        """
        Get and remove highest priority experiment.
        
        Returns:
            ActionCandidate or None if empty
        """
        if not self._heap:
            return None
        
        entry = heappop(self._heap)
        self._total_cost -= entry.action.cost
        return entry.action
    
    def peek(self) -> Optional[ActionCandidate]:
        """
        Get highest priority without removing.
        
        Returns:
            ActionCandidate or None if empty
        """
        if not self._heap:
            return None
        return self._heap[0].action
    
    def remove(self, action_id: str) -> bool:
        """
        Remove an action from the queue.
        
        Args:
            action_id: ID of action to remove
            
        Returns:
            True if removed
        """
        for i, entry in enumerate(self._heap):
            if entry.action.id == action_id:
                self._total_cost -= entry.action.cost
                self._heap.pop(i)
                # Re-heapify
                import heapq
                heapq.heapify(self._heap)
                return True
        return False
    
    def get_all(self) -> list[ActionCandidate]:
        """Get all actions in priority order."""
        sorted_entries = sorted(self._heap, key=lambda x: x.priority, reverse=True)
        return [e.action for e in sorted_entries]
    
    def __len__(self) -> int:
        return len(self._heap)
    
    @property
    def remaining_budget(self) -> Optional[float]:
        """Remaining budget."""
        if self.budget is None:
            return None
        return self.budget - self._total_cost


class VoIPlanner:
    """
    Value of Information based experiment planner.
    
    Implements Algorithm 4.3 for prioritizing experiments
    by expected information gain with cost constraints.
    
    Example:
        >>> planner = VoIPlanner(config=PlannerConfig(budget=100))
        >>> queue = planner.plan(graph, candidates)
        >>> for action in queue.get_all():
        ...     print(f"{action.name}: EIG={action.eig:.3f}, utility={action.utility:.3f}")
    """
    
    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        eig_estimator: Optional[EIGEstimator] = None,
    ):
        """
        Initialize planner.
        
        Args:
            config: Planner configuration
            eig_estimator: EIG estimator (creates default if None)
        """
        self.config = config or PlannerConfig()
        self.eig_estimator = eig_estimator or EIGEstimator()
    
    def plan(
        self,
        graph: "CDAG",
        candidates: list[ActionCandidate],
    ) -> ExperimentQueue:
        """
        Create prioritized experiment queue.
        
        Implements Algorithm 4.3:
        1. For each candidate, estimate EIG
        2. Compute utility: u_k = EIG(a_k) - λ × Cost(a_k)
        3. Sort by utility
        4. Apply budget constraint (knapsack)
        
        Args:
            graph: The C-DAG graph
            candidates: Action candidates to evaluate
            
        Returns:
            Prioritized ExperimentQueue
        """
        # Get priors for all propositions
        priors: dict[str, float] = {}
        for prop in graph.get_all_propositions():
            priors[prop.id] = prop.prior
        
        # Rank actions by EIG
        ranked = rank_actions_by_eig(
            actions=candidates,
            priors=priors,
            cost_weight=self.config.cost_weight,
        )
        
        # Filter by minimum EIG
        filtered = [a for a in ranked if a.eig >= self.config.min_eig]
        
        # Apply risk aversion if configured
        if self.config.risk_aversion > 0:
            filtered = self._apply_risk_aversion(filtered, priors)
        
        # Create queue with budget constraint
        queue = ExperimentQueue(
            max_size=self.config.max_queue_size,
            budget=self.config.budget,
        )
        
        # Apply knapsack if budget constrained
        if self.config.budget is not None:
            selected = self._knapsack_selection(
                filtered,
                self.config.budget,
            )
            for action in selected:
                queue.push(action)
        else:
            # Just add top-k by utility
            for action in filtered[:self.config.max_queue_size]:
                queue.push(action)
        
        logger.info(
            f"Planned {len(queue)} experiments from {len(candidates)} candidates"
        )
        
        return queue
    
    def _apply_risk_aversion(
        self,
        actions: list[ActionCandidate],
        priors: dict[str, float],
    ) -> list[ActionCandidate]:
        """
        Apply risk aversion to action utilities.
        
        Risk-aware utility (Eq. 4.10):
            EU_risk = EU - ρ × σ(U)
        
        Args:
            actions: Action candidates
            priors: Proposition priors
            
        Returns:
            Actions with adjusted utilities
        """
        for action in actions:
            # Estimate utility variance (simplified)
            # Higher EIG usually means higher variance
            variance = action.eig * 0.5  # Heuristic scaling
            std = np.sqrt(variance)
            
            # Adjust utility with risk penalty
            risk_penalty = self.config.risk_aversion * std
            action.utility = action.utility - risk_penalty
        
        # Re-sort by adjusted utility
        return sorted(actions, key=lambda a: a.utility, reverse=True)
    
    def _knapsack_selection(
        self,
        actions: list[ActionCandidate],
        budget: float,
    ) -> list[ActionCandidate]:
        """
        Select actions using 0/1 knapsack algorithm.
        
        Maximizes total utility subject to budget constraint.
        
        Args:
            actions: Candidate actions
            budget: Total budget
            
        Returns:
            Selected actions
        """
        n = len(actions)
        
        if n == 0:
            return []
        
        # Scale values and weights to integers for DP
        scale = 100
        scaled_budget = int(budget * scale)
        
        # Value and weight arrays
        values = [a.utility * 1000 for a in actions]  # Scale up
        weights = [int(a.cost * scale) for a in actions]
        
        # DP table
        dp = [[0] * (scaled_budget + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(scaled_budget + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        dp[i - 1][w - weights[i - 1]] + values[i - 1],
                    )
                else:
                    dp[i][w] = dp[i - 1][w]
        
        # Backtrack to find selected items
        selected = []
        w = scaled_budget
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(actions[i - 1])
                w -= weights[i - 1]
        
        # Sort by utility for final ordering
        selected.sort(key=lambda a: a.utility, reverse=True)
        
        return selected
    
    def what_if_analysis(
        self,
        graph: "CDAG",
        action: ActionCandidate,
        outcomes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Analyze potential outcomes of an action.
        
        Args:
            graph: The C-DAG graph
            action: Action to analyze
            outcomes: Possible outcomes with probabilities
            
        Returns:
            Analysis results for each outcome
        """
        results = []
        
        for outcome in outcomes:
            # Simulate outcome
            prob = outcome.get("probability", 0.5)
            effect = outcome.get("effect", {})
            
            result = {
                "outcome": outcome,
                "probability": prob,
                "posteriors": {},
                "eig_after": {},
            }
            
            # For each affected proposition
            for prop_id in action.target_props:
                prop = graph.get_proposition(prop_id)
                if not prop:
                    continue
                
                # Current posterior
                current = prop.posterior
                
                # Simulated posterior after outcome
                delta = effect.get(prop_id, 0.0)
                simulated = max(0.0, min(1.0, current + delta))
                
                result["posteriors"][prop_id] = {
                    "current": current,
                    "simulated": simulated,
                    "change": simulated - current,
                }
                
                # EIG after this outcome
                result["eig_after"][prop_id] = estimate_eig(simulated)
            
            results.append(result)
        
        return results
    
    def stopping_criteria_met(
        self,
        graph: "CDAG",
        min_posterior: float = 0.95,
        max_eig: float = 0.01,
    ) -> tuple[bool, str]:
        """
        Check if stopping criteria are met.
        
        Stopping when:
        - All propositions have high posterior (>95%)
        - Or maximum remaining EIG is below threshold
        
        Args:
            graph: The C-DAG graph
            min_posterior: Minimum posterior for stopping
            max_eig: Maximum EIG threshold for stopping
            
        Returns:
            (should_stop, reason)
        """
        propositions = graph.get_all_propositions()
        
        if not propositions:
            return True, "No propositions"
        
        # Check posteriors
        all_decided = True
        for prop in propositions:
            if 0.05 < prop.posterior < 0.95:
                all_decided = False
                break
        
        if all_decided:
            return True, "All propositions decided (>95% or <5%)"
        
        # Check remaining EIG
        max_remaining_eig = 0.0
        for prop in propositions:
            eig = estimate_eig(prop.posterior)
            max_remaining_eig = max(max_remaining_eig, eig)
        
        if max_remaining_eig < max_eig:
            return True, f"Maximum EIG below threshold ({max_remaining_eig:.4f} < {max_eig})"
        
        return False, "Continuing"
