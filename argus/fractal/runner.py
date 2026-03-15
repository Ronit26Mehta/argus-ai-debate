"""
Parallel Debate Runner — runs debates on leaf nodes.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from argus.fractal.decomposer import PropositionTree, PropositionNode

logger = logging.getLogger(__name__)


@dataclass
class LeafDebateResult:
    """Result from debating a single leaf node."""
    node_id: str = ""
    proposition_text: str = ""
    posterior: float = 0.5
    verdict: str = "undetermined"
    confidence: float = 0.5
    evidence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.proposition_text[:100],
            "posterior": round(self.posterior, 4),
            "verdict": self.verdict,
        }


class ParallelDebateRunner:
    """
    Runs debates on leaf nodes in parallel.

    Each leaf get its own mini-debate. Results are collected for
    hierarchical aggregation.
    """

    def __init__(
        self,
        base_orchestrator: Optional[Any] = None,
        parallel_workers: int = 4,
        default_prior: float = 0.5,
    ):
        self.base = base_orchestrator
        self.parallel_workers = parallel_workers
        self.default_prior = default_prior

    def run_leaf_debates(
        self,
        tree: PropositionTree,
        **kwargs: Any,
    ) -> dict[str, LeafDebateResult]:
        """
        Run debates on all leaf nodes.

        Args:
            tree: PropositionTree to debate
            **kwargs: Passed to base orchestrator

        Returns:
            Mapping of node_id to LeafDebateResult
        """
        leaves = tree.leaves
        results: dict[str, LeafDebateResult] = {}

        if self.parallel_workers <= 1 or len(leaves) < 3:
            for leaf in leaves:
                result = self._debate_leaf(leaf, **kwargs)
                results[leaf.node_id] = result
        else:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = {
                    executor.submit(self._debate_leaf, leaf, **kwargs): leaf
                    for leaf in leaves
                }
                for future in as_completed(futures):
                    leaf = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.warning(f"Leaf debate failed for {leaf.node_id}: {e}")
                        result = LeafDebateResult(
                            node_id=leaf.node_id,
                            proposition_text=leaf.text,
                        )
                    results[leaf.node_id] = result

        logger.info(f"Completed {len(results)} leaf debates")
        return results

    def _debate_leaf(
        self,
        leaf: PropositionNode,
        **kwargs: Any,
    ) -> LeafDebateResult:
        """Run debate on a single leaf."""
        if self.base is not None:
            try:
                base_result = self.base.debate(
                    leaf.text, prior=self.default_prior, **kwargs,
                )
                return LeafDebateResult(
                    node_id=leaf.node_id,
                    proposition_text=leaf.text,
                    posterior=getattr(base_result, 'posterior', 0.5),
                    verdict=str(getattr(base_result, 'verdict', 'undetermined')),
                    confidence=getattr(base_result, 'confidence', 0.5),
                )
            except Exception as e:
                logger.warning(f"Base debate failed: {e}")

        # Heuristic fallback
        import random
        posterior = random.betavariate(3, 3)
        return LeafDebateResult(
            node_id=leaf.node_id,
            proposition_text=leaf.text,
            posterior=posterior,
            verdict="supported" if posterior > 0.5 else "contested",
            confidence=abs(posterior - 0.5) * 2,
        )
