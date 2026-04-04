"""
Tournament Tree Engine — HANNIBAL's bracket data structure.

Pre-constructs the full empty Tournament Tree from a BattleMap, then fills
results bottom-up as battles complete:

    Skirmishes → Engagements → Theatres → Campaign Root

The tree is immutable in structure once built; only result fields are
updated as battles are adjudicated.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from argus.hannibal.models import (
    BattleMap,
    CampaignVerdict,
    CampaignVerdictLabel,
    EngagementResult,
    ForceType,
    HannibalSessionConfig,
    SkirmishResult,
    TheatreResult,
    TheatreSpec,
    TournamentNode,
    TournamentNodeType,
    VictoryStrength,
    _uid,
)

logger = logging.getLogger(__name__)


class TournamentTree:
    """Hierarchical bracket structure for a HANNIBAL campaign.

    The tree is pre-constructed from a BattleMap, with node slots
    created for every skirmish, engagement, theatre, and the campaign root.
    Result fields are filled as battles complete, propagating upward.
    """

    def __init__(self, config: HannibalSessionConfig | None = None):
        self._nodes: dict[str, TournamentNode] = {}
        self._root_id: str = ""
        self._skirmish_ids: list[str] = []
        self._engagement_ids: list[str] = []
        self._theatre_ids: list[str] = []
        self._config = config or HannibalSessionConfig()

    @property
    def root(self) -> Optional[TournamentNode]:
        return self._nodes.get(self._root_id)

    @property
    def skirmish_nodes(self) -> list[TournamentNode]:
        return [self._nodes[nid] for nid in self._skirmish_ids
                if nid in self._nodes]

    @property
    def engagement_nodes(self) -> list[TournamentNode]:
        return [self._nodes[nid] for nid in self._engagement_ids
                if nid in self._nodes]

    @property
    def theatre_nodes(self) -> list[TournamentNode]:
        return [self._nodes[nid] for nid in self._theatre_ids
                if nid in self._nodes]

    @property
    def total_skirmishes(self) -> int:
        return len(self._skirmish_ids)

    @property
    def resolved_skirmishes(self) -> int:
        return sum(1 for nid in self._skirmish_ids
                   if self._nodes.get(nid, TournamentNode()).is_resolved)

    @property
    def progress_fraction(self) -> float:
        total = self.total_skirmishes
        return self.resolved_skirmishes / total if total > 0 else 0.0

    # ── Tree Construction ──────────────────────────────────────────

    def build_tree(self, battle_map: BattleMap) -> TournamentNode:
        """Pre-construct the full Tournament Tree from a BattleMap.

        Structure (height = 4):
            Campaign Root
              └── Theatre 1
                   └── Engagement 1.1
                        └── Skirmish 1.1.1
                        └── Skirmish 1.1.2
                   └── Engagement 1.2
                        └── Skirmish 1.2.1
              └── Theatre 2
                   └── Engagement 2.1 ...

        For height = 3 (no explicit theatre layer):
            Campaign Root
              └── Engagement 1
                   └── Skirmish 1.1
              └── Engagement 2 ...

        For height = 2 (quick battle):
            Campaign Root
              └── Skirmish 1
              └── Skirmish 2

        Returns:
            The Campaign Root node.
        """
        self._nodes.clear()
        self._skirmish_ids.clear()
        self._engagement_ids.clear()
        self._theatre_ids.clear()

        forces = battle_map.force_designations
        num_forces = len(forces)

        # Number of pairwise matchups per engagement
        matchups_per_engagement = min(
            self._config.max_skirmishes_per_engagement,
            max(1, num_forces - 1),
        )

        # Create Campaign Root
        root = TournamentNode(
            id=_uid("cr"),
            node_type=TournamentNodeType.CAMPAIGN_ROOT,
            label="Campaign Root",
        )
        self._nodes[root.id] = root
        self._root_id = root.id

        height = battle_map.tree_height

        if height >= 4:
            # Full structure: Root → Theatres → Engagements → Skirmishes
            for t_idx, theatre_spec in enumerate(battle_map.theatres):
                theatre_node = self._create_theatre_node(
                    theatre_spec, root.id, t_idx,
                )
                root.child_ids.append(theatre_node.id)

                for e_idx in range(theatre_spec.engagement_count):
                    eng_node = self._create_engagement_node(
                        theatre_node.id, theatre_spec.id, t_idx, e_idx,
                        theatre_spec.topical_scope,
                    )
                    theatre_node.child_ids.append(eng_node.id)

                    for s_idx in range(matchups_per_engagement):
                        pair = self._get_force_pair(forces, s_idx)
                        skirmish = self._create_skirmish_node(
                            eng_node.id, pair, t_idx, e_idx, s_idx,
                            theatre_spec.topical_scope,
                        )
                        eng_node.child_ids.append(skirmish.id)

        elif height == 3:
            # Root → Engagements → Skirmishes (theatres implicit)
            for t_idx, theatre_spec in enumerate(battle_map.theatres):
                for e_idx in range(theatre_spec.engagement_count):
                    eng_node = self._create_engagement_node(
                        root.id, theatre_spec.id, t_idx, e_idx,
                        theatre_spec.topical_scope,
                    )
                    root.child_ids.append(eng_node.id)

                    for s_idx in range(matchups_per_engagement):
                        pair = self._get_force_pair(forces, s_idx)
                        skirmish = self._create_skirmish_node(
                            eng_node.id, pair, t_idx, e_idx, s_idx,
                            theatre_spec.topical_scope,
                        )
                        eng_node.child_ids.append(skirmish.id)

        else:
            # Quick Battle: Root → Skirmishes directly
            for t_idx, theatre_spec in enumerate(battle_map.theatres):
                for s_idx in range(matchups_per_engagement):
                    pair = self._get_force_pair(forces, s_idx)
                    skirmish = self._create_skirmish_node(
                        root.id, pair, t_idx, 0, s_idx,
                        theatre_spec.topical_scope,
                    )
                    root.child_ids.append(skirmish.id)

        logger.info(
            "Tournament tree built: %d skirmishes, %d engagements, %d theatres, "
            "height=%d",
            len(self._skirmish_ids), len(self._engagement_ids),
            len(self._theatre_ids), height,
        )
        return root

    # ── Node Creation ──────────────────────────────────────────────

    def _create_theatre_node(
        self,
        spec: TheatreSpec,
        parent_id: str,
        t_idx: int,
    ) -> TournamentNode:
        node = TournamentNode(
            id=_uid("tt"),
            node_type=TournamentNodeType.THEATRE,
            parent_id=parent_id,
            label=f"Theatre {t_idx+1}: {spec.name}",
            topical_scope=spec.topical_scope,
            theatre_id=spec.id,
            engagement_weight=spec.epistemic_importance,
        )
        self._nodes[node.id] = node
        self._theatre_ids.append(node.id)
        return node

    def _create_engagement_node(
        self,
        parent_id: str,
        theatre_id: str,
        t_idx: int,
        e_idx: int,
        topic: str,
    ) -> TournamentNode:
        node = TournamentNode(
            id=_uid("en"),
            node_type=TournamentNodeType.ENGAGEMENT,
            parent_id=parent_id,
            label=f"Engagement {t_idx+1}.{e_idx+1}",
            theatre_id=theatre_id,
            topic_cluster=topic[:80],
            topic_scope=topic,
        )
        self._nodes[node.id] = node
        self._engagement_ids.append(node.id)
        return node

    def _create_skirmish_node(
        self,
        parent_id: str,
        force_pair: tuple[ForceType, ForceType],
        t_idx: int,
        e_idx: int,
        s_idx: int,
        topic: str,
    ) -> TournamentNode:
        node = TournamentNode(
            id=_uid("sk"),
            node_type=TournamentNodeType.SKIRMISH,
            parent_id=parent_id,
            label=f"Skirmish {t_idx+1}.{e_idx+1}.{s_idx+1}",
            force_a_type=force_pair[0],
            force_b_type=force_pair[1],
            topic_scope=topic,
        )
        self._nodes[node.id] = node
        self._skirmish_ids.append(node.id)
        return node

    # ── Result Propagation ─────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[TournamentNode]:
        return self._nodes.get(node_id)

    def update_skirmish_result(
        self, node_id: str, result: SkirmishResult,
    ) -> None:
        """Fill a skirmish node with its result."""
        node = self._nodes.get(node_id)
        if not node:
            logger.warning("Skirmish node %s not found", node_id)
            return
        node.winner_force = result.winner_force
        node.confidence = result.confidence_score
        node.margin = abs(result.ecs_winner - result.ecs_loser)
        node.is_resolved = True

    def resolve_engagement(self, engagement_id: str) -> EngagementResult:
        """Resolve an engagement by aggregating its child skirmishes.

        Winner = force with the greater Engagement Margin (EM):
            EM(Force) = Σ_s (Confidence_s × [1 if Force won Skirmish s else 0])
        """
        eng_node = self._nodes.get(engagement_id)
        if not eng_node:
            return EngagementResult(engagement_id=engagement_id)

        # Tally by force
        force_scores: dict[str, float] = {}
        child_results: list[SkirmishResult] = []

        for child_id in eng_node.child_ids:
            child = self._nodes.get(child_id)
            if not child or not child.is_resolved or not child.winner_force:
                continue
            fv = child.winner_force.value
            force_scores[fv] = force_scores.get(fv, 0.0) + child.confidence

        if not force_scores:
            return EngagementResult(engagement_id=engagement_id)

        winner_val = max(force_scores, key=force_scores.get)        # type: ignore[arg-type]
        total = sum(force_scores.values())
        margin = force_scores[winner_val] / total if total > 0 else 0.0

        try:
            winner_force = ForceType(winner_val)
        except ValueError:
            winner_force = ForceType.PROPOSITION

        result = EngagementResult(
            engagement_id=engagement_id,
            winner_force=winner_force,
            margin=margin,
        )

        eng_node.winner_force = winner_force
        eng_node.confidence = margin
        eng_node.margin = margin
        eng_node.is_resolved = True

        return result

    def resolve_theatre(self, theatre_id: str) -> TheatreResult:
        """Resolve a theatre by aggregating its child engagements."""
        theatre_node = self._nodes.get(theatre_id)
        if not theatre_node:
            return TheatreResult(theatre_id=theatre_id)

        force_scores: dict[str, float] = {}
        eng_results: list[EngagementResult] = []

        for child_id in theatre_node.child_ids:
            child = self._nodes.get(child_id)
            if not child or not child.is_resolved or not child.winner_force:
                continue
            fv = child.winner_force.value
            force_scores[fv] = force_scores.get(fv, 0.0) + child.margin

        if not force_scores:
            return TheatreResult(theatre_id=theatre_id)

        winner_val = max(force_scores, key=force_scores.get)        # type: ignore[arg-type]
        total = sum(force_scores.values())
        score = force_scores[winner_val] / total if total > 0 else 0.0

        try:
            winner_force = ForceType(winner_val)
        except ValueError:
            winner_force = ForceType.PROPOSITION

        result = TheatreResult(
            theatre_id=theatre_id,
            theatre_name=theatre_node.label,
            winner_force=winner_force,
            theatre_score=score,
        )

        theatre_node.winner_force = winner_force
        theatre_node.confidence = score
        theatre_node.margin = score
        theatre_node.is_resolved = True

        return result

    def resolve_campaign(self) -> CampaignVerdict:
        """Resolve the campaign root from its children.

        Aggregates theatre or engagement winners (depending on tree height).
        """
        root = self.root
        if not root:
            return CampaignVerdict()

        force_scores: dict[str, float] = {}
        for child_id in root.child_ids:
            child = self._nodes.get(child_id)
            if not child or not child.is_resolved or not child.winner_force:
                continue
            fv = child.winner_force.value
            weight = child.engagement_weight if child.engagement_weight > 0 else 1.0
            force_scores[fv] = force_scores.get(fv, 0.0) + child.margin * weight

        if not force_scores:
            return CampaignVerdict()

        winner_val = max(force_scores, key=force_scores.get)        # type: ignore[arg-type]
        total = sum(force_scores.values())
        strength = force_scores[winner_val] / total if total > 0 else 0.0

        try:
            winner_force = ForceType(winner_val)
        except ValueError:
            winner_force = ForceType.PROPOSITION

        # Verdict label
        if winner_force in (ForceType.PROPOSITION,):
            label = CampaignVerdictLabel.SUPPORTED
        elif winner_force in (ForceType.OPPOSITION,):
            label = CampaignVerdictLabel.CHALLENGED
        else:
            label = CampaignVerdictLabel.QUALIFIED

        root.winner_force = winner_force
        root.confidence = strength
        root.margin = strength
        root.is_resolved = True

        return CampaignVerdict(
            verdict_label=label,
            winning_force=winner_force,
            campaign_strength_score=strength,
            campaign_strength_label=VictoryStrength.from_score(strength),
        )

    # ── Query Methods ──────────────────────────────────────────────

    def all_nodes(self) -> list[TournamentNode]:
        return list(self._nodes.values())

    def get_bracket_state(self) -> dict[str, Any]:
        """Serialisable tree state for visualisation."""
        return {
            "root_id": self._root_id,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "skirmish_ids": self._skirmish_ids,
            "engagement_ids": self._engagement_ids,
            "theatre_ids": self._theatre_ids,
            "total_skirmishes": self.total_skirmishes,
            "resolved_skirmishes": self.resolved_skirmishes,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.get_bracket_state()

    # ── Internal Helpers ───────────────────────────────────────────

    @staticmethod
    def _get_force_pair(
        forces: list[ForceType],
        pair_idx: int,
    ) -> tuple[ForceType, ForceType]:
        """Get a (force_a, force_b) pair for a skirmish.

        For bipolar: always PF vs OF.
        For multipolar: cycles through all pairwise combinations.
        """
        if len(forces) <= 2:
            return (forces[0], forces[1]) if len(forces) == 2 else (forces[0], forces[0])

        # Generate all pairwise combinations
        pairs: list[tuple[ForceType, ForceType]] = []
        for i in range(len(forces)):
            for j in range(i + 1, len(forces)):
                pairs.append((forces[i], forces[j]))

        if not pairs:
            return (forces[0], forces[-1])

        idx = pair_idx % len(pairs)
        return pairs[idx]
