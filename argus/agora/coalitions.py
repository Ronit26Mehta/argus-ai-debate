"""
Coalition Detection Engine — discovers implicit voting blocs in the Senate.

Analyses the Position Similarity Matrix (PSM) across all senators to
identify clusters of agents that share epistemic premises. Flags coalitions
whose collective weight might be an artefact of shared (non-independent)
evidence rather than genuine convergence.

Key outputs:
    - CoalitionInfo objects with names, members, and EIS
    - Live updates per round for the sandbox
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from argus.agora.models import (
    CoalitionInfo,
    DocketItem,
    SenateRecordEntry,
    RecordEntryType,
    SessionPhase,
    _utcnow,
    _uid,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for empty vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return dot / (mag_a * mag_b)


class CoalitionDetectionEngine:
    """Detects and tracks coalitions of senators.

    Uses a position vector per senator (built from evidence polarities
    and position statements) and clusters by cosine similarity.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        llm: "BaseLLM | None" = None,
    ):
        self.threshold = threshold
        self.llm = llm

        # senator_id -> position vector
        # Each element corresponds to a dimension of the proposition space
        self._position_vectors: dict[str, list[float]] = {}
        # senator_id -> name mapping
        self._names: dict[str, str] = {}
        # detected coalitions
        self._coalitions: list[CoalitionInfo] = []

    def update_position(
        self,
        senator_id: str,
        senator_name: str,
        position_signal: float,
    ) -> None:
        """Record a position signal for a senator.

        Each call appends to the senator's position vector.
        A position signal is a float in [-1, 1]:
            +1 = full support
             0 = neutral
            -1 = full opposition
        """
        if senator_id not in self._position_vectors:
            self._position_vectors[senator_id] = []
            self._names[senator_id] = senator_name
        self._position_vectors[senator_id].append(position_signal)

    def compute_psm(self) -> dict[str, dict[str, float]]:
        """Compute the full Position Similarity Matrix.

        Returns {senator_a: {senator_b: similarity, ...}, ...}
        """
        senator_ids = list(self._position_vectors.keys())
        n = len(senator_ids)
        psm: dict[str, dict[str, float]] = {}

        # Normalize vector lengths (pad shorter vectors with 0.0)
        max_len = max((len(v) for v in self._position_vectors.values()), default=0)
        padded: dict[str, list[float]] = {}
        for sid, vec in self._position_vectors.items():
            padded[sid] = vec + [0.0] * (max_len - len(vec))

        for i in range(n):
            sid_a = senator_ids[i]
            psm[sid_a] = {}
            for j in range(n):
                sid_b = senator_ids[j]
                if i == j:
                    psm[sid_a][sid_b] = 1.0
                elif sid_b in psm and sid_a in psm[sid_b]:
                    psm[sid_a][sid_b] = psm[sid_b][sid_a]
                else:
                    psm[sid_a][sid_b] = _cosine_similarity(
                        padded[sid_a], padded[sid_b],
                    )
        return psm

    def detect_coalitions(self) -> list[CoalitionInfo]:
        """Run coalition detection and return newly detected coalitions.

        Uses single-linkage clustering: two senators are in the same
        coalition if their PSM similarity exceeds the threshold. Coalitions
        must have at least 2 members.
        """
        psm = self.compute_psm()
        senator_ids = list(psm.keys())

        # Union-Find for clustering
        parent: dict[str, str] = {sid: sid for sid in senator_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, sid_a in enumerate(senator_ids):
            for j in range(i + 1, len(senator_ids)):
                sid_b = senator_ids[j]
                if psm[sid_a][sid_b] >= self.threshold:
                    union(sid_a, sid_b)

        # Group clusters
        clusters: dict[str, list[str]] = defaultdict(list)
        for sid in senator_ids:
            clusters[find(sid)].append(sid)

        # Build CoalitionInfo for clusters with >= 2 members
        new_coalitions: list[CoalitionInfo] = []
        for _, members in clusters.items():
            if len(members) < 2:
                continue

            member_names = [self._names.get(m, m) for m in members]

            # Average similarity within coalition
            sims = []
            for i, ma in enumerate(members):
                for j in range(i + 1, len(members)):
                    mb = members[j]
                    sims.append(psm[ma][mb])
            avg_sim = sum(sims) / max(len(sims), 1)

            # Epistemic Independence Score (EIS):
            # lower if members share highly similar position vectors
            # EIS = 1 - (avg_pairwise_similarity - threshold) / (1.0 - threshold)
            eis_raw = 1.0 - max(0, (avg_sim - self.threshold)) / max(1.0 - self.threshold, 0.01)
            eis = max(0.0, min(1.0, eis_raw))

            # Name the coalition
            coalition_name = self._name_coalition(member_names)

            coalition = CoalitionInfo(
                name=coalition_name,
                member_ids=list(members),
                member_names=member_names,
                shared_premise=f"Senators share epistemic premises (avg similarity: {avg_sim:.2f})",
                epistemic_independence_score=eis,
                similarity_score=avg_sim,
            )
            new_coalitions.append(coalition)

        self._coalitions = new_coalitions
        return new_coalitions

    def _name_coalition(self, member_names: list[str]) -> str:
        """Generate a descriptive name for the coalition.

        Uses LLM if available, else constructs from member names.
        """
        if self.llm and len(member_names) >= 2:
            try:
                prompt = (
                    f"Given these senators who share similar positions: "
                    f"{', '.join(member_names)}. "
                    f"Generate a short (2-4 word) coalition name. "
                    f"Output ONLY the name, nothing else."
                )
                response = self.llm.generate(
                    prompt=prompt,
                    temperature=0.5, max_tokens=50,
                )
                name = response.content.strip().strip('"').strip()
                if 2 <= len(name.split()) <= 5:
                    return name
            except Exception:
                pass

        # Fallback: "X-Y Bloc"
        if len(member_names) <= 3:
            parts = [n.split()[-1] for n in member_names]
            return f"{'-'.join(parts)} Bloc"
        return f"Coalition of {len(member_names)}"

    @property
    def current_coalitions(self) -> list[CoalitionInfo]:
        return list(self._coalitions)

    def make_coalition_record(
        self,
        coalition: CoalitionInfo,
        phase: SessionPhase,
        round_num: int,
    ) -> SenateRecordEntry:
        """Generate a record entry for a detected coalition."""
        eis_warning = ""
        if coalition.is_low_independence:
            eis_warning = " ⚠ LOW EPISTEMIC INDEPENDENCE — weight may be inflated."
        return SenateRecordEntry(
            entry_type=RecordEntryType.COALITION_DETECTED,
            phase=phase,
            round_num=round_num,
            content=(
                f"Coalition detected: {coalition.name} "
                f"({coalition.size} members: {', '.join(coalition.member_names)}). "
                f"Avg similarity: {coalition.similarity_score:.2f}. "
                f"EIS: {coalition.epistemic_independence_score:.2f}.{eis_warning}"
            ),
            metadata=coalition.to_dict(),
        )
