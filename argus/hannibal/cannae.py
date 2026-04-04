"""
CANNAE Multipolar Encirclement Engine — HANNIBAL Protocol.

Handles resolution for campaigns with three or more Forces (tripolar,
quadrupolar).  Instead of simple bilateral scoring, CANNAE builds a
pairwise dominance matrix and finds the Condorcet winner (the Force
that dominates every other Force in bilateral encounters).

If no Condorcet winner exists (non-transitive cycle), falls back to
Borda aggregation and produces an Encirclement Conclusion verdict.

Named after the Battle of Cannae (216 BC), where Hannibal defeated
a numerically superior Roman army through double encirclement.

Pure algorithmic implementation — no numpy dependency required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from argus.hannibal.models import (
    CampaignVerdict,
    CampaignVerdictLabel,
    ForceSpec,
    ForceType,
    TheatreResult,
    VictoryStrength,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class CANNAEEngine:
    """CANNAE Multipolar Encirclement Engine.

    Resolves campaigns with 3+ forces by computing pairwise dominance.

    Pipeline:
        1. Build Dominance Matrix from pairwise theatre/engagement results
        2. Search for Condorcet Winner
        3. If none → Borda Count aggregation
        4. Generate encirclement narrative (LLM)
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def resolve(
        self,
        theatre_results: list[TheatreResult],
        forces: list[ForceSpec],
        all_skirmish_records: list[dict[str, Any]] | None = None,
    ) -> tuple[CampaignVerdict, dict[str, Any]]:
        """Resolve a multipolar campaign.

        Args:
            theatre_results: Results from all theatres.
            forces: All deployed forces.
            all_skirmish_records: Optional raw skirmish data for detailed analysis.

        Returns:
            (CampaignVerdict, encirclement_report_dict)
        """
        force_types = [f.force_type for f in forces]
        n = len(force_types)

        if n < 3:
            logger.warning("CANNAE called with fewer than 3 forces; use standard resolution.")
            # Fall through to basic resolution
            if theatre_results:
                winner = theatre_results[0].winner_force
            else:
                winner = force_types[0]
            return (
                CampaignVerdict(
                    verdict_label=CampaignVerdictLabel.SUPPORTED,
                    winning_force=winner,
                ),
                {},
            )

        # Step 1: Build dominance matrix
        matrix = self._build_dominance_matrix(theatre_results, force_types)
        logger.info("CANNAE: Dominance matrix built (%dx%d)", n, n)

        # Step 2: Find Condorcet winner
        condorcet = self._find_condorcet_winner(matrix, force_types)

        if condorcet:
            logger.info("CANNAE: Condorcet winner found: %s", condorcet.value)
            # Compute strength from dominance margin
            winner_idx = force_types.index(condorcet)
            total_dominance = sum(matrix[winner_idx])
            max_possible = (n - 1) * 1.0
            strength = total_dominance / max_possible if max_possible > 0 else 0.0

            # Find position description
            position = ""
            for f in forces:
                if f.force_type == condorcet:
                    position = f.position_description

            verdict = CampaignVerdict(
                verdict_label=self._label_from_force(condorcet),
                winning_force=condorcet,
                position_description=position,
                campaign_strength_score=strength,
                campaign_strength_label=VictoryStrength.from_score(strength),
            )
        else:
            # Step 3: Borda aggregation (no Condorcet winner)
            logger.info("CANNAE: No Condorcet winner — performing Borda aggregation")
            winner, borda_scores, strength = self._borda_aggregation(
                matrix, force_types,
            )

            position = ""
            for f in forces:
                if f.force_type == winner:
                    position = f.position_description

            verdict = CampaignVerdict(
                verdict_label=CampaignVerdictLabel.ENCIRCLEMENT_CONCLUSION,
                winning_force=winner,
                position_description=position,
                campaign_strength_score=strength,
                campaign_strength_label=VictoryStrength.from_score(strength),
            )

        # Step 4: Build encirclement report
        report = self._build_report(
            matrix, force_types, forces, condorcet is not None,
        )

        # Step 5: Generate narrative
        verdict.narrative = self._generate_narrative(
            verdict, matrix, force_types, forces,
        )

        return verdict, report

    # ── Dominance Matrix ───────────────────────────────────────────

    def _build_dominance_matrix(
        self,
        theatre_results: list[TheatreResult],
        force_types: list[ForceType],
    ) -> list[list[float]]:
        """Build an N×N dominance matrix.

        D[i][j] = proportion of theatre encounters where force i beat force j.
        D[i][i] = 0 (diagonal).

        Since our theatre results give a single winner per theatre, we
        approximate pairwise dominance from how often each force won vs
        each other force across theatres.
        """
        n = len(force_types)
        # Wins matrix: wins[i][j] = count of theatres where i beat j
        wins: list[list[int]] = [[0] * n for _ in range(n)]
        encounters: list[list[int]] = [[0] * n for _ in range(n)]

        for result in theatre_results:
            winner_idx = None
            for idx, ft in enumerate(force_types):
                if ft == result.winner_force:
                    winner_idx = idx
                    break
            if winner_idx is None:
                continue

            # Count this as a win against all other forces
            for j in range(n):
                if j != winner_idx:
                    wins[winner_idx][j] += 1
                    encounters[winner_idx][j] += 1
                    encounters[j][winner_idx] += 1

        # Compute dominance proportions
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0.0
                elif encounters[i][j] > 0:
                    total = encounters[i][j]
                    matrix[i][j] = wins[i][j] / total
                else:
                    matrix[i][j] = 0.5  # No data → assume even

        return matrix

    # ── Condorcet Winner ───────────────────────────────────────────

    def _find_condorcet_winner(
        self,
        matrix: list[list[float]],
        force_types: list[ForceType],
    ) -> Optional[ForceType]:
        """Find the Condorcet winner — the force that dominates every other.

        A force is the Condorcet winner if D[i][j] > 0.5 for ALL j ≠ i.
        """
        n = len(force_types)
        for i in range(n):
            dominates_all = True
            for j in range(n):
                if i == j:
                    continue
                if matrix[i][j] <= 0.5:
                    dominates_all = False
                    break
            if dominates_all:
                return force_types[i]
        return None

    # ── Borda Aggregation ──────────────────────────────────────────

    def _borda_aggregation(
        self,
        matrix: list[list[float]],
        force_types: list[ForceType],
    ) -> tuple[ForceType, dict[str, float], float]:
        """Borda-style aggregation when no Condorcet winner exists.

        Each force's Borda score = sum of its row in the dominance matrix.
        Winner = force with highest Borda score.
        """
        n = len(force_types)
        borda_scores: dict[str, float] = {}

        for i in range(n):
            score = sum(matrix[i])
            borda_scores[force_types[i].value] = score

        if not borda_scores:
            return force_types[0], borda_scores, 0.0

        winner_val = max(borda_scores, key=borda_scores.get)        # type: ignore
        winner = ForceType(winner_val)

        total = sum(borda_scores.values())
        strength = borda_scores[winner_val] / total if total > 0 else 0.0

        return winner, borda_scores, strength

    # ── Report Building ────────────────────────────────────────────

    def _build_report(
        self,
        matrix: list[list[float]],
        force_types: list[ForceType],
        forces: list[ForceSpec],
        has_condorcet: bool,
    ) -> dict[str, Any]:
        """Build the encirclement analysis report."""
        n = len(force_types)
        labels = [ft.display_name for ft in force_types]

        # Build readable matrix
        readable_matrix: dict[str, dict[str, float]] = {}
        for i in range(n):
            row_label = labels[i]
            readable_matrix[row_label] = {}
            for j in range(n):
                col_label = labels[j]
                readable_matrix[row_label][col_label] = round(matrix[i][j], 3)

        # Detect cycles
        cycles = self._detect_cycles(matrix, force_types)

        # Borda scores
        borda: dict[str, float] = {}
        for i in range(n):
            borda[force_types[i].display_name] = round(sum(matrix[i]), 3)

        return {
            "num_forces": n,
            "has_condorcet_winner": has_condorcet,
            "dominance_matrix": readable_matrix,
            "borda_scores": borda,
            "non_transitive_cycles": cycles,
            "force_labels": labels,
        }

    def _detect_cycles(
        self,
        matrix: list[list[float]],
        force_types: list[ForceType],
    ) -> list[str]:
        """Detect non-transitive cycles (A > B > C > A)."""
        n = len(force_types)
        cycles: list[str] = []

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Check A > B > C > A
                    if (matrix[i][j] > 0.5 and
                            matrix[j][k] > 0.5 and
                            matrix[k][i] > 0.5):
                        cycles.append(
                            f"{force_types[i].abbreviation} > "
                            f"{force_types[j].abbreviation} > "
                            f"{force_types[k].abbreviation} > "
                            f"{force_types[i].abbreviation}"
                        )
                    # Check A < B < C < A
                    if (matrix[j][i] > 0.5 and
                            matrix[k][j] > 0.5 and
                            matrix[i][k] > 0.5):
                        cycles.append(
                            f"{force_types[j].abbreviation} > "
                            f"{force_types[k].abbreviation} > "
                            f"{force_types[i].abbreviation} > "
                            f"{force_types[j].abbreviation}"
                        )
        return cycles

    # ── Narrative Generation ───────────────────────────────────────

    def _generate_narrative(
        self,
        verdict: CampaignVerdict,
        matrix: list[list[float]],
        force_types: list[ForceType],
        forces: list[ForceSpec],
    ) -> str:
        """Generate an encirclement narrative via LLM."""
        try:
            matrix_text = self._format_matrix_text(matrix, force_types)
            positions = "\n".join(
                f"  {f.force_type.abbreviation}: {f.position_description[:100]}"
                for f in forces
            )
            prompt = (
                f"Campaign verdict: {verdict.verdict_label.value}\n"
                f"Winner: {verdict.winning_force.display_name}\n"
                f"Strength: {verdict.campaign_strength_label.value}\n\n"
                f"Force positions:\n{positions}\n\n"
                f"Dominance matrix:\n{matrix_text}\n\n"
                f"Write a 3-4 sentence narrative explaining why the "
                f"{verdict.winning_force.display_name} prevailed across "
                f"the multipolar competition.  Reference the dominance "
                f"relationships between forces."
            )
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You are HANNIBAL's CANNAE Engine narrator.  "
                    "Write concise, strategic narratives about multipolar "
                    "campaign outcomes."
                ),
                temperature=0.5,
                max_tokens=512,
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("CANNAE narrative failed: %s", exc)

        return (
            f"The {verdict.winning_force.display_name} achieved "
            f"{verdict.campaign_strength_label.value.lower()} dominance "
            f"across the multipolar campaign."
        )

    @staticmethod
    def _format_matrix_text(
        matrix: list[list[float]],
        force_types: list[ForceType],
    ) -> str:
        """Format dominance matrix as readable text."""
        labels = [ft.abbreviation for ft in force_types]
        header = "     " + "  ".join(f"{l:>6}" for l in labels)
        lines = [header]
        for i, row in enumerate(matrix):
            vals = "  ".join(f"{v:>6.2f}" for v in row)
            lines.append(f"{labels[i]:>4} {vals}")
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _label_from_force(force_type: ForceType) -> CampaignVerdictLabel:
        if force_type == ForceType.PROPOSITION:
            return CampaignVerdictLabel.SUPPORTED
        elif force_type == ForceType.OPPOSITION:
            return CampaignVerdictLabel.CHALLENGED
        else:
            return CampaignVerdictLabel.QUALIFIED
