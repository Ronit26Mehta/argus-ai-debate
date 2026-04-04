"""
Proposition Depth Analyzer (PDA) — HANNIBAL's first operational component.

Runs before any Force is deployed.  Analyses the proposition's polarity
structure, epistemic depth (factual / normative / inferential), determines
force count, theatre count, and tree height, then produces a BattleMap.

Uses the same LLM interaction pattern as :mod:`argus.agora.senate_gen`:
JSON-over-LLM with graceful fallback defaults.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from argus.core.json_repair import (
    extract_json_array as _extract_json_array,
    repair_json as _try_repair_json,
)

from argus.hannibal.models import (
    BattleMap,
    EpistemicDepthScore,
    ForceType,
    HannibalSessionConfig,
    PolarityStructure,
    TheatreSpec,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# LLM Prompts
# ══════════════════════════════════════════════════════════════════════

_POLARITY_SYSTEM = """\
You are HANNIBAL's Proposition Depth Analyzer.  Given a proposition, determine \
its POLARITY STRUCTURE — how many fundamentally distinct positions exist:

- bipolar: only two sides (FOR and AGAINST)
- tripolar: three distinct positions (often includes a conditional/partial view)
- quadrupolar: four distinct positions (rare, for deeply contested topics)

Also determine each position's brief stance description and a unique, highly descriptive \
faction name (e.g. "Technocratic Autocrats", "Neo-Luddites", "Free-Market Capitalists"). \
Do NOT use generic names like "Proposition Force" or "Faction 1".

Output ONLY valid JSON:
{
  "polarity": "bipolar | tripolar | quadrupolar",
  "positions": [
    {"force_label": "proposition", "dynamic_name": "...", "stance": "..."},
    {"force_label": "opposition", "dynamic_name": "...", "stance": "..."}
  ]
}

For tripolar, add a "faction_1" entry.
For quadrupolar, add "faction_1" and "faction_2" entries.
"""

_DEPTH_SYSTEM = """\
You are HANNIBAL's Proposition Depth Analyzer.  Score the epistemic DEPTH of \
a proposition on three independent axes, each 0.0 to 1.0:

- factual: how much does this depend on empirical, verifiable facts?
- normative: how much does this involve value judgements, ethics, or policy preferences?
- inferential: how complex are the logical chains needed to evaluate this?

Higher scores indicate deeper complexity on that axis.

Output ONLY valid JSON:
{"factual": 0.0, "normative": 0.0, "inferential": 0.0}
"""

_THEATRE_SYSTEM = """\
You are HANNIBAL's Proposition Depth Analyzer.  Given a proposition and the \
number of Theatres needed, break the proposition into distinct argumentation \
theatres — independent sub-fronts where evidence battles should occur.

Each theatre should have:
- name: short label (3-5 words)
- topical_scope: what subset of the argument this theatre covers
- engagement_count: how many engagement rounds (1–3) based on depth

Output ONLY a valid JSON array:
[{{"name": "...", "topical_scope": "...", "engagement_count": 2}}, ...]

Generate EXACTLY {count} theatres.
"""


# ══════════════════════════════════════════════════════════════════════
# Proposition Depth Analyzer
# ══════════════════════════════════════════════════════════════════════

class PropositionDepthAnalyzer:
    """Analyses a proposition and produces a BattleMap specification.

    The BattleMap drives everything downstream: tree construction,
    force sizing, and CANNAE activation.
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def analyze(
        self,
        proposition: str,
        config: HannibalSessionConfig | None = None,
    ) -> BattleMap:
        """Run full analysis pipeline and return a BattleMap.

        Pipeline:
            1. Polarity detection  → bipolar / tri / quad
            2. Depth scoring       → factual / normative / inferential
            3. Front determination → number of theatres
            4. Theatre specification → scoped argument zones
            5. Force size calibration → per-force agent budgets
            6. BattleMap assembly

        Args:
            proposition: The proposition text.
            config: Optional session config overrides.

        Returns:
            A complete BattleMap ready for Force deployment.
        """
        config = config or HannibalSessionConfig()
        battle_map = BattleMap(proposition=proposition)

        # Step 1: Polarity detection
        polarity, faction_positions, faction_names = self._detect_polarity(proposition)
        battle_map.polarity = polarity
        battle_map.faction_positions = faction_positions
        battle_map.faction_names = faction_names
        battle_map.force_designations = self._get_force_designations(polarity)
        logger.info("PDA: Polarity=%s (%d forces)",
                     polarity.value, polarity.force_count)

        # Step 2: Depth scoring
        depth = self._score_depth(proposition)
        battle_map.depth_score = depth
        logger.info("PDA: Depth F=%.2f N=%.2f I=%.2f (agg=%.2f)",
                     depth.factual, depth.normative,
                     depth.inferential, depth.aggregate)

        # Step 3: Determine front count (number of theatres)
        num_theatres = self._determine_fronts(polarity, depth, config)
        logger.info("PDA: Theatres=%d", num_theatres)

        # Step 4: Specify theatres
        theatres = self._specify_theatres(proposition, num_theatres, config)
        battle_map.theatres = theatres

        # Step 5: Tree height
        tree_height = min(depth.tree_height, config.max_tree_height)
        battle_map.tree_height = tree_height

        # Step 6: Force size calibration
        force_sizes = self._calibrate_force_sizes(
            depth, num_theatres, polarity, config,
        )
        battle_map.force_sizes = force_sizes
        battle_map.estimated_total_agents = sum(force_sizes.values())

        # Step 7: Skirmish count estimation
        total_engagements = sum(t.engagement_count for t in theatres)
        skirmishes_per_engagement = min(
            config.max_skirmishes_per_engagement,
            max(1, polarity.force_count - 1),
        )
        battle_map.estimated_skirmish_count = total_engagements * skirmishes_per_engagement

        # Step 8: CANNAE activation
        battle_map.cannae_activated = polarity.force_count >= 3

        logger.info(
            "PDA: BattleMap ready — tree_height=%d, skirmishes=%d, agents=%d, CANNAE=%s",
            tree_height, battle_map.estimated_skirmish_count,
            battle_map.estimated_total_agents,
            "ON" if battle_map.cannae_activated else "OFF",
        )
        return battle_map

    # ── Step 1: Polarity Detection ─────────────────────────────────

    def _detect_polarity(
        self, proposition: str,
    ) -> tuple[PolarityStructure, dict[str, str], dict[str, str]]:
        """Detect polarity structure via LLM."""
        faction_positions: dict[str, str] = {}
        faction_names: dict[str, str] = {}
        try:
            response = self.llm.generate(
                prompt=f"Proposition: {proposition}",
                system_prompt=_POLARITY_SYSTEM,
                temperature=0.3,
                max_tokens=600,
            )
            text = response.content.strip()
            logger.debug("PDA polarity raw: %s", text[:200])

            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                polarity_str = data.get("polarity", "bipolar").lower()
                try:
                    polarity = PolarityStructure(polarity_str)
                except ValueError:
                    polarity = PolarityStructure.BIPOLAR

                # Extract faction positions and dynamic names
                for pos in data.get("positions", []):
                    label = pos.get("force_label", "")
                    stance = pos.get("stance", "")
                    dyn_name = pos.get("dynamic_name", "")
                    if label and stance:
                        faction_positions[label] = stance
                        if dyn_name:
                            faction_names[label] = dyn_name

                return polarity, faction_positions, faction_names
        except Exception as exc:
            logger.warning("PDA polarity LLM failed: %s", exc)
        return PolarityStructure.BIPOLAR, {}, {}

    # ── Step 2: Depth Scoring ──────────────────────────────────────

    def _score_depth(self, proposition: str) -> EpistemicDepthScore:
        """Score epistemic depth on three axes via LLM."""
        try:
            response = self.llm.generate(
                prompt=f"Proposition: {proposition}",
                system_prompt=_DEPTH_SYSTEM,
                temperature=0.3,
                max_tokens=512,
            )
            text = response.content.strip()
            logger.debug("PDA depth raw: %s", text[:200])

            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                return EpistemicDepthScore(
                    factual=max(0.0, min(1.0, float(data.get("factual", 0.5)))),
                    normative=max(0.0, min(1.0, float(data.get("normative", 0.5)))),
                    inferential=max(0.0, min(1.0, float(data.get("inferential", 0.5)))),
                )
        except Exception as exc:
            logger.warning("PDA depth LLM failed: %s", exc)
        return EpistemicDepthScore()

    # ── Step 3: Front Determination ────────────────────────────────

    def _determine_fronts(
        self,
        polarity: PolarityStructure,
        depth: EpistemicDepthScore,
        config: HannibalSessionConfig,
    ) -> int:
        """Determine the number of theatres (fronts).

        Heuristic:
            base = 1
            + 1 if depth.aggregate > 0.5
            + 1 if depth.aggregate > 0.7 and normative > 0.4
            Clamped to [1, 3] for i3 constraint
        """
        base = 1
        if depth.aggregate > 0.5:
            base += 1
        if depth.aggregate > 0.7 and depth.normative > 0.4:
            base += 1
        return max(1, min(base, 3))

    # ── Step 4: Theatre Specification ──────────────────────────────

    def _specify_theatres(
        self,
        proposition: str,
        num_theatres: int,
        config: HannibalSessionConfig,
    ) -> list[TheatreSpec]:
        """Specify theatre scopes via LLM."""
        try:
            system = _THEATRE_SYSTEM.format(count=num_theatres)
            response = self.llm.generate(
                prompt=f"Proposition: {proposition}",
                system_prompt=system,
                temperature=0.5,
                max_tokens=1024,
            )
            logger.debug("PDA theatres raw: %s", response.content[:200])

            raw = _extract_json_array(response.content)
            theatres: list[TheatreSpec] = []
            for item in raw[:num_theatres]:
                eng_count = min(
                    config.max_engagements_per_theatre,
                    max(1, int(item.get("engagement_count", 2))),
                )
                theatres.append(TheatreSpec(
                    name=item.get("name", f"Theatre-{len(theatres)+1}"),
                    topical_scope=item.get("topical_scope", "General scope"),
                    engagement_count=eng_count,
                ))
            if theatres:
                return theatres
        except Exception as exc:
            logger.warning("PDA theatre LLM failed: %s", exc)

        # Fallback defaults
        return self._default_theatres(proposition, num_theatres, config)

    def _default_theatres(
        self,
        proposition: str,
        num_theatres: int,
        config: HannibalSessionConfig,
    ) -> list[TheatreSpec]:
        """Generate fallback theatre specs."""
        defaults = [
            TheatreSpec(
                name="Core Evidence Theatre",
                topical_scope=(
                    f"Primary empirical and factual evidence for/against: "
                    f"{proposition[:100]}"
                ),
                engagement_count=min(2, config.max_engagements_per_theatre),
            ),
            TheatreSpec(
                name="Implications Theatre",
                topical_scope=(
                    "Broader implications, normative considerations, and "
                    "downstream consequences of the proposition."
                ),
                engagement_count=min(2, config.max_engagements_per_theatre),
            ),
            TheatreSpec(
                name="Methodology Theatre",
                topical_scope=(
                    "Methodological evaluation — quality and reliability of "
                    "evidence sources, logical structure, and inferential validity."
                ),
                engagement_count=1,
            ),
        ]
        return defaults[:num_theatres]

    # ── Step 5: Force Size Calibration ─────────────────────────────

    def _calibrate_force_sizes(
        self,
        depth: EpistemicDepthScore,
        num_theatres: int,
        polarity: PolarityStructure,
        config: HannibalSessionConfig,
    ) -> dict[str, int]:
        """Calculate per-force agent count.

        Each force gets:
            1 Commander + N Vanguards (one per theatre) + 1 Flanking + 1 IO
            + Reserves if budget allows

        Total capped at config.max_total_agents for i3 constraint.
        """
        per_force_base = 1 + num_theatres + 1 + 1  # C + V*T + F + IO
        per_force = min(per_force_base, config.max_force_size)

        num_forces = polarity.force_count
        total = per_force * num_forces

        # Enforce total agent cap
        if total > config.max_total_agents:
            per_force = max(config.min_force_size,
                            config.max_total_agents // num_forces)

        force_types = self._get_force_designations(polarity)
        sizes: dict[str, int] = {}
        for ft in force_types:
            sizes[ft.value] = per_force

        return sizes

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _get_force_designations(
        polarity: PolarityStructure,
    ) -> list[ForceType]:
        """Return the list of ForceTypes for a given polarity."""
        _MAP = {
            PolarityStructure.BIPOLAR: [
                ForceType.PROPOSITION, ForceType.OPPOSITION,
            ],
            PolarityStructure.TRIPOLAR: [
                ForceType.PROPOSITION, ForceType.OPPOSITION,
                ForceType.FACTION_1,
            ],
            PolarityStructure.QUADRUPOLAR: [
                ForceType.PROPOSITION, ForceType.OPPOSITION,
                ForceType.FACTION_1, ForceType.FACTION_2,
            ],
        }
        return _MAP.get(polarity, [ForceType.PROPOSITION, ForceType.OPPOSITION])
