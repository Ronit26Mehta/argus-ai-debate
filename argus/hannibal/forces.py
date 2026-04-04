"""
Force Deployment Engine — HANNIBAL Protocol.

Deploys organised Forces from a BattleMap.  Each Force receives:
  - 1 Commander     (strategic direction)
  - N Vanguards     (frontline — one per theatre when possible)
  - 1 Flanking      (counter-evidence specialist)
  - 1 Intelligence Officer (cross-force monitoring)
  - 0+ Reserves     (deployed if budget allows and engagements stall)

Uses LLM for persona generation with fallback to persona banks.
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
    AgentRoleSpec,
    BattleMap,
    ForceSpec,
    ForceType,
    HannibalSessionConfig,
    MilitaryRole,
    _uid,
)
from argus.hannibal.personas.commander_bank import get_commander_template
from argus.hannibal.personas.vanguard_bank import get_vanguard_templates
from argus.hannibal.personas.flanking_bank import get_flanking_templates

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# LLM Prompts
# ══════════════════════════════════════════════════════════════════════

_FORCE_PERSONA_SYSTEM = """\
You are HANNIBAL's Force Deployment Engine.  Generate agent personas for a \
military-style debate Force.

Force: {force_name} ({force_type})
Position: {position}
Domain: {domain}

Generate {count} agent specifications.  Each agent must have:
  - name: realistic professional name
  - role: one of commander, vanguard, flanking, intelligence_officer, reserve
  - domain_expertise: specific expertise area
  - epistemic_prior: float {prior_low:.2f}–{prior_high:.2f}
  - evidence_sources: list of evidence source types
  - persona_description: 1-sentence role description

Required role distribution:
{role_distribution}

Output ONLY a valid JSON array.  No extra text.
"""


# ══════════════════════════════════════════════════════════════════════
# Force Deployment Engine
# ══════════════════════════════════════════════════════════════════════

class ForceDeploymentEngine:
    """Creates and deploys Forces based on a BattleMap specification.

    Each Force is an organised team of agents with defined military
    roles, domain expertise, and calibrated epistemic priors.
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def deploy(
        self,
        battle_map: BattleMap,
        config: HannibalSessionConfig | None = None,
    ) -> list[ForceSpec]:
        """Deploy all Forces for the campaign.

        Args:
            battle_map: The deployment specification from PDA.
            config: Optional session config.

        Returns:
            List of fully-staffed ForceSpec objects.
        """
        config = config or HannibalSessionConfig()
        forces: list[ForceSpec] = []

        for force_type in battle_map.force_designations:
            force_size = battle_map.force_sizes.get(
                force_type.value,
                config.min_force_size,
            )
            force_size = max(config.min_force_size,
                             min(force_size, config.max_force_size))

            force = self._create_force(
                force_type=force_type,
                force_size=force_size,
                battle_map=battle_map,
                config=config,
            )
            forces.append(force)
            logger.info(
                "Force deployed: %s with %d agents (prior=%.2f)",
                force_type.display_name, force.force_size, force.force_prior,
            )

        return forces

    def _create_force(
        self,
        force_type: ForceType,
        force_size: int,
        battle_map: BattleMap,
        config: HannibalSessionConfig,
    ) -> ForceSpec:
        """Create a single Force with all its agents."""
        # Determine position description
        position = battle_map.faction_positions.get(
            force_type.value,
            self._default_position(force_type, battle_map.proposition),
        )

        # Calculate prior range
        prior, prior_low, prior_high = self._configure_priors(force_type)

        # Try LLM persona generation
        agents = self._generate_agents_llm(
            force_type=force_type,
            force_size=force_size,
            position=position,
            battle_map=battle_map,
            prior_low=prior_low,
            prior_high=prior_high,
        )

        # Fallback to template-based generation
        if len(agents) < force_size:
            agents = self._generate_agents_template(
                force_type=force_type,
                force_size=force_size,
                battle_map=battle_map,
                prior=prior,
            )

        force_name = battle_map.faction_names.get(force_type.value, force_type.display_name)

        return ForceSpec(
            force_type=force_type,
            force_name=force_name,
            position_description=position,
            agents=agents[:force_size],
            force_prior=prior,
            force_posterior=prior,
        )

    # ── LLM-Based Persona Generation ──────────────────────────────

    def _generate_agents_llm(
        self,
        force_type: ForceType,
        force_size: int,
        position: str,
        battle_map: BattleMap,
        prior_low: float,
        prior_high: float,
    ) -> list[AgentRoleSpec]:
        """Attempt LLM-based agent persona generation."""
        role_dist = self._role_distribution(force_size)
        role_dist_str = ", ".join(
            f"{count}x {role.value}" for role, count in role_dist.items()
        )

        try:
            domain = "general"
            if battle_map.theatres:
                domain = battle_map.theatres[0].topical_scope[:50]

            system = _FORCE_PERSONA_SYSTEM.format(
                force_name=battle_map.faction_names.get(force_type.value, force_type.display_name),
                force_type=force_type.abbreviation,
                position=position[:200],
                domain=domain,
                count=force_size,
                prior_low=prior_low,
                prior_high=prior_high,
                role_distribution=role_dist_str,
            )
            response = self.llm.generate(
                prompt=f"Proposition: {battle_map.proposition}",
                system_prompt=system,
                temperature=0.5,
                max_tokens=2048,
            )
            logger.debug("Force gen raw (%s): %s",
                          force_type.abbreviation, response.content[:200])

            raw = _extract_json_array(response.content)
            agents = self._parse_agent_list(raw, force_type, prior_low, prior_high)

            if len(agents) >= force_size * 0.7:
                # Ensure role coverage
                agents = self._ensure_role_coverage(
                    agents, force_type, battle_map, role_dist,
                    prior_low, prior_high,
                )
                return agents[:force_size]

        except Exception as exc:
            logger.warning("Force LLM gen failed (%s): %s",
                            force_type.abbreviation, exc)
        return []

    def _parse_agent_list(
        self,
        raw: list[dict],
        force_type: ForceType,
        prior_low: float,
        prior_high: float,
    ) -> list[AgentRoleSpec]:
        """Parse LLM-generated agent list into AgentRoleSpec objects."""
        agents: list[AgentRoleSpec] = []
        for item in raw:
            role_str = item.get("role", "vanguard")
            try:
                role = MilitaryRole(role_str)
            except ValueError:
                role = MilitaryRole.VANGUARD

            prior_val = float(item.get("epistemic_prior", 0.5))
            prior_val = max(prior_low, min(prior_high, prior_val))

            sources = item.get("evidence_sources", [])
            if isinstance(sources, str):
                sources = [s.strip() for s in sources.split(",")]

            agents.append(AgentRoleSpec(
                name=item.get("name", f"Agent-{len(agents)+1}"),
                role=role,
                force_type=force_type,
                domain_expertise=item.get("domain_expertise", ""),
                epistemic_prior=prior_val,
                evidence_source_priority=sources,
                persona_description=item.get("persona_description", ""),
            ))
        return agents

    # ── Template-Based Fallback ────────────────────────────────────

    def _generate_agents_template(
        self,
        force_type: ForceType,
        force_size: int,
        battle_map: BattleMap,
        prior: float,
    ) -> list[AgentRoleSpec]:
        """Generate agents from persona bank templates."""
        agents: list[AgentRoleSpec] = []
        domain = "general"
        if battle_map.theatres:
            scope = battle_map.theatres[0].topical_scope.lower()
            for d in ("science", "policy", "economics", "ethics",
                       "technology", "history", "medicine", "law"):
                if d in scope:
                    domain = d
                    break

        # 1. Commander
        cmd_template = get_commander_template(force_type, domain)
        agents.append(AgentRoleSpec(
            name=cmd_template["name"],
            role=MilitaryRole.COMMANDER,
            force_type=force_type,
            domain_expertise=cmd_template.get("domain", domain),
            epistemic_prior=prior,
            persona_description=cmd_template["persona"],
        ))

        # 2. Vanguards (one per theatre, up to remaining budget)
        remaining = force_size - 1  # 1 for commander
        vanguard_count = min(len(battle_map.theatres), remaining - 2)
        vanguard_count = max(1, vanguard_count)

        vanguard_templates = get_vanguard_templates(domain, vanguard_count)
        for i in range(vanguard_count):
            tmpl = vanguard_templates[i % len(vanguard_templates)]
            theatre_id = ""
            if i < len(battle_map.theatres):
                theatre_id = battle_map.theatres[i].id

            sources = tmpl.get("evidence_sources", "general_web")
            if isinstance(sources, str):
                sources = [s.strip() for s in sources.split(",")]

            agents.append(AgentRoleSpec(
                name=tmpl["name"],
                role=MilitaryRole.VANGUARD,
                force_type=force_type,
                domain_expertise=tmpl.get("expertise", domain),
                epistemic_prior=prior,
                evidence_source_priority=sources,
                persona_description=tmpl.get("persona", ""),
                assigned_theatre_id=theatre_id,
            ))
            remaining -= 1

        # 3. Flanking
        if remaining >= 1:
            flanking_templates = get_flanking_templates(count=1)
            if flanking_templates:
                fl = flanking_templates[0]
                sources = fl.get("evidence_sources", "general_web")
                if isinstance(sources, str):
                    sources = [s.strip() for s in sources.split(",")]
                agents.append(AgentRoleSpec(
                    name=fl["name"],
                    role=MilitaryRole.FLANKING,
                    force_type=force_type,
                    domain_expertise=fl.get("expertise", "counter-argument"),
                    epistemic_prior=prior,
                    evidence_source_priority=sources,
                    persona_description=fl.get("persona", ""),
                ))
                remaining -= 1

        # 4. Intelligence Officer
        if remaining >= 1:
            agents.append(AgentRoleSpec(
                name=f"IO-{force_type.abbreviation}",
                role=MilitaryRole.INTELLIGENCE_OFFICER,
                force_type=force_type,
                domain_expertise="Cross-force intelligence analysis",
                epistemic_prior=0.5,  # Neutral
                persona_description=(
                    "Intelligence officer monitoring opponent Force patterns "
                    "and identifying strategic vulnerabilities."
                ),
            ))
            remaining -= 1

        # 5. Reserves (remaining budget)
        for r in range(remaining):
            agents.append(AgentRoleSpec(
                name=f"Rsv-{force_type.abbreviation}-{r+1}",
                role=MilitaryRole.RESERVE,
                force_type=force_type,
                domain_expertise=domain,
                epistemic_prior=prior,
                persona_description=(
                    "Reserve agent held back for deployment when engagements "
                    "stall or fresh evidence is needed."
                ),
                is_deployed=False,
            ))

        return agents

    # ── Role Distribution & Coverage ───────────────────────────────

    @staticmethod
    def _role_distribution(force_size: int) -> dict[MilitaryRole, int]:
        """Compute required role distribution for a given force size."""
        dist: dict[MilitaryRole, int] = {
            MilitaryRole.COMMANDER: 1,
        }
        remaining = force_size - 1

        # IO: 1 if budget allows
        if remaining >= 3:
            dist[MilitaryRole.INTELLIGENCE_OFFICER] = 1
            remaining -= 1

        # Flanking: 1 if budget allows
        if remaining >= 2:
            dist[MilitaryRole.FLANKING] = 1
            remaining -= 1

        # Vanguards take the bulk
        vanguard_count = max(1, remaining)
        dist[MilitaryRole.VANGUARD] = vanguard_count

        return dist

    def _ensure_role_coverage(
        self,
        agents: list[AgentRoleSpec],
        force_type: ForceType,
        battle_map: BattleMap,
        role_dist: dict[MilitaryRole, int],
        prior_low: float,
        prior_high: float,
    ) -> list[AgentRoleSpec]:
        """Ensure all required roles are filled."""
        current_roles = {a.role for a in agents}
        prior_mid = (prior_low + prior_high) / 2

        for role, count in role_dist.items():
            existing = sum(1 for a in agents if a.role == role)
            needed = count - existing
            for _ in range(needed):
                agents.append(AgentRoleSpec(
                    name=f"{role.abbreviation}-{force_type.abbreviation}-{len(agents)+1}",
                    role=role,
                    force_type=force_type,
                    domain_expertise="General",
                    epistemic_prior=prior_mid,
                    persona_description=f"Auto-generated {role.display_name} for {force_type.display_name}",
                ))
        return agents

    # ── Prior Configuration ────────────────────────────────────────

    @staticmethod
    def _configure_priors(
        force_type: ForceType,
    ) -> tuple[float, float, float]:
        """Return (prior, prior_low, prior_high) for a force type.

        PF: high commitment to proposition truth (0.85–0.95 → prior=0.90)
        OF: high commitment to proposition falsity (0.05–0.15 → prior=0.10)
        FF: moderate/conditional (0.35–0.65 → prior=0.50)
        """
        _MAP: dict[str, tuple[float, float, float]] = {
            ForceType.PROPOSITION.value: (0.90, 0.85, 0.95),
            ForceType.OPPOSITION.value:  (0.10, 0.05, 0.15),
            ForceType.FACTION_1.value:   (0.50, 0.35, 0.65),
            ForceType.FACTION_2.value:   (0.50, 0.35, 0.65),
            ForceType.FACTION_3.value:   (0.50, 0.35, 0.65),
        }
        return _MAP.get(force_type.value, (0.50, 0.30, 0.70))

    @staticmethod
    def _default_position(force_type: ForceType, proposition: str) -> str:
        """Generate a default position description."""
        short = proposition[:100]
        _MAP = {
            ForceType.PROPOSITION.value: f"This proposition is TRUE: {short}",
            ForceType.OPPOSITION.value: f"This proposition is FALSE: {short}",
            ForceType.FACTION_1.value: f"This proposition is conditionally true under specific circumstances: {short}",
            ForceType.FACTION_2.value: f"The proposition as stated is ill-formed; the truth is more nuanced: {short}",
            ForceType.FACTION_3.value: f"Insufficient evidence exists to determine the truth of: {short}",
        }
        return _MAP.get(force_type.value, f"Position on: {short}")
