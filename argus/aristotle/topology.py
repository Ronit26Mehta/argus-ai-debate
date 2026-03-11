"""
Layer 2 — Dynamic Topology Builder.

Takes the :class:`DebateFrame` from L1 and constructs the entire agent
ecosystem from scratch:
    - Specialist agent personas with calibrated epistemic priors
    - Refuter configuration scaled to controversy
    - Jury architecture selection per debate type
    - Round count and budget estimation
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from argus.core.json_repair import (
    extract_json_array as _extract_json_array,
    repair_json as _try_repair_json,
)

from argus.aristotle.models import (
    AgentSpec,
    DebateFrame,
    DebateType,
    JuryArchitecture,
    JurySpec,
    RefuterIntensity,
    TopologySpec,
)
from argus.aristotle.personas.specialist_bank import get_specialists_for_domain
from argus.aristotle.personas.backup_agents import get_backup_agent

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_TOKENS_PER_AGENT_PER_ROUND = 2500  # conservative average

# ── Specialist count scaling ───────────────────────────────────────────

def _specialist_count(controversy: float) -> int:
    if controversy < 0.3:
        return 2
    if controversy < 0.5:
        return 3
    if controversy < 0.7:
        return 4
    return min(int(3 + controversy * 4), 6)


# ── Refuter intensity ─────────────────────────────────────────────────

def _refuter_intensity(controversy: float) -> RefuterIntensity:
    if controversy < 0.3:
        return RefuterIntensity.LOW
    if controversy < 0.6:
        return RefuterIntensity.MEDIUM
    return RefuterIntensity.HIGH


def _refuter_count(controversy: float) -> int:
    if controversy < 0.3:
        return 1
    if controversy < 0.7:
        return 2
    return 3


# ── Jury architecture per debate type ─────────────────────────────────

_JURY_MAP: dict[DebateType, tuple[JuryArchitecture, int, str]] = {
    DebateType.BINARY_FACTUAL: (
        JuryArchitecture.BAYESIAN, 3,
        "Empirical factual claims use Bayesian confidence aggregation",
    ),
    DebateType.CAUSAL: (
        JuryArchitecture.BAYESIAN, 3,
        "Causal claims use Bayesian aggregation with confidence weighting",
    ),
    DebateType.COMPARATIVE: (
        JuryArchitecture.DELIBERATIVE, 5,
        "Comparative claims benefit from deliberative jury multi-perspective evaluation",
    ),
    DebateType.PREDICTIVE: (
        JuryArchitecture.BAYESIAN, 3,
        "Predictive claims use Bayesian posterior convergence",
    ),
    DebateType.NORMATIVE: (
        JuryArchitecture.DELIBERATIVE, 5,
        "Normative/values claims require deliberative multi-perspective assessment",
    ),
    DebateType.DEFINITIONAL: (
        JuryArchitecture.VALIDITY_WEIGHTED, 5,
        "Definitional disputes weight jurors by argument quality (Wu et al. 2025)",
    ),
}


# ── Round count estimation ─────────────────────────────────────────────

def _estimate_rounds(frame: DebateFrame) -> tuple[int, str]:
    """
    Base rounds (2) + controversy bonus (0-3) + evidence density bonus (0-2)
    - sub-claim adjustment.  Capped at [1, 8].
    """
    base = 2

    # controversy bonus
    if frame.controversy_score < 0.3:
        c_bonus = 0
    elif frame.controversy_score < 0.5:
        c_bonus = 1
    elif frame.controversy_score < 0.7:
        c_bonus = 2
    else:
        c_bonus = 3

    # evidence density bonus
    density_map = {"sparse": 0, "moderate": 1, "rich": 2}
    e_bonus = density_map.get(frame.evidence_density, 1)

    # sub-claim adjustment
    parallel_subs = sum(
        1 for sc in frame.sub_claims if not sc.is_primary and not sc.depends_on
    )
    sub_adj = min(parallel_subs, 2)

    total = max(1, min(base + c_bonus + e_bonus - sub_adj, 8))

    reason = (
        f"base {base} + controversy {c_bonus} + evidence density {e_bonus}"
        f" - parallel sub-claims {sub_adj} = {total}"
    )
    return total, reason


# ── LLM prompt for persona generation ─────────────────────────────────

_PERSONA_SYSTEM = """\
You are ARISTOTLE's Topology Builder.  Given a debate frame, generate a JSON
array of specialist agent specifications.  Each element:

{{
  "name": "<realistic academic name, e.g. Dr. Amara Osei>",
  "domain_mandate": "<specific aspect of the evidence landscape this agent owns>",
  "evidence_source_priority": ["<PubMed|arXiv|legal|financial|general_web>"],
  "epistemic_prior": <float 0.15-0.85 — calibrated to generate initial disagreement>,
  "evidence_type_focus": ["<empirical|meta_analysis|case_report|expert_consensus|computational>"],
  "persona_description": "<1 sentence role description>"
}}

Rules:
- Generate EXACTLY {count} specialists.
- Priors MUST span a range ensuring meaningful disagreement (min spread ≥ 0.20).
- At least one agent should start skeptical (prior < 0.45).
- Evidence source priorities should differ across agents.
- Names should signal expertise to the user.
- Output ONLY valid JSON array, no extra text.
"""

_REFUTER_SYSTEM = """\
You are ARISTOTLE's Topology Builder.  Given a debate frame, generate {count}
refuter agent specification(s) as a JSON array.  Each element:

{{
  "name": "<realistic name>",
  "domain_mandate": "<what this refuter specifically challenges>",
  "evidence_source_priority": ["<PubMed|arXiv|legal|financial|general_web>"],
  "epistemic_prior": <float — opposing to the specialists' average prior>,
  "evidence_type_focus": ["<methodological_critique|statistical_reanalysis|alternative_explanation>"],
  "persona_description": "<1 sentence>"
}}

Intensity level: {intensity}.
- LOW: polite challenge
- MEDIUM: active rebuttal
- HIGH: adversarial cross-examination

Output ONLY valid JSON array.
"""


# ── Builder ────────────────────────────────────────────────────────────

class TopologyBuilder:
    """
    Layer 2 of ARISTOTLE — constructs the full agent ecosystem
    dynamically from a :class:`DebateFrame`.
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def build(self, frame: DebateFrame) -> TopologySpec:
        """Build a :class:`TopologySpec` from a completed :class:`DebateFrame`."""

        spec = TopologySpec(frame_id=frame.id)

        # 1. Specialist count
        n_specialists = _specialist_count(frame.controversy_score)

        # 2. Generate specialist personas via LLM
        spec.specialists = self._generate_specialists(frame, n_specialists)

        # 3. Refuter configuration
        spec.refuter_intensity = _refuter_intensity(frame.controversy_score)
        n_refuters = _refuter_count(frame.controversy_score)
        spec.refuters = self._generate_refuters(
            frame, n_refuters, spec.refuter_intensity, spec.specialists,
        )

        # 4. Jury selection
        arch, size, rationale = _JURY_MAP.get(
            frame.debate_type,
            (JuryArchitecture.BAYESIAN, 3, "Default Bayesian jury"),
        )
        # Override to validity-weighted for contested science
        if (
            frame.controversy_score >= 0.7
            and frame.debate_type in (DebateType.BINARY_FACTUAL, DebateType.CAUSAL)
        ):
            arch = JuryArchitecture.VALIDITY_WEIGHTED
            size = 5
            rationale = (
                "Contested science domain requires argument-quality weighting "
                "(Wu et al. 2025)"
            )
        spec.jury = JurySpec(architecture=arch, size=size, rationale=rationale)

        # 5. Round estimation
        rounds, reason = _estimate_rounds(frame)
        spec.estimated_rounds = rounds
        spec.round_reasoning = reason
        spec.max_rounds = 8
        spec.min_rounds = 1

        # 6. Cost estimation
        total_agents = spec.agent_count + spec.jury.size
        spec.estimated_tokens = (
            total_agents * spec.estimated_rounds * _TOKENS_PER_AGENT_PER_ROUND
        )
        spec.estimated_runtime_seconds = spec.estimated_rounds * 18.0
        spec.budget_cap_tokens = max(spec.estimated_tokens * 2, 100_000)

        logger.info(
            "Topology built: %d specialists, %d refuters, %s jury(%d), "
            "%d rounds, ~%d tokens",
            len(spec.specialists),
            len(spec.refuters),
            spec.jury.architecture.value,
            spec.jury.size,
            spec.estimated_rounds,
            spec.estimated_tokens,
        )
        return spec

    # ── persona generation ────────────────────────────────────────────

    def _generate_specialists(
        self, frame: DebateFrame, count: int,
    ) -> list[AgentSpec]:
        """Generate specialist agent specs via LLM."""
        prompt = (
            f"Debate frame:\n"
            f"  Query: {frame.raw_query}\n"
            f"  Domain: {frame.primary_domain}\n"
            f"  Secondary domains: {', '.join(frame.secondary_domains)}\n"
            f"  Debate type: {frame.debate_type.value}\n"
            f"  Controversy: {frame.controversy_score}\n"
            f"  Proposition: {frame.primary_proposition}\n"
            f"  Stance positions: {[s.label for s in frame.stance_positions]}\n"
        )
        system = _PERSONA_SYSTEM.format(count=count)
        response = self.llm.generate(
            prompt=prompt, system_prompt=system,
            temperature=0.5, max_tokens=16384,
        )
        logger.debug("Specialist persona raw LLM response (%d chars): %.500s",
                     len(response.content), response.content)
        try:
            raw = _extract_json_array(response.content)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Specialist persona parse failed; generating defaults — %s", exc)
            logger.info("Specialist persona raw content was: %.800s", response.content)
            raw = self._default_specialists(frame, count)

        agents: list[AgentSpec] = []
        for item in raw[:count]:
            agents.append(AgentSpec(
                name=item.get("name", f"Specialist-{len(agents)+1}"),
                role="specialist",
                domain_mandate=item.get("domain_mandate", frame.primary_domain),
                evidence_source_priority=item.get("evidence_source_priority", ["general_web"]),
                epistemic_prior=float(item.get("epistemic_prior", 0.5)),
                evidence_type_focus=item.get("evidence_type_focus", ["empirical"]),
                persona_description=item.get("persona_description", ""),
            ))

        # Ensure diversity: if prior spread < 0.20, adjust
        if len(agents) >= 2:
            priors = [a.epistemic_prior for a in agents]
            if max(priors) - min(priors) < 0.20:
                agents[0].epistemic_prior = max(0.15, min(priors) - 0.10)
                agents[-1].epistemic_prior = min(0.85, max(priors) + 0.10)

        return agents

    def _generate_refuters(
        self,
        frame: DebateFrame,
        count: int,
        intensity: RefuterIntensity,
        specialists: list[AgentSpec],
    ) -> list[AgentSpec]:
        """Generate refuter agent specs via LLM."""
        avg_prior = (
            sum(s.epistemic_prior for s in specialists) / max(len(specialists), 1)
        )
        prompt = (
            f"Debate frame:\n"
            f"  Query: {frame.raw_query}\n"
            f"  Domain: {frame.primary_domain}\n"
            f"  Proposition: {frame.primary_proposition}\n"
            f"  Average specialist prior: {avg_prior:.2f}\n"
        )
        system = _REFUTER_SYSTEM.format(count=count, intensity=intensity.value)
        response = self.llm.generate(
            prompt=prompt, system_prompt=system,
            temperature=0.5, max_tokens=16384,
        )
        logger.debug("Refuter persona raw LLM response (%d chars): %.500s",
                     len(response.content), response.content)
        try:
            raw = _extract_json_array(response.content)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Refuter persona parse failed; generating defaults — %s", exc)
            logger.info("Refuter persona raw content was: %.800s", response.content)
            raw = self._default_refuters(frame, count, avg_prior)

        agents: list[AgentSpec] = []
        for item in raw[:count]:
            agents.append(AgentSpec(
                name=item.get("name", f"Refuter-{len(agents)+1}"),
                role="refuter",
                domain_mandate=item.get("domain_mandate", "challenge methodology"),
                evidence_source_priority=item.get(
                    "evidence_source_priority", ["general_web"]
                ),
                epistemic_prior=float(item.get("epistemic_prior", 1.0 - avg_prior)),
                evidence_type_focus=item.get(
                    "evidence_type_focus", ["methodological_critique"]
                ),
                persona_description=item.get("persona_description", ""),
            ))
        return agents

    # ── fallback defaults (dynamic — draws from specialist_bank) ─────

    @staticmethod
    def _default_specialists(
        frame: DebateFrame, count: int,
    ) -> list[dict[str, Any]]:
        """Pull fallback specialists from the domain-aware specialist bank."""
        bank_templates = get_specialists_for_domain(
            frame.primary_domain, count,
        )
        specs: list[dict[str, Any]] = []
        for i in range(count):
            prior = round(0.35 + (i / max(count - 1, 1)) * 0.30, 2)
            if i < len(bank_templates):
                template = bank_templates[i]
                specs.append({
                    "name": template["name"],
                    "domain_mandate": template.get(
                        "domain_mandate", frame.primary_domain,
                    ),
                    "evidence_source_priority": template.get(
                        "evidence_source_priority", ["general_web"],
                    ),
                    "epistemic_prior": prior,
                    "evidence_type_focus": template.get(
                        "evidence_type_focus", ["empirical"],
                    ),
                    "persona_description": template.get(
                        "persona_description",
                        f"Domain expert #{i+1} on {frame.primary_domain}",
                    ),
                })
            else:
                # Exhaust bank → draw from backup_agents pool
                exclude = [s["name"] for s in specs]
                backup = get_backup_agent(exclude_names=exclude, prior=prior)
                specs.append({
                    "name": backup["name"],
                    "domain_mandate": backup.get(
                        "domain_mandate", frame.primary_domain,
                    ),
                    "evidence_source_priority": backup.get(
                        "evidence_source_priority", ["general_web"],
                    ),
                    "epistemic_prior": prior,
                    "evidence_type_focus": backup.get(
                        "evidence_type_focus", ["empirical"],
                    ),
                    "persona_description": backup.get(
                        "persona_description",
                        f"Domain expert #{i+1} on {frame.primary_domain}",
                    ),
                })
        return specs

    @staticmethod
    def _default_refuters(
        frame: DebateFrame, count: int, avg_prior: float,
    ) -> list[dict[str, Any]]:
        """Generate fallback refuters dynamically from the backup agent pool."""
        specs: list[dict[str, Any]] = []
        used_names: list[str] = []
        for i in range(count):
            backup = get_backup_agent(
                exclude_names=used_names,
                prior=round(1.0 - avg_prior, 2),
            )
            used_names.append(backup["name"])
            specs.append({
                "name": backup["name"],
                "domain_mandate": (
                    f"Challenge methodology for {frame.primary_domain}"
                ),
                "evidence_source_priority": backup.get(
                    "evidence_source_priority", ["general_web"],
                ),
                "epistemic_prior": round(1.0 - avg_prior, 2),
                "evidence_type_focus": ["methodological_critique"],
                "persona_description": backup.get(
                    "persona_description", f"Adversarial refuter #{i+1}",
                ),
            })
        return specs
