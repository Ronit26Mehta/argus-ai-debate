"""
Senate Generation Engine — AGORA's first and most foundational component.

Runs before a single deliberating agent is instantiated. Analyses the
proposition's domain, controversy profile, epistemic structure, and
required balance, then generates a Senate Composition Specification (SCS).

Pipeline:
    1. Domain Detection
    2. Controversy Scoring (3-axis)
    3. Evidence Landscape Assessment
    4. Stance Space Mapping
    5. N Calculation
    6. Balance Mandate Enforcement
    7. Senator Persona Generation
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

from argus.core.json_repair import (
    extract_json_array as _extract_json_array,
    repair_json as _try_repair_json,
)

from argus.agora.models import (
    AgoraSessionConfig,
    ControversyVector,
    DocketEvidenceType,
    EvidenceLandscape,
    SenateSpec,
    SenatorCategory,
    SenatorSpec,
    StancePosition,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── Tokens per senator per round (cost estimation) ────────────────────
_TOKENS_PER_SENATOR_PER_ROUND = 3000

# ══════════════════════════════════════════════════════════════════════
# Default senator templates (fallback when LLM generation fails)
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_SENATOR_BANK: dict[SenatorCategory, list[dict[str, str]]] = {
    SenatorCategory.DOMAIN_EXPERT: [
        {"name": "Dr. Amara Osei", "desc": "Primary domain authority with deep empirical expertise"},
        {"name": "Prof. Henrik Larsson", "desc": "Secondary domain specialist with quantitative focus"},
        {"name": "Dr. Mei-Ling Chen", "desc": "Cross-disciplinary expert bridging theory and practice"},
    ],
    SenatorCategory.ADVERSARIAL_CHALLENGER: [
        {"name": "Dr. Viktor Petrov", "desc": "Methodological critic challenging evidentiary standards"},
        {"name": "Prof. Diana Reyes", "desc": "Adversarial examiner probing inferential weaknesses"},
    ],
    SenatorCategory.SYNTHESIS_AGENT: [
        {"name": "Sen. Integra-1", "desc": "Evidence landscape integrator tracking full docket state"},
        {"name": "Sen. Integra-2", "desc": "Position-aware synthesiser mapping convergence patterns"},
    ],
    SenatorCategory.NORMATIVE_ANALYST: [
        {"name": "Prof. Anika Sharma", "desc": "Ethics and values analyst surfacing hidden assumptions"},
        {"name": "Dr. James Okonkwo", "desc": "Stakeholder impact assessor examining normative implications"},
    ],
    SenatorCategory.HISTORICAL_CONTEXTUALIST: [
        {"name": "Prof. Eleanor Whitfield", "desc": "Precedent analyst drawing from historical case study"},
        {"name": "Dr. Takeshi Mori", "desc": "Comparative historian examining cross-cultural parallels"},
    ],
    SenatorCategory.DEVILS_ADVOCATE: [
        {"name": "Sen. Contrarius", "desc": "Dynamic position holder defending the least-represented view"},
    ],
    SenatorCategory.EPISTEMIC_AUDITOR: [
        {"name": "Sen. Veritas-1", "desc": "Evidence governance officer monitoring logical fallacies and source quality"},
        {"name": "Sen. Veritas-2", "desc": "Methodological auditor checking confirmation bias patterns"},
    ],
    SenatorCategory.CROSS_DOMAIN_INTEGRATOR: [
        {"name": "Dr. Sofia Andersson", "desc": "Cross-domain bridge identifying inter-field implications"},
    ],
}

# ══════════════════════════════════════════════════════════════════════
# LLM Prompts
# ══════════════════════════════════════════════════════════════════════

_DOMAIN_SYSTEM = """\
You are AGORA's Senate Generation Engine. Given a proposition, identify:
1. The PRIMARY domain (one of: science, policy, economics, ethics, history, \
technology, law, social_science, medicine, environment, philosophy, general)
2. Any SECONDARY domains (up to 3)

Output ONLY valid JSON:
{"primary_domain": "...", "secondary_domains": ["...", "..."]}
"""

_CONTROVERSY_SYSTEM = """\
You are AGORA's Controversy Scorer. Given a proposition, score its controversy \
on three axes, each 0.0 to 1.0:
- empirical: is the factual record genuinely contested?
- normative: do reasonable people disagree on values?
- epistemic: is the evidence itself uncertain or incomplete?

Output ONLY valid JSON:
{"empirical": 0.0, "normative": 0.0, "epistemic": 0.0}
"""

_STANCE_SYSTEM = """\
You are AGORA's Stance Space Mapper. Given a proposition, identify all \
legitimate positions a reasonable agent could hold. Not just binary for/against — \
include partial, conditional, and abstention positions.

Output ONLY a valid JSON array:
[{"label": "...", "description": "...", "estimated_support": 0.0}, ...]

Generate between 3 and 7 positions.
"""

_SENATOR_SYSTEM = """\
You are AGORA's Senate Generation Engine. Generate senator specifications for \
a deliberative assembly. Each senator must have:

{{
  "name": "<realistic academic/professional name>",
  "category": "<one of: domain_expert, adversarial_challenger, synthesis_agent, \
normative_analyst, historical_contextualist, devils_advocate, epistemic_auditor, \
cross_domain_integrator>",
  "domain_expertise": "<specific expertise area>",
  "prior_position": <float 0.15–0.85>,
  "evidence_gathering_mandate": "<what evidence this senator is tasked to find>",
  "evidence_sources": ["<source_type>"],
  "deliberative_temperament": "<aggressive|measured|cautious>",
  "persona_description": "<1 sentence role description>"
}}

Rules:
- Generate EXACTLY {count} senators
- Follow category distribution strictly: {distribution}
- Priors MUST span a range ensuring meaningful disagreement (min spread >= 0.25)
- At least one senator should start skeptical (prior < 0.40)
- At least one senator should start supportive (prior > 0.60)
- Exactly ONE devils_advocate
- Evidence sources should vary across senators
- Output ONLY valid JSON array, no extra text.
"""


# ══════════════════════════════════════════════════════════════════════
# Senate Generation Engine
# ══════════════════════════════════════════════════════════════════════

class SenateGenerator:
    """
    Generates a Senate Composition Specification for a given proposition.

    The Senate is dynamically calibrated per proposition — no two
    propositions produce the same Senate.
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def generate(
        self,
        proposition: str,
        config: AgoraSessionConfig | None = None,
    ) -> SenateSpec:
        """Generate a full SenateSpec for the given proposition.

        Args:
            proposition: The proposition to deliberate.
            config: Optional session configuration overrides.

        Returns:
            A complete SenateSpec ready for session launch.
        """
        config = config or AgoraSessionConfig()
        spec = SenateSpec(proposition=proposition)

        # Step 1: Domain Detection
        spec.primary_domain, spec.secondary_domains = self._detect_domain(proposition)
        logger.info("Domain: %s (secondary: %s)", spec.primary_domain, spec.secondary_domains)

        # Step 2: Controversy Scoring
        spec.controversy = self._score_controversy(proposition)
        logger.info("Controversy: %s", spec.controversy.to_dict())

        # Step 3: Evidence Landscape Assessment
        spec.evidence_landscape = self._assess_evidence_landscape(
            proposition, spec.primary_domain, spec.controversy,
        )

        # Step 4: Stance Space Mapping
        spec.stance_space = self._map_stance_space(proposition)
        logger.info("Stance space: %d positions", len(spec.stance_space))

        # Step 5: N Calculation
        n, reasoning = self._calculate_n(
            spec.controversy, spec.evidence_landscape,
            spec.stance_space, config,
        )
        spec.n_calculation_reasoning = reasoning
        logger.info("Senate size N=%d (%s)", n, reasoning)

        # Step 6 + 7: Generate senators with Balance Mandate
        category_dist = self._compute_category_distribution(
            n, spec.controversy, spec.evidence_landscape,
            len(spec.secondary_domains) > 0,
        )
        spec.senators = self._generate_senators(
            proposition, n, category_dist,
            spec.primary_domain, spec.secondary_domains,
            spec.controversy,
        )

        # Cost estimation
        rounds_estimate = config.max_rounds
        spec.estimated_tokens = n * rounds_estimate * _TOKENS_PER_SENATOR_PER_ROUND
        spec.estimated_runtime_seconds = rounds_estimate * n * 8.0

        logger.info(
            "Senate generated: %d senators, ~%d tokens, ~%.0fs",
            spec.senate_size, spec.estimated_tokens, spec.estimated_runtime_seconds,
        )
        return spec

    # ── Step 1: Domain Detection ──────────────────────────────────────

    def _detect_domain(self, proposition: str) -> tuple[str, list[str]]:
        """Detect primary and secondary domains via LLM."""
        try:
            response = self.llm.generate(
                prompt=f"Proposition: {proposition}",
                system_prompt=_DOMAIN_SYSTEM,
                temperature=0.3, max_tokens=512,
            )
            text = response.content.strip()
            print(f"\n[DEBUG] Raw Domain output: {repr(text)}\n")
            # Try to parse JSON
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                primary = data.get("primary_domain", "general")
                secondary = data.get("secondary_domains", [])
                return primary, secondary[:3]
        except Exception as exc:
            logger.warning("Domain detection LLM failed: %s", exc)
        return "general", []

    # ── Step 2: Controversy Scoring ───────────────────────────────────

    def _score_controversy(self, proposition: str) -> ControversyVector:
        """Score controversy on three axes via LLM."""
        try:
            response = self.llm.generate(
                prompt=f"Proposition: {proposition}",
                system_prompt=_CONTROVERSY_SYSTEM,
                temperature=0.3, max_tokens=512,
            )
            text = response.content.strip()
            print(f"\n[DEBUG] Raw Controversy output: {repr(text)}\n")
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                return ControversyVector(
                    empirical=max(0.0, min(1.0, float(data.get("empirical", 0.5)))),
                    normative=max(0.0, min(1.0, float(data.get("normative", 0.5)))),
                    epistemic=max(0.0, min(1.0, float(data.get("epistemic", 0.5)))),
                )
        except Exception as exc:
            logger.warning("Controversy scoring LLM failed: %s", exc)
        return ControversyVector()

    # ── Step 3: Evidence Landscape ────────────────────────────────────

    def _assess_evidence_landscape(
        self,
        proposition: str,
        domain: str,
        controversy: ControversyVector,
    ) -> EvidenceLandscape:
        """Assess evidence landscape based on domain and controversy."""
        # Heuristic — could be enhanced with LLM
        all_types = list(DocketEvidenceType)

        # Select relevant types based on domain
        domain_type_map: dict[str, list[DocketEvidenceType]] = {
            "science": [DocketEvidenceType.QUANTITATIVE, DocketEvidenceType.EXPERIMENTAL, DocketEvidenceType.THEORETICAL],
            "medicine": [DocketEvidenceType.QUANTITATIVE, DocketEvidenceType.EXPERIMENTAL, DocketEvidenceType.QUALITATIVE],
            "policy": [DocketEvidenceType.QUALITATIVE, DocketEvidenceType.QUANTITATIVE, DocketEvidenceType.HISTORICAL],
            "economics": [DocketEvidenceType.QUANTITATIVE, DocketEvidenceType.THEORETICAL, DocketEvidenceType.HISTORICAL],
            "ethics": [DocketEvidenceType.QUALITATIVE, DocketEvidenceType.THEORETICAL, DocketEvidenceType.ANECDOTAL],
            "history": [DocketEvidenceType.HISTORICAL, DocketEvidenceType.QUALITATIVE, DocketEvidenceType.ANECDOTAL],
            "law": [DocketEvidenceType.LEGAL, DocketEvidenceType.HISTORICAL, DocketEvidenceType.QUALITATIVE],
            "technology": [DocketEvidenceType.QUANTITATIVE, DocketEvidenceType.EXPERIMENTAL, DocketEvidenceType.THEORETICAL],
        }
        selected = domain_type_map.get(domain, all_types[:4])

        # Density from controversy aggregate
        agg = controversy.aggregate
        if agg < 0.3:
            density, score = "sparse", 0.3
        elif agg < 0.6:
            density, score = "moderate", 0.5
        else:
            density, score = "rich", 0.8

        return EvidenceLandscape(
            available_types=selected,
            density=density,
            density_score=score,
        )

    # ── Step 4: Stance Space Mapping ──────────────────────────────────

    def _map_stance_space(self, proposition: str) -> list[StancePosition]:
        """Map the full stance space via LLM."""
        try:
            response = self.llm.generate(
                prompt=f"Proposition: {proposition}",
                system_prompt=_STANCE_SYSTEM,
                temperature=0.5, max_tokens=1024,
            )
            print(f"\n[DEBUG] Raw Stance space output: {repr(response.content)}\n")
            raw = _extract_json_array(response.content)
            positions = []
            for item in raw[:7]:
                positions.append(StancePosition(
                    label=item.get("label", "Unknown"),
                    description=item.get("description", ""),
                    estimated_support=float(item.get("estimated_support", 0.5)),
                ))
            if len(positions) >= 2:
                return positions
        except Exception as exc:
            logger.warning("Stance space LLM failed: %s", exc)

        # Fallback: binary + qualified
        return [
            StancePosition(label="Strongly Support", description="Full agreement with the proposition", estimated_support=0.3),
            StancePosition(label="Qualified Support", description="Agreement with caveats or conditions", estimated_support=0.25),
            StancePosition(label="Neutral / Undecided", description="Insufficient evidence to take a position", estimated_support=0.15),
            StancePosition(label="Qualified Opposition", description="Disagreement with caveats", estimated_support=0.15),
            StancePosition(label="Strongly Oppose", description="Full disagreement with the proposition", estimated_support=0.15),
        ]

    # ── Step 5: N Calculation ─────────────────────────────────────────

    def _calculate_n(
        self,
        controversy: ControversyVector,
        evidence: EvidenceLandscape,
        stances: list[StancePosition],
        config: AgoraSessionConfig,
    ) -> tuple[int, str]:
        """Calculate the optimal number of senators N.

        Formula:
            base = 7
            + controversy bonus (0–8 based on aggregate)
            + stance breadth bonus (0–4 based on number of positions)
            + evidence density bonus (0–3)
            Clamped to [config.min_senators, config.max_senators]
        """
        base = 7
        controversy_bonus = int(controversy.aggregate * 8)
        stance_bonus = min(len(stances) - 2, 4) if len(stances) > 2 else 0
        density_map = {"sparse": 0, "moderate": 1, "rich": 3}
        density_bonus = density_map.get(evidence.density, 1)

        n = base + controversy_bonus + stance_bonus + density_bonus
        n = max(config.min_senators, min(n, config.max_senators))

        reasoning = (
            f"base {base} + controversy {controversy_bonus} "
            f"+ stances {stance_bonus} + evidence density {density_bonus} "
            f"= {n} (clamped to [{config.min_senators}, {config.max_senators}])"
        )
        return n, reasoning

    # ── Step 6: Category Distribution (Balance Mandate) ───────────────

    def _compute_category_distribution(
        self,
        n: int,
        controversy: ControversyVector,
        evidence: EvidenceLandscape,
        is_multi_domain: bool,
    ) -> dict[SenatorCategory, int]:
        """Compute category distribution enforcing the Balance Mandate.

        Constraints:
            - No single category > 35% of seats
            - SA + EA together >= 20% of seats
            - Exactly 1 DA
            - At least 1 DE, 1 AC, 1 SA, 1 EA
            - CDI only if multi-domain
            - NA only if normative controversy > 0.4
            - HC only if historical evidence available
        """
        max_per_cat = max(1, int(n * 0.35))
        min_sa_ea = max(2, int(math.ceil(n * 0.20)))

        dist: dict[SenatorCategory, int] = {}

        # Fixed: 1 DA always
        dist[SenatorCategory.DEVILS_ADVOCATE] = 1
        remaining = n - 1

        # SA and EA: enforce >= 20% combined
        sa_count = max(1, min_sa_ea // 2)
        ea_count = max(1, min_sa_ea - sa_count)
        dist[SenatorCategory.SYNTHESIS_AGENT] = sa_count
        dist[SenatorCategory.EPISTEMIC_AUDITOR] = ea_count
        remaining -= (sa_count + ea_count)

        # AC: at least 1
        ac_count = max(1, min(2, remaining // 3))
        dist[SenatorCategory.ADVERSARIAL_CHALLENGER] = ac_count
        remaining -= ac_count

        # Optional categories
        if is_multi_domain and remaining >= 1:
            dist[SenatorCategory.CROSS_DOMAIN_INTEGRATOR] = 1
            remaining -= 1

        if controversy.normative > 0.4 and remaining >= 1:
            na_count = min(2, remaining)
            dist[SenatorCategory.NORMATIVE_ANALYST] = na_count
            remaining -= na_count

        has_historical = DocketEvidenceType.HISTORICAL in evidence.available_types
        if has_historical and remaining >= 1:
            dist[SenatorCategory.HISTORICAL_CONTEXTUALIST] = 1
            remaining -= 1

        # All remaining go to Domain Expert (primary contributors)
        dist[SenatorCategory.DOMAIN_EXPERT] = max(1, remaining)

        # Enforce max_per_cat
        for cat in dist:
            if cat != SenatorCategory.DEVILS_ADVOCATE:
                dist[cat] = min(dist[cat], max_per_cat)

        # Verify total == n, adjust DE
        total = sum(dist.values())
        if total < n:
            dist[SenatorCategory.DOMAIN_EXPERT] += (n - total)
        elif total > n:
            excess = total - n
            for cat in [SenatorCategory.DOMAIN_EXPERT, SenatorCategory.ADVERSARIAL_CHALLENGER]:
                if excess <= 0:
                    break
                reduction = min(excess, dist.get(cat, 0) - 1)
                if reduction > 0:
                    dist[cat] -= reduction
                    excess -= reduction

        return dist

    # ── Step 7: Senator Generation ────────────────────────────────────

    def _generate_senators(
        self,
        proposition: str,
        n: int,
        distribution: dict[SenatorCategory, int],
        primary_domain: str,
        secondary_domains: list[str],
        controversy: ControversyVector,
    ) -> list[SenatorSpec]:
        """Generate senator personas via LLM with fallback defaults."""
        dist_str = ", ".join(
            f"{cat.abbreviation}={count}" for cat, count in distribution.items()
        )

        try:
            prompt = (
                f"Proposition: {proposition}\n"
                f"Primary domain: {primary_domain}\n"
                f"Secondary domains: {', '.join(secondary_domains)}\n"
                f"Controversy: empirical={controversy.empirical:.2f}, "
                f"normative={controversy.normative:.2f}, "
                f"epistemic={controversy.epistemic:.2f}\n"
            )
            system = _SENATOR_SYSTEM.format(count=n, distribution=dist_str)
            response = self.llm.generate(
                prompt=prompt, system_prompt=system,
                temperature=0.5, max_tokens=2048,
            )
            logger.debug("Senator generation LLM response (%d chars)", len(response.content))
            print(f"\n[DEBUG] Raw Senator gen output: {repr(response.content)}\n")

            raw = _extract_json_array(response.content)
            senators = self._parse_senator_list(raw, n, distribution)
            if len(senators) >= n * 0.7:
                # Fill missing slots with defaults
                senators = self._fill_missing_senators(senators, n, distribution, primary_domain)
                return senators[:n]

        except Exception as exc:
            logger.warning("Senator generation LLM failed, using defaults: %s", exc)

        # Full fallback
        return self._default_senators(n, distribution, primary_domain)

    def _parse_senator_list(
        self,
        raw: list[dict[str, Any]],
        n: int,
        distribution: dict[SenatorCategory, int],
    ) -> list[SenatorSpec]:
        """Parse LLM-generated senator list into SenatorSpec objects."""
        senators: list[SenatorSpec] = []
        for item in raw[:n]:
            cat_str = item.get("category", "domain_expert")
            try:
                category = SenatorCategory(cat_str)
            except ValueError:
                category = SenatorCategory.DOMAIN_EXPERT

            senators.append(SenatorSpec(
                name=item.get("name", f"Senator-{len(senators)+1}"),
                category=category,
                domain_expertise=item.get("domain_expertise", ""),
                prior_position=max(0.15, min(0.85, float(item.get("prior_position", 0.5)))),
                evidence_gathering_mandate=item.get("evidence_gathering_mandate", ""),
                evidence_sources=item.get("evidence_sources", ["general_web"]),
                deliberative_temperament=item.get("deliberative_temperament", "measured"),
                persona_description=item.get("persona_description", ""),
            ))

        # Ensure prior diversity
        if len(senators) >= 2:
            priors = [s.prior_position for s in senators]
            if max(priors) - min(priors) < 0.25:
                senators[0].prior_position = max(0.15, min(priors) - 0.15)
                senators[-1].prior_position = min(0.85, max(priors) + 0.15)

        return senators

    def _fill_missing_senators(
        self,
        senators: list[SenatorSpec],
        n: int,
        distribution: dict[SenatorCategory, int],
        domain: str,
    ) -> list[SenatorSpec]:
        """Fill missing category slots with default senators."""
        current_dist: dict[SenatorCategory, int] = {}
        for s in senators:
            current_dist[s.category] = current_dist.get(s.category, 0) + 1

        used_names = {s.name for s in senators}

        for cat, required in distribution.items():
            current = current_dist.get(cat, 0)
            needed = required - current
            if needed <= 0:
                continue

            templates = _DEFAULT_SENATOR_BANK.get(cat, [])
            for i in range(needed):
                if i < len(templates):
                    template = templates[i]
                    name = template["name"]
                    if name in used_names:
                        name = f"{name} II"
                    used_names.add(name)
                    senators.append(SenatorSpec(
                        name=name,
                        category=cat,
                        domain_expertise=domain,
                        prior_position=round(0.3 + (i / max(needed - 1, 1)) * 0.4, 2),
                        evidence_gathering_mandate=f"Gather {cat.abbreviation}-relevant evidence for {domain}",
                        evidence_sources=["general_web"],
                        deliberative_temperament="measured",
                        persona_description=template["desc"],
                    ))
                else:
                    senators.append(SenatorSpec(
                        name=f"{cat.abbreviation}-{len(senators)+1}",
                        category=cat,
                        domain_expertise=domain,
                        prior_position=0.5,
                        evidence_gathering_mandate=f"{cat.display_name} on {domain}",
                        evidence_sources=["general_web"],
                        deliberative_temperament="measured",
                        persona_description=f"Auto-generated {cat.display_name}",
                    ))
        return senators

    def _default_senators(
        self,
        n: int,
        distribution: dict[SenatorCategory, int],
        domain: str,
    ) -> list[SenatorSpec]:
        """Generate fully default senators from the template bank."""
        senators: list[SenatorSpec] = []
        used_names: set[str] = set()

        for cat, count in distribution.items():
            templates = _DEFAULT_SENATOR_BANK.get(cat, [])
            for i in range(count):
                prior = round(0.25 + (i / max(count - 1, 1)) * 0.50, 2)
                if i < len(templates):
                    name = templates[i]["name"]
                    desc = templates[i]["desc"]
                else:
                    name = f"{cat.abbreviation}-{i+1}"
                    desc = f"Auto-generated {cat.display_name} #{i+1}"

                if name in used_names:
                    name = f"{name} II"
                used_names.add(name)

                senators.append(SenatorSpec(
                    name=name,
                    category=cat,
                    domain_expertise=domain,
                    prior_position=prior,
                    evidence_gathering_mandate=f"Gather evidence as {cat.display_name} for {domain}",
                    evidence_sources=["general_web"],
                    deliberative_temperament="measured",
                    persona_description=desc,
                ))

        return senators[:n]
