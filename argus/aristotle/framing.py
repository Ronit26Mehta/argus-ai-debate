"""
Layer 1 — Intent Parsing & Framing Engine.

Activated when the user submits a message.  Performs seven analytical
operations using a lightweight fast-inference model:

    (a) Domain classification
    (b) Debate type detection
    (c) Controversy level scoring
    (d) Evidence density estimation
    (e) Stance space mapping
    (f) Claim decomposition
    (g) Prior probability calibration
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from argus.core.json_repair import (
    extract_json_object as _extract_json,
    repair_json as _try_repair_json,
)

from argus.aristotle.models import (
    DebateFrame,
    DebateType,
    LiteratureProbe,
    StancePosition,
    SubClaim,
)

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── Hedging language indicators ────────────────────────────────────────

_HEDGING_TOKENS = {
    "actually", "really", "overhyped", "just a", "or is it", "truly",
    "supposedly", "allegedly", "myth", "hoax", "debunked", "overblown",
    "exaggerated", "moral panic", "hype", "controversial", "disputed",
}

# ── Domain priors (controversy base-rate per domain) ──────────────────

_DOMAIN_CONTROVERSY_PRIORS: dict[str, float] = {
    "nutrition": 0.70,
    "epidemiology": 0.55,
    "psychology": 0.60,
    "climate_science": 0.50,
    "economics": 0.65,
    "medicine": 0.45,
    "physics": 0.15,
    "mathematics": 0.10,
    "computer_science": 0.30,
    "sociology": 0.65,
    "political_science": 0.75,
    "philosophy": 0.70,
    "law": 0.55,
    "education": 0.55,
    "general": 0.50,
}

# ── Structured LLM prompt for the seven framing operations ────────────

_FRAMING_SYSTEM_PROMPT = """\
You are ARISTOTLE, an expert debate framing engine.  Given a user's
natural-language question you MUST return a single JSON object (no markdown
fences) with EXACTLY these keys:

{
  "primary_domain": "<string — most relevant knowledge domain>",
  "secondary_domains": ["<string>", ...],
  "debate_type": "<binary_factual|causal|comparative|predictive|normative|definitional>",
  "controversy_signals": ["<string — phrases signalling controversy>", ...],
  "evidence_density": "<sparse|moderate|rich>",
  "evidence_density_score": <float 0-1>,
  "stance_positions": [
      {"label": "<short label>", "description": "<1-2 sentences>", "estimated_support": <float 0-1>}
  ],
  "sub_claims": [
      {"text": "<atomic sub-proposition>", "is_primary": <bool>}
  ],
  "primary_proposition": "<the single falsifiable proposition under debate>",
  "framing_narrative": "<3-5 sentence plain-English summary of how you understood the query>"
}

Guidelines:
- Identify at least 2 stance positions.
- Decompose into at least 1 sub-claim (the primary proposition itself counts).
- The primary_proposition must be falsifiable and specific.
- Output ONLY valid JSON, no extra text.
"""

_LITERATURE_PROBE_PROMPT = """\
You are a rapid literature scanner.  For the claim below, imagine you have
access to the last 5 years of published research.  Estimate the following
and reply with ONLY a JSON object:

Claim: {proposition}

{{
  "num_support": <int — studies that support the claim>,
  "num_against": <int — studies that challenge the claim>,
  "num_partial": <int — studies with qualified/partial support>,
  "total_scanned": <int — total considered>,
  "summaries": ["<1-line summary of key finding>", ...],
  "direction_confidence": <float 0-1 — how confident the literature leans one way>
}}
"""


# ── Controversy scorer ─────────────────────────────────────────────────

def _score_controversy(
    query: str,
    domain: str,
    llm_signals: list[str],
) -> float:
    """
    Compute a 0.0-1.0 controversy score from:
      - hedging language in query
      - domain-specific prior
      - LLM-identified controversy signals
    """
    query_lower = query.lower()

    # hedging contribution
    hedge_hits = sum(1 for t in _HEDGING_TOKENS if t in query_lower)
    hedge_score = min(hedge_hits * 0.15, 0.4)

    # domain prior
    domain_prior = _DOMAIN_CONTROVERSY_PRIORS.get(
        domain.lower().replace(" ", "_"),
        _DOMAIN_CONTROVERSY_PRIORS["general"],
    )

    # LLM signal contribution
    signal_score = min(len(llm_signals) * 0.10, 0.3)

    raw = 0.30 * domain_prior + 0.35 * hedge_score + 0.35 * signal_score
    # Normalise to [0, 1] with a floor from the domain prior
    score = max(min(raw + domain_prior * 0.3, 1.0), 0.1)
    return round(score, 2)


# ── Prior calibration ──────────────────────────────────────────────────

def _calibrate_prior(
    probe: LiteratureProbe,
    controversy: float,
    query: str,
) -> float:
    """
    Set the Bayesian prior using:
      - literature probe direction
      - controversy level
      - uncertainty language in the query
    """
    if probe.total_scanned == 0:
        return 0.50

    support_ratio = probe.num_support / max(probe.total_scanned, 1)
    against_ratio = probe.num_against / max(probe.total_scanned, 1)

    # Base from literature direction
    base = 0.50 + 0.30 * (support_ratio - against_ratio)

    # Controversy pulls toward 0.50
    base = base * (1 - controversy * 0.3) + 0.50 * (controversy * 0.3)

    # Uncertainty language in query pulls toward 0.50
    query_lower = query.lower()
    uncertainty_tokens = {"maybe", "perhaps", "unclear", "debatable", "uncertain"}
    if any(t in query_lower for t in uncertainty_tokens):
        base = base * 0.8 + 0.50 * 0.2

    return round(max(min(base, 0.85), 0.15), 2)


# ── Engine ─────────────────────────────────────────────────────────────

class FramingEngine:
    """
    Layer 1 of ARISTOTLE — parses a natural-language query into a
    :class:`DebateFrame` driving the rest of the pipeline.
    """

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def frame(self, query: str) -> DebateFrame:
        """
        Execute all seven framing operations.

        Returns a fully-populated :class:`DebateFrame`.
        """
        # Step 1 — main structured analysis via LLM
        analysis = self._run_structured_analysis(query)

        # Step 2 — literature probe
        proposition = analysis.get("primary_proposition", query)
        probe = self._run_literature_probe(proposition)

        # Step 3 — controversy scoring
        domain = analysis.get("primary_domain", "general")
        signals = analysis.get("controversy_signals", [])
        controversy = _score_controversy(query, domain, signals)

        # Step 4 — prior calibration
        prior = _calibrate_prior(probe, controversy, query)

        # Build DebateFrame
        frame = DebateFrame(
            raw_query=query,
            primary_domain=domain,
            secondary_domains=analysis.get("secondary_domains", []),
            debate_type=self._parse_debate_type(
                analysis.get("debate_type", "binary_factual")
            ),
            controversy_score=controversy,
            controversy_signals=signals,
            evidence_density=analysis.get("evidence_density", "moderate"),
            evidence_density_score=float(
                analysis.get("evidence_density_score", 0.5)
            ),
            stance_positions=[
                StancePosition(
                    label=s.get("label", ""),
                    description=s.get("description", ""),
                    estimated_support=float(s.get("estimated_support", 0.5)),
                )
                for s in analysis.get("stance_positions", [])
            ],
            sub_claims=[
                SubClaim(
                    text=c.get("text", ""),
                    is_primary=bool(c.get("is_primary", False)),
                )
                for c in analysis.get("sub_claims", [])
            ],
            primary_proposition=proposition,
            prior_probability=prior,
            literature_probe=probe,
            framing_narrative=analysis.get("framing_narrative", ""),
        )

        # Guarantee at least one primary sub-claim
        if not any(sc.is_primary for sc in frame.sub_claims):
            frame.sub_claims.insert(
                0,
                SubClaim(text=frame.primary_proposition, is_primary=True),
            )

        logger.info(
            "Framing complete: type=%s  controversy=%.2f  prior=%.2f",
            frame.debate_type.value,
            frame.controversy_score,
            frame.prior_probability,
        )
        return frame

    # ── internal helpers ──────────────────────────────────────────────

    def _run_structured_analysis(self, query: str) -> dict[str, Any]:
        """Call the LLM for the main structured framing analysis."""
        response = self.llm.generate(
            prompt=f"User question:\n{query}",
            system_prompt=_FRAMING_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=16384,
        )
        logger.debug("Framing raw LLM response (%d chars): %.500s",
                     len(response.content), response.content)
        try:
            return _extract_json(response.content)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse framing JSON; using defaults — %s", exc)
            logger.info("Framing raw content was: %.800s", response.content)
            return {
                "primary_domain": "general",
                "secondary_domains": [],
                "debate_type": "binary_factual",
                "controversy_signals": [],
                "evidence_density": "moderate",
                "evidence_density_score": 0.5,
                "stance_positions": [
                    {"label": "Support", "description": "The claim is supported.", "estimated_support": 0.5},
                    {"label": "Against", "description": "The claim is not supported.", "estimated_support": 0.5},
                ],
                "sub_claims": [{"text": query, "is_primary": True}],
                "primary_proposition": query,
                "framing_narrative": f'Analysing the query: "{query}"',
            }

    def _run_literature_probe(self, proposition: str) -> LiteratureProbe:
        """Rapid n=10 literature scan via LLM."""
        prompt = _LITERATURE_PROBE_PROMPT.format(proposition=proposition)
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a research literature scanner.",
            temperature=0.2,
            max_tokens=16384,
        )
        logger.debug("Literature probe raw LLM response (%d chars): %.500s",
                     len(response.content), response.content)
        try:
            data = _extract_json(response.content)
            return LiteratureProbe(
                num_support=int(data.get("num_support", 0)),
                num_against=int(data.get("num_against", 0)),
                num_partial=int(data.get("num_partial", 0)),
                total_scanned=int(data.get("total_scanned", 0)),
                summaries=data.get("summaries", []),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Literature probe parse failed; using defaults — %s", exc)
            logger.info("Literature probe raw content was: %.800s", response.content)
            return LiteratureProbe()

    @staticmethod
    def _parse_debate_type(raw: str) -> DebateType:
        """Resolve a raw string to a DebateType enum."""
        mapping = {v.value: v for v in DebateType}
        normalised = raw.strip().lower().replace(" ", "_")
        return mapping.get(normalised, DebateType.BINARY_FACTUAL)
