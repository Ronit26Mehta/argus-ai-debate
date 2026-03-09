"""
Backup agents for mid-debate replacement.

When :mod:`monitor.py` detects an agent performing below threshold
(Evidence Drought), it replaces that agent with a backup from this bank.
Backup agents are pre-configured with a briefing mechanism so they can
join the debate mid-stream with full context of prior rounds.
"""

from __future__ import annotations

from typing import Any

# Pool of backup agents keyed by broad evidence focus
BACKUP_AGENTS: list[dict[str, Any]] = [
    {
        "name": "Dr. Yuki Tanaka",
        "domain_mandate": "Systematic Reviews and Evidence Synthesis",
        "evidence_source_priority": ["PubMed", "Cochrane"],
        "evidence_type_focus": ["meta_analysis", "empirical"],
        "persona_description": "Systematic review specialist who synthesises large evidence bodies",
    },
    {
        "name": "Prof. Anil Deshmukh",
        "domain_mandate": "Statistical Methodology and Replication",
        "evidence_source_priority": ["arXiv", "PubMed"],
        "evidence_type_focus": ["computational", "empirical"],
        "persona_description": "Biostatistician focused on p-curve analysis and replication",
    },
    {
        "name": "Dr. Clara Johansson",
        "domain_mandate": "Cross-Cultural Comparative Evidence",
        "evidence_source_priority": ["general_web", "JSTOR"],
        "evidence_type_focus": ["empirical", "case_report"],
        "persona_description": "Cross-cultural researcher comparing outcomes across populations",
    },
    {
        "name": "Prof. David Okonkwo",
        "domain_mandate": "Policy Impact Assessment",
        "evidence_source_priority": ["NBER", "general_web"],
        "evidence_type_focus": ["empirical", "expert_consensus"],
        "persona_description": "Policy analyst evaluating real-world intervention outcomes",
    },
    {
        "name": "Dr. Maria Santos",
        "domain_mandate": "Qualitative Evidence and Case Studies",
        "evidence_source_priority": ["general_web", "PubMed"],
        "evidence_type_focus": ["case_report", "expert_consensus"],
        "persona_description": "Qualitative researcher adding depth to quantitative evidence",
    },
    {
        "name": "Prof. Isaac Park",
        "domain_mandate": "Historical and Longitudinal Data Patterns",
        "evidence_source_priority": ["general_web", "arXiv"],
        "evidence_type_focus": ["empirical"],
        "persona_description": "Historian of science and long-run trend analyst",
    },
]


def get_backup_agent(
    exclude_names: list[str] | None = None,
    prior: float = 0.5,
) -> dict[str, Any]:
    """
    Return the next available backup agent spec, skipping any whose name
    appears in *exclude_names*.  The ``epistemic_prior`` is set to *prior*.
    """
    exclude = set(exclude_names or [])
    for agent in BACKUP_AGENTS:
        if agent["name"] not in exclude:
            return {**agent, "epistemic_prior": prior}
    # If all are exhausted, recycle the first with a modified name
    fallback = {**BACKUP_AGENTS[0], "epistemic_prior": prior}
    fallback["name"] = fallback["name"] + " (II)"
    return fallback
