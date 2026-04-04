"""
Base VAPT agent class for fsociety.

Extends ARGUS BaseAgent with VAPT-specific analysis utilities
and common system prompt patterns.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import Optional, Any, TYPE_CHECKING
from enum import Enum

from argus.agents.base import BaseAgent, AgentConfig, AgentRole, AgentResponse

if TYPE_CHECKING:
    from argus.core.llm.base import BaseLLM
    from argus.cdag.graph import CDAG
    from fsociety.vkg import VulnerabilityKnowledgeGraph

logger = logging.getLogger(__name__)


class AgentTier(str, Enum):
    """Agent tier in the fsociety hierarchy."""
    CORE = "tier1_core"
    SPECIALIST = "tier2_specialist"
    OUTPUT = "tier3_output"


class VAPTAgentConfig(AgentConfig):
    """Configuration for VAPT agents."""
    persona_name: str = "Unknown"
    vapt_domain: str = "General"
    tier: AgentTier = AgentTier.CORE


class VAPTAgent(BaseAgent):
    """
    Base class for all fsociety VAPT agents.

    Each agent has a distinct persona (Mr. Robot character),
    a VAPT domain of expertise, and a tier in the debate lifecycle.
    """

    # Subclasses MUST override these
    PERSONA_NAME: str = "Unknown"
    PERSONA_QUOTE: str = ""
    VAPT_DOMAIN: str = "General"
    TIER: AgentTier = AgentTier.CORE
    RDC_ROLE: str = "Specialist"

    def __init__(
        self,
        llm: "BaseLLM",
        config: Optional[VAPTAgentConfig] = None,
    ):
        cfg = config or VAPTAgentConfig(
            persona_name=self.PERSONA_NAME,
            vapt_domain=self.VAPT_DOMAIN,
            tier=self.TIER,
            role=AgentRole.SPECIALIST,
        )
        super().__init__(llm=llm, config=cfg, name=self.PERSONA_NAME)
        self.vapt_domain = self.VAPT_DOMAIN
        self.tier = self.TIER

    @abstractmethod
    def get_vapt_system_prompt(self) -> str:
        """Get the VAPT-specific system prompt for this agent."""
        ...

    def get_system_prompt(self) -> str:
        """Build full system prompt combining persona + VAPT domain."""
        return (
            f"You are {self.PERSONA_NAME}, a VAPT specialist agent in the fsociety system.\n"
            f"Domain: {self.VAPT_DOMAIN}\n"
            f"Role in RDC: {self.RDC_ROLE}\n"
            f'"{self.PERSONA_QUOTE}"\n\n'
            f"{self.get_vapt_system_prompt()}"
        )

    def act(self, graph: "CDAG", context: dict[str, Any]) -> AgentResponse:
        """Perform the agent's main action on the VKG."""
        system_prompt = self.get_system_prompt()
        prompt = self._build_analysis_prompt(graph, context)

        response = self.generate(prompt, system_prompt=system_prompt)

        self.log_action(
            action=f"{self.PERSONA_NAME}_analysis",
            details={"domain": self.VAPT_DOMAIN, "response_len": len(response)},
        )

        return AgentResponse(
            success=True,
            content=response,
            data={
                "agent_name": self.PERSONA_NAME,
                "tier": self.TIER.value,
                "domain": self.VAPT_DOMAIN,
            },
        )

    def _build_analysis_prompt(self, graph: "CDAG", context: dict[str, Any]) -> str:
        """Build the analysis prompt from context."""
        parts = [f"Analyze the following target from the perspective of {self.VAPT_DOMAIN}.\n"]

        if "code_chunks" in context:
            parts.append(f"Code chunks to analyze:\n{context['code_chunks'][:3000]}\n")
        if "dependencies" in context:
            parts.append(f"Dependencies:\n{context['dependencies'][:1000]}\n")
        if "git_insights" in context:
            parts.append(f"Git insights:\n{context['git_insights'][:500]}\n")
        if "previous_findings" in context:
            parts.append(f"Previous agent findings:\n{context['previous_findings'][:1500]}\n")

        parts.append(
            "\nList each security finding as a bullet point in this exact format:\n"
            "- [P0|P1|P2|P3] <description of the vulnerability> (CWE-XXX)\n\n"
            "Example:\n"
            "- [P2] SQL injection in login query via unsanitized user input (CWE-89)\n"
            "- [P1] Hardcoded database password in config.py (CWE-798)\n\n"
            "List all findings you can identify:"
        )
        return "\n".join(parts)

    def parse_findings(self, response: str) -> list[dict[str, Any]]:
        """Parse structured findings from LLM response (JSON or free-text)."""
        findings: list[dict[str, Any]] = []

        # Attempt 1: Try JSON parsing
        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                candidate = text.split("```")[1].split("```")[0].strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    text = candidate

            data = json.loads(text)
            if isinstance(data, dict) and "findings" in data:
                return data["findings"]
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

        # Attempt 2: Parse bullet-point format "- [P1] description (CWE-XXX)"
        import re
        pattern = re.compile(
            r"[-*•]\s*\[?(P[0-3]|CRITICAL|HIGH|MEDIUM|LOW|INFO)\]?\s*[:\-]?\s*(.+?)(?:\(?(CWE-\d+)\)?)?$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in pattern.finditer(response):
            sev = match.group(1).upper()
            text = match.group(2).strip().rstrip("(").strip()
            cwe = match.group(3) or ""
            if len(text) > 5:  # skip noise
                findings.append({
                    "text": text,
                    "severity": sev,
                    "confidence": 0.6,
                    "cwe": cwe,
                })

        # Attempt 3: Extract numbered items "1. description"
        if not findings:
            num_pattern = re.compile(
                r"\d+[.)]\s+(.{15,})", re.MULTILINE,
            )
            for match in num_pattern.finditer(response):
                text = match.group(1).strip()
                findings.append({
                    "text": text[:200],
                    "severity": "P2",
                    "confidence": 0.5,
                    "cwe": "",
                })

        # Attempt 4: If still nothing, treat the whole response as one finding
        if not findings and len(response.strip()) > 20:
            # Extract lines that look like findings
            for line in response.strip().splitlines():
                line = line.strip()
                if len(line) > 20 and any(kw in line.lower() for kw in
                    ("vulnerab", "inject", "xss", "auth", "expos", "hardcod",
                     "insecur", "danger", "risk", "flaw", "bypass", "leak",
                     "overflow", "ssrf", "csrf", "misconfigur", "credential",
                     "permiss", "secret", "token", "password", "api key")):
                    findings.append({
                        "text": line[:200],
                        "severity": "P2",
                        "confidence": 0.4,
                        "cwe": "",
                    })

        if findings:
            logger.info(f"{self.PERSONA_NAME}: Parsed {len(findings)} findings from response")
        else:
            logger.warning(f"{self.PERSONA_NAME}: No findings extracted from response ({len(response)} chars)")

        return findings
