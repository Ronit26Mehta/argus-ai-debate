"""
fsociety VAPT data models.

Extends ARGUS C-DAG node/edge types with VAPT-specific types
for the Vulnerability Knowledge Graph (VKG).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field

from argus.cdag.nodes import NodeBase, NodeStatus, generate_node_id
from argus.cdag.edges import Edge, EdgeType, EdgePolarity, generate_edge_id


# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════


class SeverityLevel(str, Enum):
    """VAPT finding severity levels."""
    CRITICAL = "P0"  # Critical — immediate action required
    HIGH = "P1"      # High — fix within days
    MEDIUM = "P2"    # Medium — fix within sprint
    LOW = "P3"       # Low — backlog
    INFO = "INFO"    # Informational


class FindingStatus(str, Enum):
    """Status of a VAPT finding after debate."""
    CONFIRMED = "CONFIRMED"
    PROBABLE = "PROBABLE"
    UNCONFIRMED = "UNCONFIRMED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class VAPTEdgeType(str, Enum):
    """Edge types in the VKG (extends ARGUS EdgeType)."""
    ENABLES = "ENABLES"
    MITIGATED_BY = "MITIGATED_BY"
    MAPPED_TO = "MAPPED_TO"
    VIOLATES = "VIOLATES"
    ATTRIBUTED_TO = "ATTRIBUTED_TO"
    CHAINS_WITH = "CHAINS_WITH"
    DETECTED_BY = "DETECTED_BY"


class AgentTier(str, Enum):
    """Agent tier classification."""
    CORE = "tier1_core"
    SPECIALIST = "tier2_specialist"
    OUTPUT = "tier3_output"


class CWECategory(str, Enum):
    """Common CWE categories for classification."""
    SQL_INJECTION = "CWE-89"
    XSS = "CWE-79"
    BROKEN_AUTH = "CWE-287"
    SENSITIVE_DATA = "CWE-200"
    BROKEN_ACCESS = "CWE-284"
    MISCONFIG = "CWE-16"
    INSECURE_DESERIALIZATION = "CWE-502"
    VULNERABLE_COMPONENTS = "CWE-1104"
    LOGGING_FAILURE = "CWE-778"
    SSRF = "CWE-918"
    PATH_TRAVERSAL = "CWE-22"
    COMMAND_INJECTION = "CWE-78"
    CRYPTO_FAILURE = "CWE-327"
    HARDCODED_CREDS = "CWE-798"
    OTHER = "CWE-OTHER"


# ═══════════════════════════════════════════════════════════════════════
# VKG Node Types
# ═══════════════════════════════════════════════════════════════════════


class VulnerabilityNode(NodeBase):
    """A specific vulnerability finding in the VKG."""

    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Severity level (P0-P3 or INFO)",
    )
    status: FindingStatus = Field(
        default=FindingStatus.UNCONFIRMED,
        description="Finding status after debate",
    )
    cwe: Optional[str] = Field(default=None, description="CWE classification ID")
    cvss_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    location: Optional[str] = Field(default=None, description="File:line or URL")
    agent_source: Optional[str] = Field(default=None, description="Agent that found it")
    exploit_scenario: Optional[str] = Field(default=None, description="Attack narrative")
    remediation: Optional[str] = Field(default=None, description="Fix guidance")
    posterior: float = Field(default=0.5, description="Bayesian posterior probability")

    @classmethod
    def create(cls, text: str, severity: SeverityLevel = SeverityLevel.MEDIUM, **kwargs) -> "VulnerabilityNode":
        """Create a new vulnerability node."""
        return cls(
            id=generate_node_id("vuln"),
            text=text,
            severity=severity,
            **kwargs,
        )


class ExploitChainNode(NodeBase):
    """A multi-step exploit chain linking multiple vulnerabilities."""

    chain_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of vulnerability IDs in the chain",
    )
    impact: str = Field(default="unknown", description="Impact: RCE, Data Exfiltration, etc.")
    blast_radius: Optional[str] = Field(default=None, description="Blast radius description")
    preconditions: list[str] = Field(
        default_factory=list,
        description="Conditions required for the chain to work",
    )

    @classmethod
    def create(cls, text: str, steps: list[str], impact: str = "unknown", **kwargs) -> "ExploitChainNode":
        return cls(
            id=generate_node_id("chain"),
            text=text,
            chain_steps=steps,
            impact=impact,
            **kwargs,
        )


class AttackSurfaceNode(NodeBase):
    """An exposed interface or entry point."""

    surface_type: str = Field(
        default="api_endpoint",
        description="Type: api_endpoint, file_io, db_call, auth, user_input, dependency",
    )
    endpoint: Optional[str] = Field(default=None, description="URL or code location")
    methods: list[str] = Field(default_factory=list, description="HTTP methods available")

    @classmethod
    def create(cls, text: str, surface_type: str = "api_endpoint", **kwargs) -> "AttackSurfaceNode":
        return cls(
            id=generate_node_id("surface"),
            text=text,
            surface_type=surface_type,
            **kwargs,
        )


class MitigationNode(NodeBase):
    """An existing security control or mitigation."""

    control_type: str = Field(default="unknown", description="WAF, rate_limit, input_validation, etc.")
    effectiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Effectiveness score")

    @classmethod
    def create(cls, text: str, control_type: str = "unknown", **kwargs) -> "MitigationNode":
        return cls(id=generate_node_id("mitig"), text=text, control_type=control_type, **kwargs)


class ThreatActorNode(NodeBase):
    """A threat group whose TTPs match the target's vulnerability profile."""

    actor_name: str = Field(default="unknown", description="APT group name")
    ttps: list[str] = Field(default_factory=list, description="MITRE ATT&CK techniques")

    @classmethod
    def create(cls, text: str, actor_name: str = "unknown", **kwargs) -> "ThreatActorNode":
        return cls(id=generate_node_id("actor"), text=text, actor_name=actor_name, **kwargs)


class CVERecordNode(NodeBase):
    """A linked CVE record."""

    cve_id: str = Field(default="", description="CVE identifier (e.g. CVE-2023-12345)")
    cvss_score: float = Field(default=0.0, ge=0.0, le=10.0)
    patch_available: bool = Field(default=False)
    patch_url: Optional[str] = Field(default=None)
    affected_versions: list[str] = Field(default_factory=list)
    has_public_exploit: bool = Field(default=False)

    @classmethod
    def create(cls, cve_id: str, cvss: float = 0.0, **kwargs) -> "CVERecordNode":
        return cls(
            id=generate_node_id("cve"),
            text=f"CVE record: {cve_id}",
            cve_id=cve_id,
            cvss_score=cvss,
            **kwargs,
        )


class ComplianceRequirementNode(NodeBase):
    """A compliance framework requirement that a finding violates."""

    framework: str = Field(default="", description="PCI-DSS, SOC2, ISO27001, NIST, HIPAA, GDPR")
    requirement_id: str = Field(default="", description="Specific requirement ID")

    @classmethod
    def create(cls, framework: str, req_id: str, text: str = "", **kwargs) -> "ComplianceRequirementNode":
        return cls(
            id=generate_node_id("compliance"),
            text=text or f"{framework} {req_id}",
            framework=framework,
            requirement_id=req_id,
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════
# Session & Scan Models
# ═══════════════════════════════════════════════════════════════════════


class ScanTarget(BaseModel):
    """A scan target definition."""

    path: Optional[str] = Field(default=None, description="Local path to codebase")
    repo_url: Optional[str] = Field(default=None, description="GitHub/GitLab repo URL")
    url: Optional[str] = Field(default=None, description="Live web target URL")

    @property
    def name(self) -> str:
        """Derive a human name from the target."""
        if self.repo_url:
            return self.repo_url.rstrip("/").split("/")[-1]
        if self.path:
            return Path(self.path).name if hasattr(Path, "__call__") else self.path.split("/")[-1]
        if self.url:
            return self.url.replace("https://", "").replace("http://", "").split("/")[0]
        return "unknown_target"

    @property
    def mode(self) -> str:
        """Determine scan mode."""
        if self.path or self.repo_url:
            if self.url:
                return "hybrid"
            return "static"
        if self.url:
            return "dynamic"
        return "unknown"


class ScanSession(BaseModel):
    """A complete scan session record."""

    session_id: str = Field(default="")
    target: ScanTarget = Field(default_factory=ScanTarget)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    total_findings: int = Field(default=0)
    severity_counts: dict[str, int] = Field(
        default_factory=lambda: {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "INFO": 0}
    )
    agents_used: list[str] = Field(default_factory=list)
    debate_rounds: int = Field(default=0)


# Import Path for ScanTarget
from pathlib import Path
