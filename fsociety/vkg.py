"""
Vulnerability Knowledge Graph (VKG).

fsociety's name for the ARGUS C-DAG, specialized for VAPT.
Extends CDAG with VAPT-specific node/edge operations.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.cdag.graph import CDAG
from argus.cdag.edges import EdgeType, Edge, create_support_edge, create_attack_edge

from fsociety.models import (
    SeverityLevel,
    FindingStatus,
    VAPTEdgeType,
    VulnerabilityNode,
    ExploitChainNode,
    AttackSurfaceNode,
    MitigationNode,
    ThreatActorNode,
    CVERecordNode,
    ComplianceRequirementNode,
)

logger = logging.getLogger(__name__)


class VulnerabilityKnowledgeGraph(CDAG):
    """
    Vulnerability Knowledge Graph — the fsociety C-DAG.

    Extends ARGUS CDAG with VAPT-specific methods for managing
    vulnerability findings, exploit chains, and security nodes.
    """

    def __init__(self, name: str = "fsociety-vkg", **kwargs):
        super().__init__(name=name, **kwargs)
        self._vulnerabilities: dict[str, VulnerabilityNode] = {}
        self._exploit_chains: dict[str, ExploitChainNode] = {}
        self._attack_surfaces: dict[str, AttackSurfaceNode] = {}
        self._mitigations: dict[str, MitigationNode] = {}
        self._cve_records: dict[str, CVERecordNode] = {}

    # ── Vulnerability management ──────────────────────────────────

    def add_vulnerability(
        self,
        text: str,
        severity: SeverityLevel = SeverityLevel.MEDIUM,
        agent_source: Optional[str] = None,
        **kwargs,
    ) -> VulnerabilityNode:
        """Add a vulnerability finding to the VKG."""
        vuln = VulnerabilityNode.create(
            text=text,
            severity=severity,
            agent_source=agent_source,
            **kwargs,
        )
        self.add_node(vuln)
        self._vulnerabilities[vuln.id] = vuln
        logger.info(f"Added vulnerability {vuln.id}: {text[:80]}")
        return vuln

    def get_vulnerability(self, vuln_id: str) -> Optional[VulnerabilityNode]:
        """Get a vulnerability by ID."""
        return self._vulnerabilities.get(vuln_id)

    def get_all_vulnerabilities(self) -> list[VulnerabilityNode]:
        """Get all vulnerabilities."""
        return list(self._vulnerabilities.values())

    def get_findings_by_severity(self, severity: SeverityLevel) -> list[VulnerabilityNode]:
        """Get all findings at a given severity level."""
        return [v for v in self._vulnerabilities.values() if v.severity == severity]

    def get_confirmed_findings(self) -> list[VulnerabilityNode]:
        """Get all confirmed findings (CONFIRMED or PROBABLE)."""
        return [
            v for v in self._vulnerabilities.values()
            if v.status in (FindingStatus.CONFIRMED, FindingStatus.PROBABLE)
        ]

    # ── Exploit chain management ──────────────────────────────────

    def add_exploit_chain(
        self,
        text: str,
        vuln_ids: list[str],
        impact: str = "unknown",
        **kwargs,
    ) -> ExploitChainNode:
        """Add an exploit chain linking multiple vulnerabilities."""
        chain = ExploitChainNode.create(
            text=text,
            steps=vuln_ids,
            impact=impact,
            **kwargs,
        )
        self.add_node(chain)
        self._exploit_chains[chain.id] = chain

        # Link each vulnerability in the chain
        for vuln_id in vuln_ids:
            if vuln_id in self._vulnerabilities:
                edge = create_support_edge(vuln_id, chain.id, confidence=0.8)
                self.add_edge(edge)

        logger.info(f"Added exploit chain {chain.id} with {len(vuln_ids)} steps")
        return chain

    def chain_vulnerabilities(self, vuln_a_id: str, vuln_b_id: str, confidence: float = 0.7) -> Optional[str]:
        """Create a CHAINS_WITH edge between two vulnerabilities."""
        if vuln_a_id in self._vulnerabilities and vuln_b_id in self._vulnerabilities:
            edge = create_support_edge(vuln_a_id, vuln_b_id, confidence=confidence)
            return self.add_edge(edge)
        return None

    def get_exploit_chains(self) -> list[ExploitChainNode]:
        """Get all exploit chains."""
        return list(self._exploit_chains.values())

    # ── Attack surface management ─────────────────────────────────

    def add_attack_surface(
        self,
        text: str,
        surface_type: str = "api_endpoint",
        **kwargs,
    ) -> AttackSurfaceNode:
        """Add an attack surface node."""
        surface = AttackSurfaceNode.create(
            text=text,
            surface_type=surface_type,
            **kwargs,
        )
        self.add_node(surface)
        self._attack_surfaces[surface.id] = surface
        return surface

    # ── Mitigation management ─────────────────────────────────────

    def add_mitigation(
        self,
        text: str,
        vuln_id: str,
        control_type: str = "unknown",
        effectiveness: float = 0.5,
    ) -> Optional[MitigationNode]:
        """Add a mitigation that reduces a vulnerability's impact."""
        mitig = MitigationNode.create(
            text=text,
            control_type=control_type,
            effectiveness=effectiveness,
        )
        self.add_node(mitig)
        self._mitigations[mitig.id] = mitig

        # MITIGATED_BY edge (attack edge — reduces severity)
        if vuln_id in self._vulnerabilities:
            edge = create_attack_edge(mitig.id, vuln_id, confidence=effectiveness)
            self.add_edge(edge)

        return mitig

    # ── CVE record management ─────────────────────────────────────

    def add_cve_record(
        self,
        cve_id: str,
        vuln_id: str,
        cvss: float = 0.0,
        patch_available: bool = False,
        has_public_exploit: bool = False,
    ) -> CVERecordNode:
        """Add a CVE record linked to a vulnerability."""
        cve = CVERecordNode.create(
            cve_id=cve_id,
            cvss=cvss,
            patch_available=patch_available,
            has_public_exploit=has_public_exploit,
        )
        self.add_node(cve)
        self._cve_records[cve.id] = cve

        # MAPPED_TO edge
        if vuln_id in self._vulnerabilities:
            edge = create_support_edge(cve.id, vuln_id, confidence=0.9)
            self.add_edge(edge)

        return cve

    # ── Summary ───────────────────────────────────────────────────

    def vkg_summary(self) -> dict[str, Any]:
        """Get VKG-specific summary statistics."""
        base = self.summary()
        confirmed = self.get_confirmed_findings()
        base.update({
            "total_vulnerabilities": len(self._vulnerabilities),
            "total_exploit_chains": len(self._exploit_chains),
            "total_attack_surfaces": len(self._attack_surfaces),
            "total_mitigations": len(self._mitigations),
            "total_cve_records": len(self._cve_records),
            "confirmed_findings": len(confirmed),
            "severity_breakdown": {
                sev.value: len(self.get_findings_by_severity(sev))
                for sev in SeverityLevel
            },
        })
        return base
