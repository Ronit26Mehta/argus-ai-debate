"""Tests for fsociety models."""

import pytest
from fsociety.models import (
    SeverityLevel,
    FindingStatus,
    VAPTEdgeType,
    AgentTier,
    VulnerabilityNode,
    ExploitChainNode,
    AttackSurfaceNode,
    MitigationNode,
    CVERecordNode,
    ComplianceRequirementNode,
    ScanTarget,
    ScanSession,
)


class TestEnums:
    def test_severity_levels(self):
        assert SeverityLevel.CRITICAL.value == "P0"
        assert SeverityLevel.HIGH.value == "P1"
        assert SeverityLevel.MEDIUM.value == "P2"
        assert SeverityLevel.LOW.value == "P3"
        assert SeverityLevel.INFO.value == "INFO"

    def test_finding_status(self):
        assert FindingStatus.CONFIRMED.value == "CONFIRMED"
        assert FindingStatus.FALSE_POSITIVE.value == "FALSE_POSITIVE"

    def test_vapt_edge_types(self):
        assert VAPTEdgeType.ENABLES.value == "ENABLES"
        assert VAPTEdgeType.MITIGATED_BY.value == "MITIGATED_BY"

    def test_agent_tiers(self):
        assert AgentTier.CORE.value == "tier1_core"
        assert AgentTier.SPECIALIST.value == "tier2_specialist"
        assert AgentTier.OUTPUT.value == "tier3_output"


class TestNodeTypes:
    def test_vulnerability_node_create(self):
        vuln = VulnerabilityNode.create(
            text="SQL Injection in login endpoint",
            severity=SeverityLevel.CRITICAL,
        )
        assert vuln.text == "SQL Injection in login endpoint"
        assert vuln.severity == SeverityLevel.CRITICAL
        assert vuln.id.startswith("vuln-")

    def test_exploit_chain_create(self):
        chain = ExploitChainNode.create(
            text="SQLi → RCE chain",
            steps=["vuln-1", "vuln-2"],
            impact="RCE",
        )
        assert len(chain.chain_steps) == 2
        assert chain.impact == "RCE"

    def test_attack_surface_create(self):
        surface = AttackSurfaceNode.create(
            text="/api/users endpoint",
            surface_type="api_endpoint",
        )
        assert surface.surface_type == "api_endpoint"

    def test_cve_record_create(self):
        cve = CVERecordNode.create(cve_id="CVE-2023-12345", cvss=9.8)
        assert cve.cve_id == "CVE-2023-12345"
        assert cve.cvss_score == 9.8


class TestScanTarget:
    def test_path_target(self):
        target = ScanTarget(path="/foo/bar")
        assert target.mode == "static"

    def test_url_target(self):
        target = ScanTarget(url="https://example.com")
        assert target.mode == "dynamic"

    def test_hybrid_target(self):
        target = ScanTarget(path="/foo", url="https://example.com")
        assert target.mode == "hybrid"
