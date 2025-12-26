"""
Unit tests for provenance module.
"""

import pytest
import json
from pathlib import Path


class TestProvenanceLedger:
    """Tests for provenance ledger."""
    
    def test_ledger_creation(self):
        """Test ledger creation."""
        from argus.provenance import ProvenanceLedger
        
        ledger = ProvenanceLedger()
        
        assert len(ledger) == 0
    
    def test_record_event(self):
        """Test recording event."""
        from argus.provenance import ProvenanceLedger, EventType
        
        ledger = ProvenanceLedger()
        
        event = ledger.record(
            EventType.PROPOSITION_ADDED,
            entity_id="prop_123",
            attributes={"text": "Test"},
        )
        
        assert event is not None
        assert event.event_type == EventType.PROPOSITION_ADDED
        assert event.entity_id == "prop_123"
        assert event.hash != ""
    
    def test_hash_chain(self):
        """Test hash chain integrity."""
        from argus.provenance import ProvenanceLedger, EventType
        
        ledger = ProvenanceLedger()
        
        e1 = ledger.record(EventType.SESSION_START)
        e2 = ledger.record(EventType.PROPOSITION_ADDED, entity_id="p1")
        e3 = ledger.record(EventType.EVIDENCE_ADDED, entity_id="e1")
        
        # Chain should be linked
        assert e2.prev_hash == e1.hash
        assert e3.prev_hash == e2.hash
    
    def test_verify_integrity(self):
        """Test integrity verification."""
        from argus.provenance import ProvenanceLedger, EventType
        
        ledger = ProvenanceLedger()
        
        ledger.record(EventType.SESSION_START)
        ledger.record(EventType.PROPOSITION_ADDED)
        ledger.record(EventType.SESSION_END)
        
        is_valid, errors = ledger.verify_integrity()
        
        assert is_valid
        assert len(errors) == 0
    
    def test_query_by_entity(self):
        """Test querying by entity."""
        from argus.provenance import ProvenanceLedger, EventType
        
        ledger = ProvenanceLedger()
        
        ledger.record(EventType.PROPOSITION_ADDED, entity_id="p1")
        ledger.record(EventType.EVIDENCE_ADDED, entity_id="e1")
        ledger.record(EventType.EVIDENCE_ADDED, entity_id="e2")
        ledger.record(EventType.VERDICT_RENDERED, entity_id="p1")
        
        results = ledger.query(entity_id="p1")
        
        assert len(results) == 2
    
    def test_ledger_persistence(self, temp_ledger):
        """Test ledger persistence."""
        from argus.provenance import EventType
        
        temp_ledger.record(EventType.SESSION_START)
        temp_ledger.record(EventType.SESSION_END)
        
        assert len(temp_ledger) == 2
    
    def test_export_prov(self):
        """Test PROV-O export."""
        from argus.provenance import ProvenanceLedger, EventType
        
        ledger = ProvenanceLedger()
        ledger.record(EventType.PROPOSITION_ADDED, entity_id="p1")
        
        prov = ledger.export_prov()
        
        assert "@context" in prov
        assert "@graph" in prov
        assert len(prov["@graph"]) == 1


class TestIntegrity:
    """Tests for integrity utilities."""
    
    def test_compute_hash(self):
        """Test hash computation."""
        from argus.provenance.integrity import compute_hash
        
        h1 = compute_hash("test content")
        h2 = compute_hash("test content")
        h3 = compute_hash("different content")
        
        assert h1 == h2
        assert h1 != h3
    
    def test_verify_hash(self):
        """Test hash verification."""
        from argus.provenance.integrity import compute_hash, verify_hash
        
        content = "test content"
        h = compute_hash(content)
        
        assert verify_hash(content, h)
        assert not verify_hash("wrong content", h)
    
    def test_attestation(self):
        """Test attestation creation."""
        from argus.provenance.integrity import create_attestation
        
        content = "This is important content"
        attestation = create_attestation(content, attester="test_agent")
        
        assert attestation.attester == "test_agent"
        assert attestation.verify(content)
        assert not attestation.verify("different content")
    
    def test_merkle_root(self):
        """Test Merkle root computation."""
        from argus.provenance.integrity import compute_hash, compute_merkle_root
        
        hashes = [
            compute_hash("a"),
            compute_hash("b"),
            compute_hash("c"),
            compute_hash("d"),
        ]
        
        root = compute_merkle_root(hashes)
        
        assert len(root) == 64  # SHA-256 hex
        
        # Same inputs = same root
        root2 = compute_merkle_root(hashes)
        assert root == root2
