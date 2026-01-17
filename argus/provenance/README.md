# ARGUS Provenance Module

## Overview

The `provenance/` module provides **PROV-O compatible** audit trails with cryptographic integrity for tracking all debate operations.

## Components

| File | Description |
|------|-------------|
| `ledger.py` | Event ledger with hash-chain |
| `integrity.py` | Cryptographic verification |

## Quick Start

```python
from argus.provenance import Ledger, Event, EventType

# Create ledger
ledger = Ledger()

# Log events
ledger.log(Event(
    type=EventType.SESSION_START,
    data={"topic": "AI Ethics", "prior": 0.5},
))

ledger.log(Event(
    type=EventType.EVIDENCE_ADDED,
    data={"text": "Research shows...", "polarity": 1},
))

# Verify integrity
is_valid = ledger.verify_integrity()
print(f"Integrity valid: {is_valid}")

# Export
ledger.save("provenance.json")
```

## Event Types

```python
from argus.provenance import EventType

EventType.SESSION_START      # Debate initialized
EventType.SESSION_END        # Debate completed
EventType.PROPOSITION_ADDED  # New proposition
EventType.EVIDENCE_ADDED     # Evidence attached
EventType.REBUTTAL_ADDED     # Rebuttal created
EventType.VERDICT_RENDERED   # Jury verdict
EventType.ROUND_COMPLETED    # Round finished
EventType.AGENT_ACTION       # Agent activity
```

## Ledger Operations

```python
from argus.provenance import Ledger

ledger = Ledger()

# Log with automatic timestamp and hash
event = ledger.log(Event(
    type=EventType.EVIDENCE_ADDED,
    data={"evidence_id": "ev-001", "source": "arxiv"},
    agent="specialist-1",
))

# Query events
all_evidence = ledger.query(type=EventType.EVIDENCE_ADDED)
agent_actions = ledger.query(agent="specialist-1")
recent = ledger.query(since=datetime(2024, 1, 1))

# Get event chain
chain = ledger.get_chain()
for event in chain:
    print(f"{event.timestamp}: {event.type} (hash: {event.hash[:8]})")

# Export PROV-O format
prov_doc = ledger.to_prov_o()
```

## Integrity Verification

```python
from argus.provenance import verify_integrity, compute_hash

# Verify entire ledger
is_valid = ledger.verify_integrity()

# Verify specific event chain
is_valid = verify_integrity(events)

# Compute content hash
hash_value = compute_hash(content)
```

## Hash Chain

Each event contains:
- `id` - Unique event ID
- `type` - Event type
- `timestamp` - ISO timestamp
- `data` - Event payload
- `agent` - Acting agent (optional)
- `prev_hash` - Hash of previous event
- `hash` - SHA-256 of this event

```python
# Manual verification
from argus.provenance.integrity import verify_chain

events = ledger.get_chain()
valid, invalid_index = verify_chain(events)
if not valid:
    print(f"Chain broken at index {invalid_index}")
```

## PROV-O Compatibility

Export to W3C PROV-O standard:

```python
# JSON-LD format
prov_doc = ledger.to_prov_o(format="json-ld")

# RDF/Turtle format  
prov_doc = ledger.to_prov_o(format="turtle")
```

## Attestations

```python
from argus.provenance import create_attestation, verify_attestation

# Create cryptographic attestation
attestation = create_attestation(
    content="Evidence text...",
    signer="analyst-1",
)

# Verify attestation
is_valid = verify_attestation(attestation)
```
