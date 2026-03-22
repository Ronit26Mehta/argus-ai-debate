"""
Senate Record — complete procedural record of every AGORA session.

Implements a Hansard-style record with:
    - Typed entries (16 entry types from the protocol)
    - Hash-chain integrity for audit
    - Export to Hansard text, JSON, and summary formats
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from argus.agora.models import (
    SenateRecordEntry,
    RecordEntryType,
    SessionPhase,
    _utcnow,
    _uid,
)

logger = logging.getLogger(__name__)


class SenateRecord:
    """Complete procedural record of an AGORA session.

    Every action taken during the session is recorded as a typed entry.
    The record can be exported in three formats:
        1. Hansard — human-readable parliamentary transcript
        2. JSON    — structured data for programmatic access
        3. Summary — key entries only for quick review
    """

    def __init__(self, proposition: str = ""):
        self.id: str = _uid("sr")
        self.proposition: str = proposition
        self._entries: list[SenateRecordEntry] = []
        self._hash_chain: list[str] = []
        self._sealed: bool = False

    def add_entry(self, entry: SenateRecordEntry) -> None:
        """Add an entry to the record.

        Each entry is hash-chained to the previous for integrity.
        """
        if self._sealed:
            logger.warning("Attempted to add entry to sealed record")
            return

        # Compute hash-chain
        prev_hash = self._hash_chain[-1] if self._hash_chain else "genesis"
        entry_hash_input = (
            f"{prev_hash}:{entry.entry_type.value}:{entry.senator_id}:"
            f"{entry.content[:100]}:{entry.timestamp.isoformat()}"
        )
        entry_hash = hashlib.sha256(entry_hash_input.encode()).hexdigest()[:16]
        self._hash_chain.append(entry_hash)
        self._entries.append(entry)

    def seal(self) -> str:
        """Seal the record — no further entries accepted.

        Returns the final hash-chain digest.
        """
        if self._sealed:
            return self._hash_chain[-1] if self._hash_chain else ""

        # Add sealing entry
        seal_entry = SenateRecordEntry(
            entry_type=RecordEntryType.RECORD_SEALED,
            content=f"Senate Record sealed. Total entries: {len(self._entries)}.",
            metadata={
                "total_entries": len(self._entries),
                "chain_length": len(self._hash_chain),
            },
        )
        self.add_entry(seal_entry)
        self._sealed = True

        final_hash = self._hash_chain[-1] if self._hash_chain else ""
        logger.info(
            "Senate Record sealed: %d entries, final hash: %s",
            len(self._entries), final_hash,
        )
        return final_hash

    @property
    def entries(self) -> list[SenateRecordEntry]:
        return list(self._entries)

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    # ── Export: Hansard format ─────────────────────────────────────────

    def to_hansard(self) -> str:
        """Export as human-readable Hansard transcript."""
        lines = [
            "═" * 70,
            "AGORA SENATE RECORD — HANSARD FORMAT",
            "═" * 70,
            f"Proposition: {self.proposition}",
            f"Record ID: {self.id}",
            f"Total Entries: {len(self._entries)}",
            f"Sealed: {'Yes' if self._sealed else 'No'}",
            "═" * 70,
            "",
        ]

        current_phase: SessionPhase | None = None
        for entry in self._entries:
            # Phase header
            if entry.phase != current_phase:
                current_phase = entry.phase
                lines.append("")
                lines.append(f"── PHASE {current_phase.phase_number}: "
                             f"{current_phase.value.upper().replace('_', ' ')} ──")
                lines.append("")

            lines.append(entry.to_hansard_line())

        if self._hash_chain:
            lines.append("")
            lines.append(f"[RECORD HASH: {self._hash_chain[-1]}]")

        return "\n".join(lines)

    # ── Export: JSON format ───────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        """Export as structured JSON."""
        return {
            "id": self.id,
            "proposition": self.proposition,
            "total_entries": len(self._entries),
            "sealed": self._sealed,
            "final_hash": self._hash_chain[-1] if self._hash_chain else None,
            "entries": [e.to_dict() for e in self._entries],
        }

    def to_json_string(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_json(), indent=2, default=str)

    # ── Export: Summary format ────────────────────────────────────────

    def to_summary(self) -> str:
        """Export key entries only (phase transitions, verdicts, coalitions)."""
        key_types = {
            RecordEntryType.PHASE_TRANSITION,
            RecordEntryType.MAJORITY_OPINION,
            RecordEntryType.MINORITY_REPORT,
            RecordEntryType.COALITION_DETECTED,
            RecordEntryType.EA_RULING,
            RecordEntryType.STOPPING_TRIGGER,
            RecordEntryType.QUORUM_UPDATE,
            RecordEntryType.RECORD_SEALED,
        }

        lines = [
            "AGORA SENATE RECORD — SUMMARY",
            "─" * 50,
            f"Proposition: {self.proposition[:100]}",
            "",
        ]

        for entry in self._entries:
            if entry.entry_type in key_types:
                lines.append(entry.to_hansard_line())

        return "\n".join(lines)

    # ── Query methods ─────────────────────────────────────────────────

    def get_by_type(self, entry_type: RecordEntryType) -> list[SenateRecordEntry]:
        """Get all entries of a specific type."""
        return [e for e in self._entries if e.entry_type == entry_type]

    def get_by_phase(self, phase: SessionPhase) -> list[SenateRecordEntry]:
        """Get all entries from a specific phase."""
        return [e for e in self._entries if e.phase == phase]

    def get_by_senator(self, senator_id: str) -> list[SenateRecordEntry]:
        """Get all entries involving a specific senator."""
        return [e for e in self._entries if e.senator_id == senator_id]

    def get_by_round(self, round_num: int) -> list[SenateRecordEntry]:
        """Get all entries from a specific round."""
        return [e for e in self._entries if e.round_num == round_num]
