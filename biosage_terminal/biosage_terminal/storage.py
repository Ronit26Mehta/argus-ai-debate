"""
Storage layer for BioSage Terminal.
Provides persistent local file storage using JSON files.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic
import asyncio
import aiofiles

T = TypeVar("T")


class StorageConfig:
    """Configuration for storage paths."""
    
    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path.home() / ".biosage"
        self.base_path = base_path
        self.patients_path = base_path / "patients"
        self.cases_path = base_path / "cases"
        self.evidence_path = base_path / "evidence"
        self.reports_path = base_path / "reports"
        self.plots_path = base_path / "plots"
        self.config_path = base_path / "config"
        self.audit_path = base_path / "audit"
        
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [
            self.base_path,
            self.patients_path,
            self.cases_path,
            self.evidence_path,
            self.reports_path,
            self.plots_path,
            self.config_path,
            self.audit_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class JSONStorage(Generic[T]):
    """Generic JSON file storage handler."""
    
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        
    def _get_file_path(self, entity_id: str) -> Path:
        """Get the file path for an entity."""
        return self.directory / f"{entity_id}.json"
    
    async def save(self, entity_id: str, data: dict[str, Any]) -> None:
        """Save an entity to a JSON file."""
        file_path = self._get_file_path(entity_id)
        data["_id"] = entity_id
        data["_updated_at"] = datetime.utcnow().isoformat()
        
        async with self._lock:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, default=str))
    
    async def load(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Load an entity from a JSON file."""
        file_path = self._get_file_path(entity_id)
        if not file_path.exists():
            return None
        
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    
    def load_sync(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Synchronously load an entity from a JSON file."""
        file_path = self._get_file_path(entity_id)
        if not file_path.exists():
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def save_sync(self, entity_id: str, data: dict[str, Any]) -> None:
        """Synchronously save an entity to a JSON file."""
        file_path = self._get_file_path(entity_id)
        data["_id"] = entity_id
        data["_updated_at"] = datetime.utcnow().isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity file."""
        file_path = self._get_file_path(entity_id)
        if file_path.exists():
            async with self._lock:
                os.remove(file_path)
            return True
        return False
    
    def delete_sync(self, entity_id: str) -> bool:
        """Synchronously delete an entity file."""
        file_path = self._get_file_path(entity_id)
        if file_path.exists():
            os.remove(file_path)
            return True
        return False
    
    async def list_all(self) -> list[dict[str, Any]]:
        """List all entities in the directory."""
        entities = []
        for file_path in self.directory.glob("*.json"):
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    entities.append(json.loads(content))
            except (json.JSONDecodeError, IOError):
                continue
        return entities
    
    def list_all_sync(self) -> list[dict[str, Any]]:
        """Synchronously list all entities in the directory."""
        entities = []
        for file_path in self.directory.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    entities.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        return entities
    
    async def count(self) -> int:
        """Count all entities."""
        return len(list(self.directory.glob("*.json")))
    
    def count_sync(self) -> int:
        """Synchronously count all entities."""
        return len(list(self.directory.glob("*.json")))
    
    async def query(self, filter_fn: callable) -> list[dict[str, Any]]:
        """Query entities with a filter function."""
        all_entities = await self.list_all()
        return [e for e in all_entities if filter_fn(e)]
    
    def query_sync(self, filter_fn: callable) -> list[dict[str, Any]]:
        """Synchronously query entities with a filter function."""
        all_entities = self.list_all_sync()
        return [e for e in all_entities if filter_fn(e)]


class PatientStorage(JSONStorage):
    """Storage for patient records."""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config.patients_path)
    
    def generate_mrn(self) -> str:
        """Generate a unique Medical Record Number."""
        prefix = "BSG"
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"{prefix}-{unique_id}"
    
    async def find_by_mrn(self, mrn: str) -> Optional[dict[str, Any]]:
        """Find a patient by MRN."""
        results = await self.query(lambda p: p.get("mrn") == mrn)
        return results[0] if results else None
    
    def find_by_mrn_sync(self, mrn: str) -> Optional[dict[str, Any]]:
        """Synchronously find a patient by MRN."""
        results = self.query_sync(lambda p: p.get("mrn") == mrn)
        return results[0] if results else None
    
    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search patients by name or MRN."""
        query_lower = query.lower()
        return await self.query(
            lambda p: query_lower in p.get("name", "").lower()
            or query_lower in p.get("mrn", "").lower()
        )


class CaseStorage(JSONStorage):
    """Storage for diagnostic cases."""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config.cases_path)
        self._case_counter = 0
        self._load_counter()
    
    def _load_counter(self) -> None:
        """Load the case counter from disk."""
        counter_file = self.directory / "_counter.json"
        if counter_file.exists():
            with open(counter_file, "r") as f:
                data = json.load(f)
                self._case_counter = data.get("counter", 0)
    
    def _save_counter(self) -> None:
        """Save the case counter to disk."""
        counter_file = self.directory / "_counter.json"
        with open(counter_file, "w") as f:
            json.dump({"counter": self._case_counter}, f)
    
    def generate_case_id(self) -> str:
        """Generate a unique case ID."""
        self._case_counter += 1
        self._save_counter()
        return f"CASE-{self._case_counter:06d}"
    
    async def find_by_patient(self, patient_id: str) -> list[dict[str, Any]]:
        """Find all cases for a patient."""
        return await self.query(lambda c: c.get("patient_id") == patient_id)
    
    def find_by_patient_sync(self, patient_id: str) -> list[dict[str, Any]]:
        """Synchronously find all cases for a patient."""
        return self.query_sync(lambda c: c.get("patient_id") == patient_id)
    
    async def find_open_cases(self) -> list[dict[str, Any]]:
        """Find all open (undiagnosed) cases."""
        return await self.query(lambda c: c.get("status") == "open")
    
    def find_open_cases_sync(self) -> list[dict[str, Any]]:
        """Synchronously find all open cases."""
        return self.query_sync(lambda c: c.get("status") == "open")
    
    async def find_closed_cases(self) -> list[dict[str, Any]]:
        """Find all closed (diagnosed) cases."""
        return await self.query(lambda c: c.get("status") == "closed")


class EvidenceStorage(JSONStorage):
    """Storage for evidence and citations."""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config.evidence_path)
    
    async def find_by_case(self, case_id: str) -> list[dict[str, Any]]:
        """Find all evidence for a case."""
        return await self.query(lambda e: e.get("case_id") == case_id)
    
    def find_by_case_sync(self, case_id: str) -> list[dict[str, Any]]:
        """Synchronously find all evidence for a case."""
        return self.query_sync(lambda e: e.get("case_id") == case_id)
    
    async def find_by_specialist(self, specialist: str) -> list[dict[str, Any]]:
        """Find all evidence from a specialist."""
        return await self.query(lambda e: e.get("specialist") == specialist)


class AuditStorage(JSONStorage):
    """Storage for audit logs."""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config.audit_path)
    
    def log_event(
        self,
        event_type: str,
        user: str,
        action: str,
        details: Optional[dict[str, Any]] = None,
        status: str = "success",
    ) -> str:
        """Log an audit event."""
        event_id = str(uuid.uuid4())
        event = {
            "event_type": event_type,
            "user": user,
            "action": action,
            "details": details or {},
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.save_sync(event_id, event)
        return event_id
    
    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent audit events."""
        all_events = self.list_all_sync()
        sorted_events = sorted(
            all_events, 
            key=lambda e: e.get("timestamp", ""), 
            reverse=True
        )
        return sorted_events[:limit]


class DataStore:
    """Main data store providing access to all storage components."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.config.ensure_directories()
        
        self.patients = PatientStorage(self.config)
        self.cases = CaseStorage(self.config)
        self.evidence = EvidenceStorage(self.config)
        self.audit = AuditStorage(self.config)
        
    def get_reports_path(self) -> Path:
        """Get the reports directory path."""
        return self.config.reports_path
    
    def get_plots_path(self) -> Path:
        """Get the plots directory path."""
        return self.config.plots_path
    
    def save_report(self, report_id: str, content: str, format: str = "md") -> Path:
        """Save a report to disk."""
        file_path = self.config.reports_path / f"{report_id}.{format}"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path
    
    def save_plot(self, plot_id: str, content: str, format: str = "txt") -> Path:
        """Save a plot to disk."""
        file_path = self.config.plots_path / f"{plot_id}.{format}"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path


# Singleton instance
_data_store: Optional[DataStore] = None


def get_data_store() -> DataStore:
    """Get the singleton data store instance."""
    global _data_store
    if _data_store is None:
        _data_store = DataStore()
    return _data_store
