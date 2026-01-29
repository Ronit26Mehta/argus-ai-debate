"""
Data models and schemas for BioSage Terminal.
Uses Pydantic for validation and serialization.
"""

from datetime import datetime, date
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
import uuid


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"


class CaseStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DIAGNOSED = "diagnosed"
    CLOSED = "closed"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class Specialty(str, Enum):
    INFECTIOUS = "infectious"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    AUTOIMMUNE = "autoimmune"
    TOXICOLOGY = "toxicology"


class EmergencyContact(BaseModel):
    """Emergency contact information."""
    name: str
    relationship: str
    phone: str


class Medication(BaseModel):
    """Medication record."""
    name: str
    dosage: str
    frequency: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    notes: Optional[str] = None


class Allergy(BaseModel):
    """Allergy record."""
    allergen: str
    reaction: str
    severity: str = "moderate"


class MedicalCondition(BaseModel):
    """Medical condition record."""
    name: str
    diagnosed_date: Optional[date] = None
    status: str = "active"
    notes: Optional[str] = None


class Vitals(BaseModel):
    """Vital signs record."""
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class LabResult(BaseModel):
    """Laboratory test result."""
    test_name: str
    value: str
    unit: str
    reference_range: str
    status: str = "normal"
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class Patient(BaseModel):
    """Patient record."""
    id: str = Field(default_factory=generate_id)
    mrn: str
    name: str
    date_of_birth: date
    gender: Gender
    blood_type: Optional[BloodType] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    emergency_contact: Optional[EmergencyContact] = None
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    conditions: list[MedicalCondition] = Field(default_factory=list)
    vitals_history: list[Vitals] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_age(self) -> int:
        """Calculate patient's age."""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    def get_latest_vitals(self) -> Optional[Vitals]:
        """Get the most recent vitals."""
        if not self.vitals_history:
            return None
        return max(self.vitals_history, key=lambda v: v.recorded_at)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode="json")


class Evidence(BaseModel):
    """Evidence item supporting a diagnosis."""
    id: str = Field(default_factory=generate_id)
    case_id: str
    specialist: Specialty
    source_type: str
    source_id: Optional[str] = None
    title: str
    content: str
    relevance_score: float = 0.0
    citations: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode="json")


class DiagnosisSuggestion(BaseModel):
    """Diagnosis suggestion from a specialist."""
    diagnosis: str
    confidence: float
    rationale: str
    specialist: Specialty
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TestRecommendation(BaseModel):
    """Recommended diagnostic test."""
    test_name: str
    rationale: str
    priority: Priority
    estimated_cost: Optional[float] = None
    turnaround_time: Optional[str] = None


class TreatmentRecommendation(BaseModel):
    """Treatment recommendation."""
    recommendation: str
    rationale: str
    priority: Priority
    specialist: Specialty


class Case(BaseModel):
    """Diagnostic case."""
    id: str = Field(default_factory=generate_id)
    case_number: str
    patient_id: str
    patient_name: str
    chief_complaint: str
    symptoms: list[str] = Field(default_factory=list)
    status: CaseStatus = CaseStatus.OPEN
    priority: Priority = Priority.MEDIUM
    diagnoses: list[DiagnosisSuggestion] = Field(default_factory=list)
    test_recommendations: list[TestRecommendation] = Field(default_factory=list)
    treatment_recommendations: list[TreatmentRecommendation] = Field(default_factory=list)
    primary_diagnosis: Optional[str] = None
    primary_confidence: Optional[float] = None
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    
    def get_top_diagnosis(self) -> Optional[DiagnosisSuggestion]:
        """Get the highest confidence diagnosis."""
        if not self.diagnoses:
            return None
        return max(self.diagnoses, key=lambda d: d.confidence)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode="json")


class SpecialistAgent(BaseModel):
    """AI Specialist agent status."""
    specialty: Specialty
    name: str
    status: AgentStatus = AgentStatus.ONLINE
    confidence: float = 0.0
    cases_handled: int = 0
    recent_diagnoses: list[tuple[str, float]] = Field(default_factory=list)
    
    @property
    def icon(self) -> str:
        """Get text icon for the specialty."""
        icons = {
            Specialty.INFECTIOUS: "[INF]",
            Specialty.CARDIOLOGY: "[CAR]",
            Specialty.NEUROLOGY: "[NEU]",
            Specialty.ONCOLOGY: "[ONC]",
            Specialty.AUTOIMMUNE: "[AUT]",
            Specialty.TOXICOLOGY: "[TOX]",
        }
        return icons.get(self.specialty, "[???]")
    
    @property
    def status_indicator(self) -> str:
        """Get text status indicator."""
        indicators = {
            AgentStatus.ONLINE: "[ON]",
            AgentStatus.DEGRADED: "[DEG]",
            AgentStatus.OFFLINE: "[OFF]",
        }
        return indicators.get(self.status, "[???]")


class AuditEvent(BaseModel):
    """Audit log event."""
    id: str = Field(default_factory=generate_id)
    event_type: str
    user: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode="json")


class UserSettings(BaseModel):
    """User settings and preferences."""
    theme: str = "dark"
    default_view: str = "dashboard"
    notifications_enabled: bool = True
    auto_save: bool = True
    language: str = "en"


class AppConfig(BaseModel):
    """Application configuration."""
    llm_provider: str = "gemini"
    llm_model: str = "gemini-1.5-pro"
    api_key_env_var: str = "GEMINI_API_KEY"
    max_evidence_items: int = 10
    confidence_threshold: float = 0.7
    debug_mode: bool = False
