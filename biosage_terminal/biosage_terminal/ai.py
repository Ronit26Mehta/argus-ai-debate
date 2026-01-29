"""
AI integration module for BioSage Terminal.
Full integration with ARGUS debate framework including:
- Moderator for debate orchestration
- Specialist agents for domain-specific evidence gathering
- Refuter agents for generating challenges/rebuttals  
- Jury for rendering final verdicts via Bayesian aggregation
- CDAG (Conceptual Debate Graph) for structured argumentation

API keys and model names are read from environment variables.
Priority: Gemini > OpenAI > Anthropic > Groq > Mistral > Cohere > Ollama
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Configuration - All from Environment Variables
# =============================================================================

class LLMConfig:
    """
    Configuration for LLM providers.
    All API keys and model names come from environment variables.
    
    Environment Variables:
        GEMINI_API_KEY, GEMINI_MODEL
        OPENAI_API_KEY, OPENAI_MODEL
        ANTHROPIC_API_KEY, ANTHROPIC_MODEL
        GROQ_API_KEY, GROQ_MODEL
        MISTRAL_API_KEY, MISTRAL_MODEL
        COHERE_API_KEY, COHERE_MODEL
        OLLAMA_MODEL (no API key needed for local)
    """
    
    # Provider priority order: (provider_name, env_var_key, env_var_model, default_model)
    PROVIDER_PRIORITY = [
        ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-1.5-pro"),
        ("openai", "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o"),
        ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        ("groq", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.1-70b-versatile"),
        ("mistral", "MISTRAL_API_KEY", "MISTRAL_MODEL", "mistral-large-latest"),
        ("cohere", "COHERE_API_KEY", "COHERE_MODEL", "command-r-plus"),
        ("ollama", None, "OLLAMA_MODEL", "llama3.1"),
    ]
    
    @classmethod
    def get_available_provider(cls) -> Optional[tuple[str, str]]:
        """
        Get the first available LLM provider based on API keys.
        Returns (provider_name, model_name) or None.
        """
        for provider, env_key, env_model, default_model in cls.PROVIDER_PRIORITY:
            if env_key is None:
                # Local provider like Ollama
                if cls._check_ollama_available():
                    model = os.getenv(env_model, default_model)
                    return provider, model
            else:
                api_key = os.getenv(env_key)
                if api_key:
                    model = os.getenv(env_model, default_model)
                    return provider, model
        return None
    
    @classmethod
    def _check_ollama_available(cls) -> bool:
        """Check if Ollama is running locally."""
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
    
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get the API key for a specific provider from environment."""
        for prov, env_key, _, _ in cls.PROVIDER_PRIORITY:
            if prov == provider and env_key:
                return os.getenv(env_key)
        return None
    
    @classmethod
    def get_model(cls, provider: str) -> Optional[str]:
        """Get the model name for a provider from environment."""
        for prov, _, env_model, default_model in cls.PROVIDER_PRIORITY:
            if prov == provider:
                return os.getenv(env_model, default_model)
        return None


def get_llm(provider: Optional[str] = None, model: Optional[str] = None):
    """
    Get an LLM instance using ARGUS framework.
    
    Args:
        provider: LLM provider name (auto-detected if None)
        model: Model name (from env or default if None)
        
    Returns:
        BaseLLM instance from ARGUS
        
    Priority order: Gemini > OpenAI > Anthropic > Groq > Mistral > Cohere > Ollama
    """
    try:
        from argus import get_llm as argus_get_llm
        
        if provider is None:
            result = LLMConfig.get_available_provider()
            if result is None:
                raise ValueError(
                    "No LLM provider available. Please set an API key in environment:\n"
                    "  - GEMINI_API_KEY (recommended)\n"
                    "  - OPENAI_API_KEY\n"
                    "  - ANTHROPIC_API_KEY\n"
                    "  - GROQ_API_KEY\n"
                    "  - MISTRAL_API_KEY\n"
                    "  - COHERE_API_KEY\n"
                    "  - Or run Ollama locally\n\n"
                    "Optionally set model names:\n"
                    "  - GEMINI_MODEL, OPENAI_MODEL, etc."
                )
            provider, model = result
        elif model is None:
            model = LLMConfig.get_model(provider)
        
        logger.info(f"Initializing LLM: provider={provider}, model={model}")
        return argus_get_llm(provider, model=model)
        
    except ImportError as e:
        logger.warning(f"ARGUS not available, using fallback LLM: {e}")
        return FallbackLLM(provider, model)


class FallbackLLM:
    """Fallback LLM when ARGUS is not installed."""
    
    def __init__(self, provider: Optional[str], model: Optional[str]):
        self.provider = provider or "mock"
        self.model = model or "mock-model"
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self._client = genai.GenerativeModel(self.model)
            except ImportError:
                pass
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._client = OpenAI(api_key=api_key)
            except ImportError:
                pass
    
    def generate(self, prompt: str, **kwargs) -> "LLMResponse":
        """Generate a response from the LLM."""
        if self._client is None:
            return LLMResponse(content="[LLM not configured - please set API key]")
        
        try:
            if self.provider == "gemini":
                response = self._client.generate_content(prompt)
                return LLMResponse(content=response.text)
            elif self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return LLMResponse(content=response.choices[0].message.content)
            else:
                return LLMResponse(content="[Unknown provider]")
        except Exception as e:
            return LLMResponse(content=f"[Error: {str(e)}]")


class LLMResponse:
    """Simple LLM response wrapper for fallback."""
    
    def __init__(self, content: str):
        self.content = content


# =============================================================================
# ARGUS Debate Integration Data Structures
# =============================================================================

@dataclass
class MedicalDebateResult:
    """
    Result of a medical diagnosis debate using ARGUS framework.
    
    Contains the full debate output including:
    - Final verdict and confidence
    - All diagnoses with posteriors
    - Evidence gathered by specialists
    - Rebuttals generated by refuter
    - Complete reasoning chain
    """
    patient_id: str
    case_id: str
    
    # Primary diagnosis
    primary_diagnosis: str = ""
    primary_confidence: float = 0.0
    verdict_label: str = "undecided"  # supported, rejected, undecided
    
    # All diagnoses with scores
    diagnoses: list[dict[str, Any]] = field(default_factory=list)
    
    # Evidence from specialists
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    
    # Rebuttals from refuter
    rebuttals: list[dict[str, Any]] = field(default_factory=list)
    
    # Debate statistics
    num_rounds: int = 0
    num_evidence: int = 0
    num_rebuttals: int = 0
    
    # Reasoning
    reasoning: str = ""
    specialist_contributions: dict[str, Any] = field(default_factory=dict)
    
    # Test and treatment recommendations
    recommended_tests: list[dict[str, Any]] = field(default_factory=list)
    recommended_treatments: list[dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "patient_id": self.patient_id,
            "case_id": self.case_id,
            "primary_diagnosis": self.primary_diagnosis,
            "primary_confidence": self.primary_confidence,
            "verdict_label": self.verdict_label,
            "diagnoses": self.diagnoses,
            "evidence_items": self.evidence_items,
            "rebuttals": self.rebuttals,
            "num_rounds": self.num_rounds,
            "num_evidence": self.num_evidence,
            "num_rebuttals": self.num_rebuttals,
            "reasoning": self.reasoning,
            "specialist_contributions": self.specialist_contributions,
            "recommended_tests": self.recommended_tests,
            "recommended_treatments": self.recommended_treatments,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Medical Diagnosis Debate Engine using ARGUS
# =============================================================================

class MedicalDebateEngine:
    """
    Medical diagnosis engine using ARGUS debate framework.
    
    This engine orchestrates a full multi-agent debate for medical diagnosis:
    1. Moderator creates an agenda for the diagnosis debate
    2. Multiple Specialist agents gather evidence from their domains
    3. Refuter challenges weak evidence and generates rebuttals
    4. Jury aggregates evidence and renders final verdict
    5. CDAG tracks all propositions, evidence, and rebuttals with posteriors
    
    The debate continues until convergence or max rounds reached.
    """
    
    def __init__(self, llm=None, max_rounds: int = 2):
        """
        Initialize the medical debate engine.
        
        Args:
            llm: LLM instance (auto-detected if None)
            max_rounds: Maximum debate rounds (default: 2 for faster results)
        """
        self.llm = llm or get_llm()
        self.max_rounds = max_rounds
        
        # ARGUS components - initialized lazily
        self._moderator = None
        self._specialists: dict[str, Any] = {}
        self._refuter = None
        self._jury = None
        self._graph = None
        self._ledger = None
        
        # Medical specialties for diagnosis
        self.specialties = [
            "infectious",
            "cardiology", 
            "neurology",
            "oncology",
            "autoimmune",
            "toxicology",
        ]
        
        self._initialize_argus_components()
    
    def _initialize_argus_components(self):
        """Initialize all ARGUS debate components."""
        try:
            from argus.agents import Moderator, Specialist, Refuter, Jury
            from argus.cdag import CDAG
            from argus.provenance import ProvenanceLedger
            
            # Create moderator
            self._moderator = Moderator(self.llm)
            
            # Create refuter
            self._refuter = Refuter(self.llm)
            
            # Create jury
            self._jury = Jury(self.llm)
            
            # Create specialists for each medical domain
            for specialty in self.specialties:
                self._specialists[specialty] = Specialist(
                    self.llm,
                    domain=specialty,
                )
            
            # Create provenance ledger
            self._ledger = ProvenanceLedger()
            
            logger.info("ARGUS debate components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Could not initialize ARGUS components: {e}")
            logger.warning("Using simplified debate engine")
    
    def run_diagnosis_debate(
        self,
        patient: dict[str, Any],
        case: dict[str, Any],
    ) -> MedicalDebateResult:
        """
        Run a full diagnosis debate using ARGUS framework.
        
        This orchestrates:
        1. Creating propositions for potential diagnoses
        2. Gathering evidence from multiple specialists
        3. Generating rebuttals to challenge weak evidence
        4. Computing Bayesian posteriors
        5. Rendering final verdict via jury
        
        Args:
            patient: Patient data dictionary
            case: Case data with chief complaint and symptoms
            
        Returns:
            MedicalDebateResult with complete debate output
        """
        start_time = datetime.utcnow()
        
        patient_id = patient.get("id", "unknown")
        case_id = case.get("id", "unknown")
        
        # Check if ARGUS components are available
        if self._moderator is None:
            return self._run_simplified_diagnosis(patient, case)
        
        try:
            return self._run_argus_debate(patient, case, start_time)
        except Exception as e:
            logger.error(f"ARGUS debate failed: {e}")
            return self._run_simplified_diagnosis(patient, case)
    
    def _run_argus_debate(
        self,
        patient: dict[str, Any],
        case: dict[str, Any],
        start_time: datetime,
    ) -> MedicalDebateResult:
        """Run full ARGUS debate for medical diagnosis."""
        from argus.cdag import CDAG, Proposition, Evidence as ArgusEvidence, EdgeType
        from argus.cdag.nodes import EvidenceType
        from argus.cdag.propagation import compute_posterior, compute_all_posteriors
        from argus.provenance import EventType
        
        patient_id = patient.get("id", "unknown")
        case_id = case.get("id", "unknown")
        chief_complaint = case.get("chief_complaint", "Unknown complaint")
        symptoms = case.get("symptoms", [])
        
        # Create new CDAG for this diagnosis
        self._graph = CDAG(name=f"diagnosis_{case_id}")
        
        # Record session start
        self._ledger.record(
            EventType.SESSION_START,
            attributes={"case_id": case_id, "patient_id": patient_id},
        )
        
        # Step 1: Generate initial diagnosis hypotheses
        logger.info("Generating diagnosis hypotheses...")
        hypotheses = self._generate_diagnosis_hypotheses(patient, case)
        logger.info(f"Generated {len(hypotheses)} hypotheses: {[h['diagnosis'] for h in hypotheses]}")
        
        # Create propositions for each hypothesis
        propositions = {}
        for hyp in hypotheses:
            prop = Proposition(
                text=f"Patient has {hyp['diagnosis']}",
                prior=hyp.get("initial_confidence", 0.5),
                domain="medical",
            )
            self._graph.add_proposition(prop)
            propositions[hyp["diagnosis"]] = prop
            
            self._ledger.record(
                EventType.PROPOSITION_ADDED,
                entity_id=prop.id,
                attributes={"diagnosis": hyp["diagnosis"], "prior": prop.prior},
            )
        
        # Step 2: Run debate rounds
        all_evidence = []
        all_rebuttals = []
        round_num = 0
        
        logger.info(f"Starting debate with max_rounds={self.max_rounds}, {len(self._specialists)} specialists")
        
        while round_num < self.max_rounds:
            round_num += 1
            logger.info(f"=== Diagnosis debate round {round_num}/{self.max_rounds} ===")
            
            # Each specialist gathers evidence
            for specialty, specialist in self._specialists.items():
                logger.info(f"  Consulting {specialty} specialist...")
                spec_evidence = self._gather_specialist_evidence(
                    specialist,
                    specialty,
                    patient,
                    case,
                    propositions,
                )
                logger.info(f"  {specialty} specialist provided {len(spec_evidence)} evidence items")
                
                for ev_data in spec_evidence:
                    # Create ARGUS Evidence node
                    argus_ev = ArgusEvidence(
                        text=ev_data["content"],
                        evidence_type=EvidenceType.EXPERT,
                        polarity=1 if ev_data.get("supports", True) else -1,
                        confidence=ev_data.get("confidence", 0.7),
                        metadata={
                            "specialty": specialty,
                            "source": ev_data.get("source", "specialist_analysis"),
                        },
                    )
                    
                    # Link to relevant propositions
                    diagnosis = ev_data.get("diagnosis", "")
                    if diagnosis in propositions:
                        edge_type = EdgeType.SUPPORTS if ev_data.get("supports", True) else EdgeType.ATTACKS
                        self._graph.add_evidence(argus_ev, propositions[diagnosis].id, edge_type)
                        
                        self._ledger.record(
                            EventType.EVIDENCE_ADDED,
                            agent_id=specialist.name,
                            entity_id=argus_ev.id,
                            attributes={"specialty": specialty},
                        )
                    
                    all_evidence.append(ev_data)
            
            # Refuter generates rebuttals to challenge evidence
            for diagnosis, prop in propositions.items():
                rebuttals = self._refuter.generate_rebuttals(self._graph, prop.id)
                
                for rebuttal in rebuttals:
                    all_rebuttals.append({
                        "id": rebuttal.id,
                        "diagnosis": diagnosis,
                        "content": rebuttal.text,
                        "target_id": rebuttal.target_id,
                        "strength": rebuttal.confidence,
                    })
                    
                    self._ledger.record(
                        EventType.REBUTTAL_ADDED,
                        agent_id=self._refuter.name,
                        entity_id=rebuttal.id,
                    )
            
            # Compute posteriors for all propositions
            compute_all_posteriors(self._graph)
            
            # Check for convergence via moderator
            should_stop, reason = self._moderator.should_stop(self._graph)
            if should_stop:
                logger.info(f"Debate converged: {reason}")
                break
        
        # Step 3: Jury renders final verdicts
        verdicts = []
        for diagnosis, prop in propositions.items():
            verdict = self._jury.evaluate(self._graph, prop.id)
            verdicts.append({
                "diagnosis": diagnosis,
                "posterior": verdict.posterior,
                "label": verdict.label,
                "confidence": verdict.confidence,
                "reasoning": verdict.reasoning,
                "top_support": verdict.top_support,
                "top_attack": verdict.top_attack,
            })
            
            self._ledger.record(
                EventType.VERDICT_RENDERED,
                agent_id=self._jury.name,
                entity_id=prop.id,
                attributes={"label": verdict.label, "posterior": verdict.posterior},
            )
        
        # Sort by posterior to get primary diagnosis
        verdicts.sort(key=lambda v: v["posterior"], reverse=True)
        primary = verdicts[0] if verdicts else {"diagnosis": "Unknown", "posterior": 0.0, "label": "undecided", "reasoning": ""}
        
        # Generate test recommendations
        recommended_tests = self._generate_test_recommendations(patient, verdicts)
        
        # Generate treatment recommendations
        recommended_treatments = self._generate_treatment_recommendations(patient, primary)
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        self._ledger.record(EventType.SESSION_END)
        
        return MedicalDebateResult(
            patient_id=patient_id,
            case_id=case_id,
            primary_diagnosis=primary["diagnosis"],
            primary_confidence=primary["posterior"],
            verdict_label=primary["label"],
            diagnoses=verdicts,
            evidence_items=all_evidence,
            rebuttals=all_rebuttals,
            num_rounds=round_num,
            num_evidence=len(all_evidence),
            num_rebuttals=len(all_rebuttals),
            reasoning=primary["reasoning"],
            specialist_contributions=self._summarize_specialist_contributions(all_evidence),
            recommended_tests=recommended_tests,
            recommended_treatments=recommended_treatments,
            duration_seconds=duration,
        )
    
    def _generate_diagnosis_hypotheses(
        self,
        patient: dict[str, Any],
        case: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate initial diagnosis hypotheses based on presentation."""
        prompt = f"""You are a medical AI generating differential diagnosis hypotheses.

Patient Information:
- Age: {patient.get('age', 'Unknown')}
- Gender: {patient.get('gender', 'Unknown')}
- Chief Complaint: {case.get('chief_complaint', 'Unknown')}
- Symptoms: {', '.join(case.get('symptoms', []))}

Generate 3-5 initial diagnosis hypotheses with estimated prior probabilities.
Return as JSON array:
[
    {{"diagnosis": "Diagnosis Name", "initial_confidence": 0.35, "rationale": "Brief reason"}},
    ...
]"""
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        try:
            # Extract JSON from response
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Default hypotheses based on symptoms
        return [
            {"diagnosis": "Viral Infection", "initial_confidence": 0.3, "rationale": "Common cause"},
            {"diagnosis": "Bacterial Infection", "initial_confidence": 0.25, "rationale": "Needs investigation"},
            {"diagnosis": "Inflammatory Condition", "initial_confidence": 0.2, "rationale": "Consider autoimmune"},
        ]
    
    def _gather_specialist_evidence(
        self,
        specialist,
        specialty: str,
        patient: dict[str, Any],
        case: dict[str, Any],
        propositions: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Gather evidence from a specialist agent."""
        prompt = f"""You are a {specialty} specialist analyzing this patient case.

Patient: {patient.get('name', 'Unknown')}, {patient.get('age', '?')} y/o {patient.get('gender', 'Unknown')}
Chief Complaint: {case.get('chief_complaint', 'Unknown')}
Symptoms: {', '.join(case.get('symptoms', []))}

Current diagnostic hypotheses under consideration:
{chr(10).join(f"- {d}" for d in propositions.keys())}

Provide evidence from your specialty that supports or refutes each hypothesis.
Return as JSON array:
[
    {{
        "diagnosis": "Hypothesis Name",
        "supports": true,
        "content": "Detailed evidence text",
        "confidence": 0.75,
        "source": "clinical_observation"
    }},
    ...
]"""
        
        response = self.llm.generate(prompt, temperature=0.4)
        
        try:
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                evidence = json.loads(content[start:end])
                for ev in evidence:
                    ev["specialty"] = specialty
                return evidence
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return []
    
    def _generate_test_recommendations(
        self,
        patient: dict[str, Any],
        verdicts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate recommended diagnostic tests."""
        diagnoses_text = "\n".join(
            f"- {v['diagnosis']} (confidence: {v['posterior']:.0%})"
            for v in verdicts[:3]
        )
        
        prompt = f"""Based on these diagnostic hypotheses:
{diagnoses_text}

Patient age: {patient.get('age', 'Unknown')}

Recommend diagnostic tests to confirm or rule out these diagnoses.
Return as JSON array:
[
    {{
        "test_name": "Test Name",
        "rationale": "Why this test helps",
        "priority": "high",
        "turnaround_time": "4-6 hours"
    }},
    ...
]"""
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        try:
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return [
            {"test_name": "Complete Blood Count (CBC)", "rationale": "Baseline evaluation", "priority": "high", "turnaround_time": "4-6 hours"},
            {"test_name": "Comprehensive Metabolic Panel", "rationale": "Assess organ function", "priority": "high", "turnaround_time": "4-6 hours"},
        ]
    
    def _generate_treatment_recommendations(
        self,
        patient: dict[str, Any],
        primary_verdict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate treatment recommendations for primary diagnosis."""
        if primary_verdict.get("posterior", 0) < 0.3:
            return [{"recommendation": "Further diagnostic workup needed before treatment", "priority": "high"}]
        
        prompt = f"""Primary diagnosis: {primary_verdict.get('diagnosis', 'Unknown')}
Confidence: {primary_verdict.get('posterior', 0):.0%}
Patient age: {patient.get('age', 'Unknown')}

Provide treatment recommendations.
Return as JSON array:
[
    {{
        "recommendation": "Treatment recommendation",
        "rationale": "Why recommended",
        "priority": "high"
    }},
    ...
]"""
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        try:
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return []
    
    def _summarize_specialist_contributions(
        self,
        all_evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarize contributions from each specialist."""
        contributions = {}
        for ev in all_evidence:
            specialty = ev.get("specialty", "unknown")
            if specialty not in contributions:
                contributions[specialty] = {
                    "evidence_count": 0,
                    "supporting": 0,
                    "attacking": 0,
                }
            contributions[specialty]["evidence_count"] += 1
            if ev.get("supports", True):
                contributions[specialty]["supporting"] += 1
            else:
                contributions[specialty]["attacking"] += 1
        return contributions
    
    def _run_simplified_diagnosis(
        self,
        patient: dict[str, Any],
        case: dict[str, Any],
    ) -> MedicalDebateResult:
        """Run simplified diagnosis when ARGUS is not available."""
        start_time = datetime.utcnow()
        
        patient_id = patient.get("id", "unknown")
        case_id = case.get("id", "unknown")
        
        prompt = f"""You are a medical AI performing differential diagnosis.

Patient Information:
- Age: {patient.get('age', 'Unknown')}
- Gender: {patient.get('gender', 'Unknown')}
- Chief Complaint: {case.get('chief_complaint', 'Unknown')}
- Symptoms: {', '.join(case.get('symptoms', []))}

Provide a differential diagnosis with:
1. Top 3-5 diagnoses ranked by likelihood
2. Confidence score for each (0-100%)
3. Key supporting evidence
4. Recommended tests
5. Treatment recommendations

Return as JSON:
{{
    "diagnoses": [
        {{"diagnosis": "Name", "confidence": 0.75, "evidence": "Supporting points", "label": "supported"}}
    ],
    "reasoning": "Overall clinical reasoning",
    "tests": [{{"test_name": "Name", "rationale": "Why", "priority": "high"}}],
    "treatments": [{{"recommendation": "Treatment", "rationale": "Why", "priority": "high"}}]
}}"""
        
        response = self.llm.generate(prompt, temperature=0.3)
        
        diagnoses = []
        reasoning = ""
        tests = []
        treatments = []
        
        try:
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                diagnoses = data.get("diagnoses", [])
                reasoning = data.get("reasoning", "")
                tests = data.get("tests", [])
                treatments = data.get("treatments", [])
        except (json.JSONDecodeError, AttributeError):
            reasoning = response.content if hasattr(response, 'content') else str(response)
        
        primary = diagnoses[0] if diagnoses else {"diagnosis": "Unknown", "confidence": 0.0, "label": "undecided"}
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return MedicalDebateResult(
            patient_id=patient_id,
            case_id=case_id,
            primary_diagnosis=primary.get("diagnosis", "Unknown"),
            primary_confidence=primary.get("confidence", 0.0),
            verdict_label=primary.get("label", "undecided"),
            diagnoses=[
                {
                    "diagnosis": d.get("diagnosis", ""),
                    "posterior": d.get("confidence", 0.0),
                    "label": d.get("label", "undecided"),
                    "reasoning": d.get("evidence", ""),
                }
                for d in diagnoses
            ],
            reasoning=reasoning,
            recommended_tests=tests,
            recommended_treatments=treatments,
            duration_seconds=duration,
        )
    
    def get_specialist_analysis(
        self,
        specialty: str,
        patient: dict[str, Any],
        case: dict[str, Any],
    ) -> dict[str, Any]:
        """Get analysis from a specific specialist agent."""
        if specialty not in self._specialists:
            return {"error": f"Unknown specialty: {specialty}"}
        
        specialist = self._specialists[specialty]
        
        prompt = f"""You are a {specialty} specialist AI assistant.

Patient Case:
- Name: {patient.get('name', 'Unknown')}
- Age: {patient.get('age', 'Unknown')}
- Gender: {patient.get('gender', 'Unknown')}
- Chief Complaint: {case.get('chief_complaint', 'Unknown')}
- Symptoms: {', '.join(case.get('symptoms', []))}

Provide your specialized analysis including:
1. Most likely {specialty}-related diagnoses
2. Confidence level for each (0-100%)
3. Supporting rationale
4. Recommended {specialty}-specific tests
5. Treatment considerations

Return as JSON:
{{
    "specialty": "{specialty}",
    "diagnoses": [{{"diagnosis": "Name", "confidence": 0.75, "rationale": "Why"}}],
    "tests": [{{"test_name": "Name", "rationale": "Why"}}],
    "treatment_notes": "Treatment considerations"
}}"""
        
        response = self.llm.generate(prompt, temperature=0.4)
        
        try:
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return {
            "specialty": specialty,
            "analysis": response.content if hasattr(response, 'content') else str(response),
        }


# =============================================================================
# Convenience functions for BioSage Terminal
# =============================================================================

# Global engine instance
_engine: Optional[MedicalDebateEngine] = None


def get_diagnosis_engine(llm=None) -> MedicalDebateEngine:
    """Get or create the medical diagnosis engine."""
    global _engine
    if _engine is None or llm is not None:
        _engine = MedicalDebateEngine(llm=llm)
    return _engine


def run_diagnosis(patient: dict[str, Any], case: dict[str, Any]) -> MedicalDebateResult:
    """
    Run a medical diagnosis using ARGUS debate framework.
    
    Args:
        patient: Patient data dictionary
        case: Case data with chief_complaint and symptoms
        
    Returns:
        MedicalDebateResult with complete diagnosis
    """
    engine = get_diagnosis_engine()
    return engine.run_diagnosis_debate(patient, case)


def check_llm_availability() -> dict[str, Any]:
    """Check which LLM providers are available."""
    available = []
    unavailable = []
    
    for provider, env_key, env_model, default_model in LLMConfig.PROVIDER_PRIORITY:
        if env_key is None:
            # Check local provider
            if LLMConfig._check_ollama_available():
                model = os.getenv(env_model, default_model)
                available.append({"provider": provider, "model": model})
            else:
                unavailable.append({"provider": provider, "reason": "Ollama not running"})
        else:
            api_key = os.getenv(env_key)
            if api_key:
                model = os.getenv(env_model, default_model)
                available.append({"provider": provider, "model": model})
            else:
                unavailable.append({"provider": provider, "reason": f"{env_key} not set"})
    
    return {
        "available": available,
        "unavailable": unavailable,
        "recommended": available[0] if available else None,
        "argus_available": _check_argus_available(),
    }


def _check_argus_available() -> bool:
    """Check if ARGUS framework is available."""
    try:
        from argus import RDCOrchestrator, CDAG
        return True
    except ImportError:
        return False


def get_debate_graph_visualization(result: MedicalDebateResult) -> dict[str, Any]:
    """
    Generate visualization data for the debate graph.
    For use with plotting/reports.
    """
    nodes = []
    edges = []
    
    # Add diagnosis nodes
    for i, diag in enumerate(result.diagnoses):
        nodes.append({
            "id": f"diag_{i}",
            "type": "proposition",
            "label": diag.get("diagnosis", ""),
            "score": diag.get("posterior", 0.0),
        })
    
    # Add evidence nodes
    for i, ev in enumerate(result.evidence_items):
        nodes.append({
            "id": f"ev_{i}",
            "type": "evidence",
            "label": ev.get("content", "")[:50],
            "specialty": ev.get("specialty", ""),
        })
        
        # Find connected diagnosis
        diagnosis = ev.get("diagnosis", "")
        for j, diag in enumerate(result.diagnoses):
            if diag.get("diagnosis", "") == diagnosis:
                edges.append({
                    "source": f"ev_{i}",
                    "target": f"diag_{j}",
                    "type": "supports" if ev.get("supports", True) else "attacks",
                })
                break
    
    # Add rebuttal nodes
    for i, reb in enumerate(result.rebuttals):
        nodes.append({
            "id": f"reb_{i}",
            "type": "rebuttal",
            "label": reb.get("content", "")[:50],
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "num_propositions": len(result.diagnoses),
            "num_evidence": result.num_evidence,
            "num_rebuttals": result.num_rebuttals,
            "num_rounds": result.num_rounds,
        },
    }
