"""
Diagnosis Progress screen for BioSage Terminal.
Displays loading animation while running AI diagnosis using ARGUS debate framework.
"""

import asyncio
from datetime import datetime
from typing import Optional

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Center
from textual.widgets import Static, Footer, ProgressBar
from textual.binding import Binding
from textual.timer import Timer
from rich.text import Text

from biosage_terminal.storage import get_data_store
from biosage_terminal.ai import (
    MedicalDebateEngine,
    get_diagnosis_engine,
    check_llm_availability,
    MedicalDebateResult,
)
from biosage_terminal.models import Specialty


class DiagnosisProgressScreen(Screen):
    """Screen showing diagnosis progress with animated loading."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    
    CSS = """
    DiagnosisProgressScreen {
        background: #0F172A;
        align: center middle;
    }
    
    .progress-container {
        align: center middle;
        width: auto;
        height: auto;
        padding: 4;
        content-align: center middle;
    }
    
    .progress-title {
        text-align: center;
        width: 100%;
        content-align: center middle;
        margin-bottom: 2;
    }
    
    .progress-bar-container {
        width: 60;
        height: 3;
        margin: 2 0;
        text-align: center;
        content-align: center middle;
    }
    
    .progress-message {
        text-align: center;
        width: 100%;
        content-align: center middle;
        margin: 1 0;
    }
    
    .spinner {
        text-align: center;
        width: 100%;
        content-align: center middle;
        margin-top: 2;
    }
    
    .status-list {
        margin-top: 3;
        padding: 1;
        text-align: center;
        content-align: center middle;
    }
    """
    
    SPINNER_FRAMES = ["|", "/", "-", "\\"]
    
    DIAGNOSIS_STEPS = [
        ("Initializing ARGUS debate engine...", 0.05),
        ("Loading patient data...", 0.10),
        ("Creating diagnosis propositions...", 0.15),
        ("Moderator creating debate agenda...", 0.20),
        ("Specialists gathering evidence...", 0.35),
        ("Refuter generating rebuttals...", 0.55),
        ("Computing Bayesian posteriors...", 0.70),
        ("Jury rendering verdict...", 0.85),
        ("Generating recommendations...", 0.92),
        ("Finalizing diagnosis...", 0.98),
        ("Complete!", 1.0),
    ]
    
    def __init__(self, case_id: str = "", patient_id: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_id = case_id
        self.patient_id = patient_id
        self.progress = 0.0
        self.current_step = 0
        self.message = "Initializing..."
        self.spinner_frame = 0
        self._timer: Optional[Timer] = None
        self._running = False
        self.completed_steps = []
    
    def compose(self) -> ComposeResult:
        with Center():
            with Container(classes="progress-container"):
                yield Static(self._render_title(), id="title", classes="progress-title")
                yield Static(self._render_progress(), id="progress", classes="progress-bar-container")
                yield Static(self._render_message(), id="message", classes="progress-message")
                yield Static(self._render_spinner(), id="spinner", classes="spinner")
                yield Static(self._render_status_list(), id="status_list", classes="status-list")
        
        yield Footer()
    
    def _render_title(self) -> Text:
        """Render the title."""
        text = Text(justify="center")
        text.append("\n", style="")
        text.append("Running AI Diagnosis\n", style="bold #3B82F6")
        text.append("=" * 40, style="#334155")
        return text
    
    def _render_progress(self) -> Text:
        """Render the progress bar."""
        text = Text(justify="center")
        
        bar_width = 50
        filled = int(self.progress * bar_width)
        empty = bar_width - filled
        percentage = int(self.progress * 100)
        
        text.append("[", style="#334155")
        text.append("#" * filled, style="#3B82F6")
        text.append("-" * empty, style="#334155")
        text.append(f"] {percentage}%", style="#F1F5F9")
        
        return text
    
    def _render_message(self) -> Text:
        """Render the current message."""
        text = Text(justify="center")
        text.append(self.message, style="#94A3B8")
        return text
    
    def _render_spinner(self) -> Text:
        """Render the spinner animation."""
        text = Text(justify="center")
        if self.progress < 1.0:
            spinner = self.SPINNER_FRAMES[self.spinner_frame % len(self.SPINNER_FRAMES)]
            text.append(spinner, style="bold #06B6D4")
        else:
            text.append("[OK]", style="bold #22C55E")
        return text
    
    def _render_status_list(self) -> Text:
        """Render the list of completed steps."""
        text = Text()
        
        for step in self.completed_steps[-6:]:
            text.append("[OK] ", style="bold #22C55E")
            text.append(f"{step}\n", style="#64748B")
        
        return text
    
    def on_mount(self) -> None:
        """Start the diagnosis process when screen is mounted."""
        self._running = True
        self._timer = self.set_interval(0.1, self._update_spinner)
        self.run_worker(self._run_diagnosis())
    
    def _update_spinner(self) -> None:
        """Update the spinner animation."""
        if self.progress < 1.0:
            self.spinner_frame += 1
            try:
                self.query_one("#spinner", Static).update(self._render_spinner())
            except Exception:
                pass
    
    async def _run_diagnosis(self) -> None:
        """Run the actual diagnosis process."""
        store = get_data_store()
        
        # Load case and patient data
        case_data = store.cases.load_sync(self.case_id)
        if not case_data:
            self._update_progress(1.0, "Error: Case not found")
            await asyncio.sleep(2)
            self.app.pop_screen()
            return
        
        patient_data = store.patients.load_sync(case_data.get("patient_id", ""))
        if not patient_data:
            self._update_progress(1.0, "Error: Patient not found")
            await asyncio.sleep(2)
            self.app.pop_screen()
            return
        
        # Check LLM availability
        llm_status = check_llm_availability()
        if not llm_status["recommended"]:
            self._update_progress(1.0, "Error: No LLM configured. Set GEMINI_API_KEY.")
            await asyncio.sleep(3)
            self.app.pop_screen()
            return
        
        try:
            # Step through diagnosis using ARGUS debate framework
            
            # Initialize ARGUS debate engine
            self._update_progress(0.05, "Initializing ARGUS debate engine...")
            await asyncio.sleep(0.3)
            
            engine = get_diagnosis_engine()
            self.completed_steps.append("ARGUS debate engine initialized")
            
            # Load patient data
            self._update_progress(0.10, "Loading patient data...")
            await asyncio.sleep(0.2)
            
            # Format patient data for debate engine
            patient_dict = {
                "id": patient_data.get("id", ""),
                "name": patient_data.get("name", "Unknown"),
                "age": self._calculate_age(patient_data),
                "gender": patient_data.get("gender", "unknown"),
                "conditions": patient_data.get("conditions", []),
                "medications": patient_data.get("medications", []),
                "allergies": patient_data.get("allergies", []),
                "vitals": patient_data.get("vitals_history", [{}])[-1] if patient_data.get("vitals_history") else {},
            }
            
            # Format case data
            case_dict = {
                "id": self.case_id,
                "chief_complaint": case_data.get("chief_complaint", ""),
                "symptoms": case_data.get("symptoms", []),
            }
            
            self.completed_steps.append("Patient data loaded")
            
            # Creating propositions
            self._update_progress(0.15, "Creating diagnosis propositions...")
            await asyncio.sleep(0.3)
            self.completed_steps.append("Propositions created")
            
            # Moderator creates agenda
            self._update_progress(0.20, "Moderator creating debate agenda...")
            await asyncio.sleep(0.3)
            self.completed_steps.append("Debate agenda created")
            
            # Run the full ARGUS debate
            self._update_progress(0.35, "Specialists gathering evidence...")
            await asyncio.sleep(0.2)
            
            # Log start of debate
            self.app.log.info("Starting ARGUS debate execution...")
            
            # Execute debate in a separate thread to avoid blocking UI
            # This runs all specialists, refuter, and jury
            # Note: This may take 20-40 seconds depending on LLM speed
            loop = asyncio.get_event_loop()
            
            self.app.log.info("Calling engine.run_diagnosis_debate() in background thread...")
            
            # Start debate in background
            debate_task = loop.run_in_executor(
                None,
                engine.run_diagnosis_debate,
                patient_dict,
                case_dict
            )
            
            # Simulate progress while debate runs (since we can't get real-time updates)
            progress_messages = [
                (0.40, "Infectious disease specialist analyzing..."),
                (0.45, "Cardiology specialist analyzing..."),
                (0.50, "Neurology specialist analyzing..."),
                (0.53, "Oncology specialist analyzing..."),
                (0.56, "Autoimmune specialist analyzing..."),
                (0.59, "Toxicology specialist analyzing..."),
            ]
            
            for prog, msg in progress_messages:
                if debate_task.done():
                    break
                self._update_progress(prog, msg)
                await asyncio.sleep(1.5)
            
            # Wait for debate to complete
            self._update_progress(0.60, "Debate in progress, please wait...")
            try:
                result: MedicalDebateResult = await debate_task
                self.app.log.info(f"Debate completed: {result.num_rounds} rounds, {result.num_evidence} evidence, {result.num_rebuttals} rebuttals")
            except Exception as e:
                self.app.log.error(f"Debate execution failed: {e}")
                raise
            
            # Update progress based on debate stages
            self._update_progress(0.55, "Refuter generating rebuttals...")
            await asyncio.sleep(0.2)
            self.completed_steps.append(f"Generated {result.num_rebuttals} rebuttals")
            
            self._update_progress(0.70, "Computing Bayesian posteriors...")
            await asyncio.sleep(0.2)
            self.completed_steps.append(f"Ran {result.num_rounds} debate rounds")
            
            self._update_progress(0.85, "Jury rendering verdict...")
            await asyncio.sleep(0.2)
            self.completed_steps.append(f"Jury verdict: {result.verdict_label}")
            
            # Generate recommendations
            self._update_progress(0.92, "Generating recommendations...")
            await asyncio.sleep(0.2)
            self.completed_steps.append("Recommendations generated")
            
            # Finalize
            self._update_progress(0.98, "Finalizing diagnosis...")
            
            # Update case data with debate results
            case_data["primary_diagnosis"] = result.primary_diagnosis
            case_data["primary_confidence"] = result.primary_confidence
            case_data["verdict_label"] = result.verdict_label
            case_data["diagnoses"] = result.diagnoses
            case_data["evidence_items"] = result.evidence_items
            case_data["rebuttals"] = result.rebuttals
            case_data["test_recommendations"] = result.recommended_tests
            case_data["treatment_recommendations"] = result.recommended_treatments
            case_data["reasoning"] = result.reasoning
            case_data["specialist_contributions"] = result.specialist_contributions
            case_data["debate_statistics"] = {
                "num_rounds": result.num_rounds,
                "num_evidence": result.num_evidence,
                "num_rebuttals": result.num_rebuttals,
                "duration_seconds": result.duration_seconds,
            }
            case_data["status"] = "diagnosed"
            case_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Save case
            store.cases.save_sync(self.case_id, case_data)
            
            # Log audit event
            store.audit.log_event(
                event_type="argus_diagnosis_completed",
                user="argus_debate_system",
                action=f"ARGUS debate completed for case {case_data.get('case_number', '')}",
                details={
                    "case_id": self.case_id,
                    "primary_diagnosis": result.primary_diagnosis,
                    "primary_confidence": result.primary_confidence,
                    "verdict": result.verdict_label,
                    "num_diagnoses": len(result.diagnoses),
                    "num_evidence": result.num_evidence,
                    "num_rebuttals": result.num_rebuttals,
                    "debate_rounds": result.num_rounds,
                    "duration_seconds": result.duration_seconds,
                },
            )
            
            self.completed_steps.append("Diagnosis finalized")
            self._update_progress(1.0, "ARGUS Diagnosis complete!")
            
            await asyncio.sleep(1)
            
            # Navigate to diagnosis result view
            if self._timer:
                self._timer.stop()
            self.app.pop_screen()
            
            # Build diagnosis data for result screen
            diagnosis_data = {
                "fused": {
                    "differential": [
                        {
                            "diagnosis": d.get("diagnosis", ""),
                            "score_global": d.get("confidence", 0),
                            "why_top": d.get("rationale", ""),
                            "citations": d.get("citations", [])
                        }
                        for d in result.diagnoses
                    ],
                    "next_best_test": {
                        "name": result.recommended_tests[0].get("name", "") if result.recommended_tests else "",
                        "why": result.recommended_tests[0].get("rationale", "") if result.recommended_tests else "",
                        "linked_hypotheses": []
                    },
                    "disagreement_score": 1.0 - result.primary_confidence if result.primary_confidence else 0.2,
                    "test_plans": [
                        {"diagnosis": d.get("diagnosis", ""), "plan": d.get("rationale", "")}
                        for d in result.diagnoses[:3]
                    ]
                },
                "recommendations": [
                    {
                        "title": rec.get("name", rec.get("recommendation", "")),
                        "rationale": rec.get("rationale", ""),
                        "priority": rec.get("priority", "medium")
                    }
                    for rec in result.recommended_treatments + result.recommended_tests
                ]
            }
            
            # Push diagnosis result screen
            from biosage_terminal.screens.diagnosis_result import DiagnosisResultScreen
            result_screen = DiagnosisResultScreen(
                case_id=self.case_id,
                patient_id=self.patient_id,
                diagnosis_data=diagnosis_data
            )
            self.app.push_screen(result_screen)
            
        except Exception as e:
            self._update_progress(1.0, f"Error: {str(e)[:50]}")
            await asyncio.sleep(3)
            self.app.pop_screen()
    
    def _calculate_age(self, patient_data: dict) -> int:
        """Calculate patient age from date of birth."""
        dob = patient_data.get("date_of_birth", "")
        if not dob:
            return 0
        try:
            from datetime import date
            parts = dob.split("-")
            if len(parts) >= 3:
                dob_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
                today = date.today()
                return today.year - dob_date.year - (
                    (today.month, today.day) < (dob_date.month, dob_date.day)
                )
        except (ValueError, IndexError):
            pass
        return 0
    
    def _update_progress(self, progress: float, message: str) -> None:
        """Update progress bar and message."""
        self.progress = progress
        self.message = message
        
        try:
            self.query_one("#progress", Static).update(self._render_progress())
            self.query_one("#message", Static).update(self._render_message())
            self.query_one("#spinner", Static).update(self._render_spinner())
            self.query_one("#status_list", Static).update(self._render_status_list())
        except Exception:
            pass
    
    def action_cancel(self) -> None:
        """Cancel the diagnosis and go back."""
        self._running = False
        if self._timer:
            self._timer.stop()
        self.app.pop_screen()
