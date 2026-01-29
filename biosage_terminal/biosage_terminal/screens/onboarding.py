"""
Patient Onboarding wizard screen for BioSage Terminal.
"""

from datetime import date, datetime
from typing import Optional

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Button, Footer, Input, Select, TextArea, 
    Label, ProgressBar, ListView, ListItem
)
from textual.binding import Binding
from textual.validation import Length, Regex
from rich.text import Text

from biosage_terminal.storage import get_data_store
from biosage_terminal.models import Patient, Gender, BloodType, Vitals


class OnboardingScreen(Screen):
    """Multi-step patient onboarding wizard."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    OnboardingScreen {
        background: #0F172A;
    }
    
    .onboarding-header {
        dock: top;
        height: 3;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 0 2;
    }
    
    .progress-section {
        height: 5;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 1 2;
    }
    
    .step-content {
        height: 1fr;
        padding: 2;
        overflow-y: auto;
    }
    
    .form-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    .form-group {
        width: 1fr;
        margin: 0 1;
        height: auto;
    }
    
    .form-label {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .form-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    .form-help {
        color: #64748B;
        margin-bottom: 1;
    }
    
    .navigation-bar {
        dock: bottom;
        height: 5;
        layout: horizontal;
        align: center middle;
        padding: 1;
        border-top: solid #334155;
        background: #1E293B;
    }
    
    .nav-button {
        width: 20;
        margin: 0 2;
    }
    
    .nav-button.primary {
        background: #3B82F6;
    }
    
    .nav-button.success {
        background: #22C55E;
    }
    
    .list-section {
        height: auto;
        max-height: 15;
        border: solid #334155;
        margin: 1 0;
    }
    
    .add-button {
        margin: 1 0;
    }
    
    .error-message {
        color: #EF4444;
        margin: 1 0;
    }
    
    .step-title {
        text-style: bold;
        color: #F1F5F9;
        margin-bottom: 2;
    }
    """
    
    STEPS = [
        "Demographics",
        "Contact Info",
        "Vitals",
        "Medical History",
        "Medications",
        "Chief Complaint",
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_step = 0
        self.form_data = {
            "name": "",
            "date_of_birth": "",
            "gender": "male",
            "blood_type": "",
            "email": "",
            "phone": "",
            "address": "",
            "emergency_name": "",
            "emergency_phone": "",
            "emergency_relationship": "",
            "temperature": "",
            "heart_rate": "",
            "bp_systolic": "",
            "bp_diastolic": "",
            "respiratory_rate": "",
            "oxygen_saturation": "",
            "weight": "",
            "height": "",
            "conditions": [],
            "allergies": [],
            "medications": [],
            "chief_complaint": "",
            "symptoms": [],
        }
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="onboarding-header")
        
        with Container(classes="progress-section"):
            yield Static(self._render_progress(), id="progress_bar")
        
        with ScrollableContainer(classes="step-content", id="step_content"):
            yield from self._compose_step(0)
        
        with Horizontal(classes="navigation-bar"):
            yield Button("Previous", id="btn_prev", classes="nav-button", disabled=True)
            yield Button("Next", id="btn_next", classes="nav-button primary")
            yield Button("Submit", id="btn_submit", classes="nav-button success", disabled=True)
        
        yield Footer()
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("Patient Onboarding", style="bold #F1F5F9")
        return text
    
    def _render_progress(self) -> Text:
        """Render the progress bar."""
        text = Text()
        
        progress = (self.current_step + 1) / len(self.STEPS)
        bar_width = 50
        filled = int(progress * bar_width)
        empty = bar_width - filled
        
        text.append("[", style="#334155")
        text.append("#" * filled, style="#3B82F6")
        text.append("-" * empty, style="#334155")
        text.append("] ", style="#334155")
        text.append(f"Step {self.current_step + 1}/{len(self.STEPS)}: ", style="#94A3B8")
        text.append(self.STEPS[self.current_step], style="bold #F1F5F9")
        
        return text
    
    def _compose_step(self, step: int):
        """Compose the content for a specific step."""
        if step == 0:
            yield from self._compose_demographics_step()
        elif step == 1:
            yield from self._compose_contact_step()
        elif step == 2:
            yield from self._compose_vitals_step()
        elif step == 3:
            yield from self._compose_history_step()
        elif step == 4:
            yield from self._compose_medications_step()
        elif step == 5:
            yield from self._compose_complaint_step()
    
    def _compose_demographics_step(self):
        """Demographics step content."""
        yield Static("Patient Demographics", classes="step-title")
        
        # Name row
        row1 = Horizontal(classes="form-row")
        group1 = Vertical(classes="form-group")
        group1.compose_add_child(Label("Full Name *", classes="form-label"))
        group1.compose_add_child(Input(
            placeholder="Enter patient full name",
            id="input_name",
            classes="form-input",
            value=self.form_data["name"],
        ))
        row1.compose_add_child(group1)
        yield row1
        
        # DOB and Gender row
        row2 = Horizontal(classes="form-row")
        group2 = Vertical(classes="form-group")
        group2.compose_add_child(Label("Date of Birth *", classes="form-label"))
        group2.compose_add_child(Input(
            placeholder="YYYY-MM-DD",
            id="input_dob",
            classes="form-input",
            value=self.form_data["date_of_birth"],
        ))
        group2.compose_add_child(Static("Format: YYYY-MM-DD", classes="form-help"))
        row2.compose_add_child(group2)
        
        group3 = Vertical(classes="form-group")
        group3.compose_add_child(Label("Gender *", classes="form-label"))
        group3.compose_add_child(Select(
            [(g.value.title(), g.value) for g in Gender],
            id="select_gender",
            value=self.form_data["gender"],
        ))
        row2.compose_add_child(group3)
        yield row2
        
        # Blood type row
        row3 = Horizontal(classes="form-row")
        group4 = Vertical(classes="form-group")
        group4.compose_add_child(Label("Blood Type", classes="form-label"))
        group4.compose_add_child(Select(
            [("Unknown", ""), *[(b.value, b.value) for b in BloodType]],
            id="select_blood",
            value=self.form_data["blood_type"],
        ))
        row3.compose_add_child(group4)
        yield row3
    
    def _compose_contact_step(self):
        """Contact information step content."""
        yield Static("Contact Information", classes="step-title")
        
        # Email and Phone row
        row1 = Horizontal(classes="form-row")
        
        group1 = Vertical(classes="form-group")
        group1.compose_add_child(Label("Email", classes="form-label"))
        group1.compose_add_child(Input(
            placeholder="patient@email.com",
            id="input_email",
            classes="form-input",
            value=self.form_data["email"],
        ))
        row1.compose_add_child(group1)
        
        group2 = Vertical(classes="form-group")
        group2.compose_add_child(Label("Phone", classes="form-label"))
        group2.compose_add_child(Input(
            placeholder="(555) 123-4567",
            id="input_phone",
            classes="form-input",
            value=self.form_data["phone"],
        ))
        row1.compose_add_child(group2)
        yield row1
        
        # Address row
        row2 = Horizontal(classes="form-row")
        group3 = Vertical(classes="form-group")
        group3.compose_add_child(Label("Address", classes="form-label"))
        group3.compose_add_child(Input(
            placeholder="Street, City, State, ZIP",
            id="input_address",
            classes="form-input",
            value=self.form_data["address"],
        ))
        row2.compose_add_child(group3)
        yield row2
        
        yield Static("Emergency Contact", classes="step-title")
        
        # Emergency contact row
        row3 = Horizontal(classes="form-row")
        
        group4 = Vertical(classes="form-group")
        group4.compose_add_child(Label("Name", classes="form-label"))
        group4.compose_add_child(Input(
            placeholder="Emergency contact name",
            id="input_emergency_name",
            classes="form-input",
            value=self.form_data["emergency_name"],
        ))
        row3.compose_add_child(group4)
        
        group5 = Vertical(classes="form-group")
        group5.compose_add_child(Label("Phone", classes="form-label"))
        group5.compose_add_child(Input(
            placeholder="(555) 987-6543",
            id="input_emergency_phone",
            classes="form-input",
            value=self.form_data["emergency_phone"],
        ))
        row3.compose_add_child(group5)
        
        group6 = Vertical(classes="form-group")
        group6.compose_add_child(Label("Relationship", classes="form-label"))
        group6.compose_add_child(Input(
            placeholder="Spouse, Parent, etc.",
            id="input_emergency_rel",
            classes="form-input",
            value=self.form_data["emergency_relationship"],
        ))
        row3.compose_add_child(group6)
        yield row3
    
    def _compose_vitals_step(self):
        """Vitals step content."""
        yield Static("Current Vital Signs", classes="step-title")
        
        # Temp, HR, RR row
        row1 = Horizontal(classes="form-row")
        
        group1 = Vertical(classes="form-group")
        group1.compose_add_child(Label("Temperature (F)", classes="form-label"))
        group1.compose_add_child(Input(
            placeholder="98.6",
            id="input_temp",
            classes="form-input",
            value=self.form_data["temperature"],
        ))
        row1.compose_add_child(group1)
        
        group2 = Vertical(classes="form-group")
        group2.compose_add_child(Label("Heart Rate (bpm)", classes="form-label"))
        group2.compose_add_child(Input(
            placeholder="72",
            id="input_hr",
            classes="form-input",
            value=self.form_data["heart_rate"],
        ))
        row1.compose_add_child(group2)
        
        group3 = Vertical(classes="form-group")
        group3.compose_add_child(Label("Respiratory Rate", classes="form-label"))
        group3.compose_add_child(Input(
            placeholder="16",
            id="input_rr",
            classes="form-input",
            value=self.form_data["respiratory_rate"],
        ))
        row1.compose_add_child(group3)
        yield row1
        
        # BP and SpO2 row
        row2 = Horizontal(classes="form-row")
        
        group4 = Vertical(classes="form-group")
        group4.compose_add_child(Label("Blood Pressure (Systolic)", classes="form-label"))
        group4.compose_add_child(Input(
            placeholder="120",
            id="input_bp_sys",
            classes="form-input",
            value=self.form_data["bp_systolic"],
        ))
        row2.compose_add_child(group4)
        
        group5 = Vertical(classes="form-group")
        group5.compose_add_child(Label("Blood Pressure (Diastolic)", classes="form-label"))
        group5.compose_add_child(Input(
            placeholder="80",
            id="input_bp_dia",
            classes="form-input",
            value=self.form_data["bp_diastolic"],
        ))
        row2.compose_add_child(group5)
        
        group6 = Vertical(classes="form-group")
        group6.compose_add_child(Label("SpO2 (%)", classes="form-label"))
        group6.compose_add_child(Input(
            placeholder="98",
            id="input_spo2",
            classes="form-input",
            value=self.form_data["oxygen_saturation"],
        ))
        row2.compose_add_child(group6)
        yield row2
        
        # Weight and Height row
        row3 = Horizontal(classes="form-row")
        
        group7 = Vertical(classes="form-group")
        group7.compose_add_child(Label("Weight (kg)", classes="form-label"))
        group7.compose_add_child(Input(
            placeholder="70",
            id="input_weight",
            classes="form-input",
            value=self.form_data["weight"],
        ))
        row3.compose_add_child(group7)
        
        group8 = Vertical(classes="form-group")
        group8.compose_add_child(Label("Height (cm)", classes="form-label"))
        group8.compose_add_child(Input(
            placeholder="175",
            id="input_height",
            classes="form-input",
            value=self.form_data["height"],
        ))
        row3.compose_add_child(group8)
        yield row3
    
    def _compose_history_step(self):
        """Medical history step content."""
        yield Static("Medical History", classes="step-title")
        
        yield Label("Medical Conditions", classes="form-label")
        yield Static(self._render_list_items(self.form_data["conditions"], "conditions"), 
                    id="conditions_list", classes="list-section")
        
        row1 = Horizontal(classes="form-row")
        group1 = Vertical(classes="form-group")
        group1.compose_add_child(Input(
            placeholder="Add a medical condition...",
            id="input_condition",
            classes="form-input",
        ))
        row1.compose_add_child(group1)
        row1.compose_add_child(Button("Add", id="btn_add_condition", classes="add-button"))
        yield row1
        
        yield Label("Allergies", classes="form-label")
        yield Static(self._render_list_items(self.form_data["allergies"], "allergies"), 
                    id="allergies_list", classes="list-section")
        
        row2 = Horizontal(classes="form-row")
        group2 = Vertical(classes="form-group")
        group2.compose_add_child(Input(
            placeholder="Add an allergy...",
            id="input_allergy",
            classes="form-input",
        ))
        row2.compose_add_child(group2)
        row2.compose_add_child(Button("Add", id="btn_add_allergy", classes="add-button"))
        yield row2
    
    def _compose_medications_step(self):
        """Medications step content."""
        yield Static("Current Medications", classes="step-title")
        
        yield Static(self._render_list_items(self.form_data["medications"], "medications"), 
                    id="medications_list", classes="list-section")
        
        row1 = Horizontal(classes="form-row")
        
        group1 = Vertical(classes="form-group")
        group1.compose_add_child(Label("Medication Name", classes="form-label"))
        group1.compose_add_child(Input(
            placeholder="Medication name",
            id="input_med_name",
            classes="form-input",
        ))
        row1.compose_add_child(group1)
        
        group2 = Vertical(classes="form-group")
        group2.compose_add_child(Label("Dosage", classes="form-label"))
        group2.compose_add_child(Input(
            placeholder="10mg",
            id="input_med_dosage",
            classes="form-input",
        ))
        row1.compose_add_child(group2)
        
        group3 = Vertical(classes="form-group")
        group3.compose_add_child(Label("Frequency", classes="form-label"))
        group3.compose_add_child(Input(
            placeholder="Once daily",
            id="input_med_freq",
            classes="form-input",
        ))
        row1.compose_add_child(group3)
        yield row1
        
        yield Button("Add Medication", id="btn_add_medication", classes="add-button")
    
    def _compose_complaint_step(self):
        """Chief complaint step content."""
        yield Static("Chief Complaint & Symptoms", classes="step-title")
        
        yield Label("Chief Complaint *", classes="form-label")
        yield TextArea(
            id="input_complaint",
            classes="form-input",
        )
        yield Static("Describe the main reason for the visit", classes="form-help")
        
        yield Label("Symptoms", classes="form-label")
        yield Static(self._render_list_items(self.form_data["symptoms"], "symptoms"), 
                    id="symptoms_list", classes="list-section")
        
        row1 = Horizontal(classes="form-row")
        group1 = Vertical(classes="form-group")
        group1.compose_add_child(Input(
            placeholder="Add a symptom...",
            id="input_symptom",
            classes="form-input",
        ))
        row1.compose_add_child(group1)
        row1.compose_add_child(Button("Add", id="btn_add_symptom", classes="add-button"))
        yield row1
    
    def _render_list_items(self, items: list, item_type: str) -> Text:
        """Render a list of items."""
        text = Text()
        if not items:
            text.append("  No items added yet", style="#64748B")
        else:
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    if item_type == "medications":
                        display = f"{item.get('name', '')} {item.get('dosage', '')} - {item.get('frequency', '')}"
                    else:
                        display = str(item)
                else:
                    display = str(item)
                text.append(f"  [{i+1}] {display}\n", style="#F1F5F9")
        return text
    
    def _save_current_step(self) -> None:
        """Save data from current step inputs."""
        if self.current_step == 0:
            try:
                self.form_data["name"] = self.query_one("#input_name", Input).value
                self.form_data["date_of_birth"] = self.query_one("#input_dob", Input).value
                self.form_data["gender"] = self.query_one("#select_gender", Select).value
                self.form_data["blood_type"] = self.query_one("#select_blood", Select).value or ""
            except Exception:
                pass
        elif self.current_step == 1:
            try:
                self.form_data["email"] = self.query_one("#input_email", Input).value
                self.form_data["phone"] = self.query_one("#input_phone", Input).value
                self.form_data["address"] = self.query_one("#input_address", Input).value
                self.form_data["emergency_name"] = self.query_one("#input_emergency_name", Input).value
                self.form_data["emergency_phone"] = self.query_one("#input_emergency_phone", Input).value
                self.form_data["emergency_relationship"] = self.query_one("#input_emergency_rel", Input).value
            except Exception:
                pass
        elif self.current_step == 2:
            try:
                self.form_data["temperature"] = self.query_one("#input_temp", Input).value
                self.form_data["heart_rate"] = self.query_one("#input_hr", Input).value
                self.form_data["respiratory_rate"] = self.query_one("#input_rr", Input).value
                self.form_data["bp_systolic"] = self.query_one("#input_bp_sys", Input).value
                self.form_data["bp_diastolic"] = self.query_one("#input_bp_dia", Input).value
                self.form_data["oxygen_saturation"] = self.query_one("#input_spo2", Input).value
                self.form_data["weight"] = self.query_one("#input_weight", Input).value
                self.form_data["height"] = self.query_one("#input_height", Input).value
            except Exception:
                pass
        elif self.current_step == 5:
            try:
                self.form_data["chief_complaint"] = self.query_one("#input_complaint", TextArea).text
            except Exception:
                pass
    
    def _goto_step(self, step: int) -> None:
        """Navigate to a specific step."""
        self._save_current_step()
        self.current_step = step
        
        # Update progress
        progress_widget = self.query_one("#progress_bar", Static)
        progress_widget.update(self._render_progress())
        
        # Update content
        content = self.query_one("#step_content", ScrollableContainer)
        content.remove_children()
        content.mount(*list(self._compose_step(step)))
        
        # Update navigation buttons
        btn_prev = self.query_one("#btn_prev", Button)
        btn_next = self.query_one("#btn_next", Button)
        btn_submit = self.query_one("#btn_submit", Button)
        
        btn_prev.disabled = step == 0
        btn_next.disabled = step == len(self.STEPS) - 1
        btn_submit.disabled = step != len(self.STEPS) - 1
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_prev":
            if self.current_step > 0:
                self._goto_step(self.current_step - 1)
        elif button_id == "btn_next":
            if self.current_step < len(self.STEPS) - 1:
                self._goto_step(self.current_step + 1)
        elif button_id == "btn_submit":
            self._submit_form()
        elif button_id == "btn_add_condition":
            self._add_list_item("conditions", "input_condition", "conditions_list")
        elif button_id == "btn_add_allergy":
            self._add_list_item("allergies", "input_allergy", "allergies_list")
        elif button_id == "btn_add_symptom":
            self._add_list_item("symptoms", "input_symptom", "symptoms_list")
        elif button_id == "btn_add_medication":
            self._add_medication()
    
    def _add_list_item(self, list_key: str, input_id: str, list_id: str) -> None:
        """Add an item to a list."""
        try:
            input_widget = self.query_one(f"#{input_id}", Input)
            value = input_widget.value.strip()
            if value:
                self.form_data[list_key].append(value)
                input_widget.value = ""
                
                list_widget = self.query_one(f"#{list_id}", Static)
                list_widget.update(self._render_list_items(self.form_data[list_key], list_key))
        except Exception:
            pass
    
    def _add_medication(self) -> None:
        """Add a medication to the list."""
        try:
            name = self.query_one("#input_med_name", Input).value.strip()
            dosage = self.query_one("#input_med_dosage", Input).value.strip()
            frequency = self.query_one("#input_med_freq", Input).value.strip()
            
            if name:
                self.form_data["medications"].append({
                    "name": name,
                    "dosage": dosage,
                    "frequency": frequency,
                })
                
                self.query_one("#input_med_name", Input).value = ""
                self.query_one("#input_med_dosage", Input).value = ""
                self.query_one("#input_med_freq", Input).value = ""
                
                list_widget = self.query_one("#medications_list", Static)
                list_widget.update(self._render_list_items(self.form_data["medications"], "medications"))
        except Exception:
            pass
    
    def _submit_form(self) -> None:
        """Submit the form and create patient/case."""
        self._save_current_step()
        
        store = get_data_store()
        
        # Generate IDs
        patient_id = store.patients.generate_mrn().replace("BSG-", "P-")
        mrn = store.patients.generate_mrn()
        case_number = store.cases.generate_case_id()
        
        # Parse date of birth
        dob = self.form_data["date_of_birth"]
        if dob:
            try:
                parts = dob.split("-")
                dob = f"{parts[0]}-{parts[1]}-{parts[2]}"
            except (ValueError, IndexError):
                dob = "2000-01-01"
        else:
            dob = "2000-01-01"
        
        # Create vitals
        vitals = {}
        if self.form_data["temperature"]:
            try:
                vitals["temperature"] = float(self.form_data["temperature"])
            except ValueError:
                pass
        if self.form_data["heart_rate"]:
            try:
                vitals["heart_rate"] = int(self.form_data["heart_rate"])
            except ValueError:
                pass
        if self.form_data["bp_systolic"]:
            try:
                vitals["blood_pressure_systolic"] = int(self.form_data["bp_systolic"])
            except ValueError:
                pass
        if self.form_data["bp_diastolic"]:
            try:
                vitals["blood_pressure_diastolic"] = int(self.form_data["bp_diastolic"])
            except ValueError:
                pass
        if self.form_data["respiratory_rate"]:
            try:
                vitals["respiratory_rate"] = int(self.form_data["respiratory_rate"])
            except ValueError:
                pass
        if self.form_data["oxygen_saturation"]:
            try:
                vitals["oxygen_saturation"] = float(self.form_data["oxygen_saturation"])
            except ValueError:
                pass
        if self.form_data["weight"]:
            try:
                vitals["weight"] = float(self.form_data["weight"])
            except ValueError:
                pass
        if self.form_data["height"]:
            try:
                vitals["height"] = float(self.form_data["height"])
            except ValueError:
                pass
        
        if vitals:
            vitals["recorded_at"] = datetime.utcnow().isoformat()
        
        # Create patient data
        patient_data = {
            "mrn": mrn,
            "name": self.form_data["name"] or "Unknown Patient",
            "date_of_birth": dob,
            "gender": self.form_data["gender"],
            "blood_type": self.form_data["blood_type"],
            "email": self.form_data["email"],
            "phone": self.form_data["phone"],
            "address": self.form_data["address"],
            "emergency_contact": {
                "name": self.form_data["emergency_name"],
                "phone": self.form_data["emergency_phone"],
                "relationship": self.form_data["emergency_relationship"],
            } if self.form_data["emergency_name"] else None,
            "conditions": [{"name": c, "status": "active"} for c in self.form_data["conditions"]],
            "allergies": [{"allergen": a, "reaction": "Unknown", "severity": "moderate"} for a in self.form_data["allergies"]],
            "medications": self.form_data["medications"],
            "vitals_history": [vitals] if vitals else [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Save patient
        store.patients.save_sync(patient_id, patient_data)
        
        # Create case data
        case_data = {
            "case_number": case_number,
            "patient_id": patient_id,
            "patient_name": patient_data["name"],
            "chief_complaint": self.form_data["chief_complaint"] or "Not specified",
            "symptoms": self.form_data["symptoms"],
            "status": "open",
            "priority": "medium",
            "diagnoses": [],
            "test_recommendations": [],
            "treatment_recommendations": [],
            "notes": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Save case
        case_id = case_number.replace("CASE-", "C-")
        store.cases.save_sync(case_id, case_data)
        
        # Log audit event
        store.audit.log_event(
            event_type="patient_created",
            user="system",
            action=f"Created patient {mrn} and case {case_number}",
            details={"patient_id": patient_id, "case_id": case_id},
        )
        
        # Navigate to diagnosis progress screen
        from biosage_terminal.screens.diagnosis_progress import DiagnosisProgressScreen
        diagnosis_screen = DiagnosisProgressScreen(case_id=case_id, patient_id=patient_id)
        self.app.push_screen(diagnosis_screen)
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
