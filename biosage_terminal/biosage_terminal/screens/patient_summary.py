"""
Patient Summary screen for BioSage Terminal.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Footer, TabbedContent, TabPane, DataTable
from textual.binding import Binding
from rich.text import Text
from datetime import date

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import format_progress_bar


class PatientSummaryScreen(Screen):
    """Detailed patient summary with tabs for different data."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("d", "goto_dashboard", "Dashboard"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    PatientSummaryScreen {
        background: #0F172A;
    }
    
    .patient-header {
        dock: top;
        height: 3;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 0 2;
    }
    
    .demographics-box {
        background: #1E293B;
        border: solid #334155;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }
    
    .tab-content {
        height: 1fr;
        padding: 1;
    }
    
    .vitals-grid {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    .vital-card {
        height: 8;
        background: #1E293B;
        border: solid #334155;
        padding: 1;
        text-align: center;
    }
    
    .timeline-container {
        padding: 1;
    }
    
    .timeline-event {
        padding-left: 4;
        border-left: solid #334155;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        color: #F1F5F9;
        margin: 1 0;
    }
    """
    
    def __init__(self, patient_id: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patient_id = patient_id
        self.patient_data = {}
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="patient-header")
        
        with ScrollableContainer():
            yield Static(self._render_demographics(), id="demographics", classes="demographics-box")
            
            with TabbedContent():
                with TabPane("Overview", id="tab_overview"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(self._render_overview(), id="overview_content")
                
                with TabPane("Vitals", id="tab_vitals"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(self._render_vitals(), id="vitals_content")
                
                with TabPane("Labs", id="tab_labs"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(self._render_labs(), id="labs_content")
                
                with TabPane("History", id="tab_history"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(self._render_history(), id="history_content")
                
                with TabPane("Timeline", id="tab_timeline"):
                    with ScrollableContainer(classes="tab-content"):
                        yield Static(self._render_timeline(), id="timeline_content")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Load patient data when screen is mounted."""
        self._load_patient_data()
    
    def _load_patient_data(self) -> None:
        """Load patient data from storage."""
        store = get_data_store()
        self.patient_data = store.patients.load_sync(self.patient_id) or {}
        self._update_displays()
    
    def _update_displays(self) -> None:
        """Update all display widgets."""
        try:
            self.query_one("#demographics", Static).update(self._render_demographics())
            self.query_one("#overview_content", Static).update(self._render_overview())
            self.query_one("#vitals_content", Static).update(self._render_vitals())
            self.query_one("#labs_content", Static).update(self._render_labs())
            self.query_one("#history_content", Static).update(self._render_history())
            self.query_one("#timeline_content", Static).update(self._render_timeline())
        except Exception:
            pass
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("Patient Summary", style="bold #F1F5F9")
        return text
    
    def _render_demographics(self) -> Text:
        """Render the demographics box."""
        text = Text()
        
        name = self.patient_data.get("name", "Unknown")
        mrn = self.patient_data.get("mrn", "N/A")
        gender = self.patient_data.get("gender", "Unknown")
        blood_type = self.patient_data.get("blood_type", "Unknown")
        email = self.patient_data.get("email", "N/A")
        phone = self.patient_data.get("phone", "N/A")
        address = self.patient_data.get("address", "N/A")
        
        # Calculate age
        age = self._calculate_age()
        
        # Row 1
        text.append("Name: ", style="bold #94A3B8")
        text.append(f"{name:25}", style="#F1F5F9")
        text.append("Age: ", style="bold #94A3B8")
        text.append(f"{age:10}", style="#F1F5F9")
        text.append("Gender: ", style="bold #94A3B8")
        text.append(f"{gender}\n", style="#F1F5F9")
        
        # Row 2
        text.append("MRN: ", style="bold #94A3B8")
        text.append(f"{mrn:26}", style="#F1F5F9")
        text.append("Blood Type: ", style="bold #94A3B8")
        text.append(f"{blood_type}\n", style="#F1F5F9")
        
        # Row 3
        text.append("Phone: ", style="bold #94A3B8")
        text.append(f"{phone:24}", style="#F1F5F9")
        text.append("Email: ", style="bold #94A3B8")
        text.append(f"{email}\n", style="#F1F5F9")
        
        # Row 4
        text.append("Address: ", style="bold #94A3B8")
        text.append(f"{address}\n", style="#F1F5F9")
        
        # Emergency contact
        emergency = self.patient_data.get("emergency_contact", {})
        if emergency and emergency.get("name"):
            text.append("Emergency: ", style="bold #94A3B8")
            text.append(
                f"{emergency.get('name', '')} ({emergency.get('relationship', '')}) - {emergency.get('phone', '')}",
                style="#F1F5F9"
            )
        
        return text
    
    def _calculate_age(self) -> int:
        """Calculate patient's age."""
        dob = self.patient_data.get("date_of_birth", "")
        if not dob:
            return 0
        try:
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
    
    def _render_overview(self) -> Text:
        """Render the overview tab."""
        text = Text()
        
        text.append("=== Patient Overview ===\n\n", style="bold #F1F5F9")
        
        # Active conditions
        conditions = self.patient_data.get("conditions", [])
        text.append("Active Conditions\n", style="bold #3B82F6")
        text.append("-" * 30 + "\n", style="#334155")
        if conditions:
            for cond in conditions:
                name = cond.get("name", "Unknown")
                status = cond.get("status", "active")
                status_color = "#22C55E" if status == "active" else "#64748B"
                text.append(f"  - {name}", style="#F1F5F9")
                text.append(f" [{status}]\n", style=status_color)
        else:
            text.append("  No conditions recorded\n", style="#64748B")
        
        text.append("\n", style="")
        
        # Allergies
        allergies = self.patient_data.get("allergies", [])
        text.append("Allergies\n", style="bold #EF4444")
        text.append("-" * 30 + "\n", style="#334155")
        if allergies:
            for allergy in allergies:
                allergen = allergy.get("allergen", "Unknown")
                reaction = allergy.get("reaction", "")
                severity = allergy.get("severity", "moderate")
                severity_colors = {
                    "mild": "#22C55E",
                    "moderate": "#EAB308",
                    "severe": "#EF4444",
                }
                text.append(f"  - {allergen}", style="#F1F5F9")
                if reaction:
                    text.append(f": {reaction}", style="#94A3B8")
                text.append(f" [{severity}]\n", style=severity_colors.get(severity, "#94A3B8"))
        else:
            text.append("  NKDA (No Known Drug Allergies)\n", style="#22C55E")
        
        text.append("\n", style="")
        
        # Recent cases
        store = get_data_store()
        cases = store.cases.find_by_patient_sync(self.patient_id)
        
        text.append("Recent Cases\n", style="bold #3B82F6")
        text.append("-" * 30 + "\n", style="#334155")
        if cases:
            for case in cases[:5]:
                case_num = case.get("case_number", "")
                status = case.get("status", "open")
                diagnosis = case.get("primary_diagnosis", "Pending")
                status_colors = {
                    "open": "#EAB308",
                    "in_progress": "#3B82F6",
                    "diagnosed": "#22C55E",
                    "closed": "#64748B",
                }
                text.append(f"  {case_num}: ", style="#F1F5F9")
                text.append(f"{diagnosis}", style="#94A3B8")
                text.append(f" [{status}]\n", style=status_colors.get(status, "#64748B"))
        else:
            text.append("  No cases recorded\n", style="#64748B")
        
        return text
    
    def _render_vitals(self) -> Text:
        """Render the vitals tab."""
        text = Text()
        
        text.append("=== Current Vital Signs ===\n\n", style="bold #F1F5F9")
        
        vitals_history = self.patient_data.get("vitals_history", [])
        
        if not vitals_history:
            text.append("No vitals recorded\n", style="#64748B")
            return text
        
        # Get most recent vitals
        latest = vitals_history[-1] if vitals_history else {}
        
        vitals = [
            ("Temperature", latest.get("temperature"), "F", (97.0, 99.0)),
            ("Heart Rate", latest.get("heart_rate"), "bpm", (60, 100)),
            ("Resp. Rate", latest.get("respiratory_rate"), "/min", (12, 20)),
            ("BP Systolic", latest.get("blood_pressure_systolic"), "mmHg", (90, 140)),
            ("BP Diastolic", latest.get("blood_pressure_diastolic"), "mmHg", (60, 90)),
            ("SpO2", latest.get("oxygen_saturation"), "%", (95, 100)),
            ("Weight", latest.get("weight"), "kg", None),
            ("Height", latest.get("height"), "cm", None),
        ]
        
        for name, value, unit, normal_range in vitals:
            if value is not None:
                # Determine status
                status = "normal"
                if normal_range:
                    if value < normal_range[0] or value > normal_range[1]:
                        status = "abnormal"
                    if normal_range[0] and value < normal_range[0] * 0.8:
                        status = "critical"
                    if normal_range[1] and value > normal_range[1] * 1.2:
                        status = "critical"
                
                status_colors = {
                    "normal": "#22C55E",
                    "abnormal": "#EAB308",
                    "critical": "#EF4444",
                }
                
                text.append(f"{name:15}", style="#94A3B8")
                text.append(f"{value}", style=f"bold {status_colors[status]}")
                text.append(f" {unit}\n", style="#64748B")
        
        text.append("\n", style="")
        
        # Vitals history
        if len(vitals_history) > 1:
            text.append("Vitals History\n", style="bold #3B82F6")
            text.append("-" * 30 + "\n", style="#334155")
            for vital in vitals_history[-5:]:
                recorded = vital.get("recorded_at", "")[:16]
                temp = vital.get("temperature", "-")
                hr = vital.get("heart_rate", "-")
                text.append(f"  {recorded}: Temp {temp}F, HR {hr} bpm\n", style="#94A3B8")
        
        return text
    
    def _render_labs(self) -> Text:
        """Render the labs tab."""
        text = Text()
        
        text.append("=== Laboratory Results ===\n\n", style="bold #F1F5F9")
        
        lab_results = self.patient_data.get("lab_results", [])
        
        if not lab_results:
            text.append("No lab results recorded\n", style="#64748B")
            return text
        
        # Header
        text.append(f"{'Test':<25} {'Value':<15} {'Reference':<15} {'Status':<10}\n", style="bold #94A3B8")
        text.append("-" * 65 + "\n", style="#334155")
        
        for result in lab_results:
            test_name = result.get("test_name", "Unknown")[:24]
            value = f"{result.get('value', '-')} {result.get('unit', '')}"[:14]
            reference = result.get("reference_range", "-")[:14]
            status = result.get("status", "normal")
            
            status_colors = {
                "normal": "#22C55E",
                "abnormal": "#EAB308",
                "critical": "#EF4444",
            }
            status_display = f"[{status.upper()}]"
            
            text.append(f"{test_name:<25} {value:<15} {reference:<15}", style="#F1F5F9")
            text.append(f"{status_display}\n", style=status_colors.get(status, "#64748B"))
        
        return text
    
    def _render_history(self) -> Text:
        """Render the history tab."""
        text = Text()
        
        text.append("=== Medical History ===\n\n", style="bold #F1F5F9")
        
        # Medications
        medications = self.patient_data.get("medications", [])
        text.append("[-] Medications", style="bold #F1F5F9")
        text.append(f" ({len(medications)})\n", style="#64748B")
        if medications:
            for med in medications:
                name = med.get("name", "Unknown")
                dosage = med.get("dosage", "")
                frequency = med.get("frequency", "")
                text.append(f"    |-- {name}", style="#F1F5F9")
                if dosage:
                    text.append(f" {dosage}", style="#94A3B8")
                if frequency:
                    text.append(f" - {frequency}", style="#64748B")
                text.append("\n", style="")
        else:
            text.append("    No medications\n", style="#64748B")
        
        text.append("\n", style="")
        
        # Conditions
        conditions = self.patient_data.get("conditions", [])
        text.append("[-] Medical Conditions", style="bold #F1F5F9")
        text.append(f" ({len(conditions)})\n", style="#64748B")
        if conditions:
            for cond in conditions:
                name = cond.get("name", "Unknown")
                diagnosed_date = cond.get("diagnosed_date", "")
                text.append(f"    |-- {name}", style="#F1F5F9")
                if diagnosed_date:
                    text.append(f" (diagnosed: {diagnosed_date})", style="#64748B")
                text.append("\n", style="")
        else:
            text.append("    No conditions recorded\n", style="#64748B")
        
        text.append("\n", style="")
        
        # Allergies
        allergies = self.patient_data.get("allergies", [])
        text.append("[-] Allergies", style="bold #EF4444")
        text.append(f" ({len(allergies)})\n", style="#64748B")
        if allergies:
            for allergy in allergies:
                allergen = allergy.get("allergen", "Unknown")
                reaction = allergy.get("reaction", "")
                text.append(f"    |-- {allergen}", style="#F1F5F9")
                if reaction:
                    text.append(f": {reaction}", style="#94A3B8")
                text.append("\n", style="")
        else:
            text.append("    NKDA\n", style="#22C55E")
        
        return text
    
    def _render_timeline(self) -> Text:
        """Render the timeline tab."""
        text = Text()
        
        text.append("=== Patient Timeline ===\n\n", style="bold #F1F5F9")
        
        # Build timeline from various events
        events = []
        
        # Patient creation
        created = self.patient_data.get("created_at", "")
        if created:
            events.append((created, "Patient record created", ""))
        
        # Vitals recordings
        for vital in self.patient_data.get("vitals_history", []):
            recorded = vital.get("recorded_at", "")
            if recorded:
                temp = vital.get("temperature", "")
                hr = vital.get("heart_rate", "")
                details = f"Temp: {temp}F, HR: {hr} bpm" if temp and hr else "Vitals recorded"
                events.append((recorded, "Vitals recorded", details))
        
        # Cases
        store = get_data_store()
        cases = store.cases.find_by_patient_sync(self.patient_id)
        for case in cases:
            created = case.get("created_at", "")
            if created:
                events.append((created, f"Case {case.get('case_number', '')} opened", 
                             case.get("chief_complaint", "")))
            if case.get("status") == "diagnosed":
                updated = case.get("updated_at", "")
                if updated:
                    events.append((updated, f"Case {case.get('case_number', '')} diagnosed",
                                 case.get("primary_diagnosis", "")))
        
        # Sort by timestamp
        events.sort(key=lambda x: x[0], reverse=True)
        
        if not events:
            text.append("No timeline events\n", style="#64748B")
            return text
        
        # Render timeline
        for timestamp, title, details in events[:20]:
            text.append("   |\n", style="#334155")
            text.append("   ", style="")
            text.append("[*]", style="bold #3B82F6")
            text.append(f"--- {timestamp[:16]}: ", style="#F1F5F9")
            text.append(f"{title}\n", style="bold #F1F5F9")
            if details:
                text.append(f"   |    {details}\n", style="#94A3B8")
        
        text.append("   |\n", style="#334155")
        text.append("   V\n", style="#334155")
        
        return text
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.pop_screen()
        self.app.push_screen("dashboard")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
