"""
Case View screen for BioSage Terminal.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Footer, TabbedContent, TabPane, DataTable
from textual.binding import Binding
from rich.text import Text

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import (
    SPECIALTY_ICONS, SPECIALTY_COLORS, PRIORITY_COLORS,
    format_progress_bar, format_confidence_bar,
)


class CaseViewScreen(Screen):
    """Case view with tabs for overview, specialists, evidence, tests, and recommendations."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("r", "run_diagnosis", "Run Diagnosis"),
        Binding("t", "goto_tests", "Order Tests"),
        Binding("p", "goto_patient", "Patient Summary"),
        Binding("c", "goto_collaboration", "Collaborate"),
        Binding("h", "goto_research", "Research"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    CaseViewScreen {
        background: #0F172A;
    }
    
    .case-header {
        dock: top;
        height: 3;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 0 2;
    }
    
    .patient-info-bar {
        height: 5;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 1;
    }
    
    .tab-content {
        height: 1fr;
        padding: 1;
    }
    
    .diagnosis-card {
        background: #1E293B;
        border: solid #3B82F6;
        padding: 2;
        margin: 1;
    }
    
    .diagnosis-card.primary {
        border: solid #22C55E;
    }
    
    .specialist-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    .specialist-card {
        height: 15;
        background: #1E293B;
        border: solid #334155;
        padding: 1;
    }
    
    .evidence-item {
        background: #0F172A;
        border: solid #334155;
        padding: 1;
        margin: 1 0;
    }
    
    .test-item {
        background: #1E293B;
        border: solid #334155;
        padding: 1;
        margin: 1 0;
    }
    
    .recommendation-item {
        background: #1E293B;
        padding: 1;
        margin: 1 0;
    }
    
    .section-title {
        text-style: bold;
        color: #F1F5F9;
        margin-bottom: 1;
    }
    
    .no-data {
        color: #64748B;
        text-align: center;
        padding: 4;
    }
    """
    
    def __init__(self, case_id: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_id = case_id
        self.case_data = {}
        self.patient_data = {}
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="case-header")
        yield Static(self._render_patient_info(), id="patient_info", classes="patient-info-bar")
        
        with TabbedContent():
            with TabPane("Overview", id="tab_overview"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_overview(), id="overview_content")
            
            with TabPane("Specialists", id="tab_specialists"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_specialists(), id="specialists_content")
            
            with TabPane("Evidence", id="tab_evidence"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_evidence(), id="evidence_content")
            
            with TabPane("Tests", id="tab_tests"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_tests(), id="tests_content")
            
            with TabPane("Recommendations", id="tab_recommendations"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_recommendations(), id="recommendations_content")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Load case data when screen is mounted."""
        self._load_case_data()
    
    def _load_case_data(self) -> None:
        """Load case and patient data from storage."""
        store = get_data_store()
        
        self.case_data = store.cases.load_sync(self.case_id) or {}
        if self.case_data:
            patient_id = self.case_data.get("patient_id", "")
            self.patient_data = store.patients.load_sync(patient_id) or {}
        
        # Update displays
        self._update_displays()
    
    def _update_displays(self) -> None:
        """Update all display widgets with loaded data."""
        try:
            self.query_one("#patient_info", Static).update(self._render_patient_info())
            self.query_one("#overview_content", Static).update(self._render_overview())
            self.query_one("#specialists_content", Static).update(self._render_specialists())
            self.query_one("#evidence_content", Static).update(self._render_evidence())
            self.query_one("#tests_content", Static).update(self._render_tests())
            self.query_one("#recommendations_content", Static).update(self._render_recommendations())
        except Exception:
            pass
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("Case View", style="bold #F1F5F9")
        text.append(" | ", style="#334155")
        text.append("[r] Diagnosis  [t] Tests  [p] Patient", style="#94A3B8")
        return text
    
    def _render_patient_info(self) -> Text:
        """Render patient information bar."""
        text = Text()
        
        name = self.patient_data.get("name", "Unknown")
        mrn = self.patient_data.get("mrn", "N/A")
        case_num = self.case_data.get("case_number", self.case_id)
        complaint = self.case_data.get("chief_complaint", "Not specified")
        
        # Calculate age
        age = 0
        dob = self.patient_data.get("date_of_birth", "")
        if dob:
            try:
                from datetime import date
                parts = dob.split("-")
                if len(parts) >= 3:
                    dob_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
                    today = date.today()
                    age = today.year - dob_date.year - (
                        (today.month, today.day) < (dob_date.month, dob_date.day)
                    )
            except (ValueError, IndexError):
                pass
        
        text.append("Name: ", style="bold #94A3B8")
        text.append(f"{name}", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append("Age: ", style="bold #94A3B8")
        text.append(f"{age}", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append("MRN: ", style="bold #94A3B8")
        text.append(f"{mrn}", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append("Case: ", style="bold #94A3B8")
        text.append(f"{case_num}\n", style="#F1F5F9")
        text.append("Chief Complaint: ", style="bold #94A3B8")
        text.append(complaint, style="#F1F5F9")
        
        return text
    
    def _render_overview(self) -> Text:
        """Render the overview tab content."""
        text = Text()
        
        diagnoses = self.case_data.get("diagnoses", [])
        status = self.case_data.get("status", "open")
        
        # Status
        status_colors = {
            "open": "#EAB308",
            "in_progress": "#3B82F6",
            "diagnosed": "#22C55E",
            "closed": "#64748B",
        }
        status_color = status_colors.get(status, "#64748B")
        text.append("Status: ", style="bold #94A3B8")
        text.append(f"[{status.upper()}]\n\n", style=f"bold {status_color}")
        
        # Debate statistics if available
        debate_stats = self.case_data.get("debate_statistics", {})
        if debate_stats:
            text.append("=== ARGUS Debate Summary ===\n", style="bold #A855F7")
            text.append(f"Debate Rounds: {debate_stats.get('num_rounds', 0)}\n", style="#94A3B8")
            text.append(f"Evidence Items: {debate_stats.get('num_evidence', 0)}\n", style="#94A3B8")
            text.append(f"Rebuttals: {debate_stats.get('num_rebuttals', 0)}\n", style="#94A3B8")
            text.append(f"Duration: {debate_stats.get('duration_seconds', 0):.1f}s\n\n", style="#94A3B8")
        
        if not diagnoses:
            text.append("No diagnoses generated yet.\n", style="#64748B")
            text.append("Press [r] to run AI diagnosis.\n", style="#94A3B8")
            return text
        
        # Verdict label
        verdict_label = self.case_data.get("verdict_label", "undecided")
        verdict_colors = {
            "supported": "#22C55E",
            "rejected": "#EF4444",
            "undecided": "#EAB308",
        }
        verdict_color = verdict_colors.get(verdict_label, "#EAB308")
        text.append("Jury Verdict: ", style="bold #94A3B8")
        text.append(f"[{verdict_label.upper()}]\n\n", style=f"bold {verdict_color}")
        
        # Primary diagnosis
        text.append("=== Primary Diagnosis ===\n\n", style="bold #F1F5F9")
        
        primary_diag = self.case_data.get("primary_diagnosis", "")
        primary_conf = self.case_data.get("primary_confidence", 0) or 0
        
        if primary_diag:
            text.append(f"{primary_diag}\n", style="bold #22C55E")
            conf_bar = format_confidence_bar(primary_conf)
            text.append(f"{conf_bar}\n\n", style="#F1F5F9")
        
        # Reasoning
        reasoning = self.case_data.get("reasoning", "")
        if reasoning:
            text.append("=== Clinical Reasoning ===\n\n", style="bold #F1F5F9")
            text.append(f"{reasoning[:500]}...\n\n" if len(reasoning) > 500 else f"{reasoning}\n\n", style="#94A3B8")
        
        # Other diagnoses
        text.append("=== Differential Diagnoses ===\n\n", style="bold #F1F5F9")
        
        for diag in diagnoses[:5]:
            diagnosis = diag.get("diagnosis", "Unknown")
            posterior = diag.get("posterior", diag.get("confidence", 0))
            reasoning_text = diag.get("reasoning", diag.get("rationale", ""))
            label = diag.get("label", "undecided")
            
            # Label indicator
            label_color = verdict_colors.get(label, "#EAB308")
            text.append(f"[{label[:3].upper()}] ", style=f"bold {label_color}")
            text.append(f"{diagnosis}\n", style="bold #F1F5F9")
            
            conf_bar = format_progress_bar(posterior, width=20)
            text.append(f"   {conf_bar} {int(posterior * 100)}%\n", style="#94A3B8")
            
            if reasoning_text:
                text.append(f"   {reasoning_text[:80]}...\n", style="#64748B")
            
            text.append("\n", style="")
        
        return text
    
    def _render_specialists(self) -> Text:
        """Render the specialists tab content showing ARGUS agent contributions."""
        text = Text()
        
        text.append("=== ARGUS Specialist Contributions ===\n\n", style="bold #A855F7")
        
        specialist_contributions = self.case_data.get("specialist_contributions", {})
        
        if not specialist_contributions:
            text.append("No specialist analysis available yet.\n", style="#64748B")
            text.append("Run diagnosis to get specialist input.\n", style="#94A3B8")
            return text
        
        for specialty, data in specialist_contributions.items():
            icon = SPECIALTY_ICONS.get(specialty.lower(), "[SPE]")
            color = SPECIALTY_COLORS.get(specialty.lower(), "#3B82F6")
            
            text.append(f"\n{icon} {specialty.upper()} Specialist\n", style=f"bold {color}")
            text.append("-" * 40 + "\n", style="#334155")
            
            evidence_count = data.get("evidence_count", 0)
            propositions_supported = data.get("propositions_supported", [])
            
            text.append(f"  Evidence Items: {evidence_count}\n", style="#94A3B8")
            
            if propositions_supported:
                text.append("  Supported Diagnoses:\n", style="#94A3B8")
                for prop in propositions_supported[:3]:
                    text.append(f"    - {prop}\n", style="#22C55E")
            
            # Show a sample of the specialist's reasoning
            reasoning = data.get("reasoning", "")
            if reasoning:
                text.append(f"  Analysis: {reasoning[:120]}...\n", style="#64748B")
            
            text.append("\n", style="")
        
        # Also show debate agent summary
        text.append("=== Debate Agents ===\n\n", style="bold #F1F5F9")
        
        text.append("[MOD] Moderator: ", style="bold #A855F7")
        text.append("Orchestrated debate agenda\n", style="#94A3B8")
        
        rebuttals = self.case_data.get("rebuttals", [])
        text.append("[REF] Refuter: ", style="bold #EF4444")
        text.append(f"Generated {len(rebuttals)} rebuttals\n", style="#94A3B8")
        
        verdict_label = self.case_data.get("verdict_label", "undecided")
        text.append("[JUR] Jury: ", style="bold #22C55E")
        text.append(f"Rendered verdict: {verdict_label.upper()}\n", style="#94A3B8")
        
        return text
    
    def _render_evidence(self) -> Text:
        """Render the evidence tab content with ARGUS evidence and rebuttals."""
        text = Text()
        
        # Show ARGUS evidence items first (from debate)
        evidence_items = self.case_data.get("evidence_items", [])
        rebuttals = self.case_data.get("rebuttals", [])
        
        text.append("=== ARGUS Evidence (from Debate) ===\n\n", style="bold #22C55E")
        
        if evidence_items:
            for ev in evidence_items[:10]:
                specialist = ev.get("specialist", "unknown")
                target = ev.get("target_diagnosis", "")
                content = ev.get("content", "")
                polarity = ev.get("polarity", "supports")
                weight = ev.get("weight", 0.5)
                
                icon = SPECIALTY_ICONS.get(specialist.lower(), "[EV]")
                color = SPECIALTY_COLORS.get(specialist.lower(), "#22C55E")
                
                polarity_color = "#22C55E" if polarity == "supports" else "#EF4444"
                polarity_icon = "[+]" if polarity == "supports" else "[-]"
                
                text.append(f"{icon} ", style=f"bold {color}")
                text.append(f"{specialist.upper()} ", style=f"bold {color}")
                text.append(f"{polarity_icon} ", style=f"bold {polarity_color}")
                text.append(f"{target}\n", style="bold #F1F5F9")
                text.append(f"   Weight: {weight:.2f}\n", style="#94A3B8")
                if content:
                    text.append(f"   {content[:120]}...\n" if len(content) > 120 else f"   {content}\n", style="#64748B")
                text.append("\n", style="")
        else:
            text.append("No evidence gathered yet.\n", style="#64748B")
        
        # Show rebuttals from Refuter
        text.append("=== Rebuttals (from Refuter) ===\n\n", style="bold #EF4444")
        
        if rebuttals:
            for reb in rebuttals[:5]:
                target = reb.get("target_diagnosis", reb.get("target", ""))
                content = reb.get("content", "")
                strength = reb.get("strength", 0.5)
                
                text.append("[REF] ", style="bold #EF4444")
                text.append(f"Challenges: {target}\n", style="bold #F1F5F9")
                text.append(f"   Strength: {strength:.2f}\n", style="#94A3B8")
                if content:
                    text.append(f"   {content[:150]}...\n" if len(content) > 150 else f"   {content}\n", style="#64748B")
                text.append("\n", style="")
        else:
            text.append("No rebuttals generated.\n", style="#64748B")
        
        # Also show stored evidence if any
        store = get_data_store()
        stored_evidence = store.evidence.find_by_case_sync(self.case_id)
        
        if stored_evidence:
            text.append("=== Stored Citations ===\n\n", style="bold #06B6D4")
            for ev in stored_evidence[:5]:
                title = ev.get("title", "Untitled")
                content = ev.get("content", "")
                score = ev.get("relevance_score", 0)
                
                text.append(f"[CIT] {title}\n", style="bold #06B6D4")
                text.append(f"   Relevance: {int(score * 100)}%\n", style="#94A3B8")
                if content:
                    text.append(f"   {content[:100]}...\n", style="#64748B")
                text.append("\n", style="")
        
        return text
    
    def _render_tests(self) -> Text:
        """Render the tests tab content."""
        text = Text()
        
        text.append("=== Recommended Tests ===\n\n", style="bold #F1F5F9")
        
        tests = self.case_data.get("test_recommendations", [])
        
        if not tests:
            text.append("No tests recommended yet.\n", style="#64748B")
            text.append("Tests will be recommended after diagnosis.\n", style="#94A3B8")
            return text
        
        for i, test in enumerate(tests, 1):
            name = test.get("test_name", "Unknown Test")
            rationale = test.get("rationale", "")
            priority = test.get("priority", "medium")
            cost = test.get("estimated_cost")
            turnaround = test.get("turnaround_time", "")
            
            priority_color = PRIORITY_COLORS.get(priority.lower(), "#94A3B8")
            
            text.append(f"{i}. {name}", style="bold #F1F5F9")
            text.append(f" [{priority.upper()}]\n", style=f"bold {priority_color}")
            
            if cost is not None or turnaround:
                text.append("   ", style="")
                if cost is not None:
                    text.append(f"${cost:.0f}", style="#22C55E")
                if cost is not None and turnaround:
                    text.append(" | ", style="#334155")
                if turnaround:
                    text.append(turnaround, style="#64748B")
                text.append("\n", style="")
            
            if rationale:
                text.append(f"   {rationale}\n", style="#94A3B8")
            
            text.append("\n", style="")
        
        return text
    
    def _render_recommendations(self) -> Text:
        """Render the recommendations tab content."""
        text = Text()
        
        text.append("=== Treatment Recommendations ===\n\n", style="bold #F1F5F9")
        
        recommendations = self.case_data.get("treatment_recommendations", [])
        
        if not recommendations:
            text.append("No recommendations generated yet.\n", style="#64748B")
            text.append("Recommendations will be provided after diagnosis.\n", style="#94A3B8")
            return text
        
        for rec in recommendations:
            recommendation = rec.get("recommendation", "")
            rationale = rec.get("rationale", "")
            priority = rec.get("priority", "medium")
            specialist = rec.get("specialist", "")
            
            priority_color = PRIORITY_COLORS.get(priority.lower(), "#94A3B8")
            
            # Priority indicator
            text.append(f"[{priority.upper()}] ", style=f"bold {priority_color}")
            text.append(f"{recommendation}\n", style="bold #F1F5F9")
            
            if specialist:
                icon = SPECIALTY_ICONS.get(specialist.lower(), "")
                color = SPECIALTY_COLORS.get(specialist.lower(), "#94A3B8")
                text.append(f"   From: {icon} {specialist.title()}\n", style=color)
            
            if rationale:
                text.append(f"   {rationale}\n", style="#94A3B8")
            
            text.append("\n", style="")
        
        return text
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_run_diagnosis(self) -> None:
        """Run AI diagnosis for this case."""
        from biosage_terminal.screens.diagnosis_progress import DiagnosisProgressScreen
        
        # Get patient_id from case data
        store = get_data_store()
        case_data = store.cases.load_sync(self.case_id)
        patient_id = case_data.get("patient_id", "") if case_data else ""
        
        diagnosis_screen = DiagnosisProgressScreen(case_id=self.case_id, patient_id=patient_id)
        self.app.push_screen(diagnosis_screen)
    
    def action_goto_tests(self) -> None:
        """Navigate to tests and orders screen."""
        from biosage_terminal.screens.tests_orders import TestsOrdersScreen
        tests_screen = TestsOrdersScreen(case_id=self.case_id)
        self.app.push_screen(tests_screen)
    
    def action_goto_patient(self) -> None:
        """Navigate to patient summary screen."""
        patient_id = self.case_data.get("patient_id", "")
        if patient_id:
            from biosage_terminal.screens.patient_summary import PatientSummaryScreen
            patient_screen = PatientSummaryScreen(patient_id=patient_id)
            self.app.push_screen(patient_screen)
        else:
            self.notify("No patient linked to this case", severity="warning")
    
    def action_goto_collaboration(self) -> None:
        """Navigate to collaboration room for this case."""
        from biosage_terminal.screens.collaboration_room import CollaborationRoomScreen
        collab_screen = CollaborationRoomScreen(case_id=self.case_id)
        self.app.push_screen(collab_screen)
    
    def action_goto_research(self) -> None:
        """Navigate to research hub for this case."""
        from biosage_terminal.screens.research_hub import ResearchHubScreen
        research_screen = ResearchHubScreen(case_id=self.case_id)
        self.app.push_screen(research_screen)
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
